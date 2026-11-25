"""Unit tests on synthetic data (no downloads needed):  python -m pytest -q"""
import numpy as np
import pandas as pd

from smhi_trv_wind import Settings
from smhi_trv_wind.geo import sector, wrap90, wrap180
from smhi_trv_wind.model import combine, exposure, physics, site_factors

cfg = Settings()


def test_height_factor():
    assert abs(physics.height_factor(6.0) - 0.912) < 1e-3


def test_roughness_identities():
    # open terrain is unchanged; P and F are inverse for the same z0 at 10 m
    assert abs(physics.to_potential(0.03, cfg) - 1) < 1e-12
    assert abs(physics.to_site(0.03, cfg, z=10.0) - 1) < 1e-12
    for z0 in (0.001, 0.1, 1.0):
        assert abs(physics.to_potential(z0, cfg) * physics.to_site(z0, cfg, z=10.0) - 1) < 1e-12
    # rough upwind terrain -> station reads low -> potential wind is larger
    assert physics.to_potential(1.0, cfg) > 1 and physics.to_site(1.0, cfg) < physics.height_factor(6.0)


def test_angles():
    assert sector(359.0) == 0.0 and sector(12.0) == 22.5
    assert wrap180(350 - 10) == -20 and wrap90(170) == -10


def test_site_factor_recovery():
    rng = np.random.default_rng(0)
    n, ids, dirs = 4000, list("ABCDE"), np.arange(0, 360, 22.5)
    true = pd.DataFrame(rng.normal(0, 0.3, (5, 16)), index=ids, columns=dirs)
    true -= true.mean(0)
    wd0 = rng.uniform(0, 360, n)
    a = rng.normal(1.6, 0.3, n)
    k = sector(wd0)
    lnU = np.column_stack([a + true.loc[s].reindex(k).values + rng.normal(0, 0.1, n) for s in ids])
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    ws = pd.DataFrame(np.exp(lnU), index=idx, columns=ids)
    wd = pd.DataFrame(np.tile(wd0[:, None], 5), index=idx, columns=ids)
    sf = site_factors.fit(ws, wd, Settings(site_min_network_ws=0.0, site_min_stations=5, site_nsec=16))
    ok = sf.n.values > 30
    assert np.abs(sf.c.values[ok] - true.values[ok]).max() < 0.05


def test_exposure_recovery():
    rng = np.random.default_rng(1)
    n = 20000
    D = pd.DataFrame(dict(trv=rng.integers(0, 20, n).astype(str), tree=rng.uniform(0, 1, n), lnxh=rng.normal(2, 1, n),
                          lnU=rng.normal(1.5, 0.4, n), U3=rng.uniform(2, 10, n), U0=rng.uniform(2, 10, n),
                          wd_s=rng.uniform(0, 360, n), road=rng.uniform(0, 180, n)))
    b = np.array([-1.0, 0.2, 0.1, 0.3, -0.3])
    y = b[0] + b[1] * D.tree + b[2] * D.lnxh + (b[3] + b[4] * D.tree) * D.lnU + rng.normal(0, 0.05, n)
    D["ws"] = D.U3 * np.exp(y)
    D["wd"] = (D.wd_s - 12 - 25 * np.sin(np.radians(2 * wrap90(D.wd_s - D.road))) * D.tree + rng.normal(0, 3, n)) % 360
    c6 = Settings.round6()
    D["tree_dir"] = D.tree
    m = exposure.ExposureModel(c6).fit(D)
    assert np.allclose(m.b, b, atol=0.02)
    assert abs(m.d[0] + 12) < 0.5 and abs(m.d[1] + 25) < 0.5


def test_combination_weights():
    d = pd.Series({"a": 2.0, "b": 10.0, "c": 70.0})
    sel = combine.select(d, 60.0, 2)
    w = combine.weights(d, sel, cfg, 1.0)
    assert sel == ["a", "b"] and abs(w["a"] - (1 / 3) / (1 / 3 + 1 / 10)) < 1e-12


def test_round7_design_recovery():
    rng = np.random.default_rng(2)
    n = 20000
    D = pd.DataFrame(dict(trv=rng.integers(0, 20, n).astype(str), tree=rng.uniform(0, 1, n), tree_far=rng.uniform(0, 1, n),
                          lnxh=rng.normal(2, 1, n), lnU=rng.normal(1.5, 0.4, n), dT=rng.normal(0, 2, n),
                          U3=rng.uniform(2, 10, n), U0=rng.uniform(2, 10, n), wd_s=rng.uniform(0, 360, n), road=rng.uniform(0, 180, n)))
    D["tree_dir"] = D.tree
    b = np.array([-0.9, -0.3, 0.09, 0.28, 0.023, -0.15])     # b0, near, ln(x/h), lnU, dT, far
    y = b[0] + b[1] * D.tree + b[2] * D.lnxh + b[3] * D.lnU + b[4] * D.dT + b[5] * D.tree_far + rng.normal(0, 0.05, n)
    D["ws"] = D.U3 * np.exp(y); D["wd"] = D.wd_s
    c = Settings(use_stability=True)
    m = exposure.ExposureModel(c).fit(D)
    assert exposure.speed_terms(c) == ["b0", "b1_tree_near", "b2_lnxh", "b3_lnU", "b_dT", "b_tree_far"]
    assert exposure.speed_terms(Settings()) == ["b0", "b1_tree_near", "b2_lnxh", "b3_lnU", "b_tree_far"]
    assert np.allclose(m.b, b, atol=0.01)


def test_round6_preset():
    c = Settings.round6()
    assert c.tree_near_radius == 200 and c.tree_far_radius is None and c.lnU_tree_interaction and c.site_nsec == 16
