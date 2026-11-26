"""Evaluation helpers: cross-fitting, stuck-direction mask, leave-one-out neighbour, cross-fitted exposure model."""
import numpy as np

from smhi_trv_wind import assess


def test_crossfit_ratio_brute_force():
    rng = np.random.default_rng(0)
    H, S = 40, 5
    num, den = rng.uniform(1, 5, (H, S)), rng.uniform(1, 5, (H, S))
    num[3, 2] = np.nan
    blk = assess.blocks(H, 4)
    c = assess.crossfit_ratio(num, den, blk)
    for b in range(4):
        for j in range(S):
            m = (blk[:, None] != b) & (np.arange(S)[None, :] != j) & np.isfinite(num) & np.isfinite(den)
            assert np.isclose(c[np.where(blk == b)[0][0], j], num[m].sum() / den[m].sum())


def test_stuck_mask_run_length():
    ws = np.full((10, 1), 5.0)
    wd = np.array([10, 20, 30, 30, 30, 40, 50, 50, 60, 70], float)[:, None]
    st = assess.stuck_mask(ws, wd, run=3)
    assert st[:, 0].tolist() == [False, False, True, True, True, False, False, False, False, False]
    assert not assess.stuck_mask(np.full((10, 1), 1.0), wd, run=3).any()        # calm hours are not flagged


def test_nearest_other_never_self():
    rng = np.random.default_rng(1)
    ws, wd = rng.uniform(0, 10, (20, 4)), rng.uniform(0, 360, (20, 4))
    ws[:, 1] = np.nan                                                       # station 1 never reports
    D = assess.distance_matrix(np.array([60.0, 60.1, 60.2, 61.0]), np.array([10.0, 10.0, 10.0, 10.0]))
    U, W, dist = assess.nearest_other(ws, wd, D)
    assert np.allclose(U[:, 0], ws[:, 2])                                   # nearest with data is station 2, not 1
    for i in range(4):
        assert not np.any(np.isclose(dist[:, i], 0))


def test_crossfit_exposure_recovers_coefficients():
    rng = np.random.default_rng(2)
    H, S = 400, 12
    base = rng.uniform(2, 10, (H, S))
    feats = {"T100": np.tile(rng.uniform(0, 1, S), (H, 1)), "lnxh": np.tile(rng.normal(0, 1, S), (H, 1)),
             "F1000": np.tile(rng.uniform(0, 0.5, S), (H, 1))}
    lnreg = np.log(base.mean(1))[:, None]
    y = -0.5 - 0.3 * feats["T100"] + 0.1 * feats["lnxh"] + 0.2 * lnreg - 1.5 * feats["F1000"] + rng.normal(0, 0.1, (H, S))
    ws = base * np.exp(y)
    est, _, coef = assess.crossfit_exposure(base, base * 0, ws, feats, assess.blocks(H, 4))
    assert np.allclose(coef.mean(0), [-0.5, -0.3, 0.1, 0.2, -1.5], atol=0.08)
    assert np.nanmedian(np.abs(np.log(est / ws))) < 0.12
