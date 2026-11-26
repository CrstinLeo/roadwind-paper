"""Mechanism diagnostics of the final model for one country (SWW_COUNTRY = NO | SE) -> SWW_OUT/mechanism_<country>.pkl

1. Partial residuals of the roadside exposure model, ln(U_road/U3) - X b + b_k x_k, binned by each predictor x_k.
   Station-cluster statistics: mean over roadside stations of the per-station bin mean, +- SE over stations.
2. Wind-direction deflection wrap180(phi_road - phi_ref) binned by the angle alpha between the reference wind and the
   road axis, for low and high near-tree share (T200 < 0.2, >= 0.5), same station-cluster statistics.
Fitted on all test hours (as the correction ladder); hours with U0 >= 2 m/s (speed) and U0 >= 3, U_road >= 1.5 (direction).
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import Settings
from smhi_trv_wind.geo import wrap180, wrap90
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine

country = os.environ.get("SWW_COUNTRY", "NO")
cfg = Settings.norway(max_pair_km=120.0) if country == "NO" else Settings(max_pair_km=150.0)
OUT = Path(os.environ.get("SWW_OUT", "./output")); OUT.mkdir(parents=True, exist_ok=True)
E = Engine(cfg, pd.read_parquet(cfg.paths.root / "features_variants.parquet"), t_split=cfg.test_end, t_end=cfg.test_end)
E.A = np.ones(len(E.H), bool)
r = E.run(RECOMMENDED, keep_rows=True)
X, y, fit, b = r["X"], r["y"], r["fit"], np.asarray(r["b"])
names = ["intercept", "T100", "lnxh", "lnUnet", "F1000"]
assert X.shape[-1] == len(names), X.shape
st = np.broadcast_to(E.pj, y.shape)                       # roadside station index of every row
res = y - (X.astype(float) @ b)                           # residual of ln(U_road/U3)

BINS = dict(T100=np.linspace(0, 1, 11), F1000=np.linspace(0, 1, 11), lnxh=np.linspace(-1, 6, 15), lnUnet=np.log([1, 2, 3, 4, 5, 6, 8, 10, 14]))


def cluster_bins(x, v, s, edges, min_rows=20):
    """per-station bin means, then mean and SE over stations"""
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1) & np.isfinite(v)
    df = pd.DataFrame(dict(b=idx[ok], s=s[ok], v=v[ok]))
    g = df.groupby(["b", "s"]).v.agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_rows]
    a = g.groupby("b")["mean"].agg(["mean", "std", "size"])
    a["se"] = a["std"] / np.sqrt(a["size"])
    a["x"] = 0.5 * (edges[a.index] + edges[a.index + 1])
    return a.reset_index()


out = dict(country=country, b=b, names=names, smear=r["smear"], partial={})
for k, nm in enumerate(names[1:], start=1):
    pr = res[fit] + b[k] * X[..., k][fit]
    out["partial"][nm] = cluster_bins(X[..., k][fit], pr, st[fit], BINS[nm])
    print(nm, "b =", round(float(b[k]), 3), "bins:", len(out["partial"][nm]), flush=True)

# station-level data for a scatter of exposure: mean observed ln(U_road/U3) per station vs its mean T100 / F1000
out["stations"] = pd.DataFrame(dict(s=st[fit], y=y[fit], T100=X[..., 1][fit], F1000=X[..., 4][fit])).groupby("s").mean()

# direction deflection
del X, res
wd_s = E.swd.values[:, E.ps]; wd = E.twd[:, E.pj]; U0 = E.sws.values[:, E.ps]; ws = E.tws[:, E.pj]
alpha = wrap90(wd_s - E.road[E.pj][None, :])
t200 = E._trv_sector(E.Vt[RECOMMENDED.dir_tree])[:, E.pj]
dd = wrap180(wd - wd_s)
m = (U0 >= 3) & (ws >= 1.5) & np.isfinite(dd) & np.isfinite(alpha) & np.isfinite(t200) & (E.pd[None, :] <= 60)
edges = np.arange(-90, 91, 15.0)
sd = np.broadcast_to(E.pj, dd.shape)
out["direction"] = {lab: cluster_bins(alpha[m & sel], dd[m & sel], sd[m & sel], edges)
                    for lab, sel in (("open (T200 < 0.2)", t200 < 0.2), ("trees (T200 >= 0.5)", t200 >= 0.5))}
out["d"] = np.asarray(r["d"])
pickle.dump(out, open(OUT / f"mechanism_{country}.pkl", "wb"))
print("wrote", OUT / f"mechanism_{country}.pkl")
