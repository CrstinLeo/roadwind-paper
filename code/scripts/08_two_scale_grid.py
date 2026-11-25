"""Grid of the two tree scales (near radius x far radius / sector width) on top of M6."""
import dataclasses, os, pickle
import numpy as np
import pandas as pd
from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import Engine, Spec, cv

cfg = Settings(max_pair_km=150.0)
V = pd.read_parquet(cfg.paths.root / "features_variants.parquet")
E = Engine(cfg, V, t_split="2026-11-19 00:00", t_end=os.environ.get("SWW_END", "2026-11-23 15:00"))
M6 = dict(interaction=False, extra=("stab",), km=150, min_n=1, power=1.5, site_nsec=8, L_smhi=100)
out = {}
for near in (50, 100, 150, 200):
    for far in ("none", "tree_R300_H22.5", "tree_R500_H22.5", "tree_R1000_H22.5", "tree_R500_H90", "tree_R1000_H90", "tree_R1000_H180", "treew_L400_H22.5"):
        ex = ("stab",) if far == "none" else ("stab", "v:" + far)
        out[(near, far)] = cv(E, Spec(**{**M6, "tree": f"tree_R{near}_H22.5", "extra": ex}))
for L in (100, 400):
    for lt in ("treew_L25_H22.5", "treew_L50_H22.5"):
        out[(lt, f"lnz0 L_trv={L}")] = cv(E, Spec(**{**M6, "tree": lt, "extra": ("stab", "v:tree_R1000_H22.5"), "L_trv": L}))
for L in (25, 50, 400, 1000):
    out[(100, f"tree_R1000_H22.5 | L_trv={L}")] = cv(E, Spec(**{**M6, "tree": "tree_R100_H22.5", "extra": ("stab", "v:tree_R1000_H22.5"), "L_trv": L}))
pickle.dump(out, open(cfg.paths.out / "two_scale_cv.pkl", "wb"))
ref = out[(100, "tree_R1000_H22.5")]["mae"]
for k, r in out.items():
    d = r["mae"] - ref; ok = np.isfinite(d)
    print(f"{str(k):60s} mean {np.nanmean(r['mae']):.4f} med {np.nanmedian(r['mae']):.4f} d_vs_M6 {d[ok].mean():+.4f}±{d[ok].std(ddof=1)/np.sqrt(ok.sum()):.4f}")
