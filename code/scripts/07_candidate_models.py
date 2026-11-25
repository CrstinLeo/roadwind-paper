"""Candidate improved models, scored by the same blocked-time x LOSO cross-validation."""
import dataclasses, os, pickle
import numpy as np
import pandas as pd
from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import Engine, Spec, cv

cfg = Settings(max_pair_km=150.0)
V = pd.read_parquet(cfg.paths.root / "features_variants.parquet")
E = Engine(cfg, V, t_split="2026-11-19 00:00", t_end=os.environ.get("SWW_END", "2026-11-23 15:00"))
two = ("v:tree_R1000_H22.5",)
DIR = dict(dir_km=60, dir_power=1.0, dir_tree="tree_R200_H22.5")   # direction part kept as in M0
C = {
    "M0 baseline (round 6)": Spec(),
    "M0 without roughness physics": Spec(roughness=False),
    "M0 without site factors": Spec(site="none"),
    "M1 = M0 − lnU×tree": Spec(interaction=False),
    "M2 = M1 + stability": Spec(interaction=False, extra=("stab",)),
    "M3 = M2 + cross-road×tree": Spec(interaction=False, extra=("stab", "cross", "cross_tree")),
    "M4 = M2 + two-scale trees (100 m + 1 km)": Spec(interaction=False, tree="tree_R100_H22.5", extra=("stab",) + two),
    "M5 = M4 + all stations, 1/d^1.5": Spec(interaction=False, tree="tree_R100_H22.5", extra=("stab",) + two, km=150, min_n=1, power=1.5, **DIR),
    "M6 = M5 + 8 sectors, L_smhi 100 m": Spec(interaction=False, tree="tree_R100_H22.5", extra=("stab",) + two, km=150, min_n=1, power=1.5, site_nsec=8, L_smhi=100, **DIR),
    "M7 = M6 + cross-road×tree": Spec(interaction=False, tree="tree_R100_H22.5", extra=("stab", "cross", "cross_tree") + two, km=150, min_n=1, power=1.5, site_nsec=8, L_smhi=100, **DIR),
    "M6 without roughness physics": Spec(interaction=False, tree="tree_R100_H22.5", extra=("stab",) + two, km=150, min_n=1, power=1.5, site_nsec=8, L_smhi=100, roughness=False, **DIR),
    "GBM upper bound (M7 features + Sx, TPI, h_mid)": Spec(interaction=True, tree="tree_R100_H22.5", extra=("stab", "cross", "sx", "tpi", "hmid") + two, km=150, min_n=1, power=1.5, site_nsec=8, L_smhi=100, learner="gbm", **DIR),
}
res = {k: cv(E, s) for k, s in C.items()}
full = {k: E.run(s) for k, s in C.items() if "GBM" not in k}            # coefficients on all hours (A = first part only)
pickle.dump(dict(res=res, specs=C, coefs={k: (v["b"], v["d"]) for k, v in full.items()}), open(cfg.paths.out / "candidates_cv.pkl", "wb"))
b = res["M0 baseline (round 6)"]["mae"]
for k, r in res.items():
    d = r["mae"] - b; ok = np.isfinite(d)
    print(f"{k:50s} mean {np.nanmean(r['mae']):.4f} med {np.nanmedian(r['mae']):.4f} d {d[ok].mean():+.4f}±{d[ok].std(ddof=1)/np.sqrt(ok.sum()):.4f} "
          f"dir {np.nanmedian(r['dir']):.1f} bias {np.nanmedian(r['bias_log']):+.3f} rmse_log {np.nanmean(r['rmse_log']):.3f} raw {np.nanmedian(r['mae_raw']):.3f} better_than_raw {(r['mae']<r['mae_raw']).sum()}")
