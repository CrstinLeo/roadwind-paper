"""Sweden (development region): every sensitivity scan re-made around the FINAL model (RECOMMENDED) on the full record
used elsewhere in the paper (SWW_TEST_END, 287 h), 4 time blocks x leave-one-station-out -> SWW_OUT/review_SE.pkl
(replaces the round-6 scans of 06_sensitivity_cv.py / 08_two_scale_grid.py, which were made around Spec()).
"""
import dataclasses
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine, cv

cfg = Settings(max_pair_km=150.0)
OUT = Path(os.environ.get("SWW_OUT", "./output")); OUT.mkdir(parents=True, exist_ok=True)
E = Engine(cfg, pd.read_parquet(cfg.paths.root / "features_variants.parquet"), t_split=cfg.test_end, t_end=cfg.test_end)
print("hours", len(E.H), "roadside", len(E.TM), flush=True)
R = dataclasses.replace
B = RECOMMENDED
scans = {
    "z_blend": [(f"{v} m", R(B, z_blend=v)) for v in (30, 40, 60, 80, 100, 150)],
    "c_tree": [(f"{v}", R(B, c_tree=v)) for v in (0.05, 0.075, 0.1, 0.125, 0.15)],
    "L_smhi": [(f"{v} m", R(B, L_smhi=v)) for v in (25, 50, 100, 200, 400, 1000)],
    "L_trv": [(f"{v} m", R(B, L_trv=v)) for v in (25, 50, 100, 200, 400, 1000)],
    "site": [(v, R(B, site=v)) for v in ("none", "mean", "sector")],
    "tree_R": [(f"{r} m", R(B, tree=f"tree_R{r}_H22.5")) for r in (50, 100, 150, 200, 300, 500)] + [("none", R(B, use_tree=False))],
    "far": [("none", R(B, extra=()))] + [(f"{r} m all dir.", R(B, extra=(f"v:tree_R{r}_H180",))) for r in (300, 500, 1000)]
           + [("1 km upwind", R(B, extra=("v:tree_R1000_H22.5",))), ("1 km +-90", R(B, extra=("v:tree_R1000_H90",)))],
    "km": [(f"{v} km", R(B, km=v)) for v in (20, 40, 80, 150)],
    "power": [(f"{v}", R(B, power=v)) for v in (0.5, 1.0, 1.5, 2.0)],
    "dir_km": [(f"{v} km", R(B, dir_km=v)) for v in (20, 40, 60, 80, 150)],
    "road_window": [(f"+-{S} m", R(B, tree=f"along:{S}", along_z0=S)) for S in (0, 100, 250, 500)],
    "structure": [("+ lnU x tree", R(B, interaction=True)), ("no ln(x/h)", R(B, use_lnxh=False)),
                  ("+ cross-road x tree", R(B, extra=B.extra + ("cross", "cross_tree"))),
                  ("+ stability x tree", R(B, extra=B.extra + ("stab", "stab_tree")))],
}
grid = {(n, f): R(B, tree=f"tree_R{n}_H22.5", extra=(() if f == "none" else (f"v:{f}",)))
        for n in (50, 100, 150, 200, 300) for f in ("none", "tree_R300_H180", "tree_R500_H180", "tree_R1000_H180", "tree_R1000_H22.5")}
F = OUT / "review_SE.pkl"
X = pickle.load(open(F, "rb")) if F.exists() else {"scans": {}, "grid": {}}
if "final" not in X:
    X["final"] = cv(E, B, 4)
for g, items in scans.items():
    for lab, sp in items:
        if (g, lab) not in X["scans"]:
            X["scans"][(g, lab)] = cv(E, sp, 4)
    d = {lab: np.nanmean(X["scans"][(g, lab)]["mae"] - X["final"]["mae"]) for lab, _ in items}
    print(g, {k: round(v, 4) for k, v in d.items()}, flush=True)
    pickle.dump(X, open(F, "wb"))
for k, sp in grid.items():
    if k not in X["grid"]:
        X["grid"][k] = cv(E, sp, 4)
pickle.dump(X, open(F, "wb"))
print("grid", {k: round(float(np.nanmean(v["mae"] - X["final"]["mae"])), 4) for k, v in X["grid"].items()})
print("done")
