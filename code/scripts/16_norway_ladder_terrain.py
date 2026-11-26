"""Norway: correction ladder with the ratio-of-means bias, and the terrain-index test relative to the final model.

Supersedes R["ladder"] of 11_norway.py and analysis/norway/terrain_test.py (which used a preliminary model).
SWW_DATA = Norway data root, SWW_OUT -> norway_extra.pkl
"""
import dataclasses
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine, cv, ladder

cfg = Settings.norway(max_pair_km=120.0)
OUT = Path(os.environ.get("SWW_OUT", "./output")); OUT.mkdir(parents=True, exist_ok=True)
E = Engine(cfg, pd.read_parquet(cfg.paths.root / "features_variants.parquet"), t_split="2025-01-01", t_end=cfg.test_end)
NB = 8
X = dict(ladder=ladder(E, RECOMMENDED))
print("ladder medians (pairs <= 80 km)", X["ladder"].filter(regex="^(mae|rbias)_").median().round(3).to_dict(), flush=True)

base = cv(E, RECOMMENDED, NB)
X["terrain"] = {"final": base}
far = RECOMMENDED.extra
for lab, ex in [("+ Sx 300 m", ("t:sx_300",)), ("+ Sx 1 km", ("t:sx_1000",)), ("+ TPI 500 m", ("t:tpi_500",)),
                ("+ TPI 2 km", ("t:tpi_2000",)), ("+ Sx 1 km + TPI 2 km + relief", ("t:sx_1000", "t:tpi_2000", "t:relief_2000"))]:
    r = cv(E, dataclasses.replace(RECOMMENDED, extra=far + ex), NB)
    X["terrain"][lab] = r
    d = r["mae"] - base["mae"]; ok = np.isfinite(d)
    print(f"{lab:34s} dMAE {d[ok].mean():+.4f} +- {d[ok].std(ddof=1) / np.sqrt(ok.sum()):.4f}", flush=True)
pickle.dump(X, open(OUT / "norway_extra.pkl", "wb"))
