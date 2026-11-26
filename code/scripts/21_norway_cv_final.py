"""Norway: recompute the final-model CV (cv_final, pred_final, obs) in norway_results.pkl with the corrected
nearest-station baseline (nearest reference station WITH data in each hour; before 2026-11-26 the static nearest
station was used, which left three roadside stations without a baseline). Model estimates are unchanged.
SWW_DATA = Norway data root, SWW_OUT = folder with norway_results.pkl (from 11_norway.py)."""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine, cv

cfg = Settings.norway(max_pair_km=120.0)
OUT = Path(os.environ.get("SWW_OUT", "./output"))
R = pickle.load(open(OUT / "norway_results.pkl", "rb"))
E = Engine(cfg, pd.read_parquet(cfg.paths.root / "features_variants.parquet"), t_split="2025-01-01 00:00", t_end=cfg.test_end)
month = E.H.month
G = {"leaf-on (May-Sep)": np.isin(month, [5, 6, 7, 8, 9]), "leaf-off (Nov-Mar)": np.isin(month, [11, 12, 1, 2, 3])}
old = R["cv_final"]
c = cv(E, RECOMMENDED, 8, groups=G, keep_pred=True)
R["pred_final"] = c.pop("pred"); R["cv_final"] = c
R["obs"] = dict(ws=E.tws.astype(np.float32), wd=E.twd.astype(np.float32), road=E.road, ids=list(E.TM.index), lat=E.TM.lat.values, lon=E.TM.lon.values,
                name=(E.TM["name"].values if "name" in E.TM else None), hours=E.H)
print("model MAE vs previous run: max |diff|", np.nanmax(np.abs(old["mae"] - c["mae"])).round(4),
      "| raw median", np.nanmedian(old["mae_raw"]).round(3), "->", np.nanmedian(c["mae_raw"]).round(3),
      "| zero raw", int((c["mae_raw"] == 0).sum()), flush=True)
pickle.dump(R, open(OUT / "norway_results.pkl", "wb"))
