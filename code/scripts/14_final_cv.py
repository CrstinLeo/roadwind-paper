"""Final model and round-6 reference: blocked-time x leave-one-station-out CV with fold coefficients,
out-of-sample estimates and air-drag errors. SWW_COUNTRY=SE|NO -> SWW_OUT/cv_full.pkl"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import Settings, drag
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine, Spec, cv

country = os.environ.get("SWW_COUNTRY", "SE")
cfg = Settings.norway(max_pair_km=120.0) if country == "NO" else Settings(max_pair_km=150.0)
OUT = Path(os.environ.get("SWW_OUT", "./output")); OUT.mkdir(parents=True, exist_ok=True)
E = Engine(cfg, pd.read_parquet(cfg.paths.root / "features_variants.parquet"), t_split=cfg.test_end, t_end=cfg.test_end)
NB = 8 if country == "NO" else 4
c = cv(E, RECOMMENDED, NB, keep_pred=True)
c0 = cv(E, Spec(), NB)
folds_b, folds_d = [], []
for b in np.array_split(np.arange(len(E.H)), NB):
    m = np.ones(len(E.H), bool); m[b] = False; E.A = m; x = E.run(RECOMMENDED); folds_b.append(x["b"]); folds_d.append(x["d"])
E.A = np.ones(len(E.H), bool); full = E.run(RECOMMENDED)
p = c.pop("pred")
v = 80 / 3.6
dm, pm = drag.station_errors(E.tws, E.twd, p["U"], p["wdm"], E.road, v)
dr, pr = drag.station_errors(E.tws, E.twd, p["U0n"], p["wd0n"], E.road, v)
out = dict(cv=c, cv0=c0, folds_b=np.array(folds_b), folds_d=np.array(folds_d), coef=dict(b=full["b"], d=full["d"], smear=full["smear"]),
           drag=dict(model_station=dm, raw_station=dr, model_pooled=pm, raw_pooled=pr, v=v), n_trv=len(E.TM), hours=len(E.H),
           pred=p, obs=dict(ws=E.tws.astype(np.float32), wd=E.twd.astype(np.float32), road=E.road))
pickle.dump(out, open(OUT / "cv_full.pkl", "wb"))
print(country, "CV final mean/median", np.nanmean(c["mae"]).round(3), np.nanmedian(c["mae"]).round(3), "| ref6", np.nanmean(c0["mae"]).round(3),
      "| raw", np.nanmedian(c["mae_raw"]).round(3), "| drag |err| median model", np.nanmedian(dm).round(3), "raw", np.nanmedian(dr).round(3),
      "| coef", np.round(full["b"], 3), np.round(full["d"], 1))
