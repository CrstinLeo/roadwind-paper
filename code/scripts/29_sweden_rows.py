"""Sweden: per-station normal equations of the final exposure model (all hours) for the station-cluster bootstrap of
the coefficients and the VIFs -> SWW_OUT/rows_SE.pkl"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine

cfg = Settings(max_pair_km=150.0)
OUT = Path(os.environ.get("SWW_OUT", "./output"))
E = Engine(cfg, pd.read_parquet(cfg.paths.root / "features_variants.parquet"), t_split=cfg.test_end, t_end=cfg.test_end)
E.A = np.ones(len(E.H), bool)
rr = E.run(RECOMMENDED, keep_rows=True)
fit = rr["fit"]; st = np.broadcast_to(E.pj, rr["y"].shape)[fit]
Xf = rr["X"][fit].astype(float); yf = rr["y"][fit]
n_t, p = len(E.TM), Xf.shape[1]
XtX = np.stack([Xf[st == j].T @ Xf[st == j] for j in range(n_t)]); Xty = np.stack([Xf[st == j].T @ yf[st == j] for j in range(n_t)])
C = np.corrcoef(Xf[:, 1:].T)
pickle.dump(dict(XtX=XtX, Xty=Xty, b=np.asarray(rr["b"]), ids=list(E.TM.index), corr=C, vif=np.diag(np.linalg.inv(C))), open(OUT / "rows_SE.pkl", "wb"))
print("b", np.round(rr["b"], 3), "VIF", np.round(np.diag(np.linalg.inv(C)), 2))
