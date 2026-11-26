"""Norway, checks requested in review round 1 -> SWW_OUT/review_NO.pkl (saved after every step)

1. honest hold-out: every model choice re-made on 2024 only (4-block CV within 2024, 1-SE rule) and the final
   configuration fitted on 2024 and scored on the untouched year 2025 (leave-one-station-out).
   (ablations and scale scans of the full CV are in 11_norway.py)
3. rows of the final model (all hours) for station-cluster bootstrap of the coefficients, VIFs and a refit without
   bridge/causeway stations: per-station normal equations are stored (small).
"""
import dataclasses
import gc
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine, cv

cfg = Settings.norway(max_pair_km=120.0)
OUT = Path(os.environ.get("SWW_OUT", "./output")); OUT.mkdir(parents=True, exist_ok=True)
F = OUT / "review_NO.pkl"
X = pickle.load(open(F, "rb")) if F.exists() else {}
V = pd.read_parquet(cfg.paths.root / "features_variants.parquet")
R = dataclasses.replace


def save():
    pickle.dump(X, open(F, "wb"))


# ---- 1a. selection on 2024 only
if "select2024" not in X:
    E = Engine(cfg, V, t_split="2024-12-31 23:00", t_end="2024-12-31 23:00")
    cand = {"final": RECOMMENDED,
            "tree 50 m": R(RECOMMENDED, tree="tree_R50_H22.5"), "tree 200 m": R(RECOMMENDED, tree="tree_R200_H22.5"),
            "far none": R(RECOMMENDED, extra=()), "far upwind": R(RECOMMENDED, extra=("v:tree_R1000_H22.5",)),
            "far 500 m": R(RECOMMENDED, extra=("v:tree_R500_H180",)),
            "km 40": R(RECOMMENDED, km=40), "km 80": R(RECOMMENDED, km=80),
            "power 1": R(RECOMMENDED, power=1.0), "power 2": R(RECOMMENDED, power=2.0),
            "site sector": R(RECOMMENDED, site="sector"), "site none": R(RECOMMENDED, site="none")}
    X["select2024"] = {}
    for k, sp in cand.items():
        X["select2024"][k] = cv(E, sp, 4)
        print("2024", k, round(float(np.nanmean(X["select2024"][k]["mae"])), 4), flush=True)
        save()
    del E; gc.collect()

# ---- 1b. fit on 2024, score 2025 (leave-one-station-out)
if "holdout2025" not in X:
    E = Engine(cfg, V, t_split="2025-01-01 00:00", t_end=cfg.test_end)
    r = E.run(RECOMMENDED, keep_pred=True)
    X["holdout2025"] = dict(B=r["B"], U=r["U"], U0n=r["U0n"], H=E.H, A=E.A)
    print("2025 holdout median MAE", np.nanmedian(r["B"]["mae"]).round(3), "raw", np.nanmedian(r["B"]["mae_raw"]).round(3), flush=True)
    save()

    # ---- 3. rows of the final model on all hours -> per-station normal equations
    E.A = np.ones(len(E.H), bool)
    rr = E.run(RECOMMENDED, keep_rows=True)
    Xr, y, fit = rr["X"], rr["y"], rr["fit"]
    st = np.broadcast_to(E.pj, y.shape)[fit]
    Xf = Xr[fit].astype(float); yf = y[fit]
    p = Xf.shape[1]; n_t = len(E.TM)
    XtX = np.zeros((n_t, p, p)); Xty = np.zeros((n_t, p)); yy = np.zeros(n_t); nn = np.zeros(n_t)
    for j in range(n_t):
        m = st == j
        XtX[j] = Xf[m].T @ Xf[m]; Xty[j] = Xf[m].T @ yf[m]; yy[j] = yf[m] @ yf[m]; nn[j] = m.sum()
    C = np.corrcoef(Xf[:, 1:].T)
    X["rows"] = dict(XtX=XtX, Xty=Xty, yy=yy, n=nn, ids=list(E.TM.index), b=np.asarray(rr["b"]), corr=C,
                     vif=np.diag(np.linalg.inv(C)))
    print("VIF", X["rows"]["vif"].round(2), flush=True)
    save()
    del E, rr, Xr, y, fit, Xf; gc.collect()

print("done")
