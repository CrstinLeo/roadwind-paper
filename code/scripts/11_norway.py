"""Norway (Trøndelag-Dovre): correction ladder, site factors, seasonal blocked CV x LOSO, sensitivity scans,
model variants and the transfer test Norway <-> Sweden. Results -> SWW_OUT/norway_results.pkl

  SWW_DATA=<norway data> SWW_SE_DATA=<sweden data> SWW_OUT=... python scripts/11_norway.py
"""
import dataclasses
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import Settings, Paths
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine, Spec, cv, ladder

T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
cfg = Settings.norway(max_pair_km=120.0)
OUT = Path(os.environ.get("SWW_OUT", "./output")); OUT.mkdir(parents=True, exist_ok=True)
V = pd.read_parquet(cfg.paths.root / "features_variants.parquet")
E = Engine(cfg, V, t_split="2025-01-01 00:00", t_end=cfg.test_end)
log("hours", len(E.H), "roadside", len(E.TM), "reference", len(E.SM), "pairs", len(E.pj))
month = E.H.month
G = {"leaf-on (May-Sep)": np.isin(month, [5, 6, 7, 8, 9]), "leaf-off (Nov-Mar)": np.isin(month, [11, 12, 1, 2, 3])}
NB = 8                                                  # ~3-month blocks
R = dict(meta=dict(hours=len(E.H), n_trv=len(E.TM), n_ref=int(np.isfinite(E.sws.values).any(0).sum()), n_pairs=len(E.pj), start=str(E.H[0]), end=str(E.H[-1]),
                   trv=list(E.TM.index), ref=list(E.SM.index)))
FINAL = RECOMMENDED
REF6 = Spec()                                           # round-6 reference configuration

R["ladder"] = ladder(E, FINAL); log("ladder", R["ladder"].filter(like="mae").median().round(3).to_dict())
R["site_c"] = E.site_factors(FINAL); log("site factors", R["site_c"].shape)
R["cv_final"] = cv(E, FINAL, NB, groups=G, keep_pred=True)
R["pred_final"] = R["cv_final"].pop("pred"); R["obs"] = dict(ws=E.tws.astype(np.float32), wd=E.twd.astype(np.float32), road=E.road, hours=E.H, ids=list(E.TM.index),
                lat=E.TM.lat.values, lon=E.TM.lon.values, name=(E.TM["name"].values if "name" in E.TM else None)); log("cv final", np.nanmean(R["cv_final"]["mae"]).round(3), np.nanmedian(R["cv_final"]["mae"]).round(3))
R["cv_ref6"] = cv(E, REF6, NB, groups=G); log("cv ref6", np.nanmean(R["cv_ref6"]["mae"]).round(3))
E.A = np.ones(len(E.H), bool)
full = E.run(FINAL); R["coef"] = dict(b=full["b"], d=full["d"], smear=full["smear"])
folds = []
for b in np.array_split(np.arange(len(E.H)), NB):
    m = np.ones(len(E.H), bool); m[b] = False; E.A = m; x = E.run(FINAL); folds.append(np.r_[x["b"], x["d"]])
R["coef_folds"] = np.array(folds); log("coef", np.round(full["b"], 3), np.round(full["d"], 1))

scans = {
    "tree_R": [(f"{r} m", dict(tree=f"tree_R{r}_H22.5")) for r in (50, 100, 200, 300, 500)] + [("none", dict(use_tree=False))],
    "far": [(l, dict(extra=e)) for l, e in (("none", ()), ("300 m", ("v:tree_R300_H180",)), ("500 m", ("v:tree_R500_H180",)),
                                             ("1 km upwind", ("v:tree_R1000_H22.5",)), ("1 km all dir.", ("v:tree_R1000_H180",)),
                                             ("1 km + ring 300 m", ("v:tree_R1000_H180", "v:tree_R300_H180")))],
    "km": [(f"{k} km", dict(km=k)) for k in (20, 40, 80, 120)],
    "power": [(f"{p}", dict(power=p)) for p in (0.5, 1.0, 1.5, 2.0)],
    "site": [(s, dict(site=s)) for s in ("none", "mean", "sector")],
    "variants": [("final", {}), ("+ cross-road x tree", dict(extra=("v:tree_R1000_H180", "cross", "cross_tree"))),
                 ("no roughness conversion", dict(roughness=False)), ("no ln(x/h)", dict(use_lnxh=False)),
                 ("+ lnU x tree", dict(interaction=True))],
}
R["scans"] = {}
for g, items in scans.items():
    for lab, kw in items:
        R["scans"][(g, lab)] = cv(E, dataclasses.replace(FINAL, **kw), NB)
    log("scan", g)
pickle.dump(R, open(OUT / "norway_results.pkl", "wb"))

# ---- transfer test: coefficients fitted in one country, applied unchanged in the other
se_root = os.environ.get("SWW_SE_DATA")
if se_root:
    cse = Settings(max_pair_km=150.0, paths=Paths(root=Path(se_root)))
    ES = Engine(cse, pd.read_parquet(Path(se_root) / "features_variants.parquet"), t_split="2026-11-19 00:00", t_end=cse.test_end)
    ES.A = np.ones(len(ES.H), bool); se_full = ES.run(FINAL)
    R["coef_se"] = dict(b=se_full["b"], d=se_full["d"], smear=se_full["smear"])
    R["se_own_cv"] = cv(ES, FINAL, 4)
    ES.A = np.zeros(len(ES.H), bool); R["se_with_no_coef"] = ES.run(FINAL, fixed=R["coef"])["B"]
    E.A = np.zeros(len(E.H), bool); R["no_with_se_coef"] = E.run(FINAL, fixed=R["coef_se"], groups=G)
    log("transfer NO->SE", np.nanmedian(R["se_with_no_coef"]["mae"]).round(3), "SE own", np.nanmedian(R["se_own_cv"]["mae"]).round(3),
        "| SE->NO", np.nanmedian(R["no_with_se_coef"]["B"]["mae"]).round(3), "NO own", np.nanmedian(R["cv_final"]["mae"]).round(3))
pickle.dump(R, open(OUT / "norway_results.pkl", "wb"))
log("DONE")
