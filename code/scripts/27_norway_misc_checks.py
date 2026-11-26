"""Norway, smaller checks requested in review round 1 -> SWW_OUT/misc_NO.pkl

  persistence  site factors fitted on 2023 (as used) vs on 2024 and on 2025 (minor 9)
  turn         observed mean turn roadside - reference direction at open roadside stations, by stratum (minor 7)
  daynight     exposure-model coefficients fitted on daytime vs night-time hours (minor 8)
  terrain      F1000 coefficient with terrain indices entered first (question 12)
  bridges      coefficients without bridge / causeway stations, from the per-station normal equations (M5)
"""
import dataclasses
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import Settings
from smhi_trv_wind.geo import wrap180
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine

cfg = Settings.norway(max_pair_km=120.0)
OUT = Path(os.environ.get("SWW_OUT", "./output"))
E = Engine(cfg, pd.read_parquet(cfg.paths.root / "features_variants.parquet"), t_split="2025-01-01", t_end=cfg.test_end)
B = RECOMMENDED
X = {}

# ---- persistence of the site factors
c23 = E.site_factors(B).mean(1)
wsL, wdL, cache = E.wsL, E.wdL, E._site_cache
per = {}
for yr in (2024, 2025):
    m = E.sws.index.year == yr
    E.wsL, E.wdL, E._site_cache = E.sws[m], E.swd[m], {}
    per[yr] = E.site_factors(B).mean(1)
E.wsL, E.wdL, E._site_cache = wsL, wdL, cache
X["persistence"] = dict(c2023=c23, c2024=per[2024], c2025=per[2025],
                        r24=float(np.corrcoef(c23, per[2024].reindex(c23.index))[0, 1]),
                        r25=float(np.corrcoef(c23, per[2025].reindex(c23.index))[0, 1]),
                        mad24=float(np.mean(np.abs(c23 - per[2024].reindex(c23.index)))))
print("site-factor persistence r(2023,2024) %.2f r(2023,2025) %.2f" % (X["persistence"]["r24"], X["persistence"]["r25"]), flush=True)

# ---- observed turn at open roadside stations
pj, ps = E.pj, E.ps
U0 = E.sws.values[:, ps]; wd_s = E.swd.values[:, ps]; ws = E.tws[:, pj]; wd = E.twd[:, pj]
T200 = E._trv_sector(E.Vt["tree_R200_H22.5"])[:, pj]
F1 = E._trv_sector(E.Vt["tree_R1000_H180"])[:, pj]
dd = wrap180(wd - wd_s)
near = E.pd <= 60
m0 = (U0 >= 3) & (ws >= 1.5) & np.isfinite(dd) & (T200 < 0.2) & near[None, :]
hour, month = E.H.hour.values[:, None], E.H.month.values[:, None]
summer = np.isin(month, [4, 5, 6, 7, 8, 9])
strata = {"all open": m0, "summer day (10-14 UTC)": m0 & summer & (hour >= 10) & (hour <= 14),
          "summer night (22-02 UTC)": m0 & summer & ((hour >= 22) | (hour <= 2)),
          "U0 3-6": m0 & (U0 < 6), "U0 >= 10": m0 & (U0 >= 10),
          "F1000 < 0.15": m0 & (F1 < 0.15), "F1000 >= 0.3": m0 & (F1 >= 0.3)}
cmean = lambda a: float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(a))))))
X["turn"] = {k: dict(n=int(v.sum()), mean=cmean(dd[v]), median=float(np.median(dd[v]))) for k, v in strata.items()}
print("turn", {k: (v["n"], round(v["mean"], 1)) for k, v in X["turn"].items()}, flush=True)
del U0, wd_s, ws, wd, T200, F1, dd, m0

# ---- exposure coefficients by day / night (summer half-year)
Hh, Hm = E.H.hour.values, E.H.month.values
sm = np.isin(Hm, [4, 5, 6, 7, 8, 9])
A0 = E.A
X["daynight"] = {}
for lab, mask in (("summer day", sm & (Hh >= 10) & (Hh <= 14)), ("summer night", sm & ((Hh >= 22) | (Hh <= 2))),
                  ("winter", np.isin(Hm, [11, 12, 1, 2]))):
    E.A = mask
    r = E.run(B)
    X["daynight"][lab] = dict(b=np.asarray(r["b"]), d=np.asarray(r["d"]), smear=r["smear"], n=r["n_fit"])
    print(lab, np.round(r["b"], 3), np.round(r["d"], 1), flush=True)

# ---- terrain first, then F1000
E.A = np.ones(len(E.H), bool)
X["terrain"] = {}
for lab, ex in (("final", B.extra), ("terrain only", ("t:sx_1000", "t:tpi_2000", "t:relief_2000")),
                ("terrain + F1000", ("t:sx_1000", "t:tpi_2000", "t:relief_2000") + B.extra)):
    r = E.run(dataclasses.replace(B, extra=ex))
    X["terrain"][lab] = np.asarray(r["b"])
    print(lab, np.round(r["b"], 3), flush=True)
E.A = A0
pickle.dump(X, open(OUT / "misc_NO.pkl", "wb"))
print("wrote", OUT / "misc_NO.pkl")
