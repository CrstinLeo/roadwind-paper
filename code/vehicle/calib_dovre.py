"""Reciprocal-pass calibration of the van anemometer on the Dovre E6 sessions (2020-10-29/31), both mirror signs
-> $NORXWIND_DIR/data/cal_dovre.json. The constants in segdata.CAL are the solution with the smaller misfit."""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

NX = os.environ.get("NORXWIND_DIR", "./norxwind")
DOVRE = os.environ.get("NORXWIND_DOVRE", os.path.join(NX, "dovre"))
os.chdir(DOVRE)
from calib import load_passes, make_pairs, true_wind  # noqa: E402  (reads files relative to DOVRE)

d, line = load_passes()
pairs = make_pairs(d)


def cost(p, sgn):
    we, wn = true_wind(d, p[0], p[1], p[1], sgn)
    m = pd.DataFrame(dict(p=d.pass_id.values, b=(d.s.values // 1000).astype(int), we=we, wn=wn)).groupby(["p", "b"])[["we", "wn"]].mean()
    c, n = 0.0, 0
    for a, b in pairs:
        diff = (m.loc[a] - m.loc[b]).dropna(); c += (diff ** 2).sum().sum(); n += len(diff)
    return c / max(n, 1)


cal = []
for sgn in (1, -1):
    a0 = min((cost((a, 1.0), sgn), a) for a in range(-180, 180, 10))[1]
    r = minimize(cost, [a0, 1.0], args=(sgn,), method="Nelder-Mead", options=dict(xatol=0.05, fatol=1e-4))
    cal.append(dict(sgn=sgn, alpha=float(((r.x[0] + 180) % 360) - 180), g=float(r.x[1]), cost=float(r.fun),
                    cost0=float(cost((0, 1), sgn)), n_pairs=len(pairs)))
pick = min(cal, key=lambda c: c["cost"])
json.dump(dict(cal=cal, pick=pick, rule="smaller reciprocal-pass misfit"), open(os.path.join(NX, "data", "cal_dovre.json"), "w"), indent=1)
print(cal, "pick", pick["sgn"])
