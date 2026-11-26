"""Re-derive the mirror sign stored in every cal_<site>.json with the day-level rule (nxsite.day_sign), so that all
stored calibration outputs follow the rule used in the paper (no reference wind involved)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nxsite  # noqa: E402

for nm, (day, *_r) in nxsite.SITES.items():
    f = os.path.join(nxsite.D, f"cal_{nm}.json")
    if not os.path.exists(f):
        continue
    J = json.load(open(f)); sgn, tot = nxsite.day_sign(day)
    old = J["pick"]["sgn"]; J["pick"] = next(c for c in J["cal"] if c["sgn"] == sgn); J["rule"] = "day-level reciprocal misfit"
    J["day_costs"] = {str(k): v for k, v in tot.items()}
    json.dump(J, open(f, "w"), default=float)
    print(nm, day, "sign", old, "->", sgn, {k: round(v, 3) for k, v in tot.items()})
