"""Build bin tables for all site-days -> $NORXWIND_DIR/data/bins_<site>.parquet

Anemometer mirror sign: the reciprocal-pass calibration gives one solution per sign. The sign is a property of the
mounting, so it is chosen per measurement day by the smallest summed reciprocal-pass cost over all site-sessions of
that day (2020-11-05: Froya + Dolmsund; Dovre days: E6 sessions). No reference wind is used. With either sign the
head/tail and the absolute crosswind component are the same (mirror about the vehicle axis), so air-drag values do
not depend on the choice; only the wind direction does."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import nxsite as ns

D = os.path.join(os.environ.get("NORXWIND_DIR", "./norxwind"), "data")
DOVRE = os.environ.get("NORXWIND_DOVRE", os.path.join(os.environ.get("NORXWIND_DIR", "./norxwind"), "dovre"))


def dovre():
    cwd = os.getcwd(); os.chdir(DOVRE)
    import segdata
    d, S = segdata.build()
    os.chdir(cwd)
    ll = d.assign(seg=(d.s // segdata.SEG).astype(int)).groupby(["pass_id", "seg"])[["lat", "lon"]].mean()
    S = S.join(ll, on=["pass_id", "seg"])
    S = ns.derive(S)
    M = ns.meps_at("dovre", S.lat.values, S.lon.values, S.t.values)
    for c in M:
        S[c] = M[c].values
    S["site"] = "dovre_" + S.day.astype(str).str[5:].str.replace("-", "")
    S.to_parquet(f"{D}/bins_dovre.parquet")
    d.to_parquet(f"{D}/pts_dovre.parquet")
    print("dovre", S.shape)


def other(name):
    d, S, line, cal, pick, mn_dir = ns.build(name)
    S["site"] = name
    S.to_parquet(f"{D}/bins_{name}.parquet")
    d.to_parquet(f"{D}/pts_{name}.parquet")
    json.dump(dict(cal=cal, pick=pick, mn_dir=float(mn_dir), line=list(map(list, line.coords)),
                   bin=ns.SITES[name][3]), open(f"{D}/cal_{name}.json", "w"), default=float)
    print(name, S.shape, pick)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        dovre() if n == "dovre" else other(n)
