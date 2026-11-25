"""Readers for the Swedish data sets (SMHI open data, Trafikverket Open API, Lantmäteriet DTM)."""
import io, glob, os
import numpy as np, pandas as pd

SE = "/home/claude/se"


def read_smhi(sid, name):
    """SMHI metobs CSV (corrected-archive + latest-months merged). Returns time (UTC), value, quality, height."""
    parts, height = [], np.nan
    for per in ["corrected-archive", "latest-months"]:
        f = f"{SE}/smhi/{sid}_{name}_{per}.csv"
        if not os.path.exists(f):
            continue
        txt = open(f, encoding="utf-8-sig").read().splitlines()
        if "Mäthöjd" in txt[0]:
            try:
                height = float(txt[1].split(";")[3])
            except Exception:
                pass
        i = next(k for k, l in enumerate(txt) if l.startswith("Datum;Tid"))
        rows = [l.split(";")[:4] for l in txt[i + 1:] if l and l[0].isdigit()]
        d = pd.DataFrame(rows, columns=["date", "time", "value", "q"])
        d["t"] = pd.to_datetime(d.date + " " + d.time, utc=True)
        d["value"] = pd.to_numeric(d.value, errors="coerce")
        parts.append(d[["t", "value", "q"]])
    if not parts:
        return pd.DataFrame(columns=["t", "value", "q"]), height
    d = pd.concat(parts).drop_duplicates("t", keep="last").sort_values("t").reset_index(drop=True)
    return d, height


def smhi_wind(sid):
    ws, h = read_smhi(sid, "ws")
    wd, _ = read_smhi(sid, "wd")
    d = ws.rename(columns={"value": "ws", "q": "q_ws"}).merge(wd.rename(columns={"value": "wd", "q": "q_wd"}), on="t", how="left")
    d.attrs["height"] = h
    return d


def trv_obs():
    d = pd.read_parquet(f"{SE}/trv/obs_7d_120km.parquet")
    d["id"] = d.id.astype(str)
    return d


def trv_meta():
    m = pd.read_csv(f"{SE}/trv/measurepoints.csv")
    m["id"] = m.id.astype(str)
    return m


def smhi_meta():
    m = pd.read_csv(f"{SE}/smhi/stations_wind.csv")
    m["id"] = m.id.astype(str)
    return m
