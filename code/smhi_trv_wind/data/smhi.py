"""SMHI open meteorological observations (metobs), no registration needed.

API:  https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{par}/station/{id}/period/{period}/data.csv
      par 4 = wind speed (10-min mean, m/s), 3 = wind direction (deg), 21 = max gust
      period 'corrected-archive' (quality-controlled history) + 'latest-months' (recent, not yet archived)
Hourly values; sensor height 10 m. Airport stations report integer m/s.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import Settings
from ..geo import haversine_km

API = "https://opendata-download-metobs.smhi.se/api/version/1.0/parameter"
PARAMS = {"ws": 4, "wd": 3, "gust": 21}
PERIODS = ["corrected-archive", "latest-months"]


# ----------------------------------------------------------------------------- download
def fetch_stations(centre=(57.72, 12.55), radius_km=120.0, active_after="2015-01-01") -> pd.DataFrame:
    """SMHI wind stations within radius_km of centre (default: RV40 Göteborg-Borås)."""
    st = requests.get(f"{API}/4.json", timeout=60).json()["station"]
    rows = []
    for s in st:
        d = haversine_km(centre[0], centre[1], s["latitude"], s["longitude"])
        to = pd.to_datetime(s["to"], unit="ms")
        if d <= radius_km and to >= pd.Timestamp(active_after):
            rows.append(dict(id=str(s["key"]), name=s["name"], lat=s["latitude"], lon=s["longitude"],
                             masl=s["height"], active=s["active"], dist_km=round(float(d), 1), to=to))
    return pd.DataFrame(rows).sort_values("dist_km")


def fetch_csv(station_id: str, out_dir: Path, params=("ws", "wd", "gust"), refresh_latest=True):
    """Download the archive once; re-download 'latest-months' on every call if refresh_latest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in params:
        for per in PERIODS:
            f = out_dir / f"{station_id}_{name}_{per}.csv"
            if f.exists() and not (refresh_latest and per == "latest-months"):
                continue
            r = requests.get(f"{API}/{PARAMS[name]}/station/{station_id}/period/{per}/data.csv", timeout=120)
            if r.status_code == 200:
                f.write_bytes(r.content)


# ----------------------------------------------------------------------------- read
def _read_one(path: Path) -> pd.DataFrame:
    txt = path.read_text(encoding="utf-8-sig").splitlines()
    i = next(k for k, l in enumerate(txt) if l.startswith("Datum;Tid"))
    rows = [l.split(";")[:4] for l in txt[i + 1:] if l and l[0].isdigit()]
    d = pd.DataFrame(rows, columns=["date", "time", "value", "q"])
    d["t"] = pd.to_datetime(d.date + " " + d.time, utc=True)
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    return d[["t", "value", "q"]]


def read_series(station_id: str, name: str, smhi_dir: Path) -> pd.Series:
    """archive + latest months merged, later file wins on duplicates. Index: UTC hour."""
    parts = [_read_one(smhi_dir / f"{station_id}_{name}_{per}.csv")
             for per in PERIODS if (smhi_dir / f"{station_id}_{name}_{per}.csv").exists()]
    if not parts:
        return pd.Series(dtype=float)
    d = pd.concat(parts).drop_duplicates("t", keep="last").sort_values("t")
    return d.set_index("t").value


def stations(cfg: Settings) -> pd.DataFrame:
    m = pd.read_csv(cfg.paths.smhi_stations, dtype={"id": str})
    return m.set_index("id")


def wind_matrices(cfg: Settings, index: pd.DatetimeIndex, ids=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """hourly speed and direction matrices [time x station] reindexed to `index`"""
    ids = stations(cfg).index if ids is None else ids
    ws = pd.DataFrame({s: read_series(s, "ws", cfg.paths.smhi_dir).reindex(index) for s in ids}, index=index)
    wd = pd.DataFrame({s: read_series(s, "wd", cfg.paths.smhi_dir).reindex(index) for s in ids}, index=index)
    return ws, wd


def history(cfg: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """training history for the site factors, hourly, cfg.train_start .. cfg.train_end"""
    idx = pd.date_range(pd.Timestamp(cfg.train_start, tz="UTC"), pd.Timestamp(cfg.train_end, tz="UTC"), freq="1h")
    ids = stations(cfg).index
    ws, wd = {}, {}
    for s in ids:
        a, b = read_series(s, "ws", cfg.paths.smhi_dir), read_series(s, "wd", cfg.paths.smhi_dir)
        ws[s], wd[s] = a[(a.index >= idx[0]) & (a.index <= idx[-1])], b[(b.index >= idx[0]) & (b.index <= idx[-1])]
    ws, wd = pd.DataFrame(ws), pd.DataFrame(wd)
    full = pd.date_range(ws.index.min(), ws.index.max(), freq="1h")
    return ws.reindex(full), wd.reindex(full)


def is_integer_resolution(ws: pd.Series, share=0.3) -> bool:
    """airport stations report whole m/s"""
    v = ws.dropna()
    return bool(len(v) and (v % 1 != 0).mean() <= share)
