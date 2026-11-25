"""Trafikverket road weather stations (Open API, Road.WeatherInfo, schema 2.1).

API: POST XML to https://api.trafikinfo.trafikverket.se/v2/data.json  (free key, register at the Trafikverket API portal)
  WeatherMeasurepoint   station list, position, sensor heights (wind 6 m, Vaisala WMT700)
  WeatherObservation    latest observations; the API keeps about 7 days (5-min rolling 10-min means,
                        plus 10- and 30-min aggregates)
Longer history is only available from the VViS archive (on request).

The API key is read from the environment variable TRV_API_KEY or from the file named in TRV_API_KEY_FILE.
It is never written to disk or printed by this module.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import Settings
from ..geo import haversine_km

URL = "https://api.trafikinfo.trafikverket.se/v2/data.json"


def _key() -> str:
    if os.environ.get("TRV_API_KEY"):
        return os.environ["TRV_API_KEY"].strip()
    f = os.environ.get("TRV_API_KEY_FILE")
    if f and Path(f).exists():
        return Path(f).read_text().strip()
    raise RuntimeError("set TRV_API_KEY or TRV_API_KEY_FILE")


def _query(obj: str, filt: str = "<FILTER/>", include: str = "") -> list:
    x = (f'<REQUEST><LOGIN authenticationkey="{_key()}"/>'
         f'<QUERY objecttype="{obj}" namespace="Road.WeatherInfo" schemaversion="2.1">{filt}{include}</QUERY></REQUEST>')
    for attempt in range(3):
        try:
            r = requests.post(URL, data=x.encode(), headers={"Content-Type": "text/xml"}, timeout=300)
            r.raise_for_status()
            return r.json()["RESPONSE"]["RESULT"][0].get(obj, [])
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Trafikverket query {obj} failed")


# ----------------------------------------------------------------------------- download
def fetch_stations(centre=(57.72, 12.55)) -> pd.DataFrame:
    rows = []
    for m in _query("WeatherMeasurepoint"):
        lon, lat = map(float, re.findall(r"[-\d.]+", m["Geometry"]["WGS84"]))
        w = (m.get("Observation") or {}).get("Wind") or [{}]
        rows.append(dict(id=str(m["Id"]), name=m["Name"], lat=lat, lon=lon,
                         dist_km=float(haversine_km(centre[0], centre[1], lat, lon)),
                         wind_height=w[0].get("Height"), wind_sensor=(w[0].get("Speed") or {}).get("SensorNames"),
                         last_sample=(m.get("Observation") or {}).get("Sample")))
    return pd.DataFrame(rows).sort_values("dist_km")


def fetch_observations(ids: list[str], batch: int = 4) -> pd.DataFrame:
    fields = ["Measurepoint.Id", "Sample", "Wind", "Aggregated10minutes.Wind", "Aggregated30minutes.Wind",
              "Air.Temperature", "Surface.Temperature"]
    inc = "".join(f"<INCLUDE>{f}</INCLUDE>" for f in fields)
    g = lambda d, k: (d.get(k) or {}).get("Value")
    rows = []
    for i in range(0, len(ids), batch):
        filt = "<FILTER><OR>" + "".join(f'<EQ name="Measurepoint.Id" value="{k}"/>' for k in ids[i:i + batch]) + "</OR></FILTER>"
        for o in _query("WeatherObservation", filt, inc):
            w = (o.get("Wind") or [{}])[0]
            a10 = (o.get("Aggregated10minutes") or {}).get("Wind", {})
            a30 = (o.get("Aggregated30minutes") or {}).get("Wind", {})
            rows.append(dict(id=str(o["Measurepoint"]["Id"]), time=o["Sample"], ws=g(w, "Speed"), wd=g(w, "Direction"),
                             height=w.get("Height"), ws10=g(a10, "SpeedAverage"), gust10=g(a10, "SpeedMax"),
                             ws30=g(a30, "SpeedAverage"), gust30=g(a30, "SpeedMax"),
                             t_air=g(o.get("Air", {}), "Temperature"), t_surf=g(o.get("Surface", {}), "Temperature")))
    d = pd.DataFrame(rows)
    d["time"] = pd.to_datetime(d.time, utc=True).dt.round("1min")
    return d


def append_observations(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """the API only keeps ~7 days: run regularly and append to the local archive"""
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    d = new.drop_duplicates(["id", "time"], keep="last").sort_values(["id", "time"])
    d.to_parquet(path)
    return d


# ----------------------------------------------------------------------------- read
def stations(cfg: Settings, max_km: float = 120.0) -> pd.DataFrame:
    m = pd.read_csv(cfg.paths.trv_stations, dtype={"id": str})
    return m[m.dist_km <= max_km].set_index("id")


def wind_matrices(cfg: Settings, index: pd.DatetimeIndex, ids=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """on-the-hour samples (10-min mean ending at the hour, same as SMHI) as [time x station]"""
    o = pd.read_parquet(cfg.paths.trv_obs)
    o["id"] = o.id.astype(str)
    if ids is not None:
        o = o[o.id.isin(ids)]
    o = o[o.time.dt.minute == 0]
    ws = o.pivot_table(index="time", columns="id", values="ws").reindex(index)
    wd = o.pivot_table(index="time", columns="id", values="wd").reindex(index)
    return ws, wd


def temperature_difference(cfg: Settings, index: pd.DatetimeIndex, ids=None) -> pd.DataFrame:
    """dT = road-surface minus air temperature [K] on the hour (stability proxy), [time x station]"""
    o = pd.read_parquet(cfg.paths.trv_obs)
    o["id"] = o.id.astype(str)
    o = o[o.time.dt.minute == 0]
    if ids is not None:
        o = o[o.id.isin(ids)]
    d = o.pivot_table(index="time", columns="id", values="t_surf") - o.pivot_table(index="time", columns="id", values="t_air")
    return d.reindex(index=index, columns=list(ids) if ids is not None else None)


def road_bearing(cfg: Settings) -> pd.Series:
    """road axis (0-180 deg) at each station, OSRM snap on the OpenStreetMap road network (features/road.py)"""
    return pd.read_csv(cfg.paths.road_bearing, dtype={"id": str}).set_index("id").road_bearing
