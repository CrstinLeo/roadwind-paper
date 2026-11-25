"""Road axis at each Trafikverket station: snap to the OpenStreetMap road network with OSRM and read the
bearing of the first route step. The axis is folded to 0-180 deg (a road has no direction).
The public OSRM demo server is rate-limited: keep >= 1 s between calls, or run a local OSRM.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

OSRM = "https://router.project-osrm.org/route/v1/driving"


def road_bearing(lat: float, lon: float, retries: int = 3):
    u = f"{OSRM}/{lon},{lat};{lon},{lat + 0.02}?steps=true&overview=false"
    for _ in range(retries):
        try:
            j = requests.get(u, timeout=20).json()
            st = j["routes"][0]["legs"][0]["steps"][0]
            return st["maneuver"]["bearing_after"] % 180, j["waypoints"][0]["distance"], st.get("name", "")
        except Exception:
            time.sleep(2)
    return np.nan, np.nan, ""


def build(trv: pd.DataFrame, pause: float = 1.0) -> pd.DataFrame:
    rows = []
    for sid, s in trv.iterrows():
        b, snap, name = road_bearing(s.lat, s.lon)
        rows.append(dict(id=str(sid), road_bearing=b, snap_m=snap, road=name))
        time.sleep(pause)
    return pd.DataFrame(rows)
