"""Terrain exposure from a 10 m terrain model window centred on the station (Kartverket DTM for Norway).

  sx_D_H     Winstral et al. (2002) upwind shelter index: maximum upwind slope angle [deg] from the sensor
             (ground + z_sensor) to terrain within D m, over rays spanning +-H deg around the wind-from direction;
             positive = sheltered, negative = exposed
  tpi_R      topographic position index: station ground minus mean terrain within R m [m]
  relief_R   standard deviation of terrain within R m [m]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from ..geo import sectors

SX_D = (100, 300, 1000)


def terrain_features(dtm: np.ndarray, src: str, sid: str, z_sensor: float = 10.0, res: float = 10.0,
                     n_sectors: int = 16, half: float = 22.5) -> list[dict]:
    n = dtm.shape[0]
    c = (n - 1) / 2
    z0 = float(np.nanmean(dtm[int(c) - 1:int(c) + 2, int(c) - 1:int(c) + 2]))
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - c, yy - c) * res
    common = {}
    for R in (500, 2000):
        m = r <= R
        common[f"tpi_{R}"] = z0 - float(np.nanmean(dtm[m]))
        common[f"relief_{R}"] = float(np.nanstd(dtm[m]))
    d = np.arange(res, max(SX_D) + res, res)
    rows = []
    for th in sectors(n_sectors):
        out = dict(src=src, id=str(sid), dir=th, **common)
        best = {D: -90.0 for D in SX_D}
        for a in np.radians(th + np.linspace(-half, half, 7)):
            zz = map_coordinates(np.nan_to_num(dtm, nan=z0), [c - d * np.cos(a) / res, c + d * np.sin(a) / res], order=1, mode="nearest")
            ang = np.degrees(np.arctan2(zz - (z0 + z_sensor), d))
            for D in SX_D:
                best[D] = max(best[D], float(ang[d <= D].max()))
        for D in SX_D:
            out[f"sx_{D}"] = best[D]
        rows.append(out)
    return rows
