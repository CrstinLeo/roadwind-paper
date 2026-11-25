"""Feature variants for the sensitivity analysis (one raster read per station).

For every station and each of the 16 wind-from sectors:
  tree_R{R}_H{H}     share of object pixels (>5 m) within R m upwind, sector half-angle H deg
  treew_L{L}_H{H}    exp(-r/L)-weighted tree share (soft radius), r <= 2.5 km
  lnz0_L{L}_H{H}     footprint ln z0 = sum w ln z0 / sum w, w = exp(-r/L), z0 = 0.1h / 0.0002 water / 0.03
  fwt_L{L}_H{H}      the weighted tree share of the same footprint (lets the tree factor 0.1 be varied:
                     ln z0(c) = ln z0(0.1) + fwt * ln(c / 0.1))
  along_tree_s{s}    tree_R200_H22.5 evaluated at a point s metres along the road axis from the station
  along_lnz0_s{s}    lnz0_L100_H22.5 at the same point
The along-road values emulate a road segment: averaging over |s| <= S gives the feature of a segment of
length 2S centred on the station.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge

from ..config import Settings
from ..geo import angdiff, sectors
from .gis import TO_SWEREF, _landcover_sources, object_height

RADII = (50, 100, 150, 200, 300, 500, 1000)
HALF = (11.25, 22.5, 45.0, 90.0, 180.0)
TREE_L = (25, 50, 100, 200, 400)
Z0_L = (25, 50, 100, 200, 400, 1000, 2000)
Z0_HALF = (22.5, 45.0)
ALONG = (-1000, -750, -500, -250, -100, 0, 100, 250, 500, 750, 1000)
HALF_WIN = 2500.0
RES = 10.0


def _polar(n, ox_m=0.0, oy_m=0.0):
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    dx, dy = (xx - c) * RES - ox_m, (c - yy) * RES - oy_m
    return np.hypot(dx, dy), np.degrees(np.arctan2(dx, dy)) % 360


def station_variants(cfg: Settings, src: str, sid: str, lat: float, lon: float, road_bearing: float | None,
                     srcs) -> list[dict]:
    x, y = TO_SWEREF.transform(lon, lat)
    L, _ = merge(srcs, bounds=(x - HALF_WIN, y - HALF_WIN, x + HALF_WIN, y + HALF_WIN), res=RES, nodata=0, method="max")
    L = L[0]
    Hm = object_height(cfg, x, y, HALF_WIN, out_shape=L.shape)
    tree = Hm > 0
    lnz0 = np.log(np.where(tree, cfg.z0_tree_factor * Hm, np.where(np.isin(L, [61, 62]), cfg.z0_water, cfg.z0_open)))
    n = L.shape[0]
    r, az = _polar(n)
    inner = (r >= 10) & (r <= HALF_WIN)
    W = {Lf: np.exp(-r / Lf) for Lf in set(TREE_L) | set(Z0_L)}
    along = []
    if road_bearing is not None and np.isfinite(road_bearing):
        b = np.radians(road_bearing)
        for s in ALONG:
            rr, aa = _polar(n, s * np.sin(b), s * np.cos(b))
            along.append((s, rr, aa))
    rows = []
    for th in sectors(cfg.n_sectors):
        out = dict(src=src, id=str(sid), dir=th)
        d = angdiff(az, th)
        for H in HALF:
            sec = d <= H
            for R in RADII:
                m = sec & (r <= R)
                out[f"tree_R{R}_H{H:g}"] = float(tree[m].mean())
        for H in (22.5, 45.0):
            sec = (d <= H) & inner
            for Lf in TREE_L:
                w = W[Lf][sec]
                out[f"treew_L{Lf}_H{H:g}"] = float(np.sum(w * tree[sec]) / np.sum(w))
        for H in Z0_HALF:
            sec = (d <= H) & inner
            for Lf in Z0_L:
                w = W[Lf][sec]
                out[f"lnz0_L{Lf}_H{H:g}"] = float(np.sum(w * lnz0[sec]) / np.sum(w))
                out[f"fwt_L{Lf}_H{H:g}"] = float(np.sum(w * tree[sec]) / np.sum(w))
        for s, rr, aa in along:
            dd = angdiff(aa, th)
            m = (dd <= 22.5) & (rr <= 200)
            out[f"along_tree_s{s}"] = float(tree[m].mean())
            m = (dd <= 22.5) & (rr >= 10) & (rr <= 2000)
            w = np.exp(-rr[m] / 100.0)
            out[f"along_lnz0_s{s}"] = float(np.sum(w * lnz0[m]) / np.sum(w))
        rows.append(out)
    return rows


def build(cfg: Settings, trv: pd.DataFrame, smhi: pd.DataFrame, road: pd.Series, path) -> pd.DataFrame:
    srcs = _landcover_sources(cfg)
    rows = []
    for src, tab in (("trv", trv), ("smhi", smhi)):
        for sid, s in tab.iterrows():
            rb = road.get(sid, np.nan) if src == "trv" else None
            rows += station_variants(cfg, src, sid, s.lat, s.lon, rb, srcs)
    D = pd.DataFrame(rows)
    D.to_parquet(path)
    return D
