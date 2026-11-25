"""Station exposure features for Norway, from open measured / official data.

Object height  Kartverket national elevation model (airborne laser scanning, nhm):
               canopy/object height = DOM (surface) - DTM (terrain), WCS 1.0.0, resampled to 10 m.
               Objects > 5 m count as trees/obstacles, as in the Swedish NMD object-height layer.
Water          ESA WorldCover 2021 v200, 10 m (class 80 = permanent water bodies), Cloud-Optimised GeoTIFF on AWS.
Grid           EPSG:25833 (UTM 33N), 10 m, 5 x 5 km window centred on the station.

The features are computed by the same functions as for Sweden (variants_from_arrays, edge_from_array).
GDAL needs the proxy CA bundle for /vsicurl: set GDAL_CURL_CA_BUNDLE (or SSL_CERT_FILE).
"""
from __future__ import annotations

import io
import math
import os
import time

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import requests
from pyproj import Transformer

from ..config import Settings
from .gis import edge_from_array
from .variants import variants_from_arrays

TO_UTM33 = Transformer.from_crs(4326, 25833, always_xy=True)
WCS = {"dom": ("https://wcs.geonorge.no/skwms1/wcs.hoyde-dom-nhm-25833", "nhm_dom_topo_25833"),
       "dtm": ("https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833", "nhm_dtm_topo_25833")}
WORLDCOVER = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
HALF, N = 2500.0, 500


def _wcs(kind: str, x: float, y: float) -> np.ndarray:
    url, cov = WCS[kind]
    p = dict(service="WCS", version="1.0.0", request="GetCoverage", coverage=cov, crs="EPSG:25833",
             bbox=f"{x - HALF},{y - HALF},{x + HALF},{y + HALF}", width=N, height=N, format="GeoTIFF")
    for a in range(4):
        try:
            r = requests.get(url, params=p, timeout=180)
            r.raise_for_status()
            with rasterio.open(io.BytesIO(r.content)) as s:
                arr = s.read(1).astype(float)
                if s.nodata is not None:
                    arr[arr == s.nodata] = np.nan
            return arr
        except Exception:
            time.sleep(5 * (a + 1))
    raise RuntimeError(f"WCS {kind} failed at {x:.0f},{y:.0f}")


def _tiles(lat0, lat1, lon0, lon1):
    out = set()
    for la in (lat0, lat1):
        for lo in (lon0, lon1):
            tla, tlo = 3 * math.floor(la / 3), 3 * math.floor(lo / 3)
            out.add(f"{'N' if tla >= 0 else 'S'}{abs(tla):02d}{'E' if tlo >= 0 else 'W'}{abs(tlo):03d}")
    return sorted(out)


def water_mask(x: float, y: float) -> np.ndarray:
    tr = rasterio.transform.from_bounds(x - HALF, y - HALF, x + HALF, y + HALF, N, N)
    back = Transformer.from_crs(25833, 4326, always_xy=True)
    lons, lats = back.transform([x - HALF, x + HALF, x - HALF, x + HALF], [y - HALF, y - HALF, y + HALF, y + HALF])
    dst = np.zeros((N, N), np.uint8)
    env = dict(GDAL_CURL_CA_BUNDLE=os.environ.get("GDAL_CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt"), GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
    with rasterio.Env(**env):
        for t in _tiles(min(lats), max(lats), min(lons), max(lons)):
            part = np.zeros((N, N), np.uint8)
            try:
                with rasterio.open(WORLDCOVER.format(tile=t)) as s:
                    rasterio.warp.reproject(rasterio.band(s, 1), part, dst_transform=tr, dst_crs="EPSG:25833",
                                            resampling=rasterio.warp.Resampling.nearest)
            except rasterio.errors.RasterioIOError:
                continue                                        # tile over open sea does not exist
            dst = np.where(dst == 0, part, dst)
    return (dst == 80) | (dst == 0)                             # 0 = outside any land tile = open sea


def station_arrays(lat: float, lon: float):
    x, y = TO_UTM33.transform(lon, lat)
    dom, dtm = _wcs("dom", x, y), _wcs("dtm", x, y)
    h = np.nan_to_num(dom - dtm, nan=0.0)
    Hm = np.where(h > 5.0, h, 0.0)
    return Hm, water_mask(x, y), dtm


def build(cfg: Settings, stations: pd.DataFrame, road: pd.Series, path_variants, path_edge, path_meta) -> None:
    """stations: index id, columns lat, lon, src ('trv' = roadside, 'smhi' = reference network)"""
    V, E, M = [], [], []
    for sid, s in stations.iterrows():
        Hm, water, dtm = station_arrays(s.lat, s.lon)
        rb = road.get(sid, np.nan) if s.src == "trv" else None
        V += variants_from_arrays(cfg, s.src, sid, Hm, water, rb)
        E += edge_from_array(cfg, s.src, sid, Hm)
        c = N // 2
        M.append(dict(id=str(sid), src=s.src, elev_dtm=float(np.nanmean(dtm[c - 1:c + 1, c - 1:c + 1])),
                      tree_1km=float((Hm[c - 100:c + 100, c - 100:c + 100] > 0).mean()), water_1km=float(water[c - 100:c + 100, c - 100:c + 100].mean())))
        print("features", s.src, sid, flush=True)
    pd.DataFrame(V).to_parquet(path_variants)
    pd.DataFrame(E).to_parquet(path_edge)
    pd.DataFrame(M).to_parquet(path_meta)
