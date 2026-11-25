"""Direction-dependent station exposure features from open GIS data (16 wind-from sectors, half-angle 22.5 deg).

Sources (all open, measured):
  NMD2023 additional layer "Objekthöjd 5-45 m" (Naturvårdsverket, airborne laser scanning, 10 m)
      class codes 10/15/20/25/30/45 = 5-10, 10-15, 15-20, 20-25, 25-30, >30 m  -> class centre height h
  NMD2018 base land cover (Naturvårdsverket, 10 m), per county; classes 61 = lake, 62 = sea

Features per station and sector (centre = wind-from direction):
  tree_near     share of pixels with objects > 5 m within 0-200 m upwind             (model step 4)
  x_edge        distance to the first obstacle > 5 m along 7 upwind rays (median), capped 1500 m
  h_edge        mean object height over the first 50 m of that obstacle (median)
  xh_edge       x_edge / h_edge  -> ln(x/h) in step 4
  z0w_fp_L      footprint-weighted roughness exp(sum w ln z0 / sum w), w = exp(-r/L), r <= 2.5 km;
                z0 = 0.1 h on objects, 0.0002 on water, 0.03 elsewhere                 (model step 2)
  dist_coast_km distance to the nearest sea pixel (NMD class 62), 50 m mosaic          (site-factor explanation)
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt, map_coordinates

from ..config import Settings
from ..geo import angdiff, sectors

CODE2H = {10: 7.5, 15: 12.5, 20: 17.5, 25: 22.5, 30: 27.5, 45: 32.5}
TO_SWEREF = Transformer.from_crs(4326, 3006, always_xy=True)          # SWEREF99 TM, metres
_LUT = np.zeros(256)
for _k, _v in CODE2H.items():
    _LUT[_k] = _v


def object_height(cfg: Settings, x: float, y: float, half: float, out_shape=None) -> np.ndarray:
    """object height [m] around (x, y), 0 where no object > 5 m"""
    with rasterio.open(cfg.paths.nmd_object_height) as r:
        H = r.read(1, window=from_bounds(x - half, y - half, x + half, y + half, r.transform), out_shape=out_shape)
    return _LUT[H.astype(int)]


def _polar(n: int, res: float = 10.0):
    c = (n - 1) / 2
    oy, ox = np.mgrid[0:n, 0:n]
    dx, dy = (ox - c) * res, (c - oy) * res
    return np.hypot(dx, dy), np.degrees(np.arctan2(dx, dy)) % 360, c


def tree_features(cfg: Settings, src: str, sid: str, lat: float, lon: float) -> list[dict]:
    x, y = TO_SWEREF.transform(lon, lat)
    Hm = object_height(cfg, x, y, 2000)
    r, az, _ = _polar(Hm.shape[0])
    tree = Hm > 0
    rows = []
    for th in sectors(cfg.n_sectors):
        sec = angdiff(az, th) <= cfg.sector_half_angle
        near = sec & (r <= cfg.tree_near_radius)
        rows.append(dict(src=src, id=str(sid), dir=th, tree_near=float(tree[near].mean())))
    return rows


def edge_features(cfg: Settings, src: str, sid: str, lat: float, lon: float) -> list[dict]:
    x, y = TO_SWEREF.transform(lon, lat)
    Hm = object_height(cfg, x, y, 2000)
    _, _, c = _polar(Hm.shape[0])
    d = np.arange(5, 1505, 5.0)
    rows = []
    for th in sectors(cfg.n_sectors):
        xs, hs = [], []
        for a in np.radians(th + np.linspace(-cfg.sector_half_angle, cfg.sector_half_angle, 7)):
            hh = map_coordinates(Hm, [c - d * np.cos(a) / 10, c + d * np.sin(a) / 10], order=0, mode="nearest")
            i = np.argmax(hh > 0)
            if hh[i] > 0:
                xs.append(d[i]); hs.append(hh[i:i + 10][hh[i:i + 10] > 0].mean())
            else:
                xs.append(1500.0); hs.append(np.nan)
        x_edge = float(np.median(xs))
        h_edge = float(np.nanmedian(hs)) if np.isfinite(hs).any() else 15.0
        rows.append(dict(src=src, id=str(sid), dir=th, x_edge=x_edge, h_edge=h_edge, xh_edge=x_edge / h_edge))
    return rows


# ----------------------------------------------------------------------------- roughness with water
def _landcover_sources(cfg: Settings):
    files = [f for g in cfg.paths.nmd_base_glob for f in glob.glob(g)]
    srcs = [rasterio.open(f) for f in files]
    for s in srcs:
        if s.crs is None:
            s._crs = CRS.from_epsg(3006)
    return srcs


def sea_distance(cfg: Settings, res: float = 50.0):
    """distance [km] to the nearest sea pixel on a res-m mosaic, and its transform"""
    arr, tr = merge(_landcover_sources(cfg), res=res, nodata=0, resampling=rasterio.enums.Resampling.mode, method="max")
    return distance_transform_edt(arr[0] != 62) * res / 1000.0, tr


def roughness_features(cfg: Settings, src: str, sid: str, lat: float, lon: float, sea_km, tr50,
                       Ls=(100, 400, 1000)) -> list[dict]:
    x, y = TO_SWEREF.transform(lon, lat)
    L, _ = merge(_landcover_sources(cfg), bounds=(x - 2500, y - 2500, x + 2500, y + 2500), res=10.0, nodata=0, method="max")
    L = L[0]
    Hm = object_height(cfg, x, y, 2500, out_shape=L.shape)
    water = np.isin(L, [61, 62])
    z0 = np.where(Hm > 0, cfg.z0_tree_factor * Hm, np.where(water, cfg.z0_water, cfg.z0_open))
    r, az, _ = _polar(L.shape[0])
    col, row = ~tr50 * (x, y)
    inside = 0 <= int(row) < sea_km.shape[0] and 0 <= int(col) < sea_km.shape[1]
    dcoast = float(sea_km[int(row), int(col)]) if inside else np.nan
    rows = []
    for th in sectors(cfg.n_sectors):
        sec = angdiff(az, th) <= cfg.sector_half_angle
        out = dict(src=src, id=str(sid), dir=th, dist_coast_km=dcoast)
        for Lfp in Ls:
            m = sec & (r >= 10) & (r <= 2500)
            w = np.exp(-r[m] / Lfp)
            out[f"z0w_fp_{Lfp}"] = float(np.exp(np.sum(w * np.log(z0[m])) / np.sum(w)))
        rows.append(out)
    return rows


def build_all(cfg: Settings, trv: pd.DataFrame, smhi: pd.DataFrame) -> None:
    """compute all station features and write the three parquet tables used by the model"""
    sea_km, tr50 = sea_distance(cfg)
    T, E, Z = [], [], []
    for src, tab in (("trv", trv), ("smhi", smhi)):
        for sid, s in tab.iterrows():
            T += tree_features(cfg, src, sid, s.lat, s.lon)
            E += edge_features(cfg, src, sid, s.lat, s.lon)
            Z += roughness_features(cfg, src, sid, s.lat, s.lon, sea_km, tr50)
    pd.DataFrame(T).to_parquet(cfg.paths.feat_tree)
    pd.DataFrame(E).to_parquet(cfg.paths.feat_edge)
    pd.DataFrame(Z).to_parquet(cfg.paths.feat_z0)


# ----------------------------------------------------------------------------- tables for the model
def sector_tables(cfg: Settings) -> dict[str, pd.DataFrame]:
    """[station x sector] tables used by the pipeline"""
    rd = lambda p: pd.read_parquet(p).assign(id=lambda d: d.id.astype(str))
    Z, T, E = rd(cfg.paths.feat_z0), rd(cfg.paths.feat_tree), rd(cfg.paths.feat_edge)
    piv = lambda D, src, col: D[D.src == src].pivot(index="id", columns="dir", values=col)
    out = dict(
        z0_smhi=piv(Z, "smhi", f"z0w_fp_{int(cfg.footprint_L_smhi)}"),
        z0_trv=piv(Z, "trv", f"z0w_fp_{int(cfg.footprint_L_trv)}"),
        z0_smhi_1km=piv(Z, "smhi", "z0w_fp_1000"),
        coast_smhi=Z[Z.src == "smhi"].groupby("id").dist_coast_km.first(),
        tree_smhi=piv(T, "smhi", "tree_near"),
        lnxh_trv=np.log(piv(E, "trv", "xh_edge").clip(*cfg.xh_clip)),
    )
    # tree shares at the configured radii come from the variants table (features/variants.py);
    # tree_R200_H22.5 there is identical to tree_near of the base table
    Vv = rd(cfg.paths.feat_variants)
    out["tree_trv"] = piv(Vv, "trv", f"tree_R{cfg.tree_near_radius}_H22.5")
    out["tree_dir_trv"] = piv(Vv, "trv", f"tree_R{cfg.tree_dir_radius}_H22.5")
    if cfg.tree_far_radius:
        out["tree_far_trv"] = piv(Vv, "trv", f"tree_R{cfg.tree_far_radius}_H22.5")
    return out
