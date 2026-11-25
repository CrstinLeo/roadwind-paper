"""Water / coast features from NMD2018 base land cover (classes 61 lake, 62 sea), counties O, N, F (EPSG:3006)."""
import numpy as np, pandas as pd, rasterio, glob
from rasterio.merge import merge
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt
from features import OBJ, CODE2H, T, DIRS, angdiff
import sedata as sd

FILES = glob.glob("/home/claude/se/nmd_base/*/*/*.tif") + glob.glob("/home/claude/rv40_gis/data/nmd_O/*/*.tif")


def _open():
    return [rasterio.open(f) for f in FILES]


def mosaic(res=50.0):
    srcs = _open()
    for s in srcs: s._crs = CRS.from_epsg(3006) if s.crs is None else s.crs
    arr, tr = merge(srcs, res=res, nodata=0, resampling=rasterio.enums.Resampling.mode, method="max")
    return arr[0], tr


def window10(x, y, half=2500):
    srcs = _open()
    arr, tr = merge(srcs, bounds=(x - half, y - half, x + half, y + half), res=10.0, nodata=0, method="max")
    return arr[0]


def site_features(src, sid, lat, lon, sea_dist, tr50):
    x, y = T.transform(lon, lat)
    L = window10(x, y)                                     # land-cover codes, 10 m, ±2.5 km
    with rasterio.open(OBJ) as r:
        from rasterio.windows import from_bounds
        H = r.read(1, window=from_bounds(x - 2500, y - 2500, x + 2500, y + 2500, r.transform), out_shape=L.shape).astype(int)
    lut = np.zeros(256); [lut.__setitem__(k, v) for k, v in CODE2H.items()]
    Hm = lut[H]; water = np.isin(L, [61, 62]); n = L.shape[0]; c = (n - 1) / 2
    z0 = np.where(Hm > 0, 0.1 * Hm, np.where(water, 0.0002, 0.03))
    oy, ox = np.mgrid[0:n, 0:n]; dx, dy = (ox - c) * 10.0, (c - oy) * 10.0
    ro = np.hypot(dx, dy); az = np.degrees(np.arctan2(dx, dy)) % 360
    col, row = ~tr50 * (x, y)
    dcoast = float(sea_dist[int(row), int(col)]) * 50 / 1000 if 0 <= int(row) < sea_dist.shape[0] and 0 <= int(col) < sea_dist.shape[1] else np.nan
    rows = []
    for th in DIRS:
        sec = angdiff(az, th) <= 22.5
        r_ = dict(src=src, id=str(sid), dir=th, dist_coast_km=dcoast)
        for Lfp in [100, 400, 1000]:
            m = sec & (ro >= 10) & (ro <= 2500); w = np.exp(-ro[m] / Lfp)
            r_[f"z0w_fp_{Lfp}"] = float(np.exp(np.sum(w * np.log(z0[m])) / np.sum(w)))
        for a, b in [(0, 500), (500, 2500)]:
            m = sec & (ro > a) & (ro <= b) & (L > 0); r_[f"water_{a}_{b}"] = float(water[m].mean()) if m.any() else np.nan
        rows.append(r_)
    return rows


if __name__ == "__main__":
    A, tr50 = mosaic(50.0)
    sea = A == 62
    sea_dist = distance_transform_edt(~sea)
    np.save("/home/claude/se/nmd_base/sea_dist50.npy", sea_dist); open("/home/claude/se/nmd_base/tr50.txt", "w").write(",".join(map(str, tr50[:6])))
    print("mosaic", A.shape, "classes", np.unique(A[::10, ::10]))
    tm, smm = sd.trv_meta(), sd.smhi_meta(); tm = tm[tm.dist_km <= 120]
    out = []
    for _, s in smm.iterrows(): out += site_features("smhi", s.id, s.lat, s.lon, sea_dist, tr50)
    for _, s in tm.iterrows(): out += site_features("trv", s.id, s.lat, s.lon, sea_dist, tr50)
    pd.DataFrame(out).to_parquet("/home/claude/se/features_water_16dir.parquet"); print("done", len(out))
