"""沿路地形与地表特征（Copernicus DEM + ESA WorldCover），按 16 个来风方向预先计算"""
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates

RES = 10.0
DIRS = np.arange(0, 360, 22.5)
WC_GROUPS = {"trees": [10], "low_veg": [20, 30, 40, 90, 100], "bare_snow": [60, 70], "water": [80], "built": [50]}


def corridor_rasters(xy, dem_path, wc_path, crs="EPSG:25832", buffer=3000):
    x0, y0 = xy.min(0) - buffer; x1, y1 = xy.max(0) + buffer
    W, H = int((x1 - x0) / RES), int((y1 - y0) / RES)
    T = from_origin(x0, y1, RES, RES)
    dem = np.zeros((H, W), "float32"); wc = np.zeros((H, W), "uint8")
    with rasterio.open(dem_path) as s:
        reproject(rasterio.band(s, 1), dem, dst_transform=T, dst_crs=crs, resampling=Resampling.bilinear)
    with rasterio.open(wc_path) as s:
        reproject(rasterio.band(s, 1), wc, dst_transform=T, dst_crs=crs, resampling=Resampling.nearest)
    return dem, wc, T


def _rc(T, x, y):
    return (T.f - y) / RES - 0.5, (x - T.c) / RES - 0.5


def disk(r_m):
    r = int(r_m / RES); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2 <= r ** 2).astype("float32")


def point_features(xy, dem, wc, T, radius=500, sx_dist=1000):
    """返回：rel_elev (n,)，以及按方向的 Sx (n,16) 和迎风扇区地表比例 {组: (n,16)}"""
    K = disk(radius)
    dem_mean = fftconvolve(dem, K, mode="same") / K.sum()
    r, c = _rc(T, xy[:, 0], xy[:, 1])
    z0 = map_coordinates(dem, [r, c], order=1)
    rel = z0 - map_coordinates(dem_mean, [r, c], order=1)
    n = len(xy)
    sx = np.zeros((n, len(DIRS))); frac = {g: np.zeros((n, len(DIRS))) for g in WC_GROUPS}
    dl = np.arange(25, radius + 1, 25.0); ds = np.arange(30, sx_dist + 1, 30.0)
    for k, th in enumerate(DIRS):
        acc_sx = 0; acc = {g: 0 for g in WC_GROUPS}
        for sp in (-15, 0, 15):
            t = np.radians(th + sp); ex, ey = np.sin(t), np.cos(t)     # 指向来风方向
            px = xy[:, 0, None] + ex * ds; py = xy[:, 1, None] + ey * ds
            rr, cc = _rc(T, px.ravel(), py.ravel())
            zz = map_coordinates(dem, [rr, cc], order=1).reshape(px.shape)
            acc_sx += np.degrees(np.arctan((zz - (z0[:, None] + 3.0)) / ds)).max(1)
            px = xy[:, 0, None] + ex * dl; py = xy[:, 1, None] + ey * dl
            rr, cc = _rc(T, px.ravel(), py.ravel())
            cl = map_coordinates(wc.astype(float), [rr, cc], order=0).reshape(px.shape)
            for g, codes in WC_GROUPS.items():
                acc[g] = acc[g] + np.isin(cl, codes).mean(1)
        sx[:, k] = acc_sx / 3
        for g in WC_GROUPS:
            frac[g][:, k] = acc[g] / 3
    return z0, rel, sx, frac


def at_direction(arr, idx, theta):
    """在 16 个方向之间做圆周线性插值。arr: (n,16)；idx: 点索引；theta: 来风方向（度）"""
    pos = (np.asarray(theta) % 360) / 22.5
    i0 = np.floor(pos).astype(int) % 16; i1 = (i0 + 1) % 16; f = pos - np.floor(pos)
    return arr[idx, i0] * (1 - f) + arr[idx, i1] * f
