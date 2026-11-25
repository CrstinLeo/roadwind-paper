"""Direction-dependent site features for every station (Trafikverket and SMHI), 16 wind-from directions.

Sources: Lantmäteriet Markhöjdmodell (8 m around each station), NMD2023 object height 5–45 m (10 m, laser).
Per station and sector (centre = wind-from direction, half-angle 22.5°):
  tree_near   share of pixels with objects > 5 m within 0–200 m upwind
  tree_mid    same within 200–1000 m
  h_mid       mean object height (class centre) of tree pixels within 0–1000 m
  z0_eff      exp(mean ln z0) within 0–1000 m, z0 = 0.1 h on tree pixels, 0.03 m elsewhere (WMO open terrain)
  sx          Winstral shelter: mean over rays of max_{r<=1 km} atan((z(r) - z_site - z_sensor) / r)  [deg]
Direction independent: elev (DTM at site), tpi_1km (site minus mean within 1 km).
"""
import glob, numpy as np, pandas as pd, rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import map_coordinates
from pyproj import Transformer
import sedata as sd

OBJ = "/home/claude/se/nmd_ext/NMD2023_Tillaggsskikt_Objekthojd_objekttackning_v1_1/NMD2023_Objekt_hojd_intervall_5_till_45_v1_1.tif"
CODE2H = {10: 7.5, 15: 12.5, 20: 17.5, 25: 22.5, 30: 27.5, 45: 32.5}
T = Transformer.from_crs(4326, 3006, always_xy=True)
DIRS = np.arange(0, 360, 22.5)


def angdiff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def station_features(src, sid, lat, lon, z_sensor):
    x, y = T.transform(lon, lat)
    with rasterio.open(f"/home/claude/se/dtm/stations/{src}_{sid}.tif") as r:
        Z = r.read(1).astype(float); Z[Z < -1000] = np.nan
        row0, col0 = rasterio.transform.rowcol(r.transform, x, y); res = r.res[0]
    zc = np.nanmean(Z[row0 - 1:row0 + 2, col0 - 1:col0 + 2])
    yy, xx = np.mgrid[0:Z.shape[0], 0:Z.shape[1]]
    rr = np.hypot((xx - col0) * res, (row0 - yy) * res)
    tpi = zc - np.nanmean(Z[rr <= 1000])
    with rasterio.open(OBJ) as r:
        H = r.read(1, window=from_bounds(x - 2000, y - 2000, x + 2000, y + 2000, r.transform)).astype(float)
    Hm = np.vectorize(lambda v: CODE2H.get(int(v), 0.0))(H)
    n = Hm.shape[0]; c = (n - 1) / 2
    oy, ox = np.mgrid[0:n, 0:n]
    dxo, dyo = (ox - c) * 10.0, (c - oy) * 10.0
    ro = np.hypot(dxo, dyo); azo = np.degrees(np.arctan2(dxo, dyo)) % 360
    rows = []
    for th in DIRS:
        sec = angdiff(azo, th) <= 22.5
        near, mid, all1 = sec & (ro <= 200), sec & (ro > 200) & (ro <= 1000), sec & (ro <= 1000)
        tree = Hm > 0
        z0 = np.where(tree, 0.1 * Hm, 0.03)
        # Winstral Sx over 7 rays
        sx = []
        for a in np.radians(th + np.linspace(-22.5, 22.5, 7)):
            d = np.arange(res * 2, 1000 + res, res)
            cc, rw = col0 + d * np.sin(a) / res, row0 - d * np.cos(a) / res
            zz = map_coordinates(np.nan_to_num(Z, nan=zc), [rw, cc], order=1, mode="nearest")
            sx.append(np.degrees(np.max(np.arctan((zz - zc - z_sensor) / d))))
        rows.append(dict(src=src, id=str(sid), dir=th, tree_near=tree[near].mean(), tree_mid=tree[mid].mean(),
                         h_mid=Hm[all1 & tree].mean() if (all1 & tree).any() else 0.0,
                         z0_eff=float(np.exp(np.log(z0[all1]).mean())), sx=float(np.mean(sx)), elev=zc, tpi_1km=tpi))
    return rows


if __name__ == "__main__":
    tm, smm = sd.trv_meta(), sd.smhi_meta()
    tm = tm[tm.dist_km <= 120]
    out = []
    for _, s in tm.iterrows():
        out += station_features("trv", s.id, s.lat, s.lon, 6.0)
    for _, s in smm.iterrows():
        out += station_features("smhi", s.id, s.lat, s.lon, 10.0)
    F = pd.DataFrame(out); F.to_parquet("/home/claude/se/features_16dir.parquet"); print(F.shape); print(F.describe().round(3).T)
