"""Extra direction-dependent features (16 wind-from sectors) for TrV and SMHI stations, from NMD2023 object height (10 m):
  x_edge      distance [m] along the upwind rays to the first pixel with objects > 5 m (median of 7 rays, capped at 1500 m)
  h_edge      mean object height over the first 50 m of that obstacle (median of rays)
  xh_edge     x_edge / h_edge (distance in obstacle heights; 1500/h if no obstacle)
  z0_fp_L     footprint-weighted effective z0: exp(sum w ln z0 / sum w), w = exp(-r/L) inside the sector, L in LS
And for TrV stations the road bearing at the snapped road point (OSRM, OpenStreetMap road network).
"""
import numpy as np, pandas as pd, rasterio, requests, time
from rasterio.windows import from_bounds
from scipy.ndimage import map_coordinates
from features import OBJ, CODE2H, T, DIRS, angdiff
import sedata as sd

LS = [25, 50, 100, 200, 400, 800]


def extra(src, sid, lat, lon):
    x, y = T.transform(lon, lat)
    with rasterio.open(OBJ) as r:
        H = r.read(1, window=from_bounds(x - 2000, y - 2000, x + 2000, y + 2000, r.transform)).astype(int)
    lut = np.zeros(256); [lut.__setitem__(k, v) for k, v in CODE2H.items()]
    Hm = lut[H]; n = Hm.shape[0]; c = (n - 1) / 2
    oy, ox = np.mgrid[0:n, 0:n]; dx, dy = (ox - c) * 10.0, (c - oy) * 10.0
    ro = np.hypot(dx, dy); az = np.degrees(np.arctan2(dx, dy)) % 360
    lnz0 = np.log(np.where(Hm > 0, 0.1 * Hm, 0.03))
    rows = []
    d = np.arange(5, 1505, 5.0)
    for th in DIRS:
        sec = (angdiff(az, th) <= 22.5) & (ro >= 10) & (ro <= 2000)
        row = dict(src=src, id=str(sid), dir=th)
        for L in LS:
            w = np.exp(-ro[sec] / L); row[f"z0_fp_{L}"] = float(np.exp(np.sum(w * lnz0[sec]) / np.sum(w)))
        xs, hs = [], []
        for a in np.radians(th + np.linspace(-22.5, 22.5, 7)):
            hh = map_coordinates(Hm, [c - d * np.cos(a) / 10, c + d * np.sin(a) / 10], order=0, mode="nearest")
            i = np.argmax(hh > 0)
            if hh[i] > 0:
                xs.append(d[i]); hs.append(hh[i:i + 10][hh[i:i + 10] > 0].mean())
            else:
                xs.append(1500.0); hs.append(np.nan)
        row["x_edge"] = float(np.median(xs)); row["h_edge"] = float(np.nanmedian(hs)) if np.isfinite(hs).any() else 15.0
        row["xh_edge"] = row["x_edge"] / row["h_edge"]
        rows.append(row)
    return rows


def road_bearing(lat, lon):
    u = f"https://router.project-osrm.org/route/v1/driving/{lon},{lat};{lon},{lat + 0.02}?steps=true&overview=false"
    for k in range(3):
        try:
            j = requests.get(u, timeout=20).json()
            st = j["routes"][0]["legs"][0]["steps"][0]
            w = j["waypoints"][0]
            return st["maneuver"]["bearing_after"] % 180, w["distance"], st.get("name", "")
        except Exception:
            time.sleep(2)
    return np.nan, np.nan, ""


if __name__ == "__main__":
    tm, smm = sd.trv_meta(), sd.smhi_meta(); tm = tm[tm.dist_km <= 120]
    out = []
    for _, s in tm.iterrows(): out += extra("trv", s.id, s.lat, s.lon)
    for _, s in smm.iterrows(): out += extra("smhi", s.id, s.lat, s.lon)
    pd.DataFrame(out).to_parquet("/home/claude/se/features2_16dir.parquet"); print("features2 done", len(out), flush=True)
    rb = []
    for _, s in tm.iterrows():
        b, dist, name = road_bearing(s.lat, s.lon); rb.append(dict(id=str(s.id), road_bearing=b, snap_m=dist, road=name)); time.sleep(1.0)
    pd.DataFrame(rb).to_csv("/home/claude/se/trv/road_bearing.csv", index=False); print("roads done", flush=True)
