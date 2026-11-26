import numpy as np, pandas as pd, xarray as xr
from calib import load_passes, true_wind
from metnordic import interp
from terrain import at_direction

# reciprocal-pass calibration of the van anemometer on the Dovre E6 sessions (calib.py; 23 opposite-direction pairs,
# 1 km bins): sgn = +1 -> alpha = -29.91 deg, g = 0.959, cost 4.648 m2 s-2; sgn = -1 -> alpha = +29.94, g = 0.961,
# cost 4.724. The smaller cost (+1) is used; the same mounting solution (+1, alpha ~ -30 deg) is found independently on
# 2020-11-05 (Froya + Dolmsund) and 2020-11-21 (Naeroysund). Drag does not depend on the sign (see build_all.py).
CAL = dict(alpha=-29.91, gx=0.959, gy=0.959, sgn=1)
SEG = 500

def build():
    d, line = load_passes()
    we, wn = true_wind(d, CAL["alpha"], CAL["gx"], CAL["gy"], CAL["sgn"])
    d["we"], d["wn"] = we, wn
    mu, mv, alt = interp(xr.open_dataset("metnordic_dovre.nc"), d.lat.values, d.lon.values, d.time.values)
    d["mu"], d["mv"] = mu, mv
    d["seg"] = (d.s // SEG).astype(int)
    g = d.groupby(["pass_id", "seg"])
    h = np.radians(d.heading.values)
    d["hs"], d["hc"] = np.sin(h), np.cos(h)
    S = g.agg(day=("day", "first"), session=("session", "first"), dirE=("dirE", "first"), t=("time", "mean"),
              t0=("t0", "first"), n=("we", "size"), we=("we", "mean"), wn=("wn", "mean"),
              mu=("mu", "mean"), mv=("mv", "mean"), v=("v", "mean"), hs=("hs", "mean"), hc=("hc", "mean"),
              s_mid=("s", "mean")).reset_index()
    S = S[S.n >= 10].copy()
    S["w_veh"] = np.hypot(S.we, S.wn); S["w_mn"] = np.hypot(S.mu, S.mv)
    S["dir_veh"] = np.degrees(np.arctan2(-S.we, -S.wn)) % 360
    S["dir_mn"] = np.degrees(np.arctan2(-S.mu, -S.mv)) % 360
    S["heading"] = np.degrees(np.arctan2(S.hs, S.hc)) % 360
    hr = np.radians(S.heading)
    S["head_veh"] = -(S.we * np.sin(hr) + S.wn * np.cos(hr))       # 顶风分量（正 = 迎面来风）
    S["head_mn"] = -(S.mu * np.sin(hr) + S.mv * np.cos(hr))
    S["r"] = np.log(S.w_veh / S.w_mn)
    # 地形/地表特征：在路段中点、按 MET Nordic 来风方向取值
    F = np.load("route_features.npz")
    k = np.clip(np.searchsorted(F["s"], S.s_mid.values), 0, len(F["s"]) - 1)
    S["elev"] = F["z0"][k]
    S["rel_elev"] = pd.Series(F["rel"][k]).values
    S["sx_up"] = at_direction(F["sx"], k, S.dir_mn.values)
    for gname in ["trees", "low_veg", "water", "built"]:
        S[f"{gname}_up"] = at_direction(F[f"frac_{gname}"], k, S.dir_mn.values)
    S = S.sort_values(["t0", "t"]).reset_index(drop=True)
    return d, S
