"""Vehicle-height validation in Norway: NOR-XWIND segment winds (van-mounted anemometer at 3.4 m, calibrated
and converted to ground-fixed wind in earlier work; Dovre E6 2020-10-29/31, Frøya 2020-11-05) against the
station-based estimate. The model is the one fitted on the Statens vegvesen stations (2024-2025, no vehicle data).

Estimates compared, all from data available without the vehicle:
  ref_raw     nearest reference station, log-law height factor 10 m -> 3.4 m (z0 = 0.03 m)
  road_raw    nearest Statens vegvesen station, same height factor
  metnordic   MET Nordic analysis interpolated to the segment (column w_mn of the bins), log-law to 3.4 m
  model       correction chain + roadside exposure model + combination of all reference stations, predicted for a
              10 m roadside anemometer and brought to 3.4 m with the same open-terrain factor as the other estimates
  model_loc   the same with the local-z0 log profile from 10 m to 3.4 m (sensitivity)
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import Settings
from smhi_trv_wind.geo import haversine_km, sector, wrap180, wrap90
from smhi_trv_wind.model import physics
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine

Z_VEH = 3.4
cfg = Settings.norway(max_pair_km=120.0)
OUT = Path(os.environ.get("SWW_OUT", "./output"))
E = Engine(cfg, pd.read_parquet(cfg.paths.root / "features_variants.parquet"), t_split="2025-01-01", t_end=cfg.test_end)
if (OUT / "norway_results.pkl").exists():
    R = pickle.load(open(OUT / "norway_results.pkl", "rb"))
    b, d, smear = R["coef"]["b"], R["coef"]["d"], R["coef"]["smear"]
else:                                                           # fit the final model on all roadside data
    E.A = np.ones(len(E.H), bool)
    full = E.run(RECOMMENDED)
    b, d, smear = full["b"], full["d"], full["smear"]
C = E.site_factors(RECOMMENDED)                                 # [ref x 8 sectors], 2023
lz_ref = E.Vs["lnz0_L100_H22.5"]                                # SMHI-role footprint L = 100 m
S = pd.read_csv(cfg.paths.root / "stations_region.csv"); S["id"] = S.id.astype(str); S = S.set_index("id")
W = pd.read_parquet(cfg.paths.root / "wind_vehicle_days.parquet"); W["id"] = W.id.astype(str)
ref_ids = [s for s in C.index if s in set(W.id)]
road_ids = [s for s in S.index[S.grp == "SVV"] if s in set(W.id)]
ws = W.pivot_table(index="time", columns="id", values="ws"); wd = W.pivot_table(index="time", columns="id", values="wd").reindex(columns=ws.columns)
F = pd.read_parquet(cfg.paths.root / "vehicle_segment_features.parquet")
piv = lambda col: F.pivot(index="id", columns="dir", values=col)
T_near, T_far, XH, LZ = piv("tree_R100_H22.5"), piv("tree_R1000_H180"), np.log(piv("xh_edge").clip(*cfg.xh_clip)), piv("lnz0_L100_H22.5")
T_dir = piv("tree_R200_H22.5")
segmeta = F.drop_duplicates("id").set_index("id")[["site", "seg", "lat", "lon", "road_bearing"]]
fh = physics.height_factor(Z_VEH, 10.0, 0.03)

rows = []
NXD = Path(os.environ.get("NORXWIND_DIR", "data/norxwind")) / "data"          # vehicle/build_all.py output
for site, f in (("dovre", NXD / "bins_dovre.parquet"), ("froya", NXD / "bins_froya.parquet")):
    B = pd.read_parquet(f)
    for _, r in B.iterrows():
        sid = f"{site}_{r.seg}"
        h = pd.Timestamp(r.t).round("1h")                   # reference hour (hourly station values)
        if h not in ws.index or sid not in segmeta.index:
            continue
        u, w = ws.loc[h, ref_ids].astype(float), wd.loc[h, ref_ids].astype(float)
        ok = u.notna() & w.notna()
        if ok.sum() < 5:
            continue
        knet = sector(physics.network_direction(u[ok].to_frame().T, w[ok].to_frame().T).iloc[0], 16)
        k8 = sector(physics.network_direction(u[ok].to_frame().T, w[ok].to_frame().T).iloc[0], 8)
        P = np.array([physics.to_potential(np.exp(lz_ref.loc[s, knet]), cfg) for s in ref_ids])
        cs = C.reindex(ref_ids)
        corr = u.values * P * np.exp(-cs.mean(axis=1).values)          # one site factor per station (final model)
        lnU_net = np.log(max(np.nanmean(corr), 0.5))
        # the exposure model predicts a 10 m roadside anemometer; it is brought to 3.4 m with the same open-terrain
        # factor as every other estimate (main), or with the local-z0 profile (sensitivity, as in the first version)
        F10 = physics.to_site(np.exp(LZ.loc[sid, knet]), cfg, z=10.0)
        Floc = physics.to_site(np.exp(LZ.loc[sid, knet]), cfg, z=Z_VEH)
        x = np.array([1.0, T_near.loc[sid, knet], XH.loc[sid, knet], lnU_net, T_far.loc[sid, knet]])
        U10 = corr * F10 * np.exp(x @ b) * smear
        dist = np.array([float(haversine_km(r.lat, r.lon, S.loc[s, "lat"], S.loc[s, "lon"])) for s in ref_ids])
        good = np.isfinite(U10) & (U10 > 0)
        wgt = 1 / np.clip(dist, 3, None) ** 1.5
        U10c = float(np.exp(np.sum(wgt[good] * np.log(U10[good])) / np.sum(wgt[good])))
        U = fh * U10c
        U_loc = U10c * Floc / F10
        alpha = wrap90(w.values - segmeta.loc[sid, "road_bearing"])
        wd4 = (w.values + d[0] + d[1] * np.sin(np.radians(2 * alpha)) * T_dir.loc[sid, knet]) % 360
        wdir = (dist <= 60) & np.isfinite(wd4)
        if wdir.sum() < 2:
            wdir = np.argsort(dist)[:2]
        wd_ = 1 / np.clip(dist[wdir], 3, None)
        phi = float(np.degrees(np.arctan2(np.sum(wd_ * np.sin(np.radians(wd4[wdir]))), np.sum(wd_ * np.cos(np.radians(wd4[wdir]))))) % 360)
        i_ref = int(np.nanargmin(np.where(ok.values, dist, np.nan)))
        dr = {s: float(haversine_km(r.lat, r.lon, S.loc[s, "lat"], S.loc[s, "lon"])) for s in road_ids if pd.notna(ws.loc[h, s])}
        s_road = min(dr, key=dr.get) if dr else None
        rows.append(dict(site=site, seg=r.seg, t=r.t, w_veh=r.w_veh, dir_veh=r.dir_veh, heading=r.heading,
                         model=U, model_loc=U_loc, model_dir=phi, ref_raw=fh * u.values[i_ref], ref_dir=w.values[i_ref], ref_km=dist[i_ref],
                         road_raw=fh * ws.loc[h, s_road] if s_road else np.nan, road_dir=wd.loc[h, s_road] if s_road else np.nan,
                         road_km=dr.get(s_road, np.nan), metnordic=fh * r.w_mn, metnordic_dir=r.dir_mn))
V = pd.DataFrame(rows)


def comps(spd, dirn, heading):
    """head (+ = against the vehicle) and cross components of a wind blowing FROM dirn"""
    a = np.radians(dirn - heading)
    return spd * np.cos(a), spd * np.sin(a)


res = []
for site, g in list(V.groupby("site")) + [("all", V)]:
    hv, cv_ = comps(g.w_veh, g.dir_veh, g.heading)
    for k, dcol in (("ref_raw", "ref_dir"), ("road_raw", "road_dir"), ("metnordic", "metnordic_dir"), ("model", "model_dir"), ("model_loc", "model_dir")):
        hm, cm = comps(g[k], g[dcol], g.heading)
        ok = np.isfinite(g[k])
        res.append(dict(site=site, estimate=k, n=int(ok.sum()), mae=float(np.mean(np.abs(g[k] - g.w_veh)[ok])),
                        bias=float(np.mean((g[k] - g.w_veh)[ok])), dir_mae=float(np.mean(np.abs(wrap180(g[dcol] - g.dir_veh))[ok & (g.w_veh >= 2)])),
                        head_mae=float(np.mean(np.abs(hm - hv)[ok])), cross_mae=float(np.mean(np.abs(cm - cv_)[ok]))))
Res = pd.DataFrame(res)
V.to_parquet(OUT / "vehicle_validation_rows.parquet"); Res.to_csv(OUT / "vehicle_validation.csv", index=False)
pd.set_option("display.width", 200)
print(Res.round(3).to_string(index=False))
