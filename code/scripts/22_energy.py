"""Air-drag energy of a tractor-semitrailer passing the roadside weather stations: how much the wind changes it, and
how well each estimator of the roadside wind predicts that change. -> SWW_NO_OUT/energy.pkl

Drag (smhi_trv_wind.drag): C_dA(0) = 5.8 m2 with the VECTO generic crosswind correction (main) or Askerdal Model 1
(upper bound). rho = 1.25 kg m-3. Every wind (observed and estimated) is referred to vehicle height 3.4 m with the same
open-terrain log factor, so the conversion does not favour any estimator; its effect on the size of the penalty is
reported separately (uncertainty band). Each station-hour is driven in both directions along the road axis.

Stations enter the energy analysis only with adequate direction coverage (>= 2000 h and >= 50 % of the speed hours in
Norway, >= 50 % in Sweden) after a stuck-direction check; all estimators are scored on the same hours.
Intervals: station-cluster bootstrap (2000 resamples) for the roadside stations; hour bootstrap for the vehicle.

Inputs: norway_results.pkl (21_norway_cv_final.py), cv_full.pkl (14_final_cv.py, Sweden), metnordic_roadside.parquet
(25_metnordic_roadside.py, optional), vehicle_validation_rows.parquet (12_vehicle_norway.py).
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import assess, drag
from smhi_trv_wind.model import physics

NO = Path(os.environ.get("SWW_NO_OUT", "output/no")); SE = Path(os.environ.get("SWW_SE_OUT", "output/se"))
DNO = Path(os.environ.get("SWW_NO_DATA", "data/no"))
RHO, Z_VEH = assess.RHO, 3.4
PAIRS = [("corrected", "nearest station", "corr"), ("corrected", "nearest road station", "corr"),
         ("corrected", "rescaled nearest", "corr"), ("corrected", "no wind", "mae"), ("corrected", "nearest road station", "mae"),
         ("corrected", "rescaled nearest", "mae"), ("corrected", "nearest station", "mae"), ("calibrated", "corrected", "mae"),
         ("corrected", "nearest road station", "skill"), ("corrected", "rescaled nearest", "skill"),
         ("corrected", "MET Nordic", "mae"), ("corrected", "MET Nordic", "skill"), ("corrected", "MET Nordic", "corr"),
         ("corrected", "no wind", "strong_mae")]
OUT = {}

for country in ("Norway", "Sweden"):
    P, O, blk, D, mn, z_road = assess.load_country(country, NO, SE, DNO)
    ests, wq, info = assess.build_estimators(P, O, blk, D, mn)
    ws, road = O["ws"].astype(float), np.asarray(O["road"], float)
    if country == "Sweden":                       # 287 h: the 2000-h threshold does not apply, keep the 50 % rule
        _, ok_se, cov, _ = assess.direction_coverage(ws, O["wd"].astype(float), min_hours=100)
        info["dir_ok"], info["coverage"] = ok_se, cov
    dir_ok = info["dir_ok"]
    fh = physics.height_factor(Z_VEH, z_road, 0.03)
    main = {k: v for k, v in ests.items() if k != "MET Nordic"}
    main["no wind"] = (0 * ws, 0 * ws)
    ok = np.isfinite(ws) & np.isfinite(wq) & np.isfinite(road)[None, :] & dir_ok[None, :]
    for k, (u, d) in main.items():
        if k != "no wind":
            ok &= np.isfinite(u) & np.isfinite(d)
    o = dict(info=info, fh=fh, n_station=int(dir_ok.sum()), n_hours=int(ok.sum()), names=O.get("name"), ids=O["ids"])
    for model, speeds in (("vecto", (60, 70, 80, 90)), ("model1", (80,))):
        for v_kmh in speeds:
            S = assess.station_sums(ws, wq, road, main, ok, fh, model=model, v=v_kmh / 3.6, keep=dir_ok)
            if v_kmh == 80:
                o[(model, v_kmh)] = assess.bootstrap(S, pairs=PAIRS); o[(model, v_kmh)]["sums"] = S
            else:
                o[(model, v_kmh)] = dict(point=assess.energy_metrics(S), sums=S)
            pt = o[(model, v_kmh)]["point"]
            print(country, model, v_kmh, "E0 %.1f penalty %.1f%%" % (S["E0"], pt["pen_mean"]),
                  {k: (round(x["mae"], 2), round(x["skill"], 2), round(x["corr"], 2)) for k, x in pt.items() if isinstance(x, dict)}, flush=True)
    if mn is not None:                            # the MET Nordic hours (every third hour), all estimators on them
        em = dict(main); em["MET Nordic"] = mn
        okm = ok & np.isfinite(mn[0]) & np.isfinite(mn[1])
        S = assess.station_sums(ws, wq, road, em, okm, fh, keep=dir_ok)
        o["metnordic_hours"] = assess.bootstrap(S, pairs=PAIRS); o["metnordic_hours"]["sums"] = S
        o["metnordic_n_hours"] = int(okm.sum())
        pt = o["metnordic_hours"]["point"]
        print(country, "MET Nordic hours", {k: (round(x["mae"], 2), round(x["skill"], 2), round(x["corr"], 2)) for k, x in pt.items() if isinstance(x, dict)}, flush=True)

    # ---- uncertainty band of the observed penalty (observed wind; same stations and hours)
    E0 = float(drag.energy_kwh100(assess.V80, 0.0, 0.0, rho=RHO))
    okb = ok

    def pen(fac, model="vecto", ti=0.0):
        e0 = float(drag.energy_kwh100(assess.V80, 0.0, 0.0, rho=RHO, model=model))
        acc, n = 0.0, 0
        x, w = (np.array([0.0]), np.array([1.0])) if ti == 0 else np.polynomial.hermite_e.hermegauss(7)
        w = w / w.sum()
        for k in (0.0, 180.0):
            u = ws * fac
            h, c = drag.components(u, wq, road[None, :] + k)
            a = np.radians(wq - (road[None, :] + k))
            su, sv = ti * u, 0.8 * ti * u             # longitudinal / lateral turbulence of the wind
            tot = 0
            for xi, wi in zip(x, w):
                for xj, wj in zip(x, w):
                    dh = xi * su * np.cos(a) - xj * sv * np.sin(a); dc = xi * su * np.sin(a) + xj * sv * np.cos(a)
                    tot = tot + wi * wj * drag.energy_kwh100(assess.V80, h + dh, c + dc, rho=RHO, model=model)
            acc += np.nansum(np.where(okb, tot, np.nan)); n += okb.sum()
        return 100 * (acc / n / e0 - 1)
    band = {"reference": pen(fh)}
    if country == "Norway":
        band["sensor 6 m"] = pen(physics.height_factor(Z_VEH, 6.0, 0.03))
    for z0 in (0.01, 0.1, 0.3):
        band[f"z0 {z0}"] = pen(physics.height_factor(Z_VEH, z_road, z0))
    band["z0 0.3 d 2"] = pen(np.log((Z_VEH - 2) / 0.3) / np.log((z_road - 2) / 0.3))
    band["no reduction"] = pen(1.0)
    band["model1"] = pen(fh, "model1")
    for ti in (0.15, 0.3):
        band[f"TI {ti}"] = pen(fh, ti=ti)
    o["band"] = band
    print(country, "band", {k: round(v, 1) for k, v in band.items()}, flush=True)
    b_ = []
    for k in (0.0, 180.0):
        h, c = drag.components(ws * fh, wq, road[None, :] + k)
        b_.append(np.degrees(np.arctan2(np.abs(c), assess.V80 + h))[okb])
    b_ = np.concatenate(b_)
    o["beta_cap_frac"], o["beta_q"] = float(np.mean(b_ > drag.BETA_PEAK)), np.percentile(b_, [50, 90, 99])
    OUT[country] = o

# ---- vehicle (NOR-XWIND): measured wind at 3.4 m; results per reference hour, hour bootstrap
V = pd.read_parquet(NO / "vehicle_validation_rows.parquet")
V["h"] = pd.to_datetime(V.t, utc=True).dt.round("1h")         # the reference hour every station estimate belongs to
rng = np.random.default_rng(3)
veh = dict(n_rows=len(V), n_hours=int(V.h.nunique()), hours_by_site=V.groupby("site").h.nunique().to_dict())
cols = {"no wind": None, "nearest station": ("ref_raw", "ref_dir"), "nearest road station": ("road_raw", "road_dir"),
        "MET Nordic": ("metnordic", "metnordic_dir"), "corrected": ("model", "model_dir")}
if "model_loc" in V:
    cols["corrected, local z0"] = ("model_loc", "model_dir")
for model in ("vecto", "model1"):
    E0m = float(drag.energy_kwh100(assess.V80, 0.0, 0.0, rho=RHO, model=model))
    Et = assess.energy(V.w_veh.values, V.dir_veh.values, V.heading.values, model)
    hourly = {}
    for k, cc in cols.items():
        Ee = np.full(len(V), E0m) if cc is None else assess.energy(V[cc[0]].values, V[cc[1]].values, V.heading.values, model)
        g = pd.DataFrame(dict(h=V.h, site=V.site, t=Et, e=Ee, ws=V.w_veh, u=(V[cc[0]] if cc else 0 * V.w_veh)))
        hourly[k] = g.groupby("h").agg(t=("t", "mean"), e=("e", "mean"), ws=("ws", "mean"), u=("u", "mean"), site=("site", "first"))
    H = len(hourly["corrected"]); boot = rng.integers(0, H, (2000, H))
    res = {}
    for k, hh in hourly.items():
        err = np.abs(hh.e - hh.t).values
        res[k] = dict(mae=float(err.mean()), mae_ci=np.percentile(err[boot].mean(1), [2.5, 97.5]),
                      bias=float(100 * (hh.e.mean() / hh.t.mean() - 1)),
                      bias_ci=np.percentile(100 * (hh.e.values[boot].mean(1) / hh.t.values[boot].mean(1) - 1), [2.5, 97.5]),
                      spd_mae=float(np.abs(hh.u - hh.ws).mean()) if k != "no wind" else np.nan,
                      spd_bias=float(100 * (hh.u.mean() / hh.ws.mean() - 1)) if k != "no wind" else np.nan)
        if k != "corrected":
            d = np.abs(hourly["corrected"].e - hourly["corrected"].t).values - err
            res[k]["corr_minus_this"] = (float(d.mean()), np.percentile(d[boot].mean(1), [2.5, 97.5]))
    veh[model] = res; veh[model + "_hourly"] = hourly
    veh[model + "_true"] = dict(E0=E0m, mean=float(np.mean(Et)), pen=float(100 * (np.mean(Et) / E0m - 1)))
    print("vehicle", model, {k: (round(v["mae"], 1), v["mae_ci"].round(1), round(v["bias"])) for k, v in res.items()}, flush=True)
near = V[V.road_km <= 0.1]                       # van within 100 m of a roadside station
veh["near_station"] = dict(n_seg=len(near), n_hours=int(near.h.nunique()), ratio=float(near.w_veh.mean() / near.road_raw.mean()),
                           implied_factor=float(near.w_veh.mean() / near.road_raw.mean() * physics.height_factor(Z_VEH, 10.0, 0.03)))
print("near station", veh["near_station"])
OUT["vehicle"] = veh
pickle.dump(OUT, open(NO / "energy.pkl", "wb"))
print("wrote", NO / "energy.pkl")
