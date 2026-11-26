"""Air-drag energy of a tractor-semitrailer passing the roadside stations: how much the wind changes it, and how well it
is predicted with no wind, with the nearest reference station and with the corrected estimate.

Drag (smhi_trv_wind.drag): C_dA(0) = 5.8 m2 (typical VECTO-certified EU 4x2 long-haul tractor-trailer) with the VECTO
generic crosswind correction (main) or Askerdal Model 1 (upper bound of the crosswind sensitivity). rho = 1.25 kg m-3.
All winds (observed and estimated) are referred to vehicle height 3.4 m with the same open-terrain log factor, so
the comparison between estimators is not affected by the conversion. Energy per distance = force, in kWh/100 km.
Each station-hour is driven in both directions along the road axis; a round trip is the sum of both.

Inputs: SWW_NO_OUT/norway_results.pkl (cv_final predictions, out of time and out of station),
        SWW_SE_OUT/cv_full.pkl, SWW_NO_OUT/vehicle_validation_rows.parquet -> SWW_NO_OUT/energy.pkl
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import drag
from smhi_trv_wind.model import physics

NO = Path(os.environ.get("SWW_NO_OUT", "/tmp/nout2")); SE = Path(os.environ.get("SWW_SE_OUT", "/tmp/sww8"))
RHO, Z_VEH = 1.25, 3.4
MODEL = "vecto"


def e_kwh100(v, speed, wfrom, heading):
    """air-drag energy [kWh/100 km] at vehicle speed v [m/s] for wind speed/direction and heading arrays"""
    h, c = drag.components(speed, wfrom, heading)
    return drag.energy_kwh100(v, h, c, rho=RHO, model=MODEL)


def e_still(v):
    return drag.energy_kwh100(v, 0.0, 0.0, rho=RHO, model=MODEL)


def country_energy(P, O, z_road, v):
    fh = physics.height_factor(Z_VEH, z_road, 0.03)
    ws, wd, road = O["ws"].astype(float) * fh, O["wd"].astype(float), np.asarray(O["road"], float)
    est = {"no wind": (np.zeros_like(ws), np.zeros_like(ws)),
           "nearest station": (P["U0n"].astype(float) * fh, P["wd0n"].astype(float)),
           "corrected": (P["U"].astype(float) * fh, P["wdm"].astype(float))}
    ok = np.isfinite(ws) & np.isfinite(wd) & np.isfinite(road)[None, :]
    for k, (u, d) in est.items():
        if k != "no wind":
            ok &= np.isfinite(u) & np.isfinite(d)
    E0 = e_still(v)
    out = dict(E0=E0, v=v, fh=fh, n=int(ok.sum()))
    Et = [np.where(ok, e_kwh100(v, ws, wd, road[None, :] + k), np.nan) for k in (0.0, 180.0)]
    Ee = {name: [np.where(ok, e_kwh100(v, u, d, road[None, :] + k), np.nan) for k in (0.0, 180.0)] for name, (u, d) in est.items()}
    one_t = np.concatenate([e[ok] for e in Et]); rt_t = (Et[0] + Et[1])[ok] / 2
    out["true_one"] = one_t; out["true_rt"] = rt_t
    # decomposition of the true wind effect: head/tail only and crosswind only
    h, c = drag.components(ws, wd, road[None, :])
    head_only = np.concatenate([drag.energy_kwh100(v, s * h, 0.0 * h, rho=RHO, model=MODEL)[ok] for s in (1, -1)])
    cross_only = drag.energy_kwh100(v, 0.0 * c, c, rho=RHO, model=MODEL)[ok]
    out["head_only_one"] = head_only; out["cross_only"] = cross_only
    st = np.broadcast_to(np.arange(ws.shape[1]), ws.shape)[ok]
    out["metrics"] = {}
    for name in est:
        one_e = np.concatenate([e[ok] for e in Ee[name]]); rt_e = (Ee[name][0] + Ee[name][1])[ok] / 2
        st2 = np.concatenate([st, st])
        df1 = pd.DataFrame(dict(s=st2, err=np.abs(one_e - one_t), e=one_e, t=one_t))
        dfr = pd.DataFrame(dict(s=st, err=np.abs(rt_e - rt_t), e=rt_e, t=rt_t))
        g1 = df1.groupby("s"); gr = dfr.groupby("s")
        mean_bias = (g1.e.mean() / g1.t.mean() - 1)                          # long-term mean energy per road
        dwt = one_t - E0; dwe = one_e - E0                                    # wind-induced part
        out["metrics"][name] = dict(
            mae_one=float(g1.err.mean().median()), mae_rt=float(gr.err.mean().median()),
            mean_bias=mean_bias.values, mean_bias_med=float(mean_bias.median()),
            abs_mean_bias_med=float(mean_bias.abs().median()),
            ss_one=float(1 - np.mean((one_e - one_t) ** 2) / np.mean((E0 - one_t) ** 2)),
            ss_rt=float(1 - np.mean((rt_e - rt_t) ** 2) / np.mean((E0 - rt_t) ** 2)),
            corr_wind=float(np.corrcoef(dwe, dwt)[0, 1]) if name != "no wind" else np.nan,
            strong_mae=float(np.mean(np.abs(one_e - one_t)[np.abs(dwt) / E0 >= 0.1])),
            one_est=one_e if v == 80 / 3.6 else None, rt_est=rt_e if v == 80 / 3.6 else None, station=st2)
    return out


R = pickle.load(open(NO / "norway_results.pkl", "rb")); S = pickle.load(open(SE / "cv_full.pkl", "rb"))
V = pd.read_parquet(NO / "vehicle_validation_rows.parquet")
OUT = {}
for MODEL in ("vecto", "model1"):
    for v_kmh in ((60, 70, 80, 90) if MODEL == "vecto" else (80,)):
        v = v_kmh / 3.6
        OUT[("Norway", v_kmh, MODEL)] = country_energy(R["pred_final"], R["obs"], 10.0, v)
        OUT[("Sweden", v_kmh, MODEL)] = country_energy(S["pred"], S["obs"], 6.0, v)
        for c in ("Norway", "Sweden"):
            o = OUT[(c, v_kmh, MODEL)]; m = o["metrics"]
            print(MODEL, c, v_kmh, "E0 %.1f wind adds %.1f%%" % (o["E0"], 100 * (np.mean(o["true_one"]) / o["E0"] - 1)),
                  {k: (round(x["mae_one"], 2), round(x["mae_rt"], 2), round(100 * x["mean_bias_med"], 1), round(x["ss_one"], 2), round(x["ss_rt"], 2)) for k, x in m.items()}, flush=True)
    # vehicle segments (NOR-XWIND): wind measured at 3.4 m, energy along the measured heading, means per hour
    v = 80 / 3.6
    Vt = e_kwh100(v, V.w_veh.values, V.dir_veh.values, V.heading.values)
    veh = {"E0": e_still(v), "rows": len(V), "true_mean": float(np.mean(Vt)), "true_q": np.percentile(Vt, [5, 50, 95])}
    H = V.t.dt.floor("h")
    for name, (u, d) in {"no wind": (0 * V.w_veh.values, 0 * V.w_veh.values), "nearest MET/Avinor station": (V.ref_raw.values, V.ref_dir.values),
                         "nearest road weather station": (V.road_raw.values, V.road_dir.values), "MET Nordic": (V.metnordic.values, V.metnordic_dir.values),
                         "corrected": (V.model.values, V.model_dir.values)}.items():
        Ve = e_kwh100(v, u, d, V.heading.values)
        hourly = pd.DataFrame(dict(h=H, t=Vt, e=Ve)).groupby("h").mean()
        veh[name] = dict(mae_seg=float(np.mean(np.abs(Ve - Vt))), mae_hour=float(np.mean(np.abs(hourly.e - hourly.t))),
                         bias=float(np.mean(Ve) / np.mean(Vt) - 1))
        print("vehicle", MODEL, name, {k: round(x, 3) for k, x in veh[name].items()}, flush=True)
    OUT[("vehicle", MODEL)] = veh
pickle.dump(OUT, open(NO / "energy.pkl", "wb"))
print("wrote", NO / "energy.pkl")
