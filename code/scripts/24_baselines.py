"""Wind-speed comparison of all roadside estimators, reliability of the corrected estimate, and direction coverage
-> SWW_NO_OUT/baselines.pkl  (review round 1: M3, M4, M9)

Estimators (smhi_trv_wind.assess.build_estimators; all out of time block and out of station):
  nearest station        nearest reference (MET/Avinor or SMHI) station with data in the hour, as measured
  rescaled nearest       the same times one constant (ratio of means learned on the other blocks and stations)
  nearest road station   the nearest OTHER roadside weather station with data in the hour (leave-one-out)
  corrected              the correction chain and exposure model of this paper
  calibrated             the corrected estimate recalibrated by bins of the estimate (cross-fitted)
  MET Nordic             1-km analysis interpolated to the station (Norway, every third hour)
"""
import os
import pickle
from pathlib import Path

import numpy as np

from smhi_trv_wind import assess

NO = Path(os.environ.get("SWW_NO_OUT", "output/no")); SE = Path(os.environ.get("SWW_SE_OUT", "output/se"))
DNO = Path(os.environ.get("SWW_NO_DATA", "data/no"))
OUT = {}
for country in ("Norway", "Sweden"):
    P, O, blk, D, mn, _, feats = assess.load_country(country, NO, SE, DNO)
    ests, wq, info = assess.build_estimators(P, O, blk, D, mn, feats)
    ws = O["ws"].astype(float)
    spd = {k: u for k, (u, d) in ests.items()}
    o = dict(info=info, names=O.get("name"), ids=O["ids"])
    o["speed_all"], o["mae_station"] = assess.speed_metrics(spd, ws)
    common = np.isfinite(ws)
    for k, u in spd.items():
        if not k.startswith("MET Nordic"):
            common &= np.isfinite(u)
    o["speed_common"], o["mae_station_common"] = assess.speed_metrics({k: np.where(common, u, np.nan) for k, u in spd.items() if not k.startswith("MET Nordic")}, ws)
    if mn is not None:
        cm = common & np.isfinite(mn[0])
        for k in spd:
            if k.startswith("MET Nordic"):
                cm &= np.isfinite(spd[k])
        o["speed_mn"], o["mae_station_mn"] = assess.speed_metrics({k: np.where(cm, u, np.nan) for k, u in spd.items()}, ws)
        o["by_obs_mn"] = assess.by_observed({k: np.where(cm, u, np.nan) for k, u in spd.items()}, ws)   # strong wind on common hours
        rng_ = np.random.default_rng(5); ms = o["mae_station_mn"]; ns_ = len(ms["corrected"])
        o["paired_mn"] = {}
        for k in ("MET Nordic", "MET Nordic, rescaled", "MET Nordic + exposure"):
            d_ = ms["corrected"] - ms[k]
            o["paired_mn"][k] = dict(mean=float(np.nanmean(d_)), se=float(np.nanstd(d_, ddof=1) / np.sqrt(np.isfinite(d_).sum())),
                                     n_better=int(np.sum(d_ < 0)), n=int(np.isfinite(d_).sum()),
                                     ci_med=np.percentile([np.nanmedian(ms["corrected"][i]) - np.nanmedian(ms[k][i]) for i in rng_.integers(0, ns_, (2000, ns_))], [2.5, 97.5]))
        o["mn_hours"] = int(cm.any(1).sum())
    ok_dir = info["dir_ok"] if country == "Norway" else assess.direction_coverage(ws, O["wd"].astype(float), min_hours=100)[1]
    o["dir"] = assess.direction_mae({k: d for k, (u, d) in ests.items()}, ws, wq, P["U0n"].astype(float), ok_dir)
    if mn is not None:
        cmd = np.isfinite(mn[1]) & cm
        o["dir_mn"] = assess.direction_mae({k: np.where(cmd, d, np.nan) for k, (u, d) in ests.items()}, ws, wq, P["U0n"].astype(float), ok_dir)
    o["reliability"] = {k: assess.reliability(spd[k], ws) for k in ("corrected", "calibrated", "nearest station", "rescaled nearest", "nearest road station")}
    o["by_obs"] = assess.by_observed(spd, ws)
    OUT[country] = o
    print(country, "\n", o["speed_all"].round(3).to_string(index=False), "\n", o["speed_common"].round(3).to_string(index=False))
    if "speed_mn" in o:
        print(o["speed_mn"].round(3).to_string(index=False))
    print("direction", {k: round(v, 1) for k, v in o["dir"].items()}, o.get("dir_mn"), flush=True)
pickle.dump(OUT, open(NO / "baselines.pkl", "wb"))
print("wrote", NO / "baselines.pkl")
