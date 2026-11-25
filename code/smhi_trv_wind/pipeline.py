"""End-to-end pipeline: data -> pair table -> steps 1-6 -> metrics.

    run(cfg) returns a Result with
      D   pair table, one row per (TrV station, SMHI station, hour) with U0..U4, wd_s, wd4, features
      P   pair metrics            (evaluate.pair_metrics)
      C   combined estimates      (model.combine)
      S   station metrics         (evaluate.station_metrics)
      V   SMHI station value      (evaluate.smhi_value)
      site, exposure  fitted SiteFactors and ExposureModel
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import evaluate
from .config import Settings
from .data import sources
from .features.gis import sector_tables
from .geo import bearing_deg, haversine_km, sector
from .model import combine as comb, exposure, physics, site_factors


@dataclass
class Result:
    D: pd.DataFrame
    P: pd.DataFrame
    C: pd.DataFrame
    S: pd.DataFrame
    V: pd.DataFrame
    site: site_factors.SiteFactors
    site_raw: site_factors.SiteFactors
    exposure: exposure.ExposureModel
    trv: pd.DataFrame
    smhi: pd.DataFrame
    k: np.ndarray
    f_height: float = 0.912


def build_pairs(cfg: Settings, tabs: dict, site: site_factors.SiteFactors):
    """steps 1-3 for every (TrV, SMHI) pair within cfg.max_pair_km, hour by hour"""
    smhi_data, trv_data = sources(cfg)
    H = cfg.hours
    SM = smhi_data.stations(cfg)
    TM = trv_data.stations(cfg)
    tws, twd = trv_data.wind_matrices(cfg, H, ids=TM.index)
    keep = tws.columns[tws.notna().sum() >= cfg.min_trv_hours]
    tws, twd, TM = tws[keep], twd[keep], TM.loc[keep]
    sws, swd = smhi_data.wind_matrices(cfg, H, ids=SM.index)

    pot, k = physics.potential_wind(sws, swd, tabs["z0_smhi"], cfg)            # step 2a: SMHI -> potential
    k_site = sector(physics.network_direction(sws, swd), cfg.site_nsec)
    corr = site.correct(pot, k_site)                                           # step 3
    dT = trv_data.temperature_difference(cfg, H, ids=TM.index) if cfg.use_stability else None
    lnU_net = np.log(corr.mean(1).clip(lower=0.5))
    road = trv_data.road_bearing(cfg)
    f_height = physics.height_factor(cfg.z_trv, cfg.z_smhi, cfg.z0_open)       # step 1

    rows = []
    for j in TM.index:
        Fj = physics.to_site(tabs["z0_trv"].loc[j].reindex(k).values, cfg)     # step 2b: potential -> TrV site
        tree = tabs["tree_trv"].loc[j].reindex(k).values
        tree_dir = tabs["tree_dir_trv"].loc[j].reindex(k).values
        tree_far = tabs["tree_far_trv"].loc[j].reindex(k).values if cfg.tree_far_radius else np.nan
        lnxh = tabs["lnxh_trv"].loc[j].reindex(k).values
        dTj = dT[j].values if dT is not None else np.nan
        for s in SM.index:
            d = float(haversine_km(TM.loc[j, "lat"], TM.loc[j, "lon"], SM.loc[s, "lat"], SM.loc[s, "lon"]))
            if d > cfg.max_pair_km:
                continue
            rows.append(pd.DataFrame(dict(
                time=H, trv=j, smhi=s, dist=d, k=k, ws=tws[j].values, wd=twd[j].values,
                U0=sws[s].values, wd_s=swd[s].values,
                U1=f_height * sws[s].values,
                U2=pot[s].values * Fj,
                U3=corr[s].values * Fj,
                tree=tree, tree_dir=tree_dir, tree_far=tree_far, dT=dTj, lnxh=lnxh, lnU=lnU_net.values,
                road=road.get(j, np.nan),
                bearing=bearing_deg(TM.loc[j, "lat"], TM.loc[j, "lon"], SM.loc[s, "lat"], SM.loc[s, "lon"]))))
    return pd.concat(rows, ignore_index=True), TM, SM, k


def run(cfg: Settings | None = None, verbose: bool = True) -> Result:
    cfg = cfg or Settings()
    log = print if verbose else (lambda *a, **k: None)
    tabs = sector_tables(cfg)
    smhi_data, trv_data = sources(cfg)
    wsL, wdL = smhi_data.history(cfg)
    potL, _ = physics.potential_wind(wsL, wdL, tabs["z0_smhi"], cfg)
    site_raw = site_factors.fit(wsL, wdL, cfg)                                  # for comparison only
    site = site_factors.fit(potL, wdL, cfg)                                     # step 3 (used)
    log(f"site factors: {site.hours} training hours; station-mean sd raw {site_raw.c.mean(1).std():.3f} -> potential {site.c.mean(1).std():.3f}")
    D, TM, SM, k = build_pairs(cfg, tabs, site)
    log(f"pairs: {D.groupby(['trv', 'smhi']).ngroups} ({D.trv.nunique()} TrV x {D.smhi.nunique()} SMHI), {len(cfg.hours)} h")
    D, model = exposure.fit_loso(D, cfg)                                        # steps 4-5
    P = evaluate.pair_metrics(D)
    C = comb.combine(D, cfg)                                                    # step 6
    fh = physics.height_factor(cfg.z_trv, cfg.z_smhi, cfg.z0_open)
    S = evaluate.station_metrics(C, fh)
    V = evaluate.smhi_value(P[P.dist <= 80], SM.name)
    return Result(D, P, C, S, V, site, site_raw, model, TM, SM, k, fh)


def save(res: Result, cfg: Settings) -> None:
    out = cfg.paths.out
    out.mkdir(parents=True, exist_ok=True)
    res.D.to_parquet(out / "pairs_hourly.parquet")
    res.P.to_csv(out / "pair_metrics.csv", index=False)
    res.C.to_parquet(out / "combined_hourly.parquet")
    res.S.to_csv(out / "station_metrics.csv")
    res.V.to_csv(out / "smhi_station_value.csv")
    res.site.c.to_csv(out / "site_factors_potential.csv")
    res.site_raw.c.to_csv(out / "site_factors_raw.csv")
    evaluate.ladder(res.P).to_csv(out / "ladder.csv", index=False)
    json.dump(dict(coefficients=res.exposure.coefficients(), summary=evaluate.summary(res.S)),
              open(out / "model.json", "w"), indent=2)
