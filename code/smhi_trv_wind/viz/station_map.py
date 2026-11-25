"""Interactive Leaflet map (single self-contained HTML; Leaflet and OSM tiles load from the internet).

Click a TrV station: lines to its SMHI candidates (width = combination weight), per-step bias table,
time series of TrV obs / nearest SMHI raw / model. SMHI squares coloured by station value.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import evaluate
from ..config import Settings
from ..geo import haversine_km
from ..model import combine as comb

TEMPLATE = Path(__file__).with_name("map_template.html")
STEP_LABELS = ["原始 SMHI 10 m", "+ 高度换算 (×0.91)", "+ 地表粗糙度换算", "+ SMHI 站点因子", "+ 路边站地理模型"]


def _r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def _list(a, n=1, mask=None):
    a = np.asarray(a, dtype=float)
    ok = np.isfinite(a) if mask is None else np.isfinite(a) & mask
    return [round(float(v), n) if o else None for v, o in zip(a, ok)]


def selection_rules(res, cfg: Settings) -> list[list[str]]:
    """station-median speed / direction error of several selection rules"""
    out = [["最近 SMHI × 0.91（未校正）", f"{res.S.mae_raw.median():.2f}", f"{res.S.dir_raw.median():.1f}°"]]
    variants = [("最近 1 站，校正模型", dict(combine_km=0.0, combine_min_n=1)),
                ("40 km 内，1/距离", dict(combine_km=40.0, combine_min_n=1)),
                (f"{cfg.combine_km:.0f} km 内（至少 {cfg.combine_min_n} 站），1/距离", {}),
                (f"{cfg.max_pair_km:.0f} km 内全部，1/距离", dict(combine_km=cfg.max_pair_km, combine_min_n=1))]
    for name, kw in variants:
        S = res.S if not kw else evaluate.station_metrics(comb.combine(res.D, dataclasses.replace(cfg, **kw)))
        out.append([name, f"{S.mae_model.median():.2f}", f"{S.dir_model.median():.1f}°"])
    return out


def build(res, cfg: Settings, path: Path, route_lonlat: np.ndarray | None = None) -> Path:
    H = cfg.hours
    TM, SM, P, C, V = res.trv, res.smhi, res.P, res.C, res.V
    tree_mean = res.D.groupby("trv").tree.mean()
    trv = []
    for j in TM.index:
        p = P[P.trv == j].sort_values("mae_U4")
        c = C[C.trv == j].set_index("time").reindex(H)
        s = res.S.loc[j]
        sel = s.selected.split(",")
        dist = p.set_index("smhi").dist
        w = comb.weights(dist, sel, cfg)
        rv = None
        if route_lonlat is not None:
            rv = float(haversine_km(TM.loc[j, "lat"], TM.loc[j, "lon"], route_lonlat[:, 1], route_lonlat[:, 0]).min())
        trv.append(dict(
            id=j, name=TM.loc[j, "name"], lat=round(TM.loc[j, "lat"], 5), lon=round(TM.loc[j, "lon"], 5),
            tree=_r(tree_mean.get(j, np.nan)), rv40=_r(rv), near=s.nearest,
            mae_model=_r(s.mae_model), mae_raw=_r(s.mae_raw), dir_model=_r(s.dir_model), dir_raw=_r(s.dir_raw),
            pairs=[dict(s=r.smhi, d=_r(r.dist), sel=r.smhi in sel, w=_r(w.get(r.smhi, 0.0)),
                        **{f"b{i}": _r(getattr(r, f"bias_U{i}")) for i in range(5)},
                        **{f"m{i}": _r(getattr(r, f"mae_U{i}")) for i in range(5)},
                        r0=_r(r.r), r4=_r(r.r4), dr=_r(r.dir_raw), dm=_r(r.dir_mod)) for r in p.itertuples()],
            ts=dict(obs=_list(c.ws), near=_list(c.U0n), mod=_list(c.U),
                    wobs=_list(c.wd, 0, c.ws.values >= 1), wnear=_list(c.wd0n, 0, c.U0n.values >= 1),
                    wmod=_list(c.wdm, 0, c.U.values >= 1))))
    smhi = []
    for sid in SM.index:
        if sid not in V.index:
            continue
        v = V.loc[sid]
        smhi.append(dict(id=sid, name=SM.loc[sid, "name"], lat=round(SM.loc[sid, "lat"], 5), lon=round(SM.loc[sid, "lon"], 5),
                         f_raw=_r(res.site_raw.factor(sid)), f_pot=_r(res.site.factor(sid)),
                         cand=int(v.candidates), top1=int(v.top1), top3=int(v.top3), raw_top1=int(v.raw_top1),
                         q=_r(v.quality_eff, 3), mae0=_r(v.med_mae0), mae4=_r(v.med_mae4), r4=_r(v.med_r4), res01=True))
    lad = evaluate.ladder(P)
    co = res.exposure.coefficients()
    sm = evaluate.summary(res.S)
    summary = dict(
        ladder=[dict(step=STEP_LABELS[i], mae=_r(r.mae), bias=_r(r.bias)) for i, r in enumerate(lad.itertuples())],
        coefs=dict(speed=dict(b0=co["speed"]["b0"], tree=co["speed"]["b1_tree"], lnxh=co["speed"]["b2_lnxh"],
                              lnU=co["speed"]["b3_lnU"], lnU_tree=co["speed"]["b4_lnU_tree"]),
                   direction=dict(d0=co["direction"]["d0"], d1=co["direction"]["d1_s2a_tree"])),
        pairs=int(len(P)), trv=len(trv), hours=len(H),
        period=f"{H[0]:%Y-%m-%d %H:%M} – {H[-1]:%Y-%m-%d %H:%M} UTC",
        sf_sd_raw=_r(res.site_raw.c.mean(1).std()), sf_sd_pot=_r(res.site.c.mean(1).std()),
        med_model=_r(sm["mae_model"]), med_raw=_r(sm["mae_raw"]),
        med_dir_model=_r(sm["dir_model"]), med_dir_raw=_r(sm["dir_raw"]), improved=sm["improved"])
    for k in summary["coefs"]:
        summary["coefs"][k] = {a: round(float(b), 3) for a, b in summary["coefs"][k].items()}
    route = [] if route_lonlat is None else [[round(float(b), 4), round(float(a), 4)] for a, b in route_lonlat[::3]]
    data = dict(trv=trv, smhi=smhi, hours=[t.strftime("%m-%d %H") for t in H], summary=summary, route=route)
    html = TEMPLATE.read_text()
    html = html.replace("__RULES__", json.dumps(selection_rules(res, cfg), ensure_ascii=False))
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path
