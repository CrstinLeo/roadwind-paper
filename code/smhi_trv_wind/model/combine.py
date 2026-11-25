"""Step 6: combine the per-pair predictions of several SMHI stations into one estimate per TrV station.

Speed      stations within cfg.speed_km (at least cfg.speed_min_n nearest), w = 1/max(d, d_min)^speed_power,
           ln U_hat = sum w ln U4 / sum w                       (round 7: all stations, p = 1.5)
Direction  stations within cfg.combine_km (at least cfg.combine_min_n), w = 1/max(d, d_min)^dir_power,
           phi_hat = weighted vector mean of phi4               (60 km, p = 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings


def select(dist: pd.Series, km: float, min_n: int) -> list[str]:
    dist = dist.sort_values()
    sel = list(dist[dist <= km].index)
    return sel if len(sel) >= min_n else list(dist.index[:min_n])


def weights(dist: pd.Series, sel: list[str], cfg: Settings, power: float = 1.0) -> pd.Series:
    w = 1 / dist[sel].clip(lower=cfg.combine_min_dist) ** power
    return w / w.sum()


def _wmean(h: pd.DataFrame, w: pd.Series):
    h = h.assign(w=h.smhi.map(w).fillna(0.0))
    ok = h.U4.notna() & (h.U4 > 0) & (h.w > 0)
    h = h.assign(wl=np.where(ok, h.w * np.log(h.U4.where(ok, 1.0)), 0.0), w_ok=np.where(ok, h.w, 0.0),
                 su=np.where(ok, h.w * np.sin(np.radians(h.wd4)), 0.0),
                 cv=np.where(ok, h.w * np.cos(np.radians(h.wd4)), 0.0))
    return h.groupby("time")[["wl", "w_ok", "su", "cv"]].sum()


def combine(D: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """D: pair table with U4, wd4. Returns one row per (TrV station, hour) with
    ws, wd (observed), U, wdm (combined model), U0n, wd0n (nearest SMHI raw), selected (speed stations)."""
    out = []
    for j, g in D.groupby("trv", sort=False):
        dist = g.groupby("smhi").dist.first()
        sel_s = select(dist, cfg.speed_km, cfg.speed_min_n)
        sel_d = select(dist, cfg.combine_km, cfg.combine_min_n)
        a = _wmean(g, weights(dist, sel_s, cfg, cfg.speed_power))
        b = _wmean(g, weights(dist, sel_d, cfg, cfg.dir_power))
        nearest = dist.idxmin()
        n = g[g.smhi == nearest].set_index("time")
        r = pd.DataFrame(dict(ws=n.ws, wd=n.wd, U0n=n.U0, wd0n=n.wd_s))
        r["U"] = np.where(a.w_ok > 0, np.exp(a.wl / a.w_ok.where(a.w_ok > 0)), np.nan)
        r["wdm"] = np.where(b.w_ok > 0, np.degrees(np.arctan2(b.su, b.cv)) % 360, np.nan)
        r = r.reset_index()
        r["trv"], r["selected"], r["selected_dir"], r["nearest"] = j, ",".join(sel_s), ",".join(sel_d), nearest
        out.append(r)
    return pd.concat(out, ignore_index=True)
