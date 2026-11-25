"""Step 6: combine the per-pair predictions of several SMHI stations into one estimate per TrV station.

    selected   = SMHI stations within cfg.combine_km, at least the cfg.combine_min_n nearest
    w_i        = 1 / max(d_i, cfg.combine_min_dist)
    ln U_hat   = sum w_i ln U4_i / sum w_i                       (log-space weighted mean)
    phi_hat    = weighted vector mean of phi4_i (unit vectors)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings


def select(dist: pd.Series, cfg: Settings) -> list[str]:
    dist = dist.sort_values()
    sel = list(dist[dist <= cfg.combine_km].index)
    return sel if len(sel) >= cfg.combine_min_n else list(dist.index[:cfg.combine_min_n])


def weights(dist: pd.Series, sel: list[str], cfg: Settings) -> pd.Series:
    w = 1 / dist[sel].clip(lower=cfg.combine_min_dist)
    return w / w.sum()


def combine(D: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """D: pair table with U4, wd4. Returns one row per (TrV station, hour) with
    ws, wd (observed), U, wdm (combined model), U0n, wd0n (nearest SMHI raw), selected."""
    out = []
    for j, g in D.groupby("trv", sort=False):
        dist = g.groupby("smhi").dist.first()
        sel = select(dist, cfg)
        w = weights(dist, sel, cfg)
        nearest = dist.idxmin()
        h = g[g.smhi.isin(sel)].assign(w=lambda x: x.smhi.map(w))
        ok = h.U4.notna() & (h.U4 > 0)
        h = h.assign(wl=np.where(ok, h.w * np.log(h.U4.where(ok, 1.0)), 0.0), w_ok=np.where(ok, h.w, 0.0),
                     su=np.where(ok, h.w * np.sin(np.radians(h.wd4)), 0.0),
                     cv=np.where(ok, h.w * np.cos(np.radians(h.wd4)), 0.0))
        a = h.groupby("time")[["wl", "w_ok", "su", "cv"]].sum()
        n = g[g.smhi == nearest].set_index("time")
        r = pd.DataFrame(dict(ws=n.ws, wd=n.wd, U0n=n.U0, wd0n=n.wd_s))
        r["U"] = np.where(a.w_ok > 0, np.exp(a.wl / a.w_ok.where(a.w_ok > 0)), np.nan)
        r["wdm"] = np.where(a.w_ok > 0, np.degrees(np.arctan2(a.su, a.cv)) % 360, np.nan)
        r = r.reset_index()
        r["trv"], r["selected"], r["nearest"] = j, ",".join(sel), nearest
        out.append(r)
    return pd.concat(out, ignore_index=True)
