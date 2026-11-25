"""Step 3: residual site factor of every SMHI station, by wind-direction sector.

    ln U_s(t) = a(t) + c_s(k(t)) + e,    sum_s c_s(k) = 0 for every k
a(t): network-common log wind of hour t; c_s(k): persistent offset of station s when the network wind
comes from sector k. Estimated by alternating least squares (two-way fixed effects) on the training
history, hours with network median speed >= cfg.site_min_network_ws.
Corrected wind: U*_s = U_s * exp(-c_s(k)).

`explain` regresses the station-mean factors on geography (leave-one-station-out), the
"common exposure law" check of round 5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Settings
from ..geo import sector, sectors
from .physics import network_direction


@dataclass
class SiteFactors:
    c: pd.DataFrame          # [station x sector] log factor
    a: pd.Series             # network term per hour (training)
    n: pd.DataFrame          # hours per station and sector
    hours: int
    resid_sd: pd.Series

    def factor(self, station: str) -> float:
        """station-mean multiplicative factor exp(mean_k c)"""
        return float(np.exp(self.c.loc[station].mean()))

    def correct(self, ws: pd.DataFrame, k) -> pd.DataFrame:
        C = pd.DataFrame({s: self.c.loc[s].reindex(k).values for s in ws.columns}, index=ws.index)
        return ws * np.exp(-C)


def fit(ws: pd.DataFrame, wd: pd.DataFrame, cfg: Settings) -> SiteFactors:
    dirs = sectors(cfg.n_sectors)
    net = ws.median(1)
    k = pd.Series(sector(network_direction(ws, wd), cfg.n_sectors), index=ws.index)
    use = (net >= cfg.site_min_network_ws) & (ws.notna().sum(1) >= min(cfg.site_min_stations, ws.shape[1]))
    L = np.log(ws[use].where(ws[use] >= cfg.site_min_ws))
    K = k[use]
    c = pd.DataFrame(0.0, index=ws.columns, columns=dirs)
    for _ in range(cfg.site_iters):
        C = pd.DataFrame({s: c.loc[s].reindex(K.values).values for s in ws.columns}, index=L.index)
        a = (L - C).mean(1)                                   # hour effect given station effects
        c = L.sub(a, axis=0).groupby(K.values).mean().T.reindex(columns=dirs)   # station x sector effect given hours
        c = c - c.mean(0)                                     # identifiability
    C = pd.DataFrame({s: c.loc[s].reindex(K.values).values for s in ws.columns}, index=L.index)
    resid = L.sub(a, axis=0) - C
    n = L.notna().groupby(K.values).sum().T.reindex(columns=dirs)
    return SiteFactors(c=c, a=a, n=n, hours=int(use.sum()), resid_sd=resid.std())


def stability(ws: pd.DataFrame, wd: pd.DataFrame, cfg: Settings, split: str = "2024-01-01") -> float:
    """correlation of the [station x sector] factors fitted on two halves of the history"""
    t = pd.Timestamp(split, tz="UTC")
    a, b = fit(ws[ws.index < t], wd[wd.index < t], cfg), fit(ws[ws.index >= t], wd[wd.index >= t], cfg)
    x, y = a.c.values.ravel(), b.c.values.ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def explain(sf: SiteFactors, geo: pd.DataFrame, cols=("tree", "ln_z0_1km", "ln1p_coast", "elev_100m")) -> pd.DataFrame:
    """c_s(k) ~ geography of station s in sector k, OLS, leave-one-station-out R^2.
    geo: long table with columns id, dir and the predictors."""
    y = sf.c.stack().rename("c").reset_index().rename(columns={"level_0": "id", "level_1": "dir"})
    y.columns = ["id", "dir", "c"]
    d = y.merge(geo, on=["id", "dir"]).dropna()
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
    pred = np.full(len(d), np.nan)
    for s in d.id.unique():
        te = (d.id == s).values
        b = np.linalg.lstsq(X[~te], d.c.values[~te], rcond=None)[0]
        pred[te] = X[te] @ b
    b = np.linalg.lstsq(X, d.c.values, rcond=None)[0]
    r2 = 1 - np.sum((d.c - pred) ** 2) / np.sum((d.c - d.c.mean()) ** 2)
    return pd.DataFrame(dict(coef=b), index=["const", *cols]).assign(loso_r2=r2)
