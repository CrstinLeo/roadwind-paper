"""Metrics.

Pair level (one TrV station x one SMHI station), per correction step L in U0..U4:
  mae_L    mean |U_L - U_trv| over all hours with both values
  rbias_L  mean(U_L) / mean(U_trv) - 1 over all hours with both values (ratio of means; positive = prediction too high)
  bias_L   exp(mean ln(U_trv / U_L)) - 1 over hours with U0 >= 2 m/s (geometric; negative = prediction too high; legacy)
  mb_L     mean (U_L - U_trv)            (arithmetic, m/s)
  r, r4    correlation of U0 / U4 with U_trv
  dir_raw, dir_mod   mean |angle error| of SMHI / model direction, hours with both speeds >= 2 m/s
Station level: combined estimate vs nearest raw SMHI x 0.91.
SMHI station value ('quality', m/s, negative = better reference), two definitions:
  method="fe"        mae_U4 = TrV-station effect + beta ln(dist) + SMHI effect (sum-to-zero), OLS     [default]
                     -> compares SMHI stations on the SAME TrV stations
  method="distance"  mean residual per SMHI station of the pooled fit mae_U4 ~ ln(dist)             [round-6 map v1]
                     -> also credits a station for serving TrV stations that are easy to predict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import wrap180

LEVELS = ["U0", "U1", "U2", "U3", "U4"]
LEVEL_NAMES = {"U0": "raw SMHI 10 m", "U1": "+ height (x0.91)", "U2": "+ roughness (Wieringa, NMD z0)",
               "U3": "+ SMHI site factor", "U4": "+ roadside exposure model"}


def pair_metrics(D: pd.DataFrame) -> pd.DataFrame:
    def f(g):
        g = g[g.ws.notna() & g.U0.notna()]
        m = (g.U0 >= 2) & (g.ws > 0.2)
        out = dict(dist=g.dist.iloc[0], n=len(g))
        for L in LEVELS:
            out[f"mae_{L}"] = float(np.mean(np.abs(g[L] - g.ws)))
            out[f"mb_{L}"] = float(np.mean(g[L] - g.ws))
            okL = g[L].notna()
            out[f"rbias_{L}"] = float(g[L][okL].mean() / g.ws[okL].mean() - 1) if okL.sum() > 10 else np.nan
            out[f"bias_{L}"] = float(np.exp(np.mean(np.log(g.ws[m] / g[L][m]))) - 1) if m.sum() > 10 else np.nan
        out["r"] = float(np.corrcoef(g.U0, g.ws)[0, 1]) if g.U0.std() > 0 else np.nan
        ok4 = g.U4.notna()
        out["r4"] = float(np.corrcoef(g.U4[ok4], g.ws[ok4])[0, 1]) if ok4.sum() > 10 else np.nan
        dm = (g.ws >= 2) & (g.U0 >= 2) & g.wd.notna() & g.wd_s.notna()
        out["dir_raw"] = float(np.mean(np.abs(wrap180(g.wd_s - g.wd)[dm]))) if dm.sum() > 10 else np.nan
        out["dir_mod"] = float(np.mean(np.abs(wrap180(g.wd4 - g.wd)[dm]))) if dm.sum() > 10 else np.nan
        return pd.Series(out)
    return D.groupby(["trv", "smhi"]).apply(f).reset_index()


def ladder(P: pd.DataFrame, max_km: float | None = 80.0) -> pd.DataFrame:
    """median over pairs (within max_km) of every correction step"""
    if max_km is not None:
        P = P[P.dist <= max_km]
    return pd.DataFrame([dict(step=LEVEL_NAMES[L], mae=P[f"mae_{L}"].median(), bias=P[f"bias_{L}"].median(), rbias=P[f"rbias_{L}"].median(),
                              mean_error=P[f"mb_{L}"].median()) for L in LEVELS])


def station_metrics(C: pd.DataFrame, f_height: float = 0.912) -> pd.DataFrame:
    """combined model vs nearest SMHI x height factor, per TrV station"""
    def f(c):
        dmod = (c.ws >= 2) & (c.U >= 2) & c.wd.notna() & c.wdm.notna()
        draw = (c.ws >= 2) & (c.U0n >= 2) & c.wd.notna() & c.wd0n.notna()
        return pd.Series(dict(
            mae_model=np.nanmean(np.abs(c.U - c.ws)), mae_raw=np.nanmean(np.abs(f_height * c.U0n - c.ws)),
            dir_model=np.mean(np.abs(wrap180(c.wdm - c.wd)[dmod])) if dmod.any() else np.nan,
            dir_raw=np.mean(np.abs(wrap180(c.wd0n - c.wd)[draw])) if draw.any() else np.nan,
            selected=c.selected.iloc[0], nearest=c.nearest.iloc[0]))
    S = C.groupby("trv").apply(f)
    S["improved"] = S.mae_model < S.mae_raw
    return S


def summary(S: pd.DataFrame) -> dict:
    return dict(stations=len(S), mae_raw=float(S.mae_raw.median()), mae_model=float(S.mae_model.median()),
                dir_raw=float(S.dir_raw.median()), dir_model=float(S.dir_model.median()),
                improved=int(S.improved.sum()))


def quality(P: pd.DataFrame, method: str = "fe") -> tuple[pd.Series, float]:
    """SMHI station effect on mae_U4 controlling for distance; returns (effect per station, slope per ln km)"""
    P = P.dropna(subset=["mae_U4"])
    ld = np.log(P.dist.clip(lower=1)).values
    if method == "distance":
        b = np.polyfit(ld, P.mae_U4.values, 1)
        return pd.Series(P.mae_U4.values - np.polyval(b, ld), index=P.index).groupby(P.smhi).mean(), float(b[0])
    trv, smhi = pd.get_dummies(P.trv, dtype=float), pd.get_dummies(P.smhi, dtype=float)
    S = smhi.values[:, :-1] - smhi.values[:, [-1]]                # sum-to-zero coding
    X = np.column_stack([trv.values, ld, S])
    b = np.linalg.lstsq(X, P.mae_U4.values, rcond=None)[0]
    eff = np.append(b[trv.shape[1] + 1:], -b[trv.shape[1] + 1:].sum())
    return pd.Series(eff, index=smhi.columns), float(b[trv.shape[1]])


def smhi_value(P: pd.DataFrame, names: pd.Series | None = None) -> pd.DataFrame:
    """value of every SMHI station as a reference (see module docstring)"""
    P = P.dropna(subset=["mae_U4"]).copy()
    P["rank4"] = P.groupby("trv").mae_U4.rank(method="first")
    P["rank0"] = P.groupby("trv").mae_U0.rank(method="first")
    V = P.groupby("smhi").agg(candidates=("trv", "size"), top1=("rank4", lambda r: int((r == 1).sum())),
                              top3=("rank4", lambda r: int((r <= 3).sum())), raw_top1=("rank0", lambda r: int((r == 1).sum())),
                              med_dist=("dist", "median"), med_mae0=("mae_U0", "median"), med_mae4=("mae_U4", "median"),
                              med_r4=("r4", "median"))
    V["quality_eff"], slope = quality(P, "fe")
    V["quality_distance_only"], _ = quality(P, "distance")
    V.attrs["dist_slope_per_ln_km"] = slope
    if names is not None:
        V["name"] = names.reindex(V.index)
    return V.sort_values("quality_eff")
