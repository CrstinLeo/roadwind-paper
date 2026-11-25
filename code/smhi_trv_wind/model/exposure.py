"""Steps 4-5: roadside exposure model (speed) and road channelling model (direction).

Speed   y = ln(U_trv / U3) = b0 + b1 tree + b2 ln(x/h) + (b3 + b4 tree) ln U_net + e
        U4 = U3 * exp(yhat) * S,   S = mean(exp(e))  (Duan 1983 smearing, log -> linear back-transform)
Dir.    dphi = wrap(phi_trv - phi_smhi) = d0 + d1 sin(2 alpha) tree,   alpha = wrap90(phi_smhi - road axis)
        phi4 = phi_smhi + dphi_hat

U3 is the physics + site-factor prediction (step 3), U_net the site-corrected network wind,
tree and ln(x/h) are taken in the sector of the network direction k.
Fitting: OLS pooled over all hours of all TrV stations; evaluation leave-one-TrV-station-out (LOSO):
the station being predicted is never in the fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Settings
from ..geo import wrap180, wrap90

SPEED_TERMS = ["tree", "lnxh", "lnU", "lnU_tree"]
SPEED_NAMES = ["b0", "b1_tree", "b2_lnxh", "b3_lnU", "b4_lnU_tree"]
DIR_NAMES = ["d0", "d1_s2a_tree"]


def design(D: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lnU_tree = D.lnU * D.tree
    Xs = np.column_stack([np.ones(len(D)), D.tree.fillna(0), D.lnxh.fillna(0), D.lnU.fillna(0), lnU_tree.fillna(0)])
    s2t = np.sin(np.radians(2 * wrap90(D.wd_s - D.road))) * D.tree
    Xd = np.column_stack([np.ones(len(D)), np.nan_to_num(s2t)])
    return Xs, Xd


def targets(D: pd.DataFrame, cfg: Settings):
    y = np.log(D.ws.clip(lower=0.1) / D.U3.clip(lower=0.1)).values
    dd = wrap180(D.wd - D.wd_s)
    s2t = np.sin(np.radians(2 * wrap90(D.wd_s - D.road))) * D.tree
    fs = ((D.U0 >= cfg.fit_min_ref_ws) & (D.ws > 0.2) & D[["tree", "lnxh", "lnU"]].notna().all(1)).values
    fd = ((D.U0 >= cfg.fit_min_ref_ws_dir) & (D.ws >= cfg.fit_min_obs_ws_dir) & np.isfinite(dd) & s2t.notna()).values
    return y, dd, fs, fd


@dataclass
class ExposureModel:
    b: np.ndarray = field(default_factory=lambda: np.zeros(5))
    d: np.ndarray = field(default_factory=lambda: np.zeros(2))
    smear: float = 1.0

    def fit(self, D: pd.DataFrame, cfg: Settings, rows: np.ndarray | None = None) -> "ExposureModel":
        Xs, Xd = design(D)
        y, dd, fs, fd = targets(D, cfg)
        keep = np.ones(len(D), bool) if rows is None else rows
        s, t = fs & keep, fd & keep
        self.b = np.linalg.lstsq(Xs[s], y[s], rcond=None)[0]
        self.smear = float(np.mean(np.exp(y[s] - Xs[s] @ self.b)))
        self.d = np.linalg.lstsq(Xd[t], dd[t], rcond=None)[0]
        return self

    def predict(self, D: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xs, Xd = design(D)
        U4 = D.U3.values * np.exp(Xs @ self.b) * self.smear
        wd4 = (D.wd_s.values + Xd @ self.d) % 360
        return U4, wd4

    def coefficients(self) -> dict:
        return dict(speed=dict(zip(SPEED_NAMES, map(float, self.b))), smearing=self.smear,
                    direction=dict(zip(DIR_NAMES, map(float, self.d))))


def fit_loso(D: pd.DataFrame, cfg: Settings) -> tuple[pd.DataFrame, ExposureModel]:
    """adds U4, wd4 predicted for every TrV station by a model fitted without that station;
    returns also the model fitted on all stations (the coefficients to report)."""
    D = D.copy()
    U4, wd4 = np.full(len(D), np.nan), np.full(len(D), np.nan)
    trv = D.trv.values
    for j in np.unique(trv):
        te = trv == j
        m = ExposureModel().fit(D, cfg, rows=~te)
        u, w = m.predict(D[te])
        U4[te], wd4[te] = u, w
    D["U4"], D["wd4"] = U4, wd4
    return D, ExposureModel().fit(D, cfg)
