"""Steps 1-2: neutral log-law height conversion and Wieringa (1986) two-step roughness conversion.

Step 1  U(z) / U(10) = ln(z / z0) / ln(10 / z0), open terrain z0 = 0.03 m  -> 10 m to 6 m: x 0.91
Step 2  through the blending height zb = 60 m, where the wind is assumed independent of the local surface:
        P(z0)  = [ln(zb/z0)   / ln(10/z0)]   * [ln(10/z0r) / ln(zb/z0r)]     station 10 m -> open-terrain 10 m ("potential wind")
        F(z0)  = [ln(zb/z0r)  / ln(10/z0r)]  * [ln(z/z0)   / ln(zb/z0)]      open-terrain 10 m -> height z over z0
        U2 = U_smhi * P(z0_smhi,k) * F(z0_trv,k)
No parameter in this file is fitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..geo import sector


def height_factor(z_to: float, z_from: float = 10.0, z0: float = 0.03) -> float:
    return float(np.log(z_to / z0) / np.log(z_from / z0))


def to_potential(z0, cfg: Settings):
    """P(z0): measured 10 m wind over upwind roughness z0 -> 10 m wind over open terrain"""
    zb, zr, z0r = cfg.z_blend, cfg.z_smhi, cfg.z0_open
    z0 = np.asarray(z0, dtype=float)
    return np.log(zb / z0) / np.log(zr / z0) * np.log(zr / z0r) / np.log(zb / z0r)


def to_site(z0, cfg: Settings, z: float | None = None):
    """F(z0): open-terrain 10 m wind -> wind at height z over upwind roughness z0"""
    zb, zr, z0r = cfg.z_blend, cfg.z_smhi, cfg.z0_open
    z = cfg.z_trv if z is None else z
    z0 = np.asarray(z0, dtype=float)
    return np.log(zb / z0r) / np.log(zr / z0r) * np.log(z / z0) / np.log(zb / z0)


def network_direction(ws: pd.DataFrame, wd: pd.DataFrame) -> pd.Series:
    """wind-from direction of the vector-mean wind over all stations (each station weighted by its speed)"""
    u = -(ws * np.sin(np.radians(wd))).mean(1)
    v = -(ws * np.cos(np.radians(wd))).mean(1)
    return pd.Series(np.degrees(np.arctan2(-u, -v)) % 360, index=ws.index)


def potential_wind(ws: pd.DataFrame, wd: pd.DataFrame, z0_smhi: pd.DataFrame, cfg: Settings):
    """U_pot,s(t) = U_s(t) * P(z0_s(k(t))), k = sector of the network direction. Returns (U_pot, k)."""
    k = sector(network_direction(ws, wd), cfg.n_sectors)
    P = pd.DataFrame({s: to_potential(z0_smhi.loc[s].reindex(k).values, cfg) for s in ws.columns}, index=ws.index)
    return ws * P, k
