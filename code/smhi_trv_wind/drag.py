"""Air drag of a tractor-semitrailer with crosswind, Model 1 of Askerdal, Fredriksson & Laine (2024, Vehicle System
Dynamics 62(5):1085-1102):

    F = 1/2 rho C_dA(theta) v_a^2,   C_dA(theta) = c1 cos^2(theta) (A_f cos(theta) + A_s sin(theta))
    v_ax = v_v + v_wx,  v_a^2 = v_ax^2 + v_wy^2,  theta = atan(|v_wy| / v_ax)

v_wx: headwind component (positive against the vehicle), v_wy: crosswind component. A_f = 10.3 m2 and A_s = 57.6 m2
for the tractor-semitrailer of the paper. c1 cancels in relative drag errors, so it is set to 1.
"""
import numpy as np

A_F, A_S = 10.3, 57.6


def cda(theta, a_f=A_F, a_s=A_S, c1=1.0):
    return c1 * np.cos(theta) ** 2 * (a_f * np.cos(theta) + a_s * np.sin(theta))


def force(v_vehicle, head, cross, rho=1.25):
    """drag force [N per unit c1] for vehicle speed [m/s] and wind components [m/s]"""
    vax = v_vehicle + head
    theta = np.arctan2(np.abs(cross), vax)
    return 0.5 * rho * cda(theta) * (vax ** 2 + cross ** 2)


def components(speed, wind_from_deg, heading_deg):
    """head (+ against the vehicle) and cross components of a wind blowing FROM wind_from_deg, vehicle moving TO heading"""
    a = np.radians(np.asarray(wind_from_deg) - np.asarray(heading_deg))
    return speed * np.cos(a), speed * np.sin(a)


def relative_error(v_vehicle, true_speed, true_dir, est_speed, est_dir, heading):
    """(F_est - F_true) / F_true for a vehicle at v_vehicle on the given heading"""
    ht, ct = components(true_speed, true_dir, heading)
    he, ce = components(est_speed, est_dir, heading)
    return force(v_vehicle, he, ce) / force(v_vehicle, ht, ct) - 1


def station_errors(obs_ws, obs_wd, est_ws, est_wd, road_bearing, v_vehicle=80 / 3.6, min_ws=0.0):
    """relative drag error for a vehicle passing each roadside station in both directions of the road.
    Arrays [hour x station]; road_bearing [station] (0-180 deg). Returns |error| per station (median over hours
    and both directions) and the pooled signed errors."""
    obs_ws, obs_wd, est_ws, est_wd = map(np.asarray, (obs_ws, obs_wd, est_ws, est_wd))
    ok = np.isfinite(obs_ws) & np.isfinite(obs_wd) & np.isfinite(est_ws) & np.isfinite(est_wd) & (obs_ws >= min_ws)
    per, pooled = [], []
    for j in range(obs_ws.shape[1]):
        m = ok[:, j]
        if m.sum() < 10 or not np.isfinite(road_bearing[j]):
            per.append(np.nan)
            continue
        e = np.concatenate([relative_error(v_vehicle, obs_ws[m, j], obs_wd[m, j], est_ws[m, j], est_wd[m, j], road_bearing[j] + k)
                            for k in (0.0, 180.0)])
        per.append(float(np.median(np.abs(e))))
        pooled.append(e)
    return np.array(per), (np.concatenate(pooled) if pooled else np.array([]))
