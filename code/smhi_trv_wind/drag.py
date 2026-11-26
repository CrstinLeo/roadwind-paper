"""Air drag of a tractor-semitrailer with crosswind.

Two drag-area models C_dA(beta) of the air attack (yaw) angle beta are provided:

"vecto" (default): the generic speed-dependent crosswind correction of the EU certification tool VECTO for
    tractor-semitrailers,  C_dA(beta) = C_dA(0) + a1 beta + a2 beta^2 + a3 beta^3  [m2, beta in degrees],
    a1 = 0.030042, a2 = 0.040817, a3 = -0.002130 (VECTO user manual, "Speed dependent correction"),
    with C_dA(0) = 5.8 m2, the typical certified value of EU 4x2 long-haul tractor-trailers (ICCT, from EEA data).
    The cubic peaks at beta = 13.4 deg; beyond the peak the value is held constant (the polynomial is fitted to
    small yaw angles only).
"model1": Model 1 of Askerdal, Fredriksson & Laine (2024, Vehicle System Dynamics 62(5)),
    C_dA(beta) = c1 cos^2(beta) (A_f cos(beta) + A_s sin(beta)), A_f = 10.3 m2, A_s = 57.6 m2, scaled to the same
    C_dA(0). It is a projected-area model fitted over 0-90 deg and gives a much stronger increase at small yaw angles;
    it is used as an upper bound of the crosswind sensitivity.

    F = 1/2 rho C_dA(beta) v_a^2,  v_ax = v_v + v_wx,  v_a^2 = v_ax^2 + v_wy^2,  beta = atan(|v_wy| / v_ax)

v_wx: headwind component (positive against the vehicle), v_wy: crosswind component.
"""
import numpy as np

A_F, A_S = 10.3, 57.6
CDA0 = 5.8
VECTO_TS = (0.030042, 0.040817, -0.002130)
MODEL = "vecto"


def _beta_peak(a=VECTO_TS):
    a1, a2, a3 = a
    return (-2 * a2 - np.sqrt(4 * a2 ** 2 - 12 * a3 * a1)) / (6 * a3)


BETA_PEAK = _beta_peak()


def cda(theta, model=None, cda0=CDA0):
    """drag area [m2] at air attack angle theta [rad]"""
    model = model or MODEL
    theta = np.abs(theta)
    if model == "vecto":
        b = np.minimum(np.degrees(theta), BETA_PEAK)
        a1, a2, a3 = VECTO_TS
        return cda0 + a1 * b + a2 * b ** 2 + a3 * b ** 3
    if model == "model1":
        return cda0 / A_F * np.cos(theta) ** 2 * (A_F * np.cos(theta) + A_S * np.sin(theta))
    raise ValueError(model)


def force(v_vehicle, head, cross, rho=1.25, model=None, cda0=CDA0):
    """drag force [N] for vehicle speed [m/s] and wind components [m/s]"""
    vax = v_vehicle + head
    theta = np.arctan2(np.abs(cross), vax)
    return 0.5 * rho * cda(theta, model, cda0) * (vax ** 2 + cross ** 2)


def energy_kwh100(v_vehicle, head, cross, **kw):
    """air-drag energy per distance [kWh/100 km] (1 N = 1 J/m; 100 km / 3.6e6 J/kWh)"""
    return force(v_vehicle, head, cross, **kw) / 36.0


def components(speed, wind_from_deg, heading_deg):
    """head (+ against the vehicle) and cross components of a wind blowing FROM wind_from_deg, vehicle moving TO heading"""
    a = np.radians(np.asarray(wind_from_deg) - np.asarray(heading_deg))
    return speed * np.cos(a), speed * np.sin(a)


def relative_error(v_vehicle, true_speed, true_dir, est_speed, est_dir, heading, model=None):
    """(F_est - F_true) / F_true for a vehicle at v_vehicle on the given heading"""
    ht, ct = components(true_speed, true_dir, heading)
    he, ce = components(est_speed, est_dir, heading)
    return force(v_vehicle, he, ce, model=model) / force(v_vehicle, ht, ct, model=model) - 1


def station_errors(obs_ws, obs_wd, est_ws, est_wd, road_bearing, v_vehicle=80 / 3.6, min_ws=0.0, model=None):
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
        e = np.concatenate([relative_error(v_vehicle, obs_ws[m, j], obs_wd[m, j], est_ws[m, j], est_wd[m, j], road_bearing[j] + k, model)
                            for k in (0.0, 180.0)])
        per.append(float(np.median(np.abs(e))))
        pooled.append(e)
    return np.array(per), (np.concatenate(pooled) if pooled else np.array([]))
