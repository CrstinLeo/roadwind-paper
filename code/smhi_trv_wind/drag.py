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
