"""Small geometry / angle helpers used everywhere."""
import numpy as np

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.radians
    a = np.sin(p(lat2 - lat1) / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(p(lon2 - lon1) / 2) ** 2
    return EARTH_R_KM * 2 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """initial bearing from point 1 to point 2, degrees from north"""
    p = np.radians
    y = np.sin(p(lon2 - lon1)) * np.cos(p(lat2))
    x = np.cos(p(lat1)) * np.sin(p(lat2)) - np.sin(p(lat1)) * np.cos(p(lat2)) * np.cos(p(lon2 - lon1))
    return np.degrees(np.arctan2(y, x)) % 360


def wrap180(a):
    """angle difference folded to [-180, 180)"""
    return (np.asarray(a, dtype=float) + 180) % 360 - 180


def wrap90(a):
    """axis difference folded to [-90, 90) (a road has no direction)"""
    return (np.asarray(a, dtype=float) + 90) % 180 - 90


def angdiff(a, b):
    return np.abs(wrap180(np.asarray(a) - np.asarray(b)))


def sector(phi, n=16):
    """centre of the n-sector containing wind-from direction phi (deg)"""
    w = 360 / n
    return (np.round(np.asarray(phi, dtype=float) / w) % n) * w


def sectors(n=16):
    return np.arange(0, 360, 360 / n)
