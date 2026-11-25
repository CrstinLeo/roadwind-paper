from . import smhi, trafikverket, frost  # noqa: F401


def sources(cfg):
    """(reference network, roadside network) data modules for the configured country"""
    if cfg.country == "NO":
        return frost.reference, frost.roadside
    return smhi, trafikverket
