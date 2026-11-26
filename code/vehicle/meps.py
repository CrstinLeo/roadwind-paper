"""MEPS (MetCoOp, 2.5 km) surface fields for stability / roughness diagnostics.

For each analysis hour h we take the latest run at least 3 h earlier (00/06/12/18 UTC),
and read lead h and h-1 (the sensible-heat and momentum fluxes are accumulated)."""
import numpy as np, pandas as pd, xarray as xr
from pathlib import Path

VARS = ["x_wind_10m", "y_wind_10m", "downward_eastward_momentum_flux_in_air",
        "downward_northward_momentum_flux_in_air", "integral_of_surface_downward_sensible_heat_flux_wrt_time",
        "air_temperature_2m", "air_temperature_0m", "surface_air_pressure", "atmosphere_boundary_layer_thickness"]


def url(run):
    if run < pd.Timestamp("2022-01-01"):
        return f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{run:%Y/%m/%d}/meps_det_2_5km_{run:%Y%m%dT%H}Z.nc"
    return f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{run:%Y/%m/%d}/meps_det_sfc_{run:%Y%m%dT%H}Z.ncml"


def fetch(hours, lat_rng, lon_rng, cache, pad=2):
    if Path(cache).exists():
        return xr.open_dataset(cache)
    hours = pd.DatetimeIndex(hours)
    runs = ((hours - pd.Timedelta("3h")).floor("6h"))
    sub = None; parts = []
    for run in sorted(set(runs)):
        ds = xr.open_dataset(url(run))
        if sub is None:
            lat, lon = ds.latitude.values, ds.longitude.values
            m = (lat >= lat_rng[0]) & (lat <= lat_rng[1]) & (lon >= lon_rng[0]) & (lon <= lon_rng[1])
            jj, ii = np.where(m)
            sub = dict(y=slice(jj.min() - pad, jj.max() + pad + 1), x=slice(ii.min() - pad, ii.max() + pad + 1))
            latlon = (ds.latitude.isel(**sub).load(), ds.longitude.isel(**sub).load())
        t0 = pd.Timestamp(ds.time.values[0])
        for h in hours[runs == run]:
            hh = h.tz_localize(None) if h.tzinfo else h
            lead = int((hh - t0) / pd.Timedelta("1h"))
            d = ds[VARS].isel(time=[lead - 1, lead], **sub).load()
            d = d.squeeze([c for c in d.dims if c.startswith("height")], drop=True)
            ACC = ["integral_of_surface_downward_sensible_heat_flux_wrt_time",
                   "downward_eastward_momentum_flux_in_air", "downward_northward_momentum_flux_in_air"]
            rec = d.isel(time=1).drop_vars(ACC)
            for v, name in zip(ACC, ["H_down", "taux", "tauy"]):          # accumulated -> hourly mean flux
                rec[name] = (d[v].isel(time=1) - d[v].isel(time=0)) / 3600.0
            rec = rec.expand_dims(time=[np.datetime64(h.tz_localize(None) if h.tzinfo else h)])
            parts.append(rec.drop_vars([v for v in rec.coords if v not in ("time", "y", "x")], errors="ignore"))
            print("MEPS", run, "lead", lead, flush=True)
    out = xr.concat(parts, "time")
    out["latitude"], out["longitude"] = latlon[0].drop_vars([c for c in latlon[0].coords if c not in ("y", "x")]), \
        latlon[1].drop_vars([c for c in latlon[1].coords if c not in ("y", "x")])
    out.to_netcdf(cache)
    return out


# ---------- surface-layer similarity (Paulson 1970 / Dyer 1974 unstable, Beljaars & Holtslag 1991 stable) ----------
KAPPA, G, CP, RD = 0.4, 9.81, 1005.0, 287.05


def psi_m(zeta):
    zeta = np.asarray(zeta, float)
    out = np.zeros_like(zeta)
    u = zeta < 0
    x = (1 - 16 * zeta[u]) ** 0.25
    out[u] = 2 * np.log((1 + x) / 2) + np.log((1 + x ** 2) / 2) - 2 * np.arctan(x) + np.pi / 2
    s = ~u
    a, b, c, d = 1.0, 2 / 3, 5.0, 0.35
    zs = np.minimum(zeta[s], 10)
    out[s] = -(a * zs + b * (zs - c / d) * np.exp(-d * zs) + b * c / d)
    return out


def diagnostics(ds):
    """u*, Obukhov length L, effective roughness z0 (from the model's own 10 m wind and stress), Ri_b proxy"""
    U10 = np.hypot(ds.x_wind_10m, ds.y_wind_10m)
    tau = np.hypot(ds.taux, ds.tauy)
    T = ds.air_temperature_2m
    rho = ds.surface_air_pressure / (RD * T)
    ustar = np.sqrt(tau / rho)
    H_up = -ds.H_down
    L = -(ustar ** 3) * T * rho * CP / (KAPPA * G * H_up.where(np.abs(H_up) > 0.5, np.sign(H_up + 1e-9) * 0.5))
    # neutral-plus-stability inversion of U10 = u*/k [ln(10/z0) - psi(10/L)]   (psi(z0/L) neglected)
    z0 = 10.0 * np.exp(-(KAPPA * U10 / ustar + xr.apply_ufunc(psi_m, 10.0 / L)))
    dT = ds.air_temperature_2m - ds.air_temperature_0m       # >0: surface colder than air (stable)
    return xr.Dataset(dict(U10=U10, ustar=ustar, H_up=H_up, L=L, z0=z0, dT=dT,
                           pblh=ds.atmosphere_boundary_layer_thickness))


def height_factor(z, z_ref, z0, L=None):
    """U(z)/U(z_ref) from the diabatic log law"""
    z0 = np.asarray(z0, float)
    if L is None:
        return np.log(z / z0) / np.log(z_ref / z0)
    L = np.asarray(L, float)
    return (np.log(z / z0) - psi_m(z / L)) / (np.log(z_ref / z0) - psi_m(z_ref / L))
