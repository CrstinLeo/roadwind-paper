import numpy as np, pandas as pd, xarray as xr
from pyproj import CRS, Transformer
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

URL = "https://thredds.met.no/thredds/dodsC/metpparchive/{d:%Y/%m/%d}/met_analysis_1_0km_nordic_{d:%Y%m%dT%H}Z.nc"

def fetch(hours, lat_rng, lon_rng, cache="metnordic_dovre.nc", pad=4):
    """下载 MET Nordic 逐小时 10 m 风（只取覆盖路线的小窗口），缓存为 NetCDF"""
    if Path(cache).exists():
        return xr.open_dataset(cache)
    first = xr.open_dataset(URL.format(d=hours[0]))
    lat, lon = first.latitude.values, first.longitude.values
    m = (lat >= lat_rng[0]) & (lat <= lat_rng[1]) & (lon >= lon_rng[0]) & (lon <= lon_rng[1])
    jj, ii = np.where(m)
    ys, xs = slice(jj.min() - pad, jj.max() + pad + 1), slice(ii.min() - pad, ii.max() + pad + 1)
    parts = []
    for h in hours:
        ds = xr.open_dataset(URL.format(d=h))
        parts.append(ds[["wind_speed_10m", "wind_direction_10m", "air_temperature_2m",
                         "air_pressure_at_sea_level"]].isel(y=ys, x=xs).load())
        print("MET Nordic", h, flush=True)
    out = xr.concat(parts, "time")
    out["altitude"] = first.altitude.isel(y=ys, x=xs).load()
    out.attrs["proj4"] = first.projection_lcc.attrs.get("proj4", "")
    out.to_netcdf(cache)
    return out

def interp(ds, lat, lon, time):
    """把逐小时网格风（分解为 u、v 分量）插值到车辆位置和时刻：空间双线性 + 时间线性"""
    tr = Transformer.from_crs(4326, CRS.from_proj4(ds.attrs["proj4"]), always_xy=True)
    X, Y = tr.transform(lon, lat)
    ws, wd = ds.wind_speed_10m.values, np.radians(ds.wind_direction_10m.values)
    u, v = -ws * np.sin(wd), -ws * np.cos(wd)             # 空气运动矢量（东、北）
    tt = (pd.to_datetime(ds.time.values).tz_localize("UTC") - pd.Timestamp("2020-01-01", tz="UTC")).total_seconds().values
    tq = (pd.to_datetime(time, utc=True) - pd.Timestamp("2020-01-01", tz="UTC")).total_seconds().values
    pts = np.column_stack([tq, Y, X])
    xs, ys = ds.x.values, ds.y.values
    if ys[0] > ys[-1]:
        ys, u, v = ys[::-1], u[:, ::-1], v[:, ::-1]
    fu = RegularGridInterpolator((tt, ys, xs), u, bounds_error=False, fill_value=None)
    fv = RegularGridInterpolator((tt, ys, xs), v, bounds_error=False, fill_value=None)
    alt = RegularGridInterpolator((ys, xs), (ds.altitude.values[::-1] if ds.y.values[0] > ds.y.values[-1]
                                             else ds.altitude.values), bounds_error=False, fill_value=None)
    return fu(pts), fv(pts), alt(np.column_stack([Y, X]))
