"""MET Nordic analysis (1 km, hourly, thredds.met.no metpparchive) at the Norwegian roadside stations.

Gridded 10 m wind is interpolated bilinearly (u, v components) to each roadside station. To keep the load on the
server reasonable every STEP-th hour of the test period is fetched (default 3). Output (data, not in git):
SWW_DATA/metnordic_roadside.parquet with columns t, id, ws, wd. Restartable: hours already present are skipped.
"""
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

URL = "https://thredds.met.no/thredds/dodsC/metpparchive/{d:%Y/%m/%d}/met_analysis_1_0km_nordic_{d:%Y%m%dT%H}Z.nc"
DATA = Path(os.environ.get("SWW_DATA", "data/no")); OUT = DATA / "metnordic_roadside.parquet"
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 3
R = pickle.load(open(Path(os.environ.get("SWW_OUT", "output/no")) / "norway_results.pkl", "rb"))
ids = list(R["meta"]["trv"]); H = R["obs"]["hours"][::STEP]
st = pd.read_csv(DATA / "stations_region.csv").set_index("id").loc[ids]

first = xr.open_dataset(URL.format(d=H[0].tz_convert(None)))
tr = Transformer.from_crs(4326, CRS.from_proj4(first.projection_lcc.attrs["proj4"]), always_xy=True)
X, Y = tr.transform(st.lon.values, st.lat.values)
xs, ys = first.x.values, first.y.values
ix = np.interp(X, xs, np.arange(len(xs))); iy = np.interp(Y, ys, np.arange(len(ys))) if ys[0] < ys[-1] else \
    len(ys) - 1 - np.interp(Y, ys[::-1], np.arange(len(ys)))
x0, x1 = int(np.floor(ix.min())) - 1, int(np.ceil(ix.max())) + 2
y0, y1 = int(np.floor(iy.min())) - 1, int(np.ceil(iy.max())) + 2
fx, fy = ix - x0, iy - y0
i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int); wx, wy = fx - i0, fy - j0


def one(h):
    for attempt in range(4):
        try:
            ds = xr.open_dataset(URL.format(d=h.tz_convert(None)))
            w = ds[["wind_speed_10m", "wind_direction_10m"]].isel(time=0, x=slice(x0, x1), y=slice(y0, y1)).load()
            ws, wd = w.wind_speed_10m.values, np.radians(w.wind_direction_10m.values)
            u, v = -ws * np.sin(wd), -ws * np.cos(wd)

            def bil(a):
                return ((1 - wx) * (1 - wy) * a[j0, i0] + wx * (1 - wy) * a[j0, i0 + 1]
                        + (1 - wx) * wy * a[j0 + 1, i0] + wx * wy * a[j0 + 1, i0 + 1])
            uu, vv = bil(u), bil(v)
            return pd.DataFrame(dict(t=h, id=ids, ws=np.hypot(uu, vv), wd=np.degrees(np.arctan2(-uu, -vv)) % 360))
        except Exception as e:                    # missing file or transient server error
            err = e; time.sleep(5 * (attempt + 1))
    print("skip", h, type(err).__name__, flush=True)
    return None


done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["t", "id", "ws", "wd"])
todo = [h for h in H if h not in set(pd.to_datetime(done.t, utc=True))]
print(len(H), "hours,", len(todo), "to fetch", flush=True)
parts = [done]
with ProcessPoolExecutor(4) as ex:                      # netCDF/OPeNDAP is not thread-safe: processes
    for k, df in enumerate(ex.map(one, todo, chunksize=4)):
        if df is not None:
            parts.append(df)
        if (k + 1) % 200 == 0:
            pd.concat(parts, ignore_index=True).to_parquet(OUT); print(k + 1, "fetched", flush=True)
pd.concat(parts, ignore_index=True).to_parquet(OUT)
print("done", OUT)
