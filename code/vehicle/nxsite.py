"""Generic pipeline for one NOR-XWIND site-day:
passes -> chainage -> reciprocal-pass anemometer calibration -> MET Nordic + MEPS at each sample -> bins."""
import sys, glob, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xarray as xr
from pyproj import Transformer, CRS
from shapely.geometry import LineString, Point
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
import metnordic, meps

NX = os.environ.get("NORXWIND_DIR", "./norxwind")
P1 = os.path.join(NX, "p1hz"); D = os.path.join(NX, "data")
TR = Transformer.from_crs(4326, 25833, always_xy=True)

SITES = {
    # name: day folder, file filter, bbox(lat0,lat1,lon0,lon1), bin (m), min speed (m/s), date shift (days)
    "froya": ("201105", lambda n: 14 <= int(n.split("_")[1]) <= 23, (63.685, 63.705, 8.45, 8.52), 250, 8),
    "dolmsund1": ("201105", lambda n: 31 <= int(n.split("_")[1]) <= 44, (63.630, 63.655, 8.81, 8.84), 100, 6),
    "dolmsund2": ("231011", lambda n: True, (63.630, 63.655, 8.81, 8.84), 100, 6),
    "naeroysund1": ("201121", lambda n: True, (64.845, 64.856, 11.20, 11.24), 100, 6),
    "naeroysund2": ("241021", lambda n: True, (64.845, 64.856, 11.20, 11.24), 100, 6),
}


def load(name):
    day, filt, bb, binm, vmin = SITES[name]
    parts = []
    for f in sorted(glob.glob(f"{P1}/{day}/*.parquet")):
        n = os.path.basename(f)[:-8]
        if not filt(n):
            continue
        d = pd.read_parquet(f).assign(session=n)
        d = d[(d.v > vmin) & d.lat.between(bb[0], bb[1]) & d.lon.between(bb[2], bb[3]) & d.U.notna()]
        if len(d) > 20:
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["x"], d["y"] = TR.transform(d.lon.values, d.lat.values)
    # reference line: principal axis, then the longest session's smoothed track
    X = np.column_stack([d.x, d.y]); c = X.mean(0)
    ax = np.linalg.svd(X - c, full_matrices=False)[2][0]
    d["p"] = (X - c) @ ax
    ref = d.loc[d.groupby("session").p.transform(lambda p: p.max() - p.min()).idxmax(), "session"]
    r = d[d.session == ref].sort_values("p")
    xs = r.x.rolling(5, center=True, min_periods=1).mean(); ys = r.y.rolling(5, center=True, min_periods=1).mean()
    line = LineString(np.column_stack([xs, ys]))
    d["s"] = [line.project(Point(a, b)) for a, b in zip(d.x, d.y)]
    d["off"] = [line.distance(Point(a, b)) for a, b in zip(d.x, d.y)]
    d = d[d.off < 30].copy()
    # keep only the dominant travel direction within each session
    d = d.sort_values(["session", "time"])
    d["ds"] = d.groupby("session").s.diff()
    sgn = d.groupby("session").ds.transform(lambda v: np.sign(v.sum()))
    d = d[(np.sign(d.ds) == sgn) | d.ds.isna()]
    d = d[d.groupby("session").s.transform("size") > 20]
    d["dirA"] = d.groupby("session").ds.transform("sum") > 0
    d["pass_id"] = d.groupby("session").time.transform("min").rank(method="dense").astype(int) - 1
    d["t0"] = d.groupby("pass_id").time.transform("min")
    return d.sort_values(["t0", "time"]).reset_index(drop=True), line


def true_wind(d, alpha, g, sgn):
    a = np.radians(alpha)
    mx, my = d.U.values, sgn * d.V.values
    rx = g * (mx * np.cos(a) - my * np.sin(a)); ry = g * (mx * np.sin(a) + my * np.cos(a))
    h = np.radians(d.heading.values)
    fe, fn, le, ln = np.sin(h), np.cos(h), -np.cos(h), np.sin(h)
    return d.v.values * fe + rx * fe + ry * le, d.v.values * fn + rx * fn + ry * ln


def pairs_of(d):
    info = d.groupby("pass_id").agg(t0=("t0", "first"), dirA=("dirA", "first")).sort_values("t0")
    ids = info.index.values
    return [(a, b) for a, b in zip(ids[:-1], ids[1:]) if info.loc[a, "dirA"] != info.loc[b, "dirA"]]


def cost(params, d, pairs, sgn, binm):
    we, wn = true_wind(d, params[0], params[1], sgn)
    m = pd.DataFrame(dict(p=d.pass_id.values, b=(d.s.values // binm).astype(int), we=we, wn=wn)) \
        .groupby(["p", "b"])[["we", "wn"]].mean()
    c, n = 0.0, 0
    for p1, p2 in pairs:
        diff = (m.loc[p1] - m.loc[p2]).dropna()
        c += (diff ** 2).sum().sum(); n += len(diff)
    return c / max(n, 1)


def calibrate(d, binm, mn_dir=None):
    pairs = pairs_of(d); best = []
    for sgn in (1, -1):
        grid = [(cost((a, 1.0), d, pairs, sgn, binm), a) for a in np.arange(-180, 180, 10)]
        a0 = min(grid)[1]
        r = minimize(cost, [a0, 1.0], args=(d, pairs, sgn, binm), method="Nelder-Mead",
                     options=dict(xatol=0.05, fatol=1e-4))
        we, wn = true_wind(d, r.x[0], r.x[1], sgn)
        dirv = np.degrees(np.arctan2(-we.mean(), -wn.mean())) % 360
        best.append(dict(sgn=sgn, alpha=((r.x[0] + 180) % 360) - 180, g=r.x[1], cost=r.fun,
                         cost0=cost((0, 1), d, pairs, sgn, binm), dir_mean=dirv, n_pairs=len(pairs)))
    return best


def circ_diff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def mn_at(name, lat, lon, time):
    ds = xr.open_dataset(f"{D}/mn_{name}.nc")
    return metnordic.interp(ds, lat, lon, time)


def meps_at(name, lat, lon, time):
    """nearest-grid MEPS diagnostics, linear in time"""
    ds = xr.open_dataset(f"{D}/meps_{name}.nc")
    dg = meps.diagnostics(ds)
    la, lo = ds.latitude.values, ds.longitude.values
    lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
    k = np.argmin((la.ravel()[None] - lat[:, None]) ** 2 + ((lo.ravel()[None] - lon[:, None]) * np.cos(np.radians(lat[:, None]))) ** 2, 1)
    jj, ii = np.unravel_index(k, la.shape)
    tt = (pd.to_datetime(ds.time.values) - pd.Timestamp("2000-01-01")).total_seconds().values
    tq = (pd.to_datetime(time, utc=True).tz_localize(None) - pd.Timestamp("2000-01-01")).total_seconds().values
    out = {}
    for v in ["ustar", "L", "z0", "dT", "pblh", "H_up", "U10"]:
        a = dg[v].values[:, jj, ii]                   # (time, n)
        out[v] = np.array([np.interp(tq[i], tt, a[:, i]) for i in range(len(tq))]) if len(tq) == len(jj) else None
    return pd.DataFrame(out)


def bins(d, binm, Z_VEH=3.4):
    d = d.copy()
    d["seg"] = (d.s // binm).astype(int)
    h = np.radians(d.heading.values); d["hs"], d["hc"] = np.sin(h), np.cos(h)
    S = d.groupby(["pass_id", "seg"]).agg(
        session=("session", "first"), dirA=("dirA", "first"), t=("time", "mean"), t0=("t0", "first"),
        n=("we", "size"), we=("we", "mean"), wn=("wn", "mean"), mu=("mu", "mean"), mv=("mv", "mean"),
        v=("v", "mean"), hs=("hs", "mean"), hc=("hc", "mean"), s_mid=("s", "mean"),
        lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
    S = S[S.n >= 3].copy()
    return derive(S)


def derive(S):
    S["w_veh"] = np.hypot(S.we, S.wn); S["w_mn"] = np.hypot(S.mu, S.mv)
    S["dir_veh"] = np.degrees(np.arctan2(-S.we, -S.wn)) % 360
    S["dir_mn"] = np.degrees(np.arctan2(-S.mu, -S.mv)) % 360
    S["heading"] = np.degrees(np.arctan2(S.hs, S.hc)) % 360
    hr = np.radians(S.heading)
    S["head_veh"] = -(S.we * np.sin(hr) + S.wn * np.cos(hr)); S["head_mn"] = -(S.mu * np.sin(hr) + S.mv * np.cos(hr))
    S["cross_veh"] = S.we * np.cos(hr) - S.wn * np.sin(hr); S["cross_mn"] = S.mu * np.cos(hr) - S.mv * np.sin(hr)
    S["r"] = np.log(S.w_veh / S.w_mn)
    S["dphi"] = circ_diff(S.dir_veh, S.dir_mn)
    return S


def day_sign(day):
    """mirror sign of the anemometer mounting on one measurement day: smallest summed reciprocal-pass cost over all
    site-sessions of that day (no reference wind involved)"""
    tot = {1: 0.0, -1: 0.0}
    for nm, (dd, *_rest) in SITES.items():
        if dd != day:
            continue
        f = f"{D}/cal_{nm}.json"
        cal = json.load(open(f))["cal"] if os.path.exists(f) else calibrate(load(nm)[0], SITES[nm][3] * 2)
        for c in cal:
            tot[c["sgn"]] += c["cost"]
    return min(tot, key=tot.get), tot


def build(name):
    day, filt, bb, binm, vmin = SITES[name]
    d, line = load(name)
    mu, mv, _ = mn_at(name, d.lat.values, d.lon.values, d.time.values)
    cal = calibrate(d, binm * 2)
    mn_dir = np.degrees(np.arctan2(-mu.mean(), -mv.mean())) % 360
    for c in cal:                                       # diagnostic only, not used for the choice
        c["dir_diff_vs_mn"] = abs(circ_diff(c["dir_mean"], mn_dir))
    sgn, _ = day_sign(day)
    pick = next(c for c in cal if c["sgn"] == sgn)
    d["we"], d["wn"] = true_wind(d, pick["alpha"], pick["g"], pick["sgn"])
    d["mu"], d["mv"] = mu, mv
    S = bins(d, binm)
    M = meps_at(name, S.lat.values, S.lon.values, S.t.values)
    for c in M:
        S[c] = M[c].values
    return d, S, line, cal, pick, mn_dir
