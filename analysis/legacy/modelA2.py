"""Model A, round 2: extra physics-motivated parameters, gust-factor roughness, model families, residual kriging, error model."""
import numpy as np, pandas as pd
import modelA as ma, sedata as sd

F2 = None


def gust_z0(min_ws=3.0, t_gust=3.0, z=6.0):
    """Wieringa (1973) gustiness model as given by Verkaik (2000, JAM 39): G = 1 + (1.42 + 0.301 ln(990/(U t) - 4)) / ln(z/z0),
    with T = 10 min (f_T = 1) and no attenuation (sonic anemometer). Median G per TrV station and 16-sector of the station's own
    wind direction, 5-min records of the rolling 10-min mean and 10-min max with U >= min_ws."""
    o = sd.trv_obs()
    o = o[(o.ws10 >= min_ws) & o.gust10.notna()].copy()
    o["G"] = o.gust10 / o.ws10
    o = o[(o.G >= 1.0) & (o.G <= 4.0)]
    o["dir"] = (np.round(o.wd / 22.5) % 16) * 22.5
    g = o.groupby(["id", "dir"]).agg(G=("G", "median"), U=("ws10", "median"), n=("G", "size")).reset_index()
    g = g[g.n >= 24]                                              # >= 2 h of 5-min records
    A = 1.42 + 0.301 * np.log(990.0 / (g.U * t_gust) - 4)
    g["z0_gust"] = z / np.exp(A / (g.G - 1))
    return g


def add_features(D):
    F2 = pd.read_parquet("/home/claude/se/features2_16dir.parquet"); F2["id"] = F2.id.astype(str)
    F1 = pd.read_parquet("/home/claude/se/features_16dir.parquet"); F1["id"] = F1.id.astype(str)
    t2 = F2[F2.src == "trv"].drop(columns="src")
    D = D.merge(t2, left_on=["id", "sector"], right_on=["id", "dir"], how="left", suffixes=("", "_f2"))
    if D.ref.iloc[0] != "MN":   # exposure of the reference station in the same wind sector
        r1 = F1[F1.src == "smhi"][["id", "dir", "tree_near", "tree_mid"]].rename(columns={"id": "ref", "dir": "sector", "tree_near": "ref_tree_near", "tree_mid": "ref_tree_mid"})
        r2 = F2[F2.src == "smhi"][["id", "dir", "z0_fp_100"]].rename(columns={"id": "ref", "dir": "sector", "z0_fp_100": "ref_z0_fp"})
        D = D.merge(r1, on=["ref", "sector"], how="left").merge(r2, on=["ref", "sector"], how="left")
    rb = pd.read_csv("/home/claude/se/trv/road_bearing.csv", dtype={"id": str})
    D = D.merge(rb[["id", "road_bearing"]], on="id", how="left")
    a = (D.wd_ref - D.road_bearing + 90) % 180 - 90                 # wind-from relative to road axis, [-90, 90)
    D["alpha_road"] = a
    D["cross_road"] = np.abs(np.sin(np.radians(a)))                 # 0 = wind along the road, 1 = across
    o = sd.trv_obs(); o = o[o.time.dt.minute == 0][["id", "time", "t_air", "t_surf"]]
    D = D.merge(o, on=["id", "time"], how="left")
    D["stab"] = D.t_air - D.t_surf                                  # > 0: road surface colder than air (stable proxy)
    loc = D.time.dt.tz_convert("Europe/Stockholm").dt.hour
    D["night"] = ((loc < 8) | (loc >= 16)).astype(float)
    D["ln_ws_ref"] = np.log(D.ws_ref)
    D["ln_xh_edge"] = np.log(np.clip(D.xh_edge, 0.2, 200))
    for L in [25, 50, 100, 200, 400, 800]:
        D[f"F_fp_{L}"] = ma.F_two_step(D[f"z0_fp_{L}"])
    return D


def loso_ols(D, feats, offset=None):
    import statsmodels.api as smapi
    off = np.zeros(len(D)) if offset is None else offset
    yy = D.y.values - off; pred = np.full(len(D), np.nan)
    ids = D.id.values
    X = smapi.add_constant(D[feats].values) if feats else np.ones((len(D), 1))
    for sid in np.unique(ids):
        te = ids == sid; tr = ~te
        b = np.linalg.lstsq(X[tr], yy[tr], rcond=None)[0]
        pred[te] = X[te] @ b + off[te]
    return pd.Series(pred, index=D.index)          # predicted y (log ratio vs F_WMO * ws_ref)


def loso_gbm(D, feats, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    pred = np.full(len(D), np.nan); ids = D.id.values; X = D[feats].values
    for sid in np.unique(ids):
        te = ids == sid
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=seed).fit(X[~te], D.y.values[~te])
        pred[te] = m.predict(X[te])
    return pd.Series(pred, index=D.index)


def residual_kriging(D, yhat, radius_km=15.0, power=2.0, min_n=1):
    """add an inverse-distance weighted mean of the OTHER stations' mean LOSO residuals (regression kriging, IDW variant)"""
    TM = sd.trv_meta().set_index("id")
    r = (D.y - yhat).groupby(D.id).mean()
    xy = TM.loc[r.index, ["lat", "lon"]]
    corr = {}
    for sid in r.index:
        d = ma.haversine(xy.loc[sid, "lat"], xy.loc[sid, "lon"], xy.lat.values, xy.lon.values)
        m = (d > 0) & (d <= radius_km)
        corr[sid] = 0.0 if m.sum() < min_n else np.sum(r.values[m] / d[m] ** power) / np.sum(1 / d[m] ** power)
    return yhat + D.id.map(corr).values


def to_ws(D, yhat):
    return ma.F_WMO * D.ws_ref * np.exp(yhat)


def station_r2(D, yhat):
    s = pd.DataFrame(dict(o=D.y, p=yhat, id=D.id)).groupby("id").mean()
    return 1 - np.sum((s.o - s.p) ** 2) / np.sum((s.o - s.o.mean()) ** 2)


def summarize(D, yhat, name):
    m = ma.metrics(D, to_ws(D, yhat), name); m["cross_station_R2"] = station_r2(D, yhat); return m


def build_all(mn_path="/home/claude/se/mn_all.nc"):
    """Hourly pairs with all features, for the SMHI-nearest reference (Ds) and the MET Nordic reference (Dm)."""
    import sys, xarray as xr
    sys.path.insert(0, "/home/claude/dovre"); import metnordic
    ds = xr.open_dataset(mn_path)
    ma.TMAX = pd.Timestamp(ds.time.values.max()).tz_localize("UTC")
    Ds = add_features(ma.build("nearest")).dropna(subset=["road_bearing", "stab"]).reset_index(drop=True)
    Dm = add_features(ma.build("mn", mn=(ds, metnordic.interp))).dropna(subset=["road_bearing", "stab"]).reset_index(drop=True)
    SM = sd.smhi_meta().set_index("id")
    u, v, _ = metnordic.interp(ds, SM.loc[Ds.ref, "lat"].values, SM.loc[Ds.ref, "lon"].values, Ds.time.values)
    Ds["mn_ref"] = np.hypot(u, v)
    TM = sd.trv_meta().set_index("id")
    u, v, _ = metnordic.interp(ds, TM.loc[Ds.id, "lat"].values, TM.loc[Ds.id, "lon"].values, Ds.time.values)
    Ds["mn_trv"] = np.hypot(u, v)
    Ds["mn_grad"] = np.log(np.clip(Ds.mn_trv, 0.3, None) / np.clip(Ds.mn_ref, 0.3, None))
    Ds["lnws_between"] = Ds.groupby("id").ln_ws_ref.transform("mean")
    return Ds, Dm
