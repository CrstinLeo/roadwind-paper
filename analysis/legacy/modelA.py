"""Model A pilot: SMHI (10 m) or MET Nordic (10 m) -> Trafikverket station (6 m), hourly, last 7 days."""
import numpy as np, pandas as pd
import sedata as sd

Z_T, Z_R, Z_B, Z0_REF = 6.0, 10.0, 60.0, 0.03
F_WMO = np.log(Z_T / Z0_REF) / np.log(Z_R / Z0_REF)


def F_two_step(z0_site, z0_ref=Z0_REF):
    return np.log(Z_B / z0_ref) / np.log(Z_R / z0_ref) * np.log(Z_T / z0_site) / np.log(Z_B / z0_site)


def haversine(lat1, lon1, lat2, lon2):
    p = np.radians
    a = np.sin(p(lat2 - lat1) / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(p(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p = np.radians
    y = np.sin(p(lon2 - lon1)) * np.cos(p(lat2))
    x = np.cos(p(lat1)) * np.sin(p(lat2)) - np.sin(p(lat1)) * np.cos(p(lat2)) * np.cos(p(lon2 - lon1))
    return np.degrees(np.arctan2(y, x)) % 360


def build(ref="nearest_auto", mn=None):
    """Hourly pairs TrV station x reference. ref: 'nearest_auto' (nearest SMHI automatic station),
    'nearest' (nearest SMHI incl. airports), or 'mn' (MET Nordic at the TrV site; needs mn=(ds, interp))."""
    TM, SM = sd.trv_meta(), sd.smhi_meta()
    TM = TM[TM.dist_km <= 120].copy()
    F = pd.read_parquet("/home/claude/se/features_16dir.parquet"); F["id"] = F.id.astype(str)
    o = sd.trv_obs(); o = o[o.time.dt.minute == 0][["id", "time", "ws", "wd"]]
    if TMAX is not None: o = o[o.time <= pd.Timestamp(TMAX)]
    SW = {s: sd.smhi_wind(s) for s in SM.id}
    res = {s: (SW[s][SW[s].t >= o.time.min()].ws % 1 != 0).mean() > 0.3 for s in SM.id}   # 0.1 m/s resolution?
    rows = []
    for _, t in TM.iterrows():
        g = o[o.id == t.id]
        if len(g) < 24: continue
        if ref == "mn":
            ds, interp = mn
            mu, mv, _ = interp(ds, np.full(len(g), t.lat), np.full(len(g), t.lon), g.time.values)
            r = pd.DataFrame(dict(time=g.time.values, ws_ref=np.hypot(mu, mv), wd_ref=np.degrees(np.arctan2(-mu, -mv)) % 360))
            r["time"] = pd.to_datetime(r.time, utc=True); rid, rdist, rbrg, relev = "MN", 0.0, np.nan, np.nan
        else:
            cand = SM.copy(); cand["d"] = haversine(t.lat, t.lon, cand.lat, cand.lon)
            if ref == "nearest_auto": cand = cand[cand.id.map(res)]
            c = cand.sort_values("d").iloc[0]
            w = SW[c.id]; r = w.rename(columns={"t": "time", "ws": "ws_ref", "wd": "wd_ref"})[["time", "ws_ref", "wd_ref"]]
            rid, rdist, rbrg = c.id, c.d, bearing(t.lat, t.lon, c.lat, c.lon)
            relev = F[(F.src == "smhi") & (F.id == c.id)].elev.iloc[0]
        m = g.merge(r, on="time").dropna()
        m["id"], m["ref"], m["ref_dist"], m["ref_bearing"], m["ref_elev"] = t.id, rid, rdist, rbrg, relev
        rows.append(m)
    D = pd.concat(rows, ignore_index=True)
    D["sector"] = (np.round(D.wd_ref / 22.5) % 16) * 22.5
    D = D.merge(F[F.src == "trv"].drop(columns="src"), left_on=["id", "sector"], right_on=["id", "dir"], how="left")
    D["dz_ref"] = D.elev - D.ref_elev
    D["upwind_angle"] = np.abs((D.ref_bearing - D.wd_ref + 180) % 360 - 180)
    D["ln_z0"] = np.log(D.z0_eff)
    D["F_phys"] = F_two_step(D.z0_eff)
    D = D[(D.ws_ref >= 2.0) & (D.ws > 0.2)].copy()
    D["y"] = np.log(D.ws / (F_WMO * D.ws_ref))
    return D


TMAX = None

FEATS = ["tree_near", "tree_mid", "h_mid", "sx", "tpi_1km"]


def loso(D, feats, base="wmo"):
    """leave-one-station-out: fit on all other stations, predict the held-out one; returns predicted ws"""
    import statsmodels.api as smapi
    pred = pd.Series(index=D.index, dtype=float)
    off = np.log(D.F_phys / F_WMO) if base == "phys" else 0.0
    yy = D.y - off
    for sid in D.id.unique():
        tr, te = D.id != sid, D.id == sid
        if feats:
            X = smapi.add_constant(D.loc[tr, feats]); f = smapi.OLS(yy[tr], X).fit()
            mu = f.predict(smapi.add_constant(D.loc[te, feats], has_constant="add"))
        else:
            mu = yy[tr].mean()
        pred[te] = F_WMO * D.loc[te, "ws_ref"] * np.exp(mu + (off[te] if base == "phys" else 0.0))
    return pred


def metrics(D, pred, name):
    e = pred - D.ws
    le = np.log(pred / D.ws)
    st = le.groupby(D.id).mean()
    return dict(model=name, n=len(D), stations=D.id.nunique(), MAE=np.mean(np.abs(e)), bias=np.mean(e),
                rmse_log=np.sqrt(np.mean(le ** 2)), station_bias_abs_median=np.median(np.abs(st)),
                station_bias_sd=st.std())
