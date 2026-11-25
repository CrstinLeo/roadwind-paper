"""Round 5: remove the site effect of every SMHI station (direction-dependent, from multi-year SMHI data), then a lean TrV model."""
import numpy as np, pandas as pd
import modelA as ma, modelA4 as m4, sedata as sd

DIRS = np.arange(0, 360, 22.5)


def smhi_long(start="2022-01-01", end="2026-11-13 19:00"):
    SM = sd.smhi_meta().set_index("id")
    ws, wd = {}, {}
    for s in SM.index:
        w = sd.smhi_wind(s); w = w[(w.t >= pd.Timestamp(start, tz="UTC")) & (w.t <= pd.Timestamp(end, tz="UTC"))].set_index("t")
        ws[s] = w.ws; wd[s] = w.wd
    ws, wd = pd.DataFrame(ws), pd.DataFrame(wd)
    idx = pd.date_range(ws.index.min(), ws.index.max(), freq="1h"); return ws.reindex(idx), wd.reindex(idx)


def network_dir(ws, wd):
    u = -(ws * np.sin(np.radians(wd))).mean(1); v = -(ws * np.cos(np.radians(wd))).mean(1)
    return pd.Series(np.degrees(np.arctan2(-u, -v)) % 360, index=ws.index)


def sector(phi):
    return (np.round(np.asarray(phi) / 22.5) % 16) * 22.5


def site_factors(ws, wd, min_net=3.0, min_ws=0.5, iters=15):
    """ln U_s(t) = a(t) + c_s(k(t)) + e;  k = 16-sector of the network wind-from direction;  sum_s c_s(k) = 0 for every k.
    Direction: phi_s(t) - phi_net(t) = delta_s(k) + e (circular mean). Uses hours with network median speed >= min_net."""
    net = ws.median(1); phi = network_dir(ws, wd); k = pd.Series(sector(phi), index=ws.index)
    use = (net >= min_net) & (ws.notna().sum(1) >= 12)
    L = np.log(ws[use].where(ws[use] >= min_ws)); K = k[use]
    c = pd.DataFrame(0.0, index=ws.columns, columns=DIRS)
    for _ in range(iters):
        C = pd.DataFrame({s: c.loc[s].reindex(K.values).values for s in ws.columns}, index=L.index)
        a = (L - C).mean(1)
        R = L.sub(a, axis=0)
        c = R.groupby(K.values).mean().T.reindex(columns=DIRS)
        c = c - c.mean(0)                                   # identifiability: network mean of c is 0 in every sector
    n = L.notna().groupby(K.values).sum().T.reindex(columns=DIRS)
    D = np.radians(wd[use].sub(phi[use], axis=0))
    delta = pd.DataFrame({s: np.degrees(np.arctan2(np.sin(D[s]).groupby(K.values).mean(), np.cos(D[s]).groupby(K.values).mean())) for s in ws.columns}).T.reindex(columns=DIRS)
    resid_sd = (L.sub(a, axis=0) - pd.DataFrame({s: c.loc[s].reindex(K.values).values for s in ws.columns}, index=L.index)).std()
    return dict(c=c, delta=delta, n=n, a=a, resid_sd=resid_sd, hours=int(use.sum()))


def standardize(ws, wd, c, delta):
    """station wind -> network-equivalent wind: U_std = U * exp(-c_s(k)), phi_std = phi - delta_s(k), k from the network direction"""
    phi = network_dir(ws, wd); k = sector(phi)
    C = pd.DataFrame({s: c.loc[s].reindex(k).values for s in ws.columns}, index=ws.index)
    Dl = pd.DataFrame({s: delta.loc[s].reindex(k).values for s in ws.columns}, index=ws.index)
    return ws * np.exp(-C), (wd - Dl) % 360


def trv_geo(R):
    """add water/coast/elevation features of the TrV site in the reference wind sector"""
    W = pd.read_parquet("/home/claude/se/features_water_16dir.parquet"); W = W[W.src == "trv"].drop(columns="src")
    R = R.merge(W, left_on=["id", "sector"], right_on=["id", "dir"], how="left", suffixes=("", "_w"))
    R["ln_dc"] = np.log1p(R.dist_coast_km); R["lnz0_1000"] = np.log(R.z0w_fp_1000); R["lnz0_100"] = np.log(R.z0w_fp_100)
    R["elev100"] = R.elev / 100.0
    return R


def forward_select(R, cands, dir_feats, start=(), tol=0.005, max_k=8):
    sel = list(start); hist = []
    base = m4.score(m4.fit_predict(R, speed_feats=sel or ["tree_near"] if not sel else sel, dir_feats=dir_feats))["MAE_ws_model"] if sel else np.inf
    while len(sel) < max_k:
        res = {}
        for k in cands:
            if k in sel: continue
            res[k] = m4.score(m4.fit_predict(R, speed_feats=sel + [k], dir_feats=dir_feats))["MAE_ws_model"]
        k = min(res, key=res.get)
        if base - res[k] < tol: break
        sel.append(k); base = res[k]; hist.append((k, base))
    return sel, hist


def reference(tws, twd, sws, swd, ss, TM, SM, speed=("idw", 250, 1.0), direction=("idw", 80, 2.0)):
    """speed reference from site-corrected SMHI speeds (ss), direction reference from raw SMHI directions"""
    a = m4.select(speed[0], tws, twd, ss, swd, TM, SM, dmax=speed[1], power=speed[2])
    b = m4.select(direction[0], tws, twd, sws, swd, TM, SM, dmax=direction[1], power=direction[2])
    a["wd_ref"] = b.wd_ref.values
    a["U_smhi"] = a.time.map(ss.mean(1))
    return trv_geo(m4.add_trv_features(a))
