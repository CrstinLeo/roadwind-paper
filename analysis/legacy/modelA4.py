"""Model A, round 4: how far is an SMHI station useful for a TrV station, reference selection, speed + direction model."""
import sys
import numpy as np, pandas as pd
import modelA as ma, sedata as sd

T0, T1 = pd.Timestamp("2026-11-13 20:00", tz="UTC"), pd.Timestamp("2026-11-21 13:00", tz="UTC")
HOURS = pd.date_range(T0, T1, freq="1h")


def wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


def circmean(a, w=None):
    a = np.radians(np.asarray(a)); w = np.ones_like(a) if w is None else w
    return np.degrees(np.arctan2(np.nansum(w * np.sin(a)), np.nansum(w * np.cos(a)))) % 360


def matrices():
    """hourly TrV and SMHI speed/direction matrices [time x station]"""
    TM = sd.trv_meta(); TM = TM[TM.dist_km <= 120].set_index("id")
    o = sd.trv_obs(); o = o[(o.time.dt.minute == 0) & o.id.isin(TM.index)]
    tws = o.pivot_table(index="time", columns="id", values="ws").reindex(HOURS)
    twd = o.pivot_table(index="time", columns="id", values="wd").reindex(HOURS)
    SM = sd.smhi_meta().set_index("id")
    sws, swd = {}, {}
    for s in SM.index:
        w = sd.smhi_wind(s).set_index("t"); sws[s] = w.ws.reindex(HOURS); swd[s] = w.wd.reindex(HOURS)
    sws, swd = pd.DataFrame(sws), pd.DataFrame(swd)
    keep = tws.columns[tws.notna().sum() >= 100]
    return tws[keep], twd[keep], sws, swd, TM.loc[keep], SM


def mn_at_trv(TM, path="/home/claude/se/mn_all.nc"):
    sys.path.insert(0, "/home/claude/dovre"); import metnordic, xarray as xr
    ds = xr.open_dataset(path)
    lat = np.repeat(TM.lat.values, len(HOURS)); lon = np.repeat(TM.lon.values, len(HOURS)); tt = np.tile(HOURS.values, len(TM))
    u, v, _ = metnordic.interp(ds, lat, lon, tt)
    ws = pd.DataFrame(np.hypot(u, v).reshape(len(TM), len(HOURS)).T, index=HOURS, columns=TM.index)
    wd = pd.DataFrame((np.degrees(np.arctan2(-u, -v)) % 360).reshape(len(TM), len(HOURS)).T, index=HOURS, columns=TM.index)
    return ws, wd


def pair_skill(y, x, yd=None, xd=None):
    """2-fold time-block CV (first/second half). Speed: OLS y = a + b x, skill vs training-mean climatology.
    Direction (hours y>=1.5, x>=2): x direction + training circular-mean offset, MAE vs constant training-mean direction."""
    m = y.notna() & x.notna(); y, x = y[m].values, x[m].values
    n = len(y)
    if n < 60: return {}
    h = np.arange(n) < n // 2
    se_m, se_c = [], []
    for tr in (h, ~h):
        te = ~tr
        b = np.polyfit(x[tr], y[tr], 1)
        se_m.append((np.polyval(b, x[te]) - y[te]) ** 2); se_c.append((y[tr].mean() - y[te]) ** 2)
    out = dict(n=n, r_ws=np.corrcoef(x, y)[0, 1], skill_ws=1 - np.concatenate(se_m).mean() / np.concatenate(se_c).mean(),
               rmse_ws=np.sqrt(np.concatenate(se_m).mean()))
    if yd is not None:
        yd_, xd_ = yd[m].values, xd[m].values
        ok = (y >= 1.5) & (x >= 2) & np.isfinite(yd_) & np.isfinite(xd_)
        if ok.sum() >= 30:
            k = np.arange(len(y)); hh = k < n // 2
            e_m, e_c = [], []
            for tr in (hh, ~hh):
                te = ~tr & ok; trm = tr & ok
                if trm.sum() < 10 or te.sum() < 10: continue
                off = circmean(wrap(yd_[trm] - xd_[trm])); cst = circmean(yd_[trm])
                e_m.append(np.abs(wrap(xd_[te] + off - yd_[te]))); e_c.append(np.abs(wrap(cst - yd_[te])))
            if e_m:
                e_m, e_c = np.concatenate(e_m), np.concatenate(e_c)
                out.update(n_dir=len(e_m), mae_dir=e_m.mean(), skill_dir=1 - e_m.mean() / e_c.mean())
    return out


def all_pairs(tws, twd, sws, swd, TM, SM):
    rows = []
    for t in tws.columns:
        for s in sws.columns:
            r = pair_skill(tws[t], sws[s], twd[t], swd[s])
            if not r: continue
            r.update(trv=t, smhi=s, dist=ma.haversine(TM.loc[t, "lat"], TM.loc[t, "lon"], SM.loc[s, "lat"], SM.loc[s, "lon"]),
                     bearing=ma.bearing(TM.loc[t, "lat"], TM.loc[t, "lon"], SM.loc[s, "lat"], SM.loc[s, "lon"]),
                     coastal_smhi=s in ("71380", "71190", "81050"))
            rows.append(r)
    return pd.DataFrame(rows)


DMAX = 40.0


def regional_dir(sws, swd):
    u = -(sws * np.sin(np.radians(swd))).mean(1); v = -(sws * np.cos(np.radians(swd))).mean(1)
    return pd.Series(np.degrees(np.arctan2(-u, -v)) % 360, index=sws.index)


def select(strategy, tws, twd, sws, swd, TM, SM, dmax=DMAX, upw=0.0, power=2.0):
    """Hourly reference for every TrV station. strategy: 'nearest' | 'upwind' | 'upwind_lag' | 'idw'.
    upwind: closest SMHI station within dmax whose bearing from the TrV site is within ±45° of the regional wind-from
    direction (vector mean of all SMHI stations); if none, the nearest station. idw: vector mean of stations within dmax, w = 1/d²."""
    rdir = regional_dir(sws, swd)
    F1 = pd.read_parquet("/home/claude/se/features_16dir.parquet"); F1["id"] = F1.id.astype(str)
    rtn = F1[F1.src == "smhi"].set_index(["id", "dir"]).tree_near
    out = []
    for t in tws.columns:
        d = pd.Series({s: ma.haversine(TM.loc[t, "lat"], TM.loc[t, "lon"], SM.loc[s, "lat"], SM.loc[s, "lon"]) for s in SM.index}).sort_values()
        b = pd.Series({s: ma.bearing(TM.loc[t, "lat"], TM.loc[t, "lon"], SM.loc[s, "lat"], SM.loc[s, "lon"]) for s in SM.index})
        near = d.index[0]
        fr = pd.DataFrame(index=tws.index); fr["id"] = t; fr["ws"] = tws[t]; fr["wd"] = twd[t]; fr["rdir"] = rdir
        if strategy in ("nearest", "upwind", "upwind_lag"):
            ref = pd.Series(near, index=tws.index)
            if strategy != "nearest":
                cand = d[d <= dmax].index
                for h in tws.index:
                    up = [s for s in cand if np.abs(wrap(b[s] - rdir[h])) <= 45 and np.isfinite(sws.at[h, s])]
                    if up: ref[h] = up[0]
            fr["ref"] = ref.values
            lag = np.zeros(len(fr), int)
            if strategy == "upwind_lag":
                along = d[ref].values * np.cos(np.radians(wrap(b[ref].values - rdir.values)))
                U = np.array([sws.at[h, s] for h, s in zip(tws.index, ref)])
                lag = np.clip(np.round(np.where(along > 0, along / (3.6 * np.maximum(U, 2.0)), 0)), 0, 3).astype(int)
            idx = np.clip(np.arange(len(fr)) - lag, 0, len(fr) - 1)
            fr["ws_ref"] = [sws[s].iloc[i] for s, i in zip(ref, idx)]
            fr["wd_ref"] = [swd[s].iloc[i] for s, i in zip(ref, idx)]
            fr["ref_dist"] = d[ref].values; fr["lag_h"] = lag
            sec = (np.round(fr.wd_ref.fillna(0) / 22.5) % 16) * 22.5
            fr["ref_tree_near"] = [rtn.get((s, q), np.nan) for s, q in zip(ref, sec)]
        else:  # idw
            cand = list(d[d <= dmax].index) or [near]
            w0 = 1 / np.maximum(d[cand].values, 3.0) ** power
            S, D = sws[cand].values, swd[cand].values
            ang = np.abs(wrap(b[cand].values[None, :] - rdir.values[:, None]))
            w = w0[None, :] * (1 + upw * np.cos(np.radians(ang))) / (1 + upw)   # upw > 0 favours upwind stations
            u = -S * np.sin(np.radians(D)); v = -S * np.cos(np.radians(D)); ok = np.isfinite(u)
            W = np.where(ok, w, 0); u, v = np.where(ok, u, 0), np.where(ok, v, 0)
            uu, vv = (W * u).sum(1) / W.sum(1), (W * v).sum(1) / W.sum(1)
            fr["ws_ref"] = (W * np.where(ok, S, 0)).sum(1) / W.sum(1)          # weighted scalar mean speed (no vector cancellation)
            fr["wd_ref"] = np.degrees(np.arctan2(-uu, -vv)) % 360
            fr["ref"] = "+".join(cand); fr["ref_dist"] = (w * d[cand].values).sum(1) / w.sum(1); fr["lag_h"] = 0
            sec = (np.round(fr.wd_ref.fillna(0) / 22.5) % 16) * 22.5
            fr["ref_tree_near"] = [np.nansum([rtn.get((s, q), np.nan) * ww for s, ww in zip(cand, wr)]) / wr.sum() for q, wr in zip(sec, w)]
        out.append(fr.reset_index(names="time"))
    R = pd.concat(out, ignore_index=True)
    R["U_smhi"] = R.time.map(sws.mean(1))
    return R


def add_trv_features(R):
    F1 = pd.read_parquet("/home/claude/se/features_16dir.parquet"); F1["id"] = F1.id.astype(str)
    F2 = pd.read_parquet("/home/claude/se/features2_16dir.parquet"); F2["id"] = F2.id.astype(str)
    F = F1[F1.src == "trv"].drop(columns="src").merge(F2[F2.src == "trv"].drop(columns="src"), on=["id", "dir"])
    R = R.copy(); R["sector"] = (np.round(R.wd_ref.fillna(0) / 22.5) % 16) * 22.5
    R = R.merge(F, left_on=["id", "sector"], right_on=["id", "dir"], how="left")
    rb = pd.read_csv("/home/claude/se/trv/road_bearing.csv", dtype={"id": str})
    R = R.merge(rb[["id", "road_bearing"]], on="id", how="left")
    R["alpha_road"] = wrap(2 * (R.wd_ref - R.road_bearing)) / 2           # [-90, 90)
    R["s2"] = np.sin(np.radians(2 * R.alpha_road)); R["cross_road"] = np.abs(np.sin(np.radians(R.alpha_road)))
    R["ln_xh_edge"] = np.log(np.clip(R.xh_edge, 0.2, 200))
    R["lnU"] = np.log(R.U_smhi.clip(lower=0.5)); R["lnU_tree"] = R.lnU * R.tree_near
    R["ln_dist"] = np.log(R.ref_dist.clip(lower=1))
    return R


SPEED_FEATS = ["tree_near", "tree_mid", "h_mid", "sx", "tpi_1km", "ln_xh_edge", "cross_road", "ref_tree_near", "lnU", "lnU_tree", "ln_dist"]
DIR_FEATS = ["tree_near", "s2", "s2_tree"]


def fit_predict(R, speed_feats=SPEED_FEATS, dir_feats=DIR_FEATS):
    """Leave-one-station-out speed (log-ratio OLS) and direction (offset OLS) predictions for every hour."""
    R = R.copy(); R["s2_tree"] = R.s2 * R.tree_near
    R["y"] = np.log(R.ws.clip(lower=0.1) / (ma.F_WMO * R.ws_ref.clip(lower=0.1)))
    R["dd"] = wrap(R.wd - R.wd_ref)
    fit_s = (R.ws_ref >= 2) & (R.ws > 0.2) & R[speed_feats].notna().all(1)
    fit_d = (R.ws_ref >= 3) & (R.ws >= 1.5) & R.dd.notna() & R[dir_feats].notna().all(1)
    Xs = np.column_stack([np.ones(len(R)), R[speed_feats].fillna(R[speed_feats].median()).values])
    Xd = np.column_stack([np.ones(len(R)), R[dir_feats].fillna(0).values])
    ps, pd_ = np.full(len(R), np.nan), np.full(len(R), np.nan); ids = R.id.values
    for sid in np.unique(ids):
        te = ids == sid
        bs = np.linalg.lstsq(Xs[fit_s.values & ~te], R.y.values[fit_s.values & ~te], rcond=None)[0]
        bd = np.linalg.lstsq(Xd[fit_d.values & ~te], R.dd.values[fit_d.values & ~te], rcond=None)[0]
        trm = fit_s.values & ~te
        sm = np.log(np.mean(np.exp(R.y.values[trm] - Xs[trm] @ bs)))      # Duan smearing: mean, not median, of the ratio
        ps[te] = Xs[te] @ bs + sm; pd_[te] = Xd[te] @ bd
    R["ws_ref6"] = ma.F_WMO * R.ws_ref
    R["ws_model"] = R.ws_ref6 * np.exp(ps)
    R["wd_model"] = (R.wd_ref + pd_) % 360
    return R


def score(R):
    ok = R.ws.notna() & R.ws_ref.notna()
    s = R[ok]
    dmask = (s.ws >= 2) & (s.ws_ref >= 2) & s.wd.notna() & s.wd_ref.notna()
    return dict(hours=len(s), MAE_ws_raw10m=np.mean(np.abs(s.ws_ref - s.ws)), MAE_ws_height=np.mean(np.abs(s.ws_ref6 - s.ws)),
                MAE_ws_model=np.mean(np.abs(s.ws_model - s.ws)), bias_model=np.mean(s.ws_model - s.ws),
                r_ws_model=np.corrcoef(s.ws_model, s.ws)[0, 1],
                n_dir=int(dmask.sum()), MAE_dir_raw=np.mean(np.abs(wrap(s.wd_ref - s.wd)[dmask])),
                MAE_dir_model=np.mean(np.abs(wrap(s.wd_model - s.wd)[dmask])))


def hybrid(tws, twd, sws, swd, TM, SM):
    """speed reference: all SMHI stations, weights 1/d (regional signal, noise averaged);
    direction reference: stations within 80 km, vector mean with weights 1/d² (direction is more local)."""
    a = select("idw", tws, twd, sws, swd, TM, SM, dmax=250, power=1.0)
    b = select("idw", tws, twd, sws, swd, TM, SM, dmax=80, power=2.0)
    a["wd_ref"] = b.wd_ref.values; a["ref"] = "speed: all/1/d, dir: ≤80 km/1/d²"
    return a


def mn_reference(tws, twd, sws, swd, TM, mws, mwd):
    out = []
    for t in tws.columns:
        fr = pd.DataFrame(dict(time=tws.index, id=t, ws=tws[t].values, wd=twd[t].values, ws_ref=mws[t].values, wd_ref=mwd[t].values,
                               ref="MET Nordic", ref_dist=1.0, lag_h=0, ref_tree_near=0.0))
        out.append(fr)
    R = pd.concat(out, ignore_index=True); R["U_smhi"] = R.time.map(sws.mean(1)); return R
