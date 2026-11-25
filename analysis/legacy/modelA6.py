"""Round 6: SMHI -> Trafikverket station by station.

Ladder for every pair (TrV station j, SMHI station s), applied hour by hour:
  L0 raw         U_s (10 m)
  L1 height      0.91 U_s                    (log law, open terrain z0 = 0.03, 10 m -> 6 m)
  L2 roughness   U_s * P_s(k) * F_j(k)       (Wieringa two-step through 60 m: SMHI upwind z0 -> open terrain -> TrV upwind z0;
                                              z0 from NMD laser object height 0.1 h, water 0.0002, else 0.03, footprint-weighted)
  L3 site        L2 * exp(-c_s(k))           (residual SMHI site factor of the POTENTIAL wind, multi-year SMHI history before the test)
  L4 model       L3 * exp(TrV geography model), fitted leave-one-TrV-station-out
Direction: raw SMHI direction, and + road channelling model (LOSO).
k = 16-sector of the network wind-from direction (vector mean of all SMHI stations).
"""
import numpy as np, pandas as pd
import modelA as ma, modelA4 as m4, modelA5 as m5, sedata as sd

ZB, Z0R = 60.0, 0.03
T_TEST0, T_TEST1 = pd.Timestamp("2026-11-13 20:00", tz="UTC"), pd.Timestamp("2026-11-22 20:00", tz="UTC")
HOURS = pd.date_range(T_TEST0, T_TEST1, freq="1h")
DMAX = 80.0
L_SMHI, L_TRV = 1000, 100        # footprint length scales: 10 m sensor sees ~1 km, 6 m roadside sensor ~100 m


def P_to_potential(z0):
    """factor converting 10 m wind over upwind roughness z0 to 10 m wind over open terrain (z0 = 0.03)"""
    return np.log(ZB / z0) / np.log(10 / z0) * np.log(10 / Z0R) / np.log(ZB / Z0R)


def F_to_site(z0, z=6.0):
    """factor converting open-terrain 10 m wind to height z over upwind roughness z0"""
    return np.log(ZB / Z0R) / np.log(10 / Z0R) * np.log(z / z0) / np.log(ZB / z0)


def tables():
    W = pd.read_parquet("/home/claude/se/features_water_16dir.parquet")
    F1 = pd.read_parquet("/home/claude/se/features_16dir.parquet"); F1["id"] = F1.id.astype(str)
    F2 = pd.read_parquet("/home/claude/se/features2_16dir.parquet"); F2["id"] = F2.id.astype(str)
    z0s = W[W.src == "smhi"].pivot(index="id", columns="dir", values=f"z0w_fp_{L_SMHI}")
    z0t = W[W.src == "trv"].pivot(index="id", columns="dir", values=f"z0w_fp_{L_TRV}")
    tree = F1[F1.src == "trv"].pivot(index="id", columns="dir", values="tree_near")
    lnxh = np.log(F2[F2.src == "trv"].pivot(index="id", columns="dir", values="xh_edge").clip(0.2, 200))
    return z0s, z0t, tree, lnxh


def potential(ws, wd, z0s):
    k = m5.sector(m5.network_dir(ws, wd))
    P = pd.DataFrame({s: P_to_potential(z0s.loc[s].reindex(k).values) for s in ws.columns}, index=ws.index)
    return ws * P, k


def run():
    z0s, z0t, tree, lnxh = tables()
    # ---------------- history: potential-wind site factors (train period only)
    wsL, wdL = m5.smhi_long(end="2026-11-13 19:00")
    potL, _ = potential(wsL, wdL, z0s)
    S_raw = m5.site_factors(wsL, wdL); S_pot = m5.site_factors(potL, wdL)
    # ---------------- test period matrices
    tws, twd, sws, swd, TM, SM = m4.matrices()
    tws, twd, sws, swd = (X.reindex(HOURS) for X in (tws, twd, sws, swd))
    for X in (tws, twd):                                    # matrices() covers the round-4 window: rebuild from raw obs
        pass
    o = sd.trv_obs(); o = o[(o.time.dt.minute == 0) & o.id.isin(TM.index)]
    tws = o.pivot_table(index="time", columns="id", values="ws").reindex(HOURS)
    twd = o.pivot_table(index="time", columns="id", values="wd").reindex(HOURS)
    sws = pd.DataFrame({s: sd.smhi_wind(s).set_index("t").ws.reindex(HOURS) for s in SM.index})
    swd = pd.DataFrame({s: sd.smhi_wind(s).set_index("t").wd.reindex(HOURS) for s in SM.index})
    keep = tws.columns[tws.notna().sum() >= 100]; tws, twd, TM = tws[keep], twd[keep], TM.loc[keep]
    pot, k = potential(sws, swd, z0s)
    Cp = pd.DataFrame({s: S_pot["c"].loc[s].reindex(k).values for s in SM.index}, index=HOURS)
    corr = pot * np.exp(-Cp)                                 # site-corrected potential wind of every SMHI station
    lnU_net = np.log(corr.mean(1).clip(lower=0.5))
    rb = pd.read_csv("/home/claude/se/trv/road_bearing.csv", dtype={"id": str}).set_index("id").road_bearing
    rows = []
    for j in TM.index:
        Fj = F_to_site(z0t.loc[j].reindex(k).values)
        tr = tree.loc[j].reindex(k).values; xh = lnxh.loc[j].reindex(k).values
        for s in SM.index:
            d = ma.haversine(TM.loc[j, "lat"], TM.loc[j, "lon"], SM.loc[s, "lat"], SM.loc[s, "lon"])
            if d > DMAX: continue
            fr = pd.DataFrame(dict(time=HOURS, trv=j, smhi=s, dist=d, ws=tws[j].values, wd=twd[j].values,
                                   U0=sws[s].values, wd_s=swd[s].values, U1=0.91 * sws[s].values,
                                   U2=pot[s].values * Fj, U3=corr[s].values * Fj, tree=tr, lnxh=xh, lnU=lnU_net.values,
                                   road=rb.get(j, np.nan), bearing=ma.bearing(TM.loc[j, "lat"], TM.loc[j, "lon"], SM.loc[s, "lat"], SM.loc[s, "lon"])))
            rows.append(fr)
    D = pd.concat(rows, ignore_index=True)
    D["k"] = np.tile(k, len(D) // len(HOURS))
    D["upwind_angle"] = np.abs(m4.wrap(D.bearing.values - D.k.values))
    return D, dict(S_raw=S_raw, S_pot=S_pot, TM=TM, SM=SM, tws=tws, twd=twd, sws=sws, swd=swd, corr=corr, k=k, z0s=z0s, z0t=z0t)


FEATS = ["tree", "lnxh", "lnU", "lnU_tree"]


def fit_trv_model(D):
    """L4: ln(U_trv / U3) = b0 + b1 tree + b2 ln(x/h) + (g0 + g1 tree) lnU_net, leave-one-TrV-station-out.
    Direction: dphi = d0 + d1 sin(2 alpha) tree, alpha = SMHI wind-from minus road axis (LOSO)."""
    D = D.copy(); D["lnU_tree"] = D.lnU * D.tree
    D["y"] = np.log(D.ws.clip(lower=0.1) / D.U3.clip(lower=0.1))
    D["s2t"] = np.sin(np.radians(2 * (m4.wrap(2 * (D.wd_s - D.road)) / 2))) * D.tree
    D["dd"] = m4.wrap(D.wd - D.wd_s)
    fs = (D.U0 >= 2) & (D.ws > 0.2) & D[FEATS].notna().all(1)
    fd = (D.U0 >= 3) & (D.ws >= 1.5) & D.dd.notna() & D.s2t.notna()
    X = np.column_stack([np.ones(len(D)), D[FEATS].fillna(0).values]); Xd = np.column_stack([np.ones(len(D)), D.s2t.fillna(0).values])
    p, pdd = np.full(len(D), np.nan), np.full(len(D), np.nan)
    for j in D.trv.unique():
        te = (D.trv == j).values
        b = np.linalg.lstsq(X[fs.values & ~te], D.y.values[fs.values & ~te], rcond=None)[0]
        r = D.y.values[fs.values & ~te] - X[fs.values & ~te] @ b
        p[te] = X[te] @ b + np.log(np.mean(np.exp(r)))
        bd = np.linalg.lstsq(Xd[fd.values & ~te], D.dd.values[fd.values & ~te], rcond=None)[0]
        pdd[te] = Xd[te] @ bd
    D["U4"] = D.U3 * np.exp(p)
    D["wd4"] = (D.wd_s + pdd) % 360
    bfull = np.linalg.lstsq(X[fs.values], D.y.values[fs.values], rcond=None)[0]
    bd_full = np.linalg.lstsq(Xd[fd.values], D.dd.values[fd.values], rcond=None)[0]
    return D, dict(speed=dict(zip(["b0"] + FEATS, bfull)), direction=dict(d0=bd_full[0], d1=bd_full[1]))


def pair_metrics(D):
    def f(g):
        ok = g.ws.notna() & g.U0.notna()
        g = g[ok]; m = (g.U0 >= 2) & (g.ws > 0.2)
        out = dict(dist=g.dist.iloc[0], n=len(g))
        for L in ["U0", "U1", "U2", "U3", "U4"]:
            out[f"mae_{L}"] = np.mean(np.abs(g[L] - g.ws))
            out[f"bias_{L}"] = float(np.exp(np.mean(np.log(g.ws[m] / g[L][m]))) - 1) if m.sum() > 10 else np.nan   # TrV/pred - 1
        out["r"] = np.corrcoef(g.U0, g.ws)[0, 1] if g.U0.std() > 0 else np.nan
        dm = (g.ws >= 2) & (g.U0 >= 2) & g.wd.notna() & g.wd_s.notna()
        out["dir_raw"] = np.mean(np.abs(m4.wrap(g.wd_s - g.wd)[dm])) if dm.sum() > 10 else np.nan
        out["dir_mod"] = np.mean(np.abs(m4.wrap(g.wd4 - g.wd)[dm])) if dm.sum() > 10 else np.nan
        return pd.Series(out)
    return D.groupby(["trv", "smhi"]).apply(f).reset_index()


def combine(D, weights="inv_d", dsel=60.0, min_n=2):
    """Combined estimate for each TrV station from the selected SMHI stations (log-space weighted mean of U4,
    vector mean of model directions). Selection: all stations within dsel km, at least the min_n nearest."""
    out = []
    for j, g in D.groupby("trv"):
        dist = g.groupby("smhi").dist.first().sort_values()
        sel = list(dist[dist <= dsel].index)
        if len(sel) < min_n: sel = list(dist.index[:min_n])
        w = (1 / dist[sel].clip(lower=3)).to_dict()
        h = g[g.smhi.isin(sel)].copy(); h["w"] = h.smhi.map(w)
        def agg(x):
            ok = x.U4.notna() & (x.U4 > 0)
            if not ok.any(): return pd.Series(dict(ws=x.ws.iloc[0], U=np.nan, wd=x.wd.iloc[0], wdm=np.nan, U0n=np.nan, wd0n=np.nan))
            ww = x.w[ok] / x.w[ok].sum()
            U = np.exp(np.sum(ww * np.log(x.U4[ok])))
            u = np.sum(ww * np.sin(np.radians(x.wd4[ok]))); v = np.sum(ww * np.cos(np.radians(x.wd4[ok])))
            near = x.loc[x.dist.idxmin()]
            return pd.Series(dict(ws=x.ws.iloc[0], U=U, wd=x.wd.iloc[0], wdm=np.degrees(np.arctan2(u, v)) % 360,
                                  U0n=near.U0, wd0n=near.wd_s))
        r = h.groupby("time").apply(agg).reset_index(); r["trv"] = j; r["selected"] = ",".join(sel)
        out.append(r)
    return pd.concat(out, ignore_index=True)
