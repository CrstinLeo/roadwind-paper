"""Evaluation helpers shared by the energy and baseline scripts (22, 24): direction coverage and QC, the comparison
estimators, cross-fitted recalibration, air-drag energy sums per station and station-cluster bootstrap.

All 'cross-fitted' quantities for a station j in time block b are learned on the other blocks AND the other stations,
the same protocol as the blocked-time x leave-one-station-out cross-validation of the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import drag
from .geo import haversine_km, wrap180

RHO, V80 = 1.25, 80 / 3.6
BINS = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, np.inf])
MIN_DIR_H, MIN_DIR_FRAC = 2000, 0.5


def blocks(n_hours: int, n_blocks: int) -> np.ndarray:
    blk = np.empty(n_hours, int)
    for b, ix in enumerate(np.array_split(np.arange(n_hours), n_blocks)):
        blk[ix] = b
    return blk


def stuck_mask(ws, wd, run=3, min_ws=2.0):
    """hours in a run of >= run identical consecutive directions while the speed is >= min_ws (stuck vane / repeated value)"""
    same = np.zeros(wd.shape, bool)
    same[1:] = (wd[1:] == wd[:-1]) & (ws[1:] >= min_ws) & (ws[:-1] >= min_ws)
    out = np.zeros_like(same)
    for j in range(wd.shape[1]):
        idx = np.flatnonzero(np.diff(np.r_[0, same[:, j].astype(int), 0]))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a >= run - 1:
                out[a - 1:b, j] = True
    return out


def direction_coverage(ws, wd, min_hours=MIN_DIR_H, min_frac=MIN_DIR_FRAC):
    """direction after QC, and stations with adequate direction coverage"""
    st = stuck_mask(ws, wd)
    wq = np.where(st, np.nan, wd)
    n_spd = np.isfinite(ws).sum(0); n_dir = (np.isfinite(ws) & np.isfinite(wq)).sum(0)
    ok = (n_dir >= min_hours) & (n_dir >= min_frac * n_spd)
    table = pd.DataFrame(dict(n_speed=n_spd, n_dir_raw=(np.isfinite(ws) & np.isfinite(wd)).sum(0), n_dir=n_dir,
                              stuck=st.sum(0), mean_ws=np.nanmean(ws, 0),
                              mean_ws_dir=np.nanmean(np.where(np.isfinite(wq), ws, np.nan), 0), dir_ok=ok))
    return wq, ok, table, float(st.sum() / max(np.isfinite(wd).sum(), 1))


def distance_matrix(lat, lon):
    n = len(lat)
    return np.array([[float(haversine_km(lat[i], lon[i], lat[j], lon[j])) for j in range(n)] for i in range(n)])


def nearest_other(ws, wd, D, k_max=15):
    """leave-one-out: wind of the nearest OTHER roadside station with speed and direction in the hour"""
    D = D.copy(); np.fill_diagonal(D, np.inf)
    U = np.full(ws.shape, np.nan); W = np.full(ws.shape, np.nan); dist = np.full(ws.shape, np.nan)
    for i in range(ws.shape[1]):
        for j in np.argsort(D[i])[:k_max]:
            m = np.isnan(U[:, i]) & np.isfinite(ws[:, j]) & np.isfinite(wd[:, j])
            U[m, i], W[m, i], dist[m, i] = ws[m, j], wd[m, j], D[i, j]
    return U, W, dist


def crossfit_ratio(num, den, blk):
    """one constant sum(num)/sum(den), learned without block b and without station j -> [hour x station]"""
    nb = blk.max() + 1
    ok = np.isfinite(num) & np.isfinite(den)
    a, b = np.where(ok, num, 0), np.where(ok, den, 0)
    A = np.stack([a[blk == k].sum(0) for k in range(nb)]); B = np.stack([b[blk == k].sum(0) for k in range(nb)])
    c = (A.sum() - A.sum(1, keepdims=True) - A.sum(0, keepdims=True) + A) / \
        (B.sum() - B.sum(1, keepdims=True) - B.sum(0, keepdims=True) + B)
    return c[blk]


def crossfit_calibration(U, ws, blk, min_n=50):
    """recalibrated estimate U r(U): r = mean(observed)/mean(estimate) in bins of the estimate, learned on the other
    blocks and stations, interpolated linearly in ln U between bin centres"""
    nb_t, nst, nq = blk.max() + 1, U.shape[1], len(BINS) - 1
    ok = np.isfinite(U) & np.isfinite(ws)
    k = np.clip(np.digitize(np.where(ok, U, 0), BINS) - 1, 0, nq - 1)
    lu = np.log(np.clip(U, 0.05, None))
    So, Se, Sl, Sn = (np.zeros((nb_t, nst, nq)) for _ in range(4))
    for b in range(nb_t):
        for q in range(nq):
            m = ok & (blk[:, None] == b) & (k == q)
            So[b, :, q] = np.where(m, ws, 0).sum(0); Se[b, :, q] = np.where(m, U, 0).sum(0)
            Sl[b, :, q] = np.where(m, lu, 0).sum(0); Sn[b, :, q] = m.sum(0)
    out = np.full(U.shape, np.nan)
    for b in range(nb_t):
        for j in range(nst):
            tr = lambda S: S.sum((0, 1)) - S[b].sum(0) - S[:, j].sum(0) + S[b, j]
            o, e, l, n = tr(So), tr(Se), tr(Sl), tr(Sn)
            g = n >= min_n
            m = (blk == b) & ok[:, j]
            out[m, j] = U[m, j] * np.interp(lu[m, j], l[g] / n[g], o[g] / e[g])
    return out


def reliability(U, ws):
    ok = np.isfinite(U) & np.isfinite(ws)
    u, o = U[ok], ws[ok]
    rows = []
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = (u >= a) & (u < b)
        if m.sum():
            rows.append(dict(lo=a, hi=b, n=int(m.sum()), est=u[m].mean(), obs=o[m].mean(), ratio=o[m].mean() / u[m].mean(),
                             q25=np.percentile(o[m], 25), q75=np.percentile(o[m], 75),
                             sd_log=float(np.std(np.log(np.clip(o[m], 0.1, None) / np.clip(u[m], 0.1, None))))))
    return pd.DataFrame(rows)


def by_observed(ests, ws, edges=(0, 2, 4, 6, 8, 10, 12, 15, np.inf)):
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m0 = np.isfinite(ws) & (ws >= a) & (ws < b)
        for k, U in ests.items():
            m = m0 & np.isfinite(U)
            if m.any():
                rows.append(dict(lo=a, hi=b, est=k, n=int(m.sum()), mae=float(np.mean(np.abs(U[m] - ws[m]))),
                                 me=float(np.mean(U[m] - ws[m]))))
    return pd.DataFrame(rows)


def energy(speed, wfrom, heading, model="vecto", v=V80, rho=RHO):
    h, c = drag.components(speed, wfrom, heading)
    return drag.energy_kwh100(v, h, c, rho=rho, model=model)


def station_sums(ws, wd, road, ests, ok, fh, model="vecto", v=V80, keep=None, cond=None):
    """per-station sums over one-way passes in both directions along the road axis. ests = {name: (speed, dir)};
    'no wind' is the still-air energy. Winds are referred to vehicle height with the factor fh."""
    E0 = float(drag.energy_kwh100(v, 0.0, 0.0, rho=RHO, model=model))
    T = [np.where(ok, energy(ws * fh, wd, road[None, :] + k, model, v), np.nan) for k in (0.0, 180.0)]
    h, c = drag.components(ws * fh, wd, road[None, :])
    head = [np.where(ok, drag.energy_kwh100(v, s * h, 0 * h, rho=RHO, model=model), np.nan) for s in (1, -1)]
    cross = np.where(ok, drag.energy_kwh100(v, 0 * c, c, rho=RHO, model=model), np.nan)
    strong = [np.abs(t - E0) >= 0.1 * E0 for t in T]
    cond = np.zeros(ws.shape, bool) if cond is None else (cond & ok)          # condition known in advance (e.g. windy hours)
    S = dict(E0=E0, n=2 * ok.sum(0), T=sum(np.nansum(t, 0) for t in T), head=sum(np.nansum(x, 0) for x in head),
             cross=2 * np.nansum(cross, 0), strong=sum(s.sum(0) for s in strong),
             large=sum((np.abs(t - E0) >= 0.2 * E0).sum(0) for t in T),
             T_rt_sq=np.nansum(((T[0] + T[1]) / 2 - E0) ** 2, 0), n_cond=2 * cond.sum(0), est={})
    for name, (u, d) in ests.items():
        Es = [np.where(ok, E0, np.nan)] * 2 if name == "no wind" else \
            [np.where(ok, energy(u * fh, d, road[None, :] + k, model, v), np.nan) for k in (0.0, 180.0)]
        s = dict(E=0, abs=0, sq=0, abs_strong=0, abs_cond=0, sq_cond=0)
        for t, e, st in zip(T, Es, strong):
            s["abs_cond"] = s["abs_cond"] + np.nansum(np.where(cond, np.abs(e - t), np.nan), 0)
            s["sq_cond"] = s["sq_cond"] + np.nansum(np.where(cond, (e - t) ** 2, np.nan), 0)
            s["E"] = s["E"] + np.nansum(e, 0); s["abs"] = s["abs"] + np.nansum(np.abs(e - t), 0)
            s["sq"] = s["sq"] + np.nansum((e - t) ** 2, 0)
            s["abs_strong"] = s["abs_strong"] + np.nansum(np.where(st, np.abs(e - t), np.nan), 0)
        rt_t, rt_e = (T[0] + T[1]) / 2, (Es[0] + Es[1]) / 2
        s["abs_rt"] = np.nansum(np.abs(rt_e - rt_t), 0); s["sq_rt"] = np.nansum((rt_e - rt_t) ** 2, 0)
        S["est"][name] = s
    if keep is not None:
        for key in ("n", "T", "head", "cross", "strong", "large", "T_rt_sq", "n_cond"):
            S[key] = S[key][keep]
        S["est"] = {k: {kk: vv[keep] for kk, vv in s.items()} for k, s in S["est"].items()}
    return S


def energy_metrics(S, idx=None):
    """metrics from per-station sums; idx = resampled station indices (bootstrap)"""
    idx = np.arange(len(S["n"])) if idx is None else idx
    n, T, E0 = S["n"][idx], S["T"][idx], S["E0"]
    pen_t = 100 * (T / n / E0 - 1)
    sq0 = S["est"]["no wind"]["sq"][idx].sum(); sq0rt = S["est"]["no wind"]["sq_rt"][idx].sum()
    sq0c = S["est"]["no wind"]["sq_cond"][idx].sum(); nc = S["n_cond"][idx].sum()
    sq0_st, abs0_st = S["est"]["no wind"]["sq"][idx], S["est"]["no wind"]["abs"][idx]
    o = dict(pen_mean=100 * (T.sum() / n.sum() / E0 - 1), pen_head=100 * (S["head"][idx].sum() / n.sum() / E0 - 1),
             pen_cross=100 * (S["cross"][idx].sum() / n.sum() / E0 - 1), pen_station=pen_t,
             strong_frac=float(S["strong"][idx].sum() / n.sum()), large_frac=float(S["large"][idx].sum() / n.sum()))
    for k, s in S["est"].items():
        pen_e = 100 * (s["E"][idx] / n / E0 - 1)
        o[k] = dict(mae=float(s["abs"][idx].sum() / n.sum()), mae_rt=float(s["abs_rt"][idx].sum() / (n.sum() / 2)),
                    skill=float(1 - s["sq"][idx].sum() / sq0), skill_rt=float(1 - s["sq_rt"][idx].sum() / sq0rt),
                    bias_med=float(np.median(100 * (s["E"][idx] / T - 1))),
                    absbias_med=float(np.median(np.abs(100 * (s["E"][idx] / T - 1)))),
                    pooled_bias=float(100 * (s["E"][idx].sum() / T.sum() - 1)),
                    corr=float(np.corrcoef(pen_t, pen_e)[0, 1]) if k != "no wind" else np.nan,
                    pen_mae=float(np.mean(np.abs(pen_e - pen_t))), pen_station=pen_e,
                    strong_mae=float(s["abs_strong"][idx].sum() / S["strong"][idx].sum()),
                    cond_mae=float(s["abs_cond"][idx].sum() / nc) if nc else np.nan,
                    cond_skill=float(1 - s["sq_cond"][idx].sum() / sq0c) if nc else np.nan,
                    st_skill_med=float(np.median(1 - s["sq"][idx] / sq0_st)),
                    n_skill_pos=int(np.sum(s["sq"][idx] < sq0_st)), n_mae_gain=int(np.sum(s["abs"][idx] < abs0_st)),
                    st_mae_gain_med=float(np.median(100 * (1 - s["abs"][idx] / abs0_st))))
    return o


def bootstrap(S, n_boot=2000, seed=1, pairs=()):
    """station-cluster bootstrap: point values, 95 % percentile intervals, and intervals of paired differences
    pairs = [(a, b, metric), ...] -> metric(a) - metric(b)"""
    rng = np.random.default_rng(seed)
    point = energy_metrics(S); n = len(S["n"])
    reps = [energy_metrics(S, rng.integers(0, n, n)) for _ in range(n_boot)]
    ci = {}
    for k, v in point.items():
        if isinstance(v, dict):
            ci[k] = {kk: np.nanpercentile([r[k][kk] for r in reps], [2.5, 97.5]) for kk in v if np.ndim(v[kk]) == 0}
        elif np.ndim(v) == 0:
            ci[k] = np.nanpercentile([r[k] for r in reps], [2.5, 97.5])
    diffs = {}
    for a, b, kk in pairs:
        if a in point and b in point:
            diffs[(a, b, kk)] = (point[a][kk] - point[b][kk], np.nanpercentile([r[a][kk] - r[b][kk] for r in reps], [2.5, 97.5]))
    return dict(point=point, ci=ci, diff=diffs)


def speed_metrics(ests, ws, n_boot=2000, seed=2):
    """per-station MAE and ratio-of-means bias; median over stations with a station bootstrap interval"""
    rng = np.random.default_rng(seed)
    rows, per = [], {}
    for k, U in ests.items():
        m = np.isfinite(U) & np.isfinite(ws)
        with np.errstate(invalid="ignore", divide="ignore"):
            mae = np.nansum(np.where(m, np.abs(U - ws), np.nan), 0) / m.sum(0)
            rb = np.nansum(np.where(m, U, np.nan), 0) / np.nansum(np.where(m, ws, np.nan), 0) - 1
        ns = len(mae)
        boot = [np.nanmedian(mae[rng.integers(0, ns, ns)]) for _ in range(n_boot)]
        rows.append(dict(est=k, mae_med=float(np.nanmedian(mae)), mae_lo=np.percentile(boot, 2.5), mae_hi=np.percentile(boot, 97.5),
                         rbias_med=float(100 * np.nanmedian(rb)), n_hours=int(m.sum())))
        per[k] = mae
    return pd.DataFrame(rows), per


def direction_mae(ests, ws, wd, U0n, ok_st):
    """median over stations (adequate coverage) of the direction MAE in hours with observed and nearest-station
    speed >= 2 m/s (model-independent hour selection)"""
    md = (ws >= 2) & (U0n >= 2) & np.isfinite(wd)
    out = {}
    for k, d in ests.items():
        m = md & np.isfinite(d)
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.nansum(np.where(m, np.abs(wrap180(d - wd)), np.nan), 0) / m.sum(0)
        out[k] = float(np.nanmedian(e[ok_st]))
    return out


def sector_features(ids, data_root, wd):
    """exposure predictors of Eq. (exposure) at every roadside station, in the 22.5-deg sector of the wind direction wd
    [hour x station]: T100 (upwind), ln(x/h) (upwind), F1000 (all directions)"""
    from pathlib import Path
    V = pd.read_parquet(Path(data_root) / "features_variants.parquet"); V["id"] = V.id.astype(str); V = V[V.src == "trv"]
    Ed = pd.read_parquet(Path(data_root) / "features2_16dir.parquet"); Ed["id"] = Ed.id.astype(str); Ed = Ed[Ed.src == "trv"]
    piv = lambda T, c: T.pivot(index="id", columns="dir", values=c).reindex(index=ids).values      # [station x 16]
    k = (np.round(np.where(np.isfinite(wd), wd, 0) / 22.5).astype(int)) % 16
    j = np.broadcast_to(np.arange(len(ids)), wd.shape)
    T100 = piv(V, "tree_R100_H22.5")[j, k]; F1000 = piv(V, "tree_R1000_H180")[j, k]
    lnxh = np.log(np.clip(piv(Ed, "xh_edge"), 0.2, 200.0))[j, k]
    bad = ~np.isfinite(wd)
    return {n: np.where(bad, np.nan, a) for n, a in (("T100", T100), ("lnxh", lnxh), ("F1000", F1000))}


def crossfit_exposure(base_u, base_d, ws, feats, blk, fit_min=2.0):
    """hybrid estimator: a gridded analysis (base_u, base_d) mapped to the roadside with the exposure model of the paper,
    ln(U_j / U_base) = c0 + c1 T100 + c2 ln(x/h) + c3 ln U_regional + c4 F1000, fitted by OLS and back-transformed with
    the smearing factor, both learned on the other time blocks AND the other stations (cross-fitted)"""
    nb, ns = blk.max() + 1, base_u.shape[1]
    lnreg = np.log(np.clip(np.nanmean(base_u, 1), 0.5, None))[:, None] * np.ones_like(base_u)
    X = np.stack([np.ones_like(base_u), feats["T100"], feats["lnxh"], lnreg, feats["F1000"]], -1)
    y = np.log(np.clip(ws, 0.1, None) / np.clip(base_u, 0.1, None))
    fit = (base_u >= fit_min) & (ws > 0.2) & np.all(np.isfinite(X), -1) & np.isfinite(y)
    p = X.shape[-1]
    XtX = np.zeros((nb, ns, p, p)); Xty = np.zeros((nb, ns, p))
    for b in range(nb):
        for j in range(ns):
            m = fit[:, j] & (blk == b)
            Xm = X[m, j]; XtX[b, j] = Xm.T @ Xm; Xty[b, j] = Xm.T @ y[m, j]
    out = np.full(base_u.shape, np.nan); coefs = []
    for b in range(nb):
        for j in range(ns):
            A_ = XtX.sum((0, 1)) - XtX[b].sum(0) - XtX[:, j].sum(0) + XtX[b, j]
            c_ = Xty.sum((0, 1)) - Xty[b].sum(0) - Xty[:, j].sum(0) + Xty[b, j]
            beta = np.linalg.solve(A_, c_); coefs.append(beta)
            tr = fit & (blk[:, None] != b); tr[:, j] = False
            smear = float(np.mean(np.exp(y[tr] - X[tr] @ beta)))
            m = (blk == b) & np.isfinite(base_u[:, j]) & np.all(np.isfinite(X[:, j]), -1)
            out[m, j] = base_u[m, j] * np.exp(X[m, j] @ beta) * smear
    return out, base_d, np.array(coefs)


def build_estimators(P, O, blk, D=None, metnordic=None, feats=None):
    """all estimators of the roadside wind compared in the paper, every one out of time block and out of station.
    P: cross-validated predictions (U, wdm, U0n, wd0n); O: observations (ws, wd); D: roadside distance matrix [km];
    metnordic: (speed, dir) arrays of the same shape or None. Returns {name: (speed, dir)} and the QC'd direction."""
    ws, wd = O["ws"].astype(float), O["wd"].astype(float)
    wq, dir_ok, cov, stuck = direction_coverage(ws, wd)
    U, wdm, U0n, wd0n = (P[k].astype(float) for k in ("U", "wdm", "U0n", "wd0n"))
    ests = {"nearest station": (U0n, wd0n),
            "rescaled nearest": (U0n * crossfit_ratio(ws, U0n, blk), wd0n),
            "corrected": (U, wdm),
            "calibrated": (crossfit_calibration(U, ws, blk), wdm)}
    info = dict(coverage=cov, stuck_frac=stuck, dir_ok=dir_ok, rescale=float(np.nanmedian(crossfit_ratio(ws, U0n, blk))))
    if D is not None:
        Ur, Wr, dr = nearest_other(ws, wq, D)
        ests["nearest road station"] = (Ur, Wr)
        info["road_km_med"] = float(np.nanmedian(dr))
    if metnordic is not None:
        ests["MET Nordic"] = metnordic
        ests["MET Nordic, rescaled"] = (metnordic[0] * crossfit_ratio(ws, metnordic[0], blk), metnordic[1])
        if feats is not None:
            u_, d_, cf = crossfit_exposure(metnordic[0], metnordic[1], ws, feats, blk)
            ests["MET Nordic + exposure"] = (u_, d_); info["hybrid_coef"] = cf
    return ests, wq, info


def load_country(country, out_no, out_se, data_no=None):
    """cross-validated predictions and observations of one country (from 11_norway.py / 14_final_cv.py),
    with the time blocks of the CV, the roadside distance matrix and, for Norway, MET Nordic at the stations"""
    import pickle
    from pathlib import Path
    if country == "Norway":
        R = pickle.load(open(Path(out_no) / "norway_results.pkl", "rb")); P, O, nb, z = R["pred_final"], R["obs"], 8, 10.0
    else:
        R = pickle.load(open(Path(out_se) / "cv_full.pkl", "rb")); P, O, nb, z = R["pred"], R["obs"], 4, 6.0
    blk = blocks(O["ws"].shape[0], nb)
    D = distance_matrix(O["lat"], O["lon"])
    mn = None
    if country == "Norway" and data_no is not None and (Path(data_no) / "metnordic_roadside.parquet").exists():
        M = pd.read_parquet(Path(data_no) / "metnordic_roadside.parquet"); M["t"] = pd.to_datetime(M.t, utc=True)
        piv = lambda c: M.pivot_table(index="t", columns="id", values=c).reindex(index=O["hours"], columns=O["ids"]).values
        mn = (piv("ws"), piv("wd"))
    feats = sector_features(O["ids"], data_no, mn[1]) if mn is not None else None
    return P, O, blk, D, mn, z, feats
