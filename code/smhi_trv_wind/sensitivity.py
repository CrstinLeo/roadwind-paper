"""Fast re-evaluation of the whole chain (steps 2-6) for many parameter settings.

Honest protocol
  * time split: period A (tuning) and period B (holdout), both inside the TrV test window;
    SMHI site factors always come from the history before A.
  * exposure model fitted on A only, leave-one-TrV-station-out; scored on A (tuning score) and B (holdout score).
  * a parameter value is chosen on A with the 1-SE rule; B is only used to confirm.
Everything is vectorised as [hour x pair] matrices; one evaluation takes a few seconds.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Settings
from .data import sources
from .geo import haversine_km, sector, wrap180, wrap90
from .model import physics
from .model.physics import network_direction


@dataclass
class Spec:
    # physics (step 2)
    z_blend: float = 60.0
    c_tree: float = 0.1                 # z0 = c * h on objects
    L_smhi: int = 1000
    H_smhi: float = 22.5
    L_trv: int = 100
    H_trv: float = 22.5
    along_z0: int | None = None         # average TrV ln z0 over a road window of +-S m
    # site factors (step 3)
    site: str = "sector"                # sector | mean | none
    site_nsec: int = 16
    site_min_net: float = 3.0
    # exposure model (step 4-5)
    tree: str = "tree_R200_H22.5"       # any tree column of features_variants, or "along:S"
    use_lnxh: bool = True
    extra: tuple = ()                   # cross, cross_tree, stab, stab_tree, night, sx, tpi, hmid, lnU_pair
    fit_min: float = 2.0
    interaction: bool = True
    # combination (step 6)
    km: float = 60.0
    min_n: int = 2
    power: float = 1.0
    weights: str = "dist"               # dist | skill
    dir_km: float | None = None
    dir_power: float | None = None      # weight exponent for the direction combination (default: power)
    dir_tree: str | None = None         # tree feature of the direction model (default: tree)
    roughness: bool = True              # False: skip step 2 (z0 = 0.03 everywhere)
    learner: str = "ols"                # ols | gbm (upper-bound check, station-grouped folds)


class Engine:
    def __init__(self, cfg: Settings, variants: pd.DataFrame, t_split: str, t_end: str):
        self.cfg = cfg
        H = pd.date_range(pd.Timestamp(cfg.test_start, tz="UTC"), pd.Timestamp(t_end, tz="UTC"), freq="1h")
        self.H, self.A = H, (H < pd.Timestamp(t_split, tz="UTC"))
        smhi_data, trv_data = sources(cfg)
        SM, TM = smhi_data.stations(cfg), trv_data.stations(cfg)
        tws, twd = trv_data.wind_matrices(cfg, H, ids=TM.index)
        keep = tws.columns[tws.notna().sum() >= cfg.min_trv_hours]
        dT = trv_data.temperature_difference(cfg, H, ids=keep)
        self.TM, self.SM = TM.loc[keep], SM
        self.tws, self.twd = tws[keep].values, twd[keep].values
        self.dT = dT.reindex(index=H, columns=keep).values
        sws, swd = smhi_data.wind_matrices(cfg, H, ids=SM.index)
        self.sws, self.swd = sws, swd
        self.k16 = sector(network_direction(sws, swd), 16)
        self.wsL, self.wdL = smhi_data.history(cfg)
        V = variants.copy(); V["id"] = V.id.astype(str)
        self.Vt = {c: V[V.src == "trv"].pivot(index="id", columns="dir", values=c) for c in V.columns if c not in ("src", "id", "dir")}
        self.Vs = {c: V[V.src == "smhi"].pivot(index="id", columns="dir", values=c) for c in V.columns if c not in ("src", "id", "dir")}
        if cfg.paths.feat_tree.exists():
            F1 = pd.read_parquet(cfg.paths.feat_tree); F1["id"] = F1.id.astype(str); F1 = F1[F1.src == "trv"]
            self.terrain = {c: F1.pivot(index="id", columns="dir", values=c) for c in ("sx", "tpi_1km", "h_mid")}
        else:                                   # terrain indices only exist for Sweden
            empty = pd.DataFrame(np.nan, index=self.TM.index, columns=np.arange(0, 360, 22.5))
            self.terrain = {c: empty for c in ("sx", "tpi_1km", "h_mid")}
        ft = cfg.paths.root / "features_terrain.parquet"     # Sx, TPI, relief from the terrain model (Norway)
        self.Tt = {}
        if ft.exists():
            TT = pd.read_parquet(ft); TT["id"] = TT.id.astype(str); TT = TT[TT.src == "trv"]
            self.Tt = {c: TT.pivot(index="id", columns="dir", values=c) for c in TT.columns if c not in ("src", "id", "dir")}
        E = pd.read_parquet(cfg.paths.feat_edge); E["id"] = E.id.astype(str)
        self.lnxh = np.log(E[E.src == "trv"].pivot(index="id", columns="dir", values="xh_edge").clip(*cfg.xh_clip))
        self.road = trv_data.road_bearing(cfg).reindex(self.TM.index).values
        # pairs
        jj, ss, dd = [], [], []
        for a, j in enumerate(self.TM.index):
            for b, s in enumerate(SM.index):
                d = float(haversine_km(self.TM.loc[j, "lat"], self.TM.loc[j, "lon"], SM.loc[s, "lat"], SM.loc[s, "lon"]))
                if d <= cfg.max_pair_km:
                    jj.append(a); ss.append(b); dd.append(d)
        self.pj, self.ps, self.pd = np.array(jj), np.array(ss), np.array(dd)
        kidx = (self.k16 / 22.5).round().astype(int) % 16
        self.kidx = kidx.values if hasattr(kidx, "values") else kidx
        hour = H.hour.values
        self.night = ((hour >= 15) | (hour < 7)).astype(float)     # November, Sweden (UTC): dark ~15-07
        self._site_cache = {}

    # ---------------------------------------------------------------- helpers
    def _trv_sector(self, table: pd.DataFrame) -> np.ndarray:
        """[hour x trv] value of a [trv x dir] table at the network sector"""
        T = table.reindex(index=self.TM.index, columns=np.arange(0, 360, 22.5)).values
        return T[:, self.kidx].T

    def _smhi_lnz0(self, sp: Spec, ids) -> pd.DataFrame:
        lz = self.Vs[f"lnz0_L{sp.L_smhi}_H{sp.H_smhi:g}"] + self.Vs[f"fwt_L{sp.L_smhi}_H{sp.H_smhi:g}"] * np.log(sp.c_tree / 0.1)
        return lz.reindex(ids)

    def _trv_lnz0(self, sp: Spec) -> pd.DataFrame:
        if sp.along_z0 is not None:
            cols = [f"along_lnz0_s{s}" for s in (-1000, -750, -500, -250, -100, 0, 100, 250, 500, 750, 1000) if abs(s) <= sp.along_z0]
            return sum(self.Vt[c] for c in cols) / len(cols)
        return self.Vt[f"lnz0_L{sp.L_trv}_H{sp.H_trv:g}"] + self.Vt[f"fwt_L{sp.L_trv}_H{sp.H_trv:g}"] * np.log(sp.c_tree / 0.1)

    def _tree(self, sp: Spec) -> pd.DataFrame:
        if sp.tree.startswith("along:"):
            S = int(sp.tree.split(":")[1])
            cols = [f"along_tree_s{s}" for s in (-1000, -750, -500, -250, -100, 0, 100, 250, 500, 750, 1000) if abs(s) <= S]
            return sum(self.Vt[c] for c in cols) / len(cols)
        return self.Vt[sp.tree]

    def site_factors(self, sp: Spec) -> pd.DataFrame:
        key = (sp.L_smhi, sp.H_smhi, sp.c_tree, sp.z_blend, sp.site, sp.site_nsec, sp.site_min_net)
        if key in self._site_cache:
            return self._site_cache[key]
        cfg = dataclasses.replace(self.cfg, z_blend=sp.z_blend)
        lz = self._smhi_lnz0(sp, self.wsL.columns)
        kL = sector(network_direction(self.wsL, self.wdL), 16)
        P = pd.DataFrame({s: physics.to_potential(np.exp(lz.loc[s].reindex(kL).values), cfg) for s in self.wsL.columns}, index=self.wsL.index)
        pot = self.wsL * P
        ksite = sector(network_direction(self.wsL, self.wdL), sp.site_nsec)
        use = (self.wsL.median(1) >= sp.site_min_net) & (self.wsL.notna().sum(1) >= 12)
        L = np.log(pot[use].where(self.wsL[use] >= 0.5)); K = pd.Series(ksite, index=self.wsL.index)[use]
        if sp.site == "none":
            c = pd.DataFrame(0.0, index=self.wsL.columns, columns=np.unique(K))
        else:
            c = pd.DataFrame(0.0, index=self.wsL.columns, columns=np.unique(K))
            for _ in range(15):
                C = pd.DataFrame({s: c.loc[s].reindex(K.values).values for s in L.columns}, index=L.index)
                a = (L - C).mean(1)
                R = L.sub(a, axis=0)
                if sp.site == "mean":
                    m = R.mean()
                    c = pd.DataFrame({k: m for k in c.columns})
                else:
                    c = R.groupby(K.values).mean().T.reindex(columns=c.columns)
                c = c - c.mean(0)
        self._site_cache[key] = c
        return c

    # ---------------------------------------------------------------- evaluation
    def run(self, sp: Spec, keep_rows: bool = False, fixed: dict | None = None, groups: dict | None = None,
            keep_pred: bool = False) -> dict:
        """fixed = dict(b=..., d=..., smear=...) applies given coefficients to every station (transfer test, no fitting)"""
        cfg = dataclasses.replace(self.cfg, z_blend=sp.z_blend)
        c = self.site_factors(sp)
        lzs = self._smhi_lnz0(sp, self.sws.columns)
        P = np.column_stack([physics.to_potential(np.exp(lzs.loc[s].values[self.kidx]), cfg) for s in self.sws.columns])
        if not sp.roughness:
            P = np.ones_like(P)
        ksite = sector(self.k16 if sp.site_nsec == 16 else network_direction(self.sws, self.swd), sp.site_nsec)
        C = np.column_stack([c.loc[s].reindex(ksite).values for s in self.sws.columns])
        corr = self.sws.values * P * np.exp(-np.nan_to_num(C))
        Fj = physics.to_site(np.exp(self._trv_sector(self._trv_lnz0(sp))), cfg)            # [hour x trv]
        if not sp.roughness:
            Fj = np.full_like(Fj, physics.height_factor(6.0))
        tree = self._trv_sector(self._tree(sp))
        lnxh = self._trv_sector(self.lnxh)
        pj, ps = self.pj, self.ps
        U0 = self.sws.values[:, ps]; wd_s = self.swd.values[:, ps]
        ws = self.tws[:, pj]; wd = self.twd[:, pj]
        U3 = corr[:, ps] * Fj[:, pj]
        lnU_net = np.log(np.clip(np.nanmean(corr, 1), 0.5, None))[:, None] * np.ones_like(U3)
        tr = tree[:, pj]
        alpha = wrap90(wd_s - self.road[pj][None, :])
        cols = [np.ones_like(U3), tr]
        if sp.use_lnxh:
            cols.append(lnxh[:, pj])
        lnU = np.log(np.clip(U3, 0.5, None)) if "lnU_pair" in sp.extra else lnU_net
        cols.append(lnU)
        if sp.interaction:
            cols.append(lnU * tr)
        cr = lambda: np.abs(np.sin(np.radians(alpha)))
        dT = lambda: np.nan_to_num(self.dT[:, pj])
        ext = dict(cross=lambda: cr(), cross_tree=lambda: cr() * tr, stab=dT, stab_tree=lambda: dT() * tr,
                   night=lambda: self.night[:, None] * np.ones_like(U3), night_tree=lambda: self.night[:, None] * tr,
                   sx=lambda: self._trv_sector(self.terrain["sx"])[:, pj], tpi=lambda: self._trv_sector(self.terrain["tpi_1km"])[:, pj],
                   hmid=lambda: self._trv_sector(self.terrain["h_mid"])[:, pj])            # built only when requested
        for e in sp.extra:
            if e.startswith("v:"):
                cols.append(np.nan_to_num(self._trv_sector(self.Vt[e[2:]])[:, pj]))
            elif e.startswith("t:"):                                   # terrain feature, e.g. t:sx_300, t:tpi_500
                cols.append(np.nan_to_num(self._trv_sector(self.Tt[e[2:]])[:, pj]))
            elif e != "lnU_pair":
                cols.append(np.nan_to_num(ext[e]()))
        X = np.stack([np.nan_to_num(x).astype(np.float32) for x in cols], -1)   # [hour x pair x p], float32 to save memory
        del cols
        y = np.log(np.clip(ws, 0.1, None) / np.clip(U3, 0.1, None))
        A = self.A[:, None]
        fit = A & (U0 >= sp.fit_min) & (ws > 0.2) & np.isfinite(U3) & np.isfinite(tr) & np.isfinite(lnxh[:, pj]) & np.isfinite(ws)
        trd = tr if sp.dir_tree is None else self._trv_sector(self.Vt[sp.dir_tree])[:, pj]
        s2t = np.sin(np.radians(2 * alpha)) * trd
        dd = wrap180(wd - wd_s)
        fitd = A & (U0 >= 3) & (ws >= 1.5) & np.isfinite(dd) & np.isfinite(s2t)
        # LOSO by TrV station through per-station normal equations
        n_t, p = len(self.TM), X.shape[-1]
        Xf, yf, jf = X[fit], y[fit], np.broadcast_to(pj, y.shape)[fit]
        XtX = np.zeros((n_t, p, p)); Xty = np.zeros((n_t, p))
        np.add.at(XtX, jf, Xf[:, :, None] * Xf[:, None, :]); np.add.at(Xty, jf, Xf * yf[:, None])
        Xd = np.stack([np.ones_like(s2t), np.nan_to_num(s2t)], -1)
        Xdf, ddf, jdf = Xd[fitd], dd[fitd], np.broadcast_to(pj, dd.shape)[fitd]
        DtD = np.zeros((n_t, 2, 2)); Dty = np.zeros((n_t, 2))
        np.add.at(DtD, jdf, Xdf[:, :, None] * Xdf[:, None, :]); np.add.at(Dty, jdf, Xdf * ddf[:, None])
        SX, Sy, SD, Sd = XtX.sum(0), Xty.sum(0), DtD.sum(0), Dty.sum(0)
        if fixed is None:
            B = np.stack([np.linalg.solve(SX - XtX[j] + 1e-9 * np.eye(p), Sy - Xty[j]) for j in range(n_t)])
            Bd = np.stack([np.linalg.solve(SD - DtD[j] + 1e-9 * np.eye(2), Sd - Dty[j]) for j in range(n_t)])
            b_all = np.linalg.solve(SX, Sy); d_all = np.linalg.solve(SD, Sd)
            smear = float(np.mean(np.exp(yf - Xf @ b_all)))
        else:
            b_all, d_all = np.asarray(fixed["b"], dtype=float), np.asarray(fixed["d"], dtype=float)
        if fixed is not None:
            B = np.tile(np.asarray(fixed["b"], dtype=float), (n_t, 1)); Bd = np.tile(np.asarray(fixed["d"], dtype=float), (n_t, 1))
            smear = float(fixed["smear"])
        yhat = np.einsum("hpk,pk->hp", X, B[pj].astype(np.float32))
        if sp.learner == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor
            grp = np.arange(n_t) % 10
            yhat = np.full(y.shape, np.nan)
            Xall = X.reshape(-1, p); jall = np.broadcast_to(pj, y.shape).reshape(-1)
            fflat = fit.reshape(-1)
            for g in range(10):
                tr_rows = fflat & (grp[jall] != g)
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200)
                m.fit(Xall[tr_rows], y.reshape(-1)[tr_rows])
                te = grp[jall] == g
                yh = yhat.reshape(-1); yh[te] = m.predict(Xall[te])
            smear = float(np.mean(np.exp(yf - yhat[fit])))
        U4 = U3 * np.exp(yhat) * smear
        wd4 = (wd_s + np.einsum("hpk,pk->hp", Xd, Bd[pj])) % 360
        # combination
        W = np.zeros((len(pj), n_t)); Wd = np.zeros((len(pj), n_t))
        if sp.weights == "skill":
            e = (y - yhat - np.log(smear)); e[~(fit)] = np.nan
            mse = np.array([np.nanmean(e[:, i] ** 2) for i in range(len(pj))])
        for j in range(n_t):
            idx = np.where(pj == j)[0]
            order = idx[np.argsort(self.pd[idx])]
            sel = [i for i in order if self.pd[i] <= sp.km] or []
            if len(sel) < sp.min_n:
                sel = list(order[:sp.min_n])
            w = 1 / np.clip(self.pd[sel], 3, None) ** sp.power
            if sp.weights == "skill":
                w = w * 0 + 1 / np.clip(mse[sel], 0.02, None)
            W[sel, j] = w
            dk = sp.dir_km if sp.dir_km is not None else sp.km
            seld = [i for i in order if self.pd[i] <= dk]
            if len(seld) < sp.min_n:
                seld = list(order[:sp.min_n])
            Wd[seld, j] = 1 / np.clip(self.pd[seld], 3, None) ** (sp.power if sp.dir_power is None else sp.dir_power)
        ok = np.isfinite(U4) & (U4 > 0)
        lu = np.where(ok, np.log(np.where(ok, U4, 1)), 0)
        U = np.exp((lu @ W) / np.where(ok @ W > 0, ok @ W, np.nan))
        okd = np.isfinite(wd4) & ok
        su = np.where(okd, np.sin(np.radians(wd4)), 0) @ Wd; cv = np.where(okd, np.cos(np.radians(wd4)), 0) @ Wd
        wdm = np.degrees(np.arctan2(su, cv)) % 360
        # nearest raw baseline
        near = np.array([np.where(pj == j)[0][np.argmin(self.pd[pj == j])] for j in range(n_t)])
        U0n = U0[:, near] * physics.height_factor(self.cfg.z_trv, self.cfg.z_smhi, self.cfg.z0_open); wd0n = wd_s[:, near]
        out = dict(spec=sp, b=b_all, d=d_all, smear=smear, n_fit=int(fit.sum()))
        periods = [("A", self.A), ("B", ~self.A)] + [(f"B:{g}", ~self.A & m) for g, m in (groups or {}).items()]
        for name, per in periods:
            m = per[:, None] & np.isfinite(self.tws) & np.isfinite(U)
            err = np.where(m, np.abs(U - self.tws), np.nan)
            mae = np.nanmean(err, 0)
            mraw = np.nanmean(np.where(per[:, None] & np.isfinite(self.tws) & np.isfinite(U0n), np.abs(U0n - self.tws), np.nan), 0)
            ml = per[:, None] & (self.tws >= 1) & np.isfinite(U)
            le = np.where(ml, np.log(self.tws / U), np.nan)
            md = per[:, None] & (self.tws >= 2) & (U0n >= 2) & np.isfinite(self.twd) & np.isfinite(wdm)   # model-independent hours
            de = np.where(md, np.abs(wrap180(wdm - self.twd)), np.nan)
            de0 = np.where(md & np.isfinite(wd0n), np.abs(wrap180(wd0n - self.twd)), np.nan)
            out[name] = dict(mae=mae, mae_raw=mraw, bias_log=np.nanmean(le, 0), rmse_log=np.sqrt(np.nanmean(le ** 2, 0)),
                             dir=np.nanmean(de, 0), sum_abs=np.nansum(err, 0), n=np.sum(np.isfinite(err), 0),
                             sum_raw=np.nansum(np.where(per[:, None] & np.isfinite(self.tws) & np.isfinite(U0n) & np.isfinite(U), np.abs(U0n - self.tws), np.nan), 0),
                             sum_dir=np.nansum(de, 0), n_dir=np.sum(np.isfinite(de), 0),
                             sum_dir_raw=np.nansum(de0, 0), n_dir_raw=np.sum(np.isfinite(de0), 0),
                             sum_le=np.nansum(le, 0), sum_le2=np.nansum(le ** 2, 0), n_le=np.sum(np.isfinite(le), 0))
        if keep_rows:
            out.update(U=U, wdm=wdm, U4=U4, U3=U3, X=X, y=y, fit=fit)
        if keep_pred:                                   # station-level estimates only (small)
            out.update(U=U.astype(np.float32), wdm=wdm.astype(np.float32), U0n=U0n.astype(np.float32), wd0n=wd0n.astype(np.float32))
        return out


def cv(engine: "Engine", sp: Spec, n_blocks: int = 4, groups: dict | None = None, fixed: dict | None = None,
       keep_pred: bool = False) -> dict:
    """leave-one-time-block-out x leave-one-TrV-station-out: every prediction is out of time and out of station.
    Returns per-station MAE etc. pooled over all held-out hours; groups = {name: hour mask} adds the same
    metrics restricted to those hours under result[name]."""
    H = engine.H
    blocks = np.array_split(np.arange(len(H)), n_blocks)
    A0 = engine.A
    keys = ("sum_abs", "n", "sum_raw", "sum_dir", "n_dir", "sum_dir_raw", "n_dir_raw", "sum_le", "sum_le2", "n_le")
    names = ["B"] + [f"B:{g}" for g in (groups or {})]
    acc = {nm: None for nm in names}
    pred = None
    try:
        for b in blocks:
            fitmask = np.ones(len(H), bool); fitmask[b] = False
            engine.A = fitmask
            res = engine.run(sp, groups=groups, fixed=fixed, keep_pred=keep_pred)
            if keep_pred:
                if pred is None:
                    pred = {k: np.full_like(res[k], np.nan) for k in ("U", "wdm", "U0n", "wd0n")}
                for k in pred:
                    pred[k][b] = res[k][b]
            for nm in names:
                r = res[nm]
                acc[nm] = {k: r[k].astype(float) for k in keys} if acc[nm] is None else {k: acc[nm][k] + r[k] for k in keys}
    finally:
        engine.A = A0

    def pack(a):
        with np.errstate(invalid="ignore", divide="ignore"):
            return dict(mae=a["sum_abs"] / a["n"], mae_raw=a["sum_raw"] / a["n"], dir=a["sum_dir"] / a["n_dir"],
                        dir_raw=a["sum_dir_raw"] / a["n_dir_raw"], n=a["n"],
                        bias_log=a["sum_le"] / a["n_le"], rmse_log=np.sqrt(a["sum_le2"] / a["n_le"]))
    out = pack(acc["B"])
    for g in (groups or {}):
        out[g] = pack(acc[f"B:{g}"])
    if keep_pred:
        out["pred"] = pred                              # out-of-time, out-of-station estimates [hour x roadside station]
    return out


def summarise(r: dict, base: dict | None = None) -> dict:
    o = {}
    for per in ("A", "B"):
        e = r[per]
        o[f"{per}_mae_mean"] = float(np.nanmean(e["mae"])); o[f"{per}_mae_med"] = float(np.nanmedian(e["mae"]))
        o[f"{per}_dir_med"] = float(np.nanmedian(e["dir"])); o[f"{per}_rmse_log"] = float(np.nanmean(e["rmse_log"]))
        o[f"{per}_bias_log"] = float(np.nanmedian(e["bias_log"]))
        if base is not None:
            d = e["mae"] - base[per]["mae"]
            d = d[np.isfinite(d)]
            o[f"{per}_dmae"] = float(d.mean()); o[f"{per}_dmae_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
    return o


def one_se_range(df: pd.DataFrame, value_col: str, score: str = "A_mae_mean", se: str = "A_dmae_se") -> tuple:
    """values whose tuning score is within 1 SE of the best (paired SE relative to the best setting)"""
    best = df.loc[df[score].idxmin()]
    ok = df[df[score] <= best[score] + max(best.get(se, 0) or 0, 1e-9)]
    return best[value_col], ok[value_col].min(), ok[value_col].max()


def ladder(engine: "Engine", sp: Spec, max_km: float = 80.0) -> pd.DataFrame:
    """correction ladder per pair (fitted and evaluated on all hours, LOSO over roadside stations):
    U0 raw, U1 height only, U2 + roughness, U3 + site factors, U4 + exposure model"""
    A0 = engine.A
    engine.A = np.ones(len(engine.H), bool)
    try:
        r = engine.run(sp, keep_rows=True)
        r2 = engine.run(dataclasses.replace(sp, site="none"), keep_rows=True)
    finally:
        engine.A = A0
    ps, pj = engine.ps, engine.pj
    U0 = engine.sws.values[:, ps]; ws = engine.tws[:, pj]
    f_h = physics.height_factor(engine.cfg.z_trv, engine.cfg.z_smhi, engine.cfg.z0_open)
    levels = dict(U0=U0, U1=U0 * f_h, U2=r2["U3"], U3=r["U3"], U4=r["U4"])
    sel = engine.pd <= max_km
    rows = []
    for i in np.where(sel)[0]:
        o = np.isfinite(ws[:, i]) & np.isfinite(U0[:, i])
        m = o & (U0[:, i] >= 2) & (ws[:, i] > 0.2)
        row = dict(pair=i, dist=engine.pd[i])
        for k, v in levels.items():
            x = v[:, i]
            ok = o & np.isfinite(x)
            row[f"mae_{k}"] = float(np.mean(np.abs(x[ok] - ws[ok, i]))) if ok.sum() > 50 else np.nan
            mm = m & np.isfinite(x) & (x > 0)
            row[f"bias_{k}"] = float(np.exp(np.mean(np.log(ws[mm, i] / x[mm]))) - 1) if mm.sum() > 50 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# Final joint model (Sweden and Norway): two tree scales (100 m upwind, forest fraction within 1 km in all
# directions), one site factor per reference station, speed combined from all stations with 1/d^1.5
RECOMMENDED = Spec(interaction=False, tree="tree_R100_H22.5", extra=("v:tree_R1000_H180",),
                   km=150, min_n=1, power=1.5, site="mean", L_smhi=100,
                   dir_km=60, dir_power=1.0, dir_tree="tree_R200_H22.5")
