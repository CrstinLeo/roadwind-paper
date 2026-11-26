"""Figures revised in review round 1 -> <repo>/figures/

  fig04_sensitivity_norway        scans around the final model (11_norway.py), incl. no near trees and 300 m radii
  fig07_two_shelter_scales_sweden grid of near-tree radius x landscape forest fraction around the final model (26)
  fig10_error_structure           station MAE, reliability diagram, MAE by observed wind for all estimators, monthly MAE
  fig13_energy                    air-drag energy: penalty and its uncertainty, station-site penalties, skill of all
                                  estimators with station-bootstrap intervals, head/cross decomposition
  fig06_vehicle_validation        vehicle height, by reference hour

Inputs: SWW_NO_OUT (norway_results.pkl, baselines.pkl, energy.pkl, vehicle_validation_rows.parquet),
        SWW_SE_OUT (review_SE.pkl, cv_full.pkl)
"""
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smhi_trv_wind import drag

NO = Path(os.environ.get("SWW_NO_OUT", "output/no")); SE = Path(os.environ.get("SWW_SE_OUT", "output/se"))
FIG = Path(os.environ.get("SWW_FIG", Path(__file__).resolve().parents[2] / "figures"))
BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#1d1d1b", "#6b6a65", "#e4e3de"
LIGHT, GOLD = "#b9b8b2", "#c98a12"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                     "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42})
W2 = 7.2
C = {"Norway": ORANGE, "Sweden": BLUE}
EST = [("no wind", "no wind", LIGHT), ("nearest station", "nearest station", MUTED), ("rescaled nearest", "rescaled nearest", GOLD),
       ("nearest road station", "nearest road station", GREEN), ("MET Nordic", "MET Nordic", PURPLE),
       ("MET Nordic, rescaled", "MET Nordic, rescaled", "#a99be8"), ("MET Nordic + exposure", "MET Nordic + exposure", "#3a2a99"),
       ("corrected", "corrected (this study)", ORANGE)]
ECOL = {k: c for k, _, c in EST}; ELAB = {k: l for k, l, _ in EST}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(NO / f"{name}.png", dpi=200); plt.close(fig); print("wrote", name)


def label(ax, s, x=-0.16, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def paired(a, b):
    d = a - b; ok = np.isfinite(d)
    return d[ok].mean(), d[ok].std(ddof=1) / np.sqrt(ok.sum())


# ------------------------------------------------------------------------------------------------ sensitivity, Norway
def fig_sens_norway():
    R = pickle.load(open(NO / "norway_results.pkl", "rb"))
    base = R["cv_final"]["mae"]
    panels = [("tree_R", "near-tree share (radius)", ["none", "50 m", "100 m", "200 m", "300 m", "500 m"], "100 m"),
              ("far", "landscape forest fraction", ["none", "300 m", "500 m", "1 km upwind", "1 km all dir.", "1 km + ring 300 m"], "1 km all dir."),
              ("km", "radius of station combination", ["20 km", "40 km", "80 km", "120 km"], "120 km"),
              ("power", "weight exponent $p$ in $d^{-p}$", ["0.5", "1.0", "1.5", "2.0"], "1.5"),
              ("site", "site factors", ["none", "mean", "sector"], "mean"),
              ("variants", "model variants", ["final", "no roughness conversion", "no ln(x/h)", "+ lnU x tree", "+ cross-road x tree"], "final")]
    short = {"no roughness conversion": "no $z_0$\nconv.", "no ln(x/h)": "no\nln(x/h)", "+ lnU x tree": "+lnU\n×tree",
             "+ cross-road x tree": "+cross\n×tree", "1 km upwind": "1 km\nupw.", "1 km all dir.": "1 km\nall", "1 km + ring 300 m": "1 km\n+300 m",
             "mean": "per\nstation", "sector": "per\nsector", "300 m": "300 m\nall", "500 m": "500 m\nall"}
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.1))
    for a, (g, title, labels, pick), s in zip(axes.ravel(), panels, "abcdef"):
        labels = [l for l in labels if (g, l) in R["scans"]]
        v = [paired(R["scans"][(g, l)]["mae"], base) for l in labels]
        x = np.arange(len(labels))
        a.errorbar(x, [m for m, _ in v], yerr=[s_ for _, s_ in v], fmt="o", color=ORANGE, ms=3.5, capsize=1.5, lw=1)
        sh = {k: v_ for k, v_ in short.items() if not (g == "tree_R" and k in ("300 m", "500 m"))}
        if g == "tree_R":
            sh.update({l: l.replace(" m", "") for l in labels if l != "none"}); a.set_xlabel("radius [m]", fontsize=7)
        if pick in labels:
            i = labels.index(pick); a.plot(i, v[i][0], "s", ms=8, mfc="none", mec=INK, mew=1)
        a.axhline(0, color=MUTED, lw=0.7); a.set_xticks(x); a.set_xticklabels([sh.get(l, l) for l in labels], fontsize=6.8)
        a.set_title(title, loc="left", fontsize=8); label(a, s, x=-0.22, y=1.02)
    for a in axes[:, 0]:
        a.set_ylabel("ΔMAE vs final [m s$^{-1}$]")
    fig.tight_layout(w_pad=0.8, h_pad=1.0); save(fig, "fig04_sensitivity_norway")


# ------------------------------------------------------------------------------------------------ Sweden grid
def fig_two_scale():
    X = pickle.load(open(SE / "review_SE.pkl", "rb"))
    ref = X["final"]["mae"]
    nears = [50, 100, 150, 200, 300]
    fars = [("none", "none"), ("tree_R300_H180", "300 m"), ("tree_R500_H180", "500 m"), ("tree_R1000_H180", "1 km"), ("tree_R1000_H22.5", "1 km\nupwind")]
    M = np.array([[paired(X["grid"][(n, f)]["mae"], ref)[0] for f, _ in fars] for n in nears])
    E = np.array([[paired(X["grid"][(n, f)]["mae"], ref)[1] for f, _ in fars] for n in nears])
    fig, ax = plt.subplots(figsize=(W2 * 0.8, 2.4))
    im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=0.025, aspect="auto")
    for i in range(len(nears)):
        for j in range(len(fars)):
            ax.text(j, i, f"{M[i, j]:+.3f}\n±{E[i, j]:.3f}", ha="center", va="center", fontsize=6.8, color=INK)
    ax.set_yticks(range(len(nears))); ax.set_yticklabels([f"near trees {n} m" for n in nears])
    ax.set_xticks(range(len(fars))); ax.set_xticklabels([l for _, l in fars]); ax.grid(False)
    ax.set_xlabel("landscape forest fraction (all directions unless stated)")
    cb = fig.colorbar(im, ax=ax, pad=0.01, extend="max"); cb.set_label("ΔMAE vs final [m s$^{-1}$]")
    save(fig, "fig07_two_shelter_scales_sweden")


# ------------------------------------------------------------------------------------------------ error structure
def fig_error_structure():
    BL = pickle.load(open(NO / "baselines.pkl", "rb"))
    R = pickle.load(open(NO / "norway_results.pkl", "rb"))
    fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0)); ax = axs.ravel()
    a = ax[0]
    for country in ("Norway", "Sweden"):
        ms = BL[country]["mae_station"]
        a.scatter(ms["nearest station"], ms["corrected"], s=7, color=C[country], alpha=0.75, lw=0, label=country)
    m = 7.0; a.plot([0, m], [0, m], color=INK, lw=0.6); a.set_xlim(0, m); a.set_ylim(0, m)
    a.set_xlabel("nearest station MAE [m s$^{-1}$]"); a.set_ylabel("corrected MAE [m s$^{-1}$]"); a.legend(frameon=False, loc="upper left")
    a = ax[1]                                                       # reliability
    for k, col, ls in (("nearest station", MUTED, "--"), ("corrected", ORANGE, "-"), ("calibrated", INK, ":")):
        t = BL["Norway"]["reliability"][k]; t = t[t.n >= 200]
        a.plot(t.est, t.obs, ls, color=col, marker="o", ms=2.5, label={"calibrated": "recalibrated"}.get(k, k))
    t = BL["Sweden"]["reliability"]["corrected"]; t = t[t.n >= 200]
    a.plot(t.est, t.obs, "-", color=BLUE, marker="o", ms=2.5, label="corrected, Sweden")
    a.plot([0, 18], [0, 18], color=INK, lw=0.5)
    a.set_xlim(0, 18); a.set_ylim(0, 18); a.set_xlabel("estimate (bin mean) [m s$^{-1}$]"); a.set_ylabel("observed (bin mean) [m s$^{-1}$]")
    a.legend(frameon=False, loc="upper left", fontsize=7.5)
    a = ax[2]                                                       # MAE by observed class, Norway, all estimators
    b = BL["Norway"]["by_obs_mn"]                                   # all estimators on the same (MET Nordic) hours
    for k in ("nearest station", "rescaled nearest", "nearest road station", "MET Nordic", "MET Nordic + exposure", "corrected"):
        t = b[(b.est == k) & (b.n >= 100)]
        xc = np.where(np.isinf(t.hi), t.lo + 2.5, 0.5 * (t.lo + t.hi))
        a.plot(xc, t.mae, "-o", ms=2.5, color=ECOL[k], label=ELAB[k], lw=1.6 if k == "corrected" else 1.0)
    a.set_xlabel("observed roadside wind [m s$^{-1}$]"); a.set_ylabel("MAE [m s$^{-1}$]"); a.legend(frameon=False, loc="upper left", fontsize=7.5)
    a = ax[3]; P, O = R["pred_final"], R["obs"]; mon = pd.DatetimeIndex(O["hours"]).month
    U, U0, ws = P["U"].astype(float), P["U0n"].astype(float), O["ws"].astype(float)
    mm = [np.nanmean(np.abs(U[mon == k] - ws[mon == k])) for k in range(1, 13)]
    m0 = [np.nanmean(np.abs(U0[mon == k] - ws[mon == k])) for k in range(1, 13)]
    a.bar(np.arange(1, 13) - 0.2, m0, 0.4, color=LIGHT, label="nearest station"); a.bar(np.arange(1, 13) + 0.2, mm, 0.4, color=ORANGE, label="corrected")
    a.set_xticks(range(1, 13)); a.set_xticklabels(list("JFMAMJJASOND")); a.set_ylabel("MAE [m s$^{-1}$]"); a.set_xlabel("month (Norway)")
    a.legend(frameon=False, loc="upper center", fontsize=6.5); a.grid(axis="x", visible=False)
    for a_, s in zip(ax, "abcd"):
        label(a_, s, x=-0.16)
    fig.tight_layout(w_pad=1.2, h_pad=1.2)
    save(fig, "fig10_error_structure")


# ------------------------------------------------------------------------------------------------ energy
def fig_energy():
    EN = pickle.load(open(NO / "energy.pkl", "rb"))
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.6), gridspec_kw=dict(width_ratios=[1.2, 1.0]))
    ax = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    a = ax[0]                                                       # penalty and its uncertainty
    labs = [("reference", "reference"), ("sensor 6 m", "sensor at 6 m"), ("z0 0.01", "$z_0$ = 0.01 m"), ("z0 0.1", "$z_0$ = 0.1 m"),
            ("z0 0.3", "$z_0$ = 0.3 m"), ("z0 0.3 d 2", "$z_0$ 0.3 m, $d$ 2 m"), ("no reduction", "no height reduction"),
            ("TI 0.15", "turbulence $I$ = 0.15"), ("TI 0.3", "turbulence $I$ = 0.3"), ("model1", "Askerdal Model 1")]
    y = np.arange(len(labs))[::-1]
    for c, dy in (("Norway", 0.15), ("Sweden", -0.15)):
        band = EN[c]["band"]
        vals = [band.get(k, np.nan) for k, _ in labs]
        a.plot(vals, y + dy, "o", color=C[c], ms=3.5, label=c)
        ci = EN[c][("vecto", 80)]["ci"]["pen_mean"]
        a.plot(ci, [y[0] + dy] * 2, "-", color=C[c], lw=2.2, alpha=0.6)
    a.set_yticks(y); a.set_yticklabels([l for _, l in labs], fontsize=8); a.set_xlabel("mean wind penalty [%]")
    a.set_xlim(0, 30); a.grid(axis="y", visible=False)
    a.legend(frameon=False, fontsize=7, loc="center right")
    a = ax[1]                                                       # station-site penalties
    for c in ("Norway", "Sweden"):
        pt = EN[c][("vecto", 80)]["point"]
        a.scatter(pt["pen_station"], pt["nearest station"]["pen_station"], s=8, facecolors="none", edgecolors=LIGHT, lw=0.7)
        a.scatter(pt["pen_station"], pt["corrected"]["pen_station"], s=8, color=C[c], lw=0, alpha=0.85)
    a.scatter([], [], s=9, facecolors="none", edgecolors=LIGHT, label="nearest station")
    a.scatter([], [], s=9, color=ORANGE, label="corrected, Norway"); a.scatter([], [], s=9, color=BLUE, label="corrected, Sweden")
    lim = 50; a.plot([0, lim], [0, lim], color=INK, lw=0.6); a.set_xlim(0, lim); a.set_ylim(0, lim + 15)
    a.set_xlabel("observed penalty at the site [%]"); a.set_ylabel("estimated penalty [%]"); a.legend(frameon=False, loc="upper left", fontsize=6.5)
    a = ax[2]                                                       # skill of all estimators with CI
    names = ["nearest station", "rescaled nearest", "nearest road station", "MET Nordic", "MET Nordic, rescaled", "MET Nordic + exposure", "corrected"]
    for i, k in enumerate(names):
        for c, dy in (("Norway", 0.17), ("Sweden", -0.17)):
            src = EN[c].get("metnordic_hours") if k.startswith("MET Nordic") else EN[c][("vecto", 80)]
            if src is None or k not in src["point"]:
                continue
            v, ci = src["point"][k]["skill"], src["ci"][k]["skill"]
            yy = len(names) - 1 - i + dy
            a.plot([max(ci[0], -1.2), ci[1]], [yy, yy], "-", color=C[c], lw=1.2)
            a.plot(max(v, -1.2), yy, "o" if v > -1.2 else "<", color=C[c], ms=3.5)
    a.axvline(0, color=INK, lw=0.6)
    a.set_yticks(range(len(names))[::-1]); a.set_yticklabels([ELAB[k] for k in names], fontsize=8); a.set_xlim(-1.25, 1.0)
    a.set_xlabel("skill vs ignoring the wind\n$1-\\mathrm{MSE}/\\mathrm{MSE}_\\mathrm{no\\ wind}$"); a.grid(axis="y", visible=False)
    a = ax[3]                                                       # decomposition
    w = 0.36
    for i, c in enumerate(("Norway", "Sweden")):
        pt = EN[c][("vecto", 80)]["point"]
        a.bar(np.arange(3) + (i - 0.5) * w, [pt["pen_mean"], pt["pen_head"], pt["pen_cross"]], w, color=C[c], alpha=0.85 if i == 0 else 0.6, label=c)
    a.set_xticks(range(3)); a.set_xticklabels(["total", "head/tail\nonly", "crosswind\nonly"], fontsize=8); a.set_ylabel("mean penalty [%]")
    a.grid(axis="x", visible=False); a.legend(frameon=False, fontsize=7)
    for a_, s, x in zip(ax, "abcd", (-0.5, -0.2, -0.5, -0.2)):
        label(a_, s, x=x)
    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    save(fig, "fig13_energy")


# ------------------------------------------------------------------------------------------------ vehicle
def fig_vehicle():
    EN = pickle.load(open(NO / "energy.pkl", "rb"))
    V = pd.read_parquet(NO / "vehicle_validation_rows.parquet")
    V["h"] = pd.to_datetime(V.t, utc=True).dt.round("1h")
    est = [("ref_raw", "ref_dir", "nearest station", MUTED), ("road_raw", "road_dir", "nearest road station", GREEN),
           ("metnordic", "metnordic_dir", "MET Nordic", PURPLE), ("model", "model_dir", "corrected (this study)", ORANGE)]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 3.8), gridspec_kw=dict(width_ratios=[1, 1.15, 1.25]))
    a = ax[0]
    g = V.groupby("h")
    for c, _, lab, col in est:
        hh = g.agg(w=("w_veh", "mean"), e=(c, "mean"), site=("site", "first"))
        for site, mk in (("dovre", "o"), ("froya", "^")):
            q = hh[hh.site == site]
            a.scatter(q.w, q.e, s=16, marker=mk, color=col, lw=0, alpha=0.85, label=lab if site == "dovre" else None)
    a.plot([0, 16], [0, 16], color=INK, lw=0.6); a.set_xlim(0, 12); a.set_ylim(0, 16)
    a.set_xlabel("van, hour mean [m s$^{-1}$]"); a.set_ylabel("estimate, hour mean [m s$^{-1}$]")
    a.text(0.97, 0.03, "○ Dovre  △ Frøya", transform=a.transAxes, ha="right", fontsize=6.5, color=MUTED)
    a.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    a = ax[1]
    res = EN["vehicle"]["vecto"]
    names = ["no wind", "nearest station", "nearest road station", "MET Nordic", "corrected"]
    for i, k in enumerate(names):
        r = res[k]; yy = len(names) - 1 - i
        a.plot(r["mae_ci"], [yy, yy], "-", color=ECOL[k], lw=1.4); a.plot(r["mae"], yy, "o", color=ECOL[k], ms=4)
    a.set_yticks(range(len(names))[::-1]); a.set_yticklabels([ELAB[k] for k in names], fontsize=8)
    a.set_xlabel("hourly energy MAE [kWh/100 km]\n(8 reference hours, 95 % CI)"); a.grid(axis="y", visible=False)
    a = ax[2]
    vv = 80 / 3.6
    combos = [("model", "model_dir", "speed: corrected\ndir.: corrected"), ("model", "metnordic_dir", "speed: corrected\ndir.: MET Nordic"),
              ("metnordic", "model_dir", "speed: MET Nordic\ndir.: corrected"), ("metnordic", "metnordic_dir", "speed: MET Nordic\ndir.: MET Nordic"),
              ("model", "dir_veh", "speed: corrected\ndir.: measured"), ("metnordic", "dir_veh", "speed: MET Nordic\ndir.: measured")]
    from smhi_trv_wind import assess
    Et = assess.energy(V.w_veh.values, V.dir_veh.values, V.heading.values)
    vals = []
    for u, d, _ in combos:
        Ee = assess.energy(V[u].values, V[d].values, V.heading.values)
        g_ = pd.DataFrame(dict(h=V.h.values, t=Et, e=Ee)).groupby("h").mean()
        vals.append(float(np.mean(np.abs(g_.e - g_.t))))
    cols = [ORANGE, ORANGE, PURPLE, PURPLE, ORANGE, PURPLE]
    a.barh(range(len(vals))[::-1], vals, color=cols, alpha=0.75, height=0.55)
    a.set_yticks(range(len(vals))[::-1]); a.set_yticklabels([c[2] for c in combos], fontsize=7.5)
    a.set_xlabel("hourly energy MAE [kWh/100 km]"); a.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        a.text(v + 0.3, len(vals) - 1 - i, f"{v:.1f}", va="center", fontsize=6.5)
    for a_, s, x in zip(ax, "abc", (-0.3, -0.75, -0.75)):
        label(a_, s, x=x)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig06_vehicle_validation")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["sens", "grid", "errors", "energy", "vehicle"]
    for k in todo:
        dict(sens=fig_sens_norway, grid=fig_two_scale, errors=fig_error_structure, energy=fig_energy, vehicle=fig_vehicle)[k]()
