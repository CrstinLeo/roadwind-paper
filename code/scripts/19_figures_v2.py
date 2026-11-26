"""Additional explanatory figures of the two-country paper -> <repo>/figures/

  fig00_concept            schematic of the problem and of the exposure predictors (side view + plan view)
  fig09_mechanisms         partial effects of the exposure predictors and the direction deflection, both countries
  fig10_error_structure    station MAE raw vs model, MAE by wind-speed class, monthly MAE (Norway), drag-error CDF
  fig11_timeseries         observed, nearest-station and corrected wind at three Norwegian roadside stations in a storm
  fig12_site_factors       site factors of the reference stations, both countries
  fig06_vehicle_validation vehicle-height validation incl. speed/direction decomposition of the drag error

Inputs: SWW_NO_OUT (norway_results.pkl, mechanism_NO.pkl, vehicle_validation_rows.parquet),
        SWW_SE_OUT (cv_full.pkl, result.pkl, mechanism_SE.pkl), SWW_NO_DATA (stations_region.csv, features_variants.parquet)
"""
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge

from smhi_trv_wind import drag
from smhi_trv_wind.geo import wrap180

NO = Path(os.environ.get("SWW_NO_OUT", "/tmp/nout2")); SE = Path(os.environ.get("SWW_SE_OUT", "/tmp/sww8"))
NOD = Path(os.environ.get("SWW_NO_DATA", "/home/claude/no"))
FIG = Path(os.environ.get("SWW_FIG", Path(__file__).resolve().parents[2] / "figures"))
BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#1d1d1b", "#6b6a65", "#e4e3de"
LIGHT = "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
W2 = 7.2                                                  # double-column width [in]
C = {"Norway": ORANGE, "Sweden": BLUE}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(NO / f"{name}.png", dpi=200); plt.close(fig); print("wrote", name)


def label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


R = pickle.load(open(NO / "norway_results.pkl", "rb"))
SEcv = pickle.load(open(SE / "cv_full.pkl", "rb"))
SEres = pickle.load(open(SE / "result.pkl", "rb"))
MECH = {"Norway": pickle.load(open(NO / "mechanism_NO.pkl", "rb")), "Sweden": pickle.load(open(SE / "mechanism_SE.pkl", "rb"))}


# ------------------------------------------------------------------------------------------------ concept
def fig_concept():
    fig = plt.figure(figsize=(W2, 2.9))
    ax = fig.add_axes([0.0, 0.0, 0.60, 1.0]); ax.set_xlim(0, 60); ax.set_ylim(-2, 30); ax.axis("off")
    # ground
    ax.add_patch(Rectangle((0, -1), 60, 1, color="#d9d7cf", lw=0))
    # reference station in open terrain
    ax.plot([5, 5], [0, 10], color=INK, lw=1.5); ax.plot([3.8, 6.2], [10, 10], color=INK, lw=1.5)
    for x in (3.8, 6.2):
        ax.add_patch(Circle((x, 10.4), 0.35, color=INK))
    ax.text(5, 11.6, "reference station\n10 m, open terrain", ha="center", fontsize=7)
    # log profile over open terrain
    z = np.linspace(0.3, 24, 60); u = np.log(z / 0.03) / np.log(24 / 0.03)
    ax.plot(10 + 6 * u, z, color=BLUE, lw=1.2); ax.text(16.3, 24.5, "open-terrain\nprofile", fontsize=6.5, color=BLUE, ha="center")
    # blending height
    ax.plot([0, 60], [24, 24], ls=(0, (4, 3)), color=MUTED, lw=0.8)
    ax.text(59.5, 24.6, "blending height $z_b$ = 60 m: wind independent of the surface", ha="right", fontsize=6.5, color=MUTED)
    # landscape forest (far) and near trees around the road
    for x0_ in np.arange(22, 34, 2.2):
        ax.add_patch(Polygon([[x0_ - 1, 0], [x0_ + 1, 0], [x0_, 7]], color="#9cc79a", lw=0))
    for x0_ in np.arange(36, 43, 2.0):
        ax.add_patch(Polygon([[x0_ - 1, 0], [x0_ + 1, 0], [x0_, 9]], color=GREEN, lw=0))
    for x0_ in np.arange(52, 60, 2.0):
        ax.add_patch(Polygon([[x0_ - 1, 0], [x0_ + 1, 0], [x0_, 9]], color=GREEN, lw=0))
    # road, roadside station and truck
    ax.add_patch(Rectangle((43.5, -0.2), 8, 0.4, color=INK, lw=0))
    ax.plot([50.5, 50.5], [0, 8], color=INK, lw=1.2); ax.add_patch(Circle((50.5, 8.3), 0.3, color=INK))
    ax.text(50.5, 12.0, "roadside\nstation", ha="center", fontsize=6.5)
    ax.add_patch(Rectangle((44.2, 0.5), 4.2, 2.6, color=BLUE, lw=0)); ax.add_patch(Rectangle((48.5, 0.5), 1.2, 2.0, color=INK, lw=0))
    ax.text(46.5, -1.8, "vehicle, 3.4 m", ha="center", fontsize=6.5)
    # reduced roadside profile
    z2 = np.linspace(0.3, 24, 60); u2 = np.clip(np.log(z2 / 1.5) / np.log(24 / 1.5), 0.05, None)
    ax.plot(44 + 6 * u2, z2, color=ORANGE, lw=1.2); ax.text(47.5, 17.5, "sheltered\nroadside\nprofile", fontsize=6.5, color=ORANGE, ha="center")
    # annotations of scales
    ax.annotate("", xy=(43.5, 15), xytext=(35.5, 15), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
    ax.text(39.5, 15.6, "near trees\n≤ 100 m", ha="center", fontsize=6.5)
    ax.annotate("", xy=(43.5, 20.5), xytext=(21, 20.5), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
    ax.text(32, 21.1, "landscape forest fraction ≤ 1 km", ha="center", fontsize=6.5)
    ax.text(1, 28.8, "a", fontsize=10, fontweight="bold")
    # plan view
    bx = fig.add_axes([0.62, 0.04, 0.37, 0.92]); bx.set_xlim(-1.25, 1.25); bx.set_ylim(-1.25, 1.25); bx.set_aspect("equal"); bx.axis("off")
    rng = np.random.default_rng(3)
    pts = rng.uniform(-1.2, 1.2, (900, 2)); keep = (np.hypot(*pts.T) < 1.15) & (np.abs(pts[:, 0] + 0.25 * pts[:, 1]) > 0.06)
    bx.scatter(pts[keep, 0], pts[keep, 1], s=3, color="#9cc79a", lw=0)
    bx.add_patch(Circle((0, 0), 1.0, fill=False, ec=INK, lw=0.7, ls=(0, (3, 2))))
    bx.text(0.72, -0.86, "1 km: $F_{1000}$", fontsize=6.5)
    bx.add_patch(Wedge((0, 0), 0.32, 90 - 22.5 - 0 + 45, 90 + 22.5 + 45, color=ORANGE, alpha=0.35, lw=0))
    bx.text(-1.22, 0.05, "100 m upwind\nsector: $T_{100}$", fontsize=6.5, color=ORANGE)
    bx.plot([-0.3, 0.3], [1.2, -1.2], color=INK, lw=2.2); bx.text(0.36, -1.12, "road", fontsize=6.5)
    bx.add_patch(FancyArrowPatch((-0.95, 0.95), (-0.42, 0.42), arrowstyle="-|>", mutation_scale=10, color=BLUE, lw=1.2))
    bx.text(-1.2, 1.02, "reference wind", fontsize=6.5, color=BLUE)
    bx.plot(0, 0, "o", ms=4, color=INK)
    bx.annotate("", xy=(-0.14, 0.14), xytext=(0, 0), arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.9))
    bx.text(-0.22, -0.18, r"$\alpha$", fontsize=8, color=PURPLE)
    bx.text(-1.2, 1.17, "b", fontsize=10, fontweight="bold")
    save(fig, "fig00_concept")


# ------------------------------------------------------------------------------------------------ mechanisms
def fig_mechanisms():
    fig, ax = plt.subplots(2, 3, figsize=(W2, 4.4))
    spec = [("T100", "near-tree share $T_{100}$", 1), ("F1000", "landscape forest fraction $F_{1000}$", 4),
            ("lnxh", "distance to forest edge $\\ln(x/h)$", 2), ("lnUnet", "regional wind $U_\\mathrm{net}$ [m s$^{-1}$]", 3)]
    pos = [(0, 0), (0, 1), (0, 2), (1, 0)]
    for (nm, xl, k), (i, j) in zip(spec, pos):
        a = ax[i, j]
        for country, M in MECH.items():
            d = M["partial"][nm]; d = d[d["size"] >= 3]
            x = np.exp(d.x) if nm == "lnUnet" else d.x
            off = d["mean"].mean() - M["b"][k] * d.x.mean()           # centre the model line on the data
            a.errorbar(x, d["mean"], yerr=d["se"], fmt="o", ms=3, color=C[country], capsize=1.5, lw=0.8, label=country)
            xx = np.linspace(d.x.min(), d.x.max(), 50)
            a.plot(np.exp(xx) if nm == "lnUnet" else xx, off + M["b"][k] * xx, color=C[country], lw=1.1, alpha=0.8)
        a.set_xlabel(xl); a.set_ylabel("partial effect on $\\ln(U_\\mathrm{road}/U_3)$" if j == 0 else "")
        if nm == "lnUnet":
            a.set_xscale("log"); a.set_xticks([1, 2, 4, 8]); a.set_xticklabels(["1", "2", "4", "8"])
    ax[0, 0].legend(frameon=False, loc="upper left")
    for j, (country, lab) in enumerate([("Sweden", "Sweden"), ("Norway", "Norway")]):
        a = ax[1, 1 + j]; M = MECH[country]
        for (grp, d), col, lab in zip(M["direction"].items(), [MUTED, GREEN], ["open ($T_{200}<0.2$)", "trees ($T_{200}\\geq 0.5$)"]):
            a.errorbar(d.x, d["mean"], yerr=d["se"], fmt="o-", ms=3, color=col, capsize=1.5, lw=1.0, label=lab)
        al = np.linspace(-90, 90, 100)
        for t, col in ((0.1, MUTED), (0.75, GREEN)):
            a.plot(al, M["d"][0] + M["d"][1] * np.sin(np.radians(2 * al)) * t, color=col, lw=0.8, ls="--")
        a.axhline(0, color=INK, lw=0.5); a.set_xticks([-90, -45, 0, 45, 90])
        a.set_xlabel("wind-to-road angle $\\alpha$ [°]"); a.set_ylabel("deflection $\\varphi_\\mathrm{road}-\\varphi_\\mathrm{ref}$ [°]" if j == 0 else "")
        a.set_title(lab, loc="right", color=C[country], fontsize=7.5)
        if j == 0:
            a.legend(frameon=False, loc="lower left", fontsize=6.5)
    for a, s in zip(ax.flat, "abcdef"):
        label(a, s)
    fig.tight_layout(w_pad=1.2, h_pad=1.6)
    save(fig, "fig09_mechanisms")


# ------------------------------------------------------------------------------------------------ error structure
def _speed_classes(U, U0, ws, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ws >= lo) & (ws < hi) & np.isfinite(U) & np.isfinite(U0)
        out.append((np.mean(np.abs(U[m] - ws[m])), np.mean(np.abs(U0[m] - ws[m])), m.sum()))
    return np.array(out)


def fig_error_structure():
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.1))
    data = {"Norway": (R["pred_final"], R["obs"], R["cv_final"]), "Sweden": (SEcv["pred"], SEcv["obs"], SEcv["cv"])}
    a = ax[0]
    for country, (P, O, c) in data.items():
        a.scatter(c["mae_raw"], c["mae"], s=7, color=C[country], alpha=0.75, lw=0, label=country)
    m = 7.0; a.plot([0, m], [0, m], color=INK, lw=0.6); a.set_xlim(0, m); a.set_ylim(0, m)
    a.set_xlabel("nearest station MAE [m s$^{-1}$]"); a.set_ylabel("corrected MAE [m s$^{-1}$]"); a.legend(frameon=False, loc="upper left")
    a = ax[1]; edges = np.array([0, 2, 4, 6, 8, 10, 14, 20])
    for country, (P, O, c) in data.items():
        s = _speed_classes(P["U"].astype(float), P["U0n"].astype(float), O["ws"].astype(float), edges)
        xc = 0.5 * (edges[:-1] + edges[1:]); ok = s[:, 2] > 200
        a.plot(xc[ok], s[ok, 1], "--", color=C[country], lw=1.0); a.plot(xc[ok], s[ok, 0], "-o", ms=3, color=C[country])
    a.plot([], [], "--", color=INK, lw=1.0, label="nearest station"); a.plot([], [], "-o", ms=3, color=INK, label="corrected")
    a.legend(frameon=False, loc="upper left"); a.set_xlabel("observed roadside wind [m s$^{-1}$]"); a.set_ylabel("MAE [m s$^{-1}$]")
    a = ax[2]; P, O, _ = data["Norway"]; mon = pd.DatetimeIndex(O["hours"]).month
    U, U0, ws = P["U"].astype(float), P["U0n"].astype(float), O["ws"].astype(float)
    mm = [np.nanmean(np.abs(U[mon == k] - ws[mon == k])) for k in range(1, 13)]
    m0 = [np.nanmean(np.abs(U0[mon == k] - ws[mon == k])) for k in range(1, 13)]
    a.bar(np.arange(1, 13) - 0.2, m0, 0.4, color=LIGHT, label="nearest station"); a.bar(np.arange(1, 13) + 0.2, mm, 0.4, color=ORANGE, label="corrected")
    a.set_xticks(range(1, 13)); a.set_xticklabels(list("JFMAMJJASOND")); a.set_ylabel("MAE [m s$^{-1}$]"); a.set_xlabel("month (Norway)")
    a.legend(frameon=False, loc="upper center", fontsize=6.5)
    a = ax[3]
    dn, _ = drag.station_errors(R["obs"]["ws"], R["obs"]["wd"], R["pred_final"]["U"], R["pred_final"]["wdm"], R["obs"]["road"])
    dr, _ = drag.station_errors(R["obs"]["ws"], R["obs"]["wd"], R["pred_final"]["U0n"], R["pred_final"]["wd0n"], R["obs"]["road"])
    for country, (mod, raw) in {"Norway": (dn, dr), "Sweden": (SEcv["drag"]["model_station"], SEcv["drag"]["raw_station"])}.items():
        for v, ls in ((raw, "--"), (mod, "-")):
            v = np.sort(100 * v[np.isfinite(v)]); a.plot(v, np.arange(1, len(v) + 1) / len(v), ls, color=C[country], lw=1.1)
    a.set_xlabel("station median |drag error| [%]"); a.set_ylabel("fraction of stations"); a.set_xlim(0, 60)
    for a, s in zip(ax, "abcd"):
        label(a, s, x=-0.24)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig10_error_structure")


# ------------------------------------------------------------------------------------------------ time series
def fig_timeseries():
    P, O = R["pred_final"], R["obs"]
    H = pd.DatetimeIndex(O["hours"]); ws = O["ws"].astype(float); U = P["U"].astype(float); U0 = P["U0n"].astype(float)
    reg = pd.Series(np.nanmean(ws, 1), index=H).rolling(72, center=True).mean()
    t0 = reg.idxmax() - pd.Timedelta("4D"); t1 = t0 + pd.Timedelta("9D")
    S = pd.read_csv(NOD / "stations_region.csv"); S["id"] = S.id.astype(str); S = S.set_index("id")
    trv = R["meta"]["trv"]; mae, raw = R["cv_final"]["mae"], R["cv_final"]["mae_raw"]
    gain = raw - mae
    order = np.argsort(gain)
    pick = [order[-1], order[len(order) // 2], order[0]]                      # largest gain, median, smallest gain
    fig, ax = plt.subplots(3, 1, figsize=(W2, 4.6), sharex=True)
    m = (H >= t0) & (H <= t1)
    for a, j, s, tag in zip(ax, pick, "abc", ("largest gain", "median gain", "largest loss")):
        a.plot(H[m], U0[m, j], color=LIGHT, lw=1.0, label="nearest reference station")
        a.plot(H[m], ws[m, j], color=INK, lw=1.0, label="observed at the roadside")
        a.plot(H[m], U[m, j], color=ORANGE, lw=1.2, label="corrected estimate")
        nm = str(S.loc[trv[j], "name"]).title() if trv[j] in S.index else trv[j]
        a.set_title(f"{nm} ({tag}): MAE {raw[j]:.2f} → {mae[j]:.2f} m s$^{{-1}}$ over the whole test period", loc="left", fontsize=7.5)
        a.set_ylabel("wind [m s$^{-1}$]"); label(a, s, x=-0.07, y=0.98)
    ax[0].legend(frameon=False, ncol=3, loc="upper right", fontsize=6.5)
    fig.autofmt_xdate(); fig.tight_layout(h_pad=0.6)
    save(fig, "fig11_timeseries")


# ------------------------------------------------------------------------------------------------ site factors
def fig_site_factors():
    S = pd.read_csv(NOD / "stations_region.csv"); S["id"] = S.id.astype(str); S = S.set_index("id")
    no = np.exp(R["site_c"].mean(axis=1)).dropna()
    no.index = [str(S.loc[i, "name"]).title().replace(" Iii", " III").replace(" Ii", " II") if i in S.index else i for i in no.index]
    se = np.exp(SEres.site.c.mean(axis=1)).dropna()
    se.index = [str(SEres.smhi.name.get(i, i)) for i in se.index]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 3.6), gridspec_kw=dict(width_ratios=[len(no), len(se)]))
    for a, (country, f) in zip(ax, (("Norway", no), ("Sweden", se))):
        f = f.sort_values()
        a.hlines(range(len(f)), 1, f.values, color=C[country], lw=1.2); a.plot(f.values, range(len(f)), "o", ms=3.5, color=C[country])
        a.axvline(1, color=INK, lw=0.6); a.set_xscale("log"); a.set_xticks([0.5, 1, 2]); a.set_xticklabels(["0.5", "1", "2"])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_yticks(range(len(f))); a.set_yticklabels(f.index, fontsize=5.8); a.set_xlabel("site factor $e^{c_s}$ (exposure relative to network)")
        a.set_title(country, loc="left", color=C[country]); a.grid(axis="y", visible=False)
    label(ax[0], "a", x=-0.55, y=1.0); label(ax[1], "b", x=-0.75, y=1.0)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig12_site_factors")


# ------------------------------------------------------------------------------------------------ vehicle
def fig_vehicle():
    V = pd.read_parquet(NO / "vehicle_validation_rows.parquet")
    est = [("ref_raw", "ref_dir", "nearest MET/Avinor station", LIGHT), ("road_raw", "road_dir", "nearest road weather station", GREEN),
           ("metnordic", "metnordic_dir", "MET Nordic analysis", PURPLE), ("model", "model_dir", "this study", ORANGE)]
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.3))
    for j, (site, g) in enumerate([("Dovre (E6, mountains)", V[V.site == "dovre"]), ("Frøya (open coast)", V[V.site == "froya"])]):
        a = ax[j]
        for c, _, lab, col in est:
            a.scatter(g.w_veh, g[c], s=4, alpha=0.5, color=col, label=lab, lw=0)
        m = max(g.w_veh.max(), g[[e[0] for e in est]].max().max()) * 1.05
        a.plot([0, m], [0, m], color=INK, lw=0.6); a.set_xlim(0, m); a.set_ylim(0, m)
        a.set_xlabel("van anemometer, 3.4 m [m s$^{-1}$]"); a.set_title(site, loc="left", fontsize=7.5)
    ax[0].set_ylabel("estimate [m s$^{-1}$]")
    vv = 80 / 3.6
    errs = [np.abs(e[np.isfinite(e)]) * 100 for e in
            (drag.relative_error(vv, V.w_veh.values, V.dir_veh.values, V[c].values, V[d].values, V.heading.values) for c, d, _, _ in est)]
    bp = ax[2].boxplot(errs, showfliers=False, widths=0.6, medianprops=dict(color=INK), patch_artist=True)
    for b, (_, _, _, col) in zip(bp["boxes"], est):
        b.set_facecolor(col); b.set_alpha(0.6)
    ax[2].set_xticks(range(1, 5)); ax[2].set_xticklabels(["MET\nstn", "road\nstn", "MET\nNordic", "this\nstudy"], fontsize=6.3)
    ax[2].set_ylabel("|drag error| at 80 km h$^{-1}$ [%]")
    combos = [("model", "model_dir", "speed: this study\ndir.: this study"), ("model", "metnordic_dir", "speed: this study\ndir.: MET Nordic"),
              ("metnordic", "model_dir", "speed: MET Nordic\ndir.: this study"), ("metnordic", "metnordic_dir", "speed: MET Nordic\ndir.: MET Nordic"),
              ("model", "dir_veh", "speed: this study\ndir.: measured"), ("metnordic", "dir_veh", "speed: MET Nordic\ndir.: measured")]
    vals = [100 * np.nanmedian(np.abs(drag.relative_error(vv, V.w_veh.values, V.dir_veh.values, V[u].values, V[d].values, V.heading.values)))
            for u, d, _ in combos]
    cols = [ORANGE, ORANGE, PURPLE, PURPLE, ORANGE, PURPLE]
    a = ax[3]; a.barh(range(len(vals))[::-1], vals, color=cols, alpha=0.75, height=0.65)
    a.set_yticks(range(len(vals))[::-1]); a.set_yticklabels([c[2] for c in combos], fontsize=5.6)
    a.set_xlabel("median |drag error| [%]"); a.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        a.text(v + 0.8, len(vals) - 1 - i, f"{v:.0f}", va="center", fontsize=6.5)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, markerscale=2.5, bbox_to_anchor=(0.42, -0.08))
    for a, s in zip(ax, "abcd"):
        label(a, s, x=-0.2 if s != "d" else -0.75)
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.8)
    save(fig, "fig06_vehicle_validation")


if __name__ == "__main__":
    for f in (fig_concept, fig_mechanisms, fig_error_structure, fig_timeseries, fig_site_factors, fig_vehicle):
        f()
