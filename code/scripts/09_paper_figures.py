"""Figures of the manuscript, written as PDF into <repo>/figures.

Inputs (all produced by the package):
  SWW_OUT/result.pkl            pipeline.run(Settings()) result (created here if missing)
  SWW_OUT/sensitivity_cv.pkl    scripts/06_sensitivity_cv.py
  SWW_OUT/two_scale_cv.pkl      scripts/08_two_scale_grid.py
  SWW_DATA/features_variants.parquet
  SWW_ROUTE (optional)          OSRM route JSON of RV40
"""
import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.merge import merge

from smhi_trv_wind import Settings, evaluate, pipeline
from smhi_trv_wind.features.gis import TO_SWEREF, _landcover_sources

cfg = Settings()
OUT = cfg.paths.out
FIG = Path(os.environ.get("SWW_FIG", Path(__file__).resolve().parents[2] / "figures"))
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#1d1d1b", "#6b6a65", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "lines.linewidth": 1.2, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
W1, W2 = 3.54, 7.48          # single / double column width [in]


def load_result():
    p = OUT / "result.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    r = pipeline.run(cfg)
    pickle.dump(r, open(p, "wb"))
    return r


def paired(a, b):
    d = a - b
    ok = np.isfinite(d)
    return d[ok].mean(), d[ok].std(ddof=1) / np.sqrt(ok.sum())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


R = load_result()

# ---------------------------------------------------------------- Fig 1: study area
def fig_map():
    TM, SM, S = R.trv, R.smhi, R.S
    tx, ty = TO_SWEREF.transform(TM.lon.values, TM.lat.values)
    sx, sy = TO_SWEREF.transform(SM.lon.values, SM.lat.values)
    x0, x1 = min(tx.min(), sx.min()) - 15e3, max(tx.max(), sx.max()) + 40e3
    y0, y1 = min(ty.min(), sy.min()) - 15e3, max(ty.max(), sy.max()) + 15e3
    res = 250.0
    lc, tr = merge(_landcover_sources(cfg), bounds=(x0, y0, x1, y1), res=res, nodata=0, resampling=rasterio.enums.Resampling.mode, method="max")
    lc = lc[0]
    with rasterio.open(cfg.paths.nmd_object_height) as src:
        win = rasterio.windows.from_bounds(x0, y0, x1, y1, src.transform)
        oh = src.read(1, window=win, out_shape=lc.shape, resampling=rasterio.enums.Resampling.average, boundless=True, fill_value=0)
    base = np.zeros(lc.shape)                               # 0 outside the land-cover mosaic, 1 open land, 2 forest, 3 water
    base[lc > 0] = 1
    base[(oh > 2) & (lc > 0)] = 2
    base[np.isin(lc, [61, 62])] = 3
    cmap = ListedColormap(["#e9e8e4", "#f7f6f2", "#d3e6cf", "#cfe1f3"])
    fig, ax = plt.subplots(figsize=(W2, W2 * (y1 - y0) / (x1 - x0) * 0.82))
    ax.imshow(base, extent=(x0, x1, y0, y1), origin="upper", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    route = os.environ.get("SWW_ROUTE")
    if route and Path(route).exists():
        c = np.array(json.load(open(route))["routes"][0]["geometry"]["coordinates"])
        rx, ry = TO_SWEREF.transform(c[:, 0], c[:, 1])
        ax.plot(rx, ry, color=INK, lw=1.6, zorder=3, label="RV40 Gothenburg–Borås")
    mae = S.mae_model.reindex(TM.index).values
    sc = ax.scatter(tx, ty, c=mae, cmap="viridis_r", vmin=0.3, vmax=1.5, s=14, edgecolor="white", linewidth=0.4, zorder=4, label="Trafikverket roadside station (6 m)")
    ax.scatter(sx, sy, marker="s", s=26, color=ORANGE, edgecolor=INK, linewidth=0.6, zorder=5, label="SMHI synoptic station (10 m)")
    for (x, y, n) in zip(sx, sy, SM.name.values):
        if x0 < x < x1 and y0 < y < y1:
            ax.annotate(n.replace(" A", "").replace(" Flygplats", "").replace("Göteborg-Landvetter", "Landvetter"), (x, y),
                        xytext=(3, 3), textcoords="offset points", fontsize=6, color=INK)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("MAE of corrected wind speed [m s$^{-1}$]")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.plot([x0 + 10e3, x0 + 60e3], [y0 + 10e3] * 2, color=INK, lw=2)
    ax.text(x0 + 35e3, y0 + 13e3, "50 km", ha="center", fontsize=7)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
    save(fig, "fig02_sweden_area")


# ---------------------------------------------------------------- Fig 3: correction ladder
def fig_ladder():
    P = R.P[R.P.dist <= 80]
    steps = ["U0", "U1", "U2", "U3", "U4"]
    lab = ["raw\n10 m", "height\n(×0.912)", "roughness\n(two-step)", "site\nfactor", "roadside\nexposure"]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    x = np.arange(5)
    for a, col, name in ((ax[0], "mae", "MAE [m s$^{-1}$]"), (ax[1], "bias", "bias  $U_{TrV}/U - 1$")):
        q = np.array([P[f"{col}_{s}"].quantile([0.25, 0.5, 0.75]).values for s in steps])
        a.fill_between(x, q[:, 0], q[:, 2], color=BLUE, alpha=0.15, lw=0, label="interquartile range")
        a.plot(x, q[:, 1], "-o", color=BLUE, ms=4, label="median over station pairs")
        a.set_xticks(x); a.set_xticklabels(lab); a.set_ylabel(name)
        for i in range(5):
            a.annotate(f"{q[i, 1]:.2f}" if col == "mae" else f"{q[i, 1]:+.0%}", (x[i], q[i, 1]), xytext=(0, 6),
                       textcoords="offset points", ha="center", fontsize=6.5)
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[0].set_ylim(bottom=0); ax[0].legend(frameon=False, loc="upper right")
    ax[0].set_title("(a) absolute error", loc="left"); ax[1].set_title("(b) bias", loc="left")
    fig.tight_layout()
    save(fig, "fig03_correction_ladder")


# ---------------------------------------------------------------- Fig 4: site factors
def fig_site():
    raw = np.exp(R.site_raw.c.mean(1)); pot = np.exp(R.site.c.mean(1))
    names = R.smhi.name.reindex(raw.index).str.replace(" A", "", regex=False).str.replace(" Flygplats", "", regex=False).str.replace("Göteborg-Landvetter", "Landvetter", regex=False)
    o = np.argsort(pot.values)
    fig = plt.figure(figsize=(W2, 3.0))
    a = fig.add_axes([0.13, 0.12, 0.33, 0.8])
    y = np.arange(len(o))
    a.hlines(y, raw.values[o], pot.values[o], color=GRID, lw=2)
    a.plot(raw.values[o], y, "o", color=MUTED, ms=4, label="measured wind")
    a.plot(pot.values[o], y, "o", color=BLUE, ms=4, label="after roughness conversion")
    a.set_yticks(y); a.set_yticklabels(names.values[o], fontsize=6.5); a.set_xscale("log")
    a.set_xticks([0.5, 0.7, 1, 1.4, 2]); a.set_xticklabels(["0.5", "0.7", "1", "1.4", "2"]); a.axvline(1, color=MUTED, lw=0.8)
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("station-mean site factor $e^{c_s}$"); a.legend(frameon=False, loc="upper left"); a.set_title("(a)", loc="left")
    b = fig.add_axes([0.58, 0.12, 0.38, 0.8])
    C = np.exp(R.site.c.values[o])
    im = b.imshow(C, aspect="auto", cmap="RdBu_r", norm=matplotlib.colors.LogNorm(vmin=0.5, vmax=2.0))
    b.set_yticks(y); b.set_yticklabels([]); b.set_xticks(range(C.shape[1]))
    b.set_xticklabels([f"{d:.0f}°" for d in R.site.c.columns], rotation=0, fontsize=6.5); b.grid(False)
    b.set_xlabel("wind-from sector of the network wind"); b.set_title("(b) $e^{c_s(k)}$ after roughness conversion", loc="left")
    cb = fig.colorbar(im, ax=b, pad=0.02); cb.set_ticks([0.5, 0.7, 1, 1.4, 2]); cb.set_ticklabels(["0.5", "0.7", "1", "1.4", "2"])
    cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.invert_yaxis(); a.set_ylim(-0.5, len(o) - 0.5)
    save(fig, "fig08_site_factors_sweden")


# ---------------------------------------------------------------- Fig 5: sensitivity panels
def fig_sensitivity():
    S = pickle.load(open(OUT / "sensitivity_cv.pkl", "rb"))["res"]
    base = S[("structure", "baseline")]["r"]["mae"]
    panels = [("L_smhi", "(a) roughness footprint, SMHI", ["25 m", "50 m", "100 m", "200 m", "400 m", "1000 m", "2000 m"], "100 m"),
              ("L_trv", "(b) roughness footprint, roadside", ["25 m", "50 m", "100 m", "200 m", "400 m", "1000 m", "2000 m"], "100 m"),
              ("z_blend", "(c) blending height", ["30 m", "40 m", "60 m", "80 m", "100 m", "150 m"], "60 m"),
              ("site", "(d) SMHI site factors", ["none", "mean", "sector"], "sector"),
              ("tree_R", "(e) radius of upwind tree share", ["50 m", "100 m", "150 m", "200 m", "300 m", "500 m", "1000 m"], "100 m"),
              ("road_window", "(f) along-road averaging ±S", ["±0 m", "±100 m", "±250 m", "±500 m", "±750 m", "±1000 m"], "±0 m"),
              ("km", "(g) radius of station combination", ["10 km", "20 km", "30 km", "40 km", "60 km", "80 km", "100 km", "150 km"], "150 km"),
              ("power", "(h) weight exponent p in $d^{-p}$", ["0.0", "0.5", "1.0", "1.5", "2.0", "3.0"], "1.5")]
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.9))
    for a, (g, title, labels, pick) in zip(axes.ravel(), panels):
        v = [paired(S[(g, l)]["r"]["mae"], base) for l in labels]
        x = np.arange(len(labels))
        a.errorbar(x, [m for m, _ in v], yerr=[s for _, s in v], fmt="o", color=BLUE, ms=3.5, capsize=1.5, lw=1)
        if pick in labels:
            i = labels.index(pick)
            a.plot(i, v[i][0], "s", ms=8, mfc="none", mec=INK, mew=1)
        a.axhline(0, color=MUTED, lw=0.7)
        a.set_xticks(x); a.set_xticklabels([l.replace(" m", "").replace(" km", "") for l in labels], rotation=45, fontsize=6.5)
        a.set_title(title, loc="left", fontsize=7.5)
    for a in axes[:, 0]:
        a.set_ylabel("ΔMAE vs round-6 model [m s$^{-1}$]")
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save(fig, "figS1_sensitivity_sweden")


# ---------------------------------------------------------------- Fig 6: two shelter scales
def fig_two_scale():
    T = pickle.load(open(OUT / "two_scale_cv.pkl", "rb"))
    ref = T[(100, "tree_R1000_H22.5")]["mae"]
    nears = [50, 100, 150, 200]
    fars = [("none", "none"), ("tree_R300_H22.5", "300 m"), ("tree_R500_H22.5", "500 m"), ("tree_R1000_H22.5", "1 km"),
            ("tree_R1000_H90", "1 km\n±90°"), ("tree_R1000_H180", "1 km\nall dir."), ("treew_L400_H22.5", "exp(−r/400 m)")]
    M = np.array([[paired(T[(n, f)]["mae"], ref)[0] for f, _ in fars] for n in nears])
    E = np.array([[paired(T[(n, f)]["mae"], ref)[1] for f, _ in fars] for n in nears])
    fig, ax = plt.subplots(figsize=(W2, 2.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.02, vmax=0.02, aspect="auto")
    for i in range(len(nears)):
        for j in range(len(fars)):
            ax.text(j, i, f"{M[i, j]:+.3f}\n±{E[i, j]:.3f}", ha="center", va="center", fontsize=6.5)
    ax.set_yticks(range(len(nears))); ax.set_yticklabels([f"near {n} m" for n in nears])
    ax.set_xticks(range(len(fars))); ax.set_xticklabels([l for _, l in fars]); ax.grid(False)
    ax.set_xlabel("far (landscape) forest fraction")
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("ΔMAE [m s$^{-1}$]")
    save(fig, "fig07_two_shelter_scales_sweden")


# ---------------------------------------------------------------- Fig 7: along-road correlation
def fig_along():
    offs = [-1000, -750, -500, -250, -100, 0, 100, 250, 500, 750, 1000]
    curves = {}
    for country, path in (("Sweden", cfg.paths.feat_variants), ("Norway", Path(os.environ.get("SWW_NO_DATA", "data/no")) / "features_variants.parquet")):
        if not Path(path).exists():
            continue
        V = pd.read_parquet(path); V = V[V.src == "trv"]
        for col in ("tree", "lnz0"):
            r = []
            for s in offs:
                a, b = V[f"along_{col}_s0"].values, V[f"along_{col}_s{s}"].values
                ok = np.isfinite(a) & np.isfinite(b)
                r.append(np.corrcoef(a[ok], b[ok])[0, 1])
            curves[(country, col)] = r
    S = pickle.load(open(OUT / "review_SE.pkl", "rb"))["scans"]          # 26_sweden_review.py, around the final model
    base = S[("road_window", "+-0 m")]["mae"]
    labs = ["+-0 m", "+-100 m", "+-250 m", "+-500 m"]
    seg = [0, 200, 500, 1000]
    v = [paired(S[("road_window", l)]["mae"], base) for l in labs]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    for (country, col), r in curves.items():
        ax[0].plot(offs, r, "-o" if col == "tree" else "--s", color=BLUE if country == "Sweden" else ORANGE, ms=3,
                   label=f"{country}: " + ("upwind tree share (200 m)" if col == "tree" else "upwind ln $z_0$ (L = 100 m)"))
    ax[0].set_xlabel("distance along the road from the station [m]"); ax[0].set_ylabel("correlation with station value")
    ax[0].set_ylim(0, 1.05); ax[0].set_title("(a)", loc="left")
    ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6.5)
    ax[1].errorbar(seg, [m for m, _ in v], yerr=[s for _, s in v], fmt="-o", color=BLUE, ms=3.5, capsize=1.5)
    ax[1].axhline(0, color=MUTED, lw=0.7)
    ax[1].set_xlabel("segment length 2S over which features are averaged [m]"); ax[1].set_ylabel("ΔMAE vs point features [m s$^{-1}$]")
    ax[1].set_title("(b)", loc="left")
    fig.tight_layout()
    save(fig, "fig05_along_road")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["map", "site", "along"]          # fig_two_scale / fig_sensitivity: superseded by 28_figures_r1.py
    for k in todo:
        dict(map=fig_map, site=fig_site, sens=fig_sensitivity, two=fig_two_scale, along=fig_along)[k]()
