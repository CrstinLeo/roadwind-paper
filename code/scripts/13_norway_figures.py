"""Figures for the Norwegian study region (and the two-country comparison) -> <repo>/figures/*.pdf

Inputs: SWW_OUT/norway_results.pkl (11_norway.py), SWW_OUT/terrain_cv.pkl (optional),
        SWW_OUT/vehicle_validation_rows.parquet (12_vehicle_norway.py), SWW_DATA (Norway) station table.
"""
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from matplotlib.colors import ListedColormap
from pyproj import Transformer

from smhi_trv_wind import Settings
from smhi_trv_wind import drag
from smhi_trv_wind.features.norway import WORLDCOVER, _tiles

cfg = Settings.norway()
OUT = Path(os.environ.get("SWW_OUT", "./output"))
FIG = Path(os.environ.get("SWW_FIG", Path(__file__).resolve().parents[2] / "figures"))
R = pickle.load(open(OUT / "norway_results.pkl", "rb"))
BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#1d1d1b", "#6b6a65", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "savefig.bbox": "tight", "pdf.fonttype": 42})
W2 = 7.48


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=200); plt.close(fig); print("wrote", name)


def paired(a, b):
    d = a - b; ok = np.isfinite(d)
    return d[ok].mean(), d[ok].std(ddof=1) / np.sqrt(ok.sum())


def fig_map_norway():
    S = pd.read_csv(cfg.paths.root / "stations_region.csv"); S["id"] = S.id.astype(str)
    T = Transformer.from_crs(4326, 25833, always_xy=True)
    S["x"], S["y"] = T.transform(S.lon.values, S.lat.values)
    x0, x1, y0, y1 = S.x.min() - 20e3, S.x.max() + 20e3, S.y.min() - 20e3, S.y.max() + 20e3
    res = 500.0
    nx_, ny_ = int((x1 - x0) / res), int((y1 - y0) / res)
    tr = rasterio.transform.from_bounds(x0, y0, x1, y1, nx_, ny_)
    back = Transformer.from_crs(25833, 4326, always_xy=True)
    lo, la = back.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    lc = np.zeros((ny_, nx_), np.uint8)
    with rasterio.Env(GDAL_CURL_CA_BUNDLE=os.environ.get("GDAL_CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")):
        for t in _tiles(min(la), max(la), min(lo), max(lo)):
            part = np.zeros_like(lc)
            try:
                with rasterio.open(WORLDCOVER.format(tile=t)) as s:
                    rasterio.warp.reproject(rasterio.band(s, 1), part, dst_transform=tr, dst_crs="EPSG:25833", resampling=rasterio.warp.Resampling.mode)
            except rasterio.errors.RasterioIOError:
                continue
            lc = np.where(lc == 0, part, lc)
    base = np.where((lc == 80) | (lc == 0), 3, np.where(lc == 10, 2, 1))
    cv_ = R["cv_final"]["mae"]
    fig, ax = plt.subplots(figsize=(W2, W2 * (y1 - y0) / (x1 - x0) * 0.8))
    ax.imshow(base, extent=(x0, x1, y0, y1), cmap=ListedColormap(["#ffffff", "#f3f2ee", "#d3e6cf", "#cfe1f3"]), vmin=0, vmax=3, interpolation="nearest")
    tr_ = S[S.grp == "SVV"].set_index("id").reindex(R["meta"]["trv"])
    sc = ax.scatter(tr_.x, tr_.y, c=cv_, cmap="viridis_r", vmin=0.5, vmax=2.5, s=16, edgecolor="white", linewidth=0.4, zorder=4,
                    label="Statens vegvesen road weather station")
    rf = S[S.grp.isin(["MET", "AVINOR"])]
    ax.scatter(rf.x, rf.y, marker="s", s=24, color=ORANGE, edgecolor=INK, linewidth=0.6, zorder=5, label="MET Norway / Avinor station")
    for site, (la_, lo_) in {"Dovre (E6)": (62.12, 9.38), "Frøya": (63.70, 8.55)}.items():
        x, y = T.transform(lo_, la_)
        ax.plot(x, y, "*", ms=11, color=PURPLE, mec="white", zorder=6)
        ax.annotate(site, (x, y), xytext=(5, -10), textcoords="offset points", fontsize=7, color=PURPLE)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01); cb.set_label("cross-validated MAE [m s$^{-1}$]")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    ax.plot([x0 + 10e3, x0 + 60e3], [y0 + 10e3] * 2, color=INK, lw=2); ax.text(x0 + 35e3, y0 + 13e3, "50 km", ha="center", fontsize=7)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")
    save(fig, "fig08_norway_area")


def fig_vehicle():
    V = pd.read_parquet(OUT / "vehicle_validation_rows.parquet")
    est = [("ref_raw", "ref_dir", "nearest MET/Avinor station"), ("road_raw", "road_dir", "nearest road weather station"),
           ("metnordic", "metnordic_dir", "MET Nordic analysis"), ("model", "model_dir", "this study")]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5))
    for j, (site, g) in enumerate([("Dovre (E6, mountain valley)", V[V.site == "dovre"]), ("Frøya (open coast)", V[V.site == "froya"])]):
        a = ax[j]
        for k, (c, _, lab) in enumerate(est):
            a.scatter(g.w_veh, g[c], s=5, alpha=0.5, color=[MUTED, GREEN, PURPLE, BLUE][k], label=lab)
        m = max(g.w_veh.max(), g[[e[0] for e in est]].max().max()) * 1.05
        a.plot([0, m], [0, m], color=INK, lw=0.8); a.set_xlim(0, m); a.set_ylim(0, m)
        a.set_xlabel("vehicle anemometer, 3.4 m [m s$^{-1}$]"); a.set_title(site, loc="left")
    ax[0].set_ylabel("estimate [m s$^{-1}$]"); ax[0].legend(frameon=False, loc="upper left", markerscale=2)
    vv = 80 / 3.6
    errs = []
    for c, dcol, lab in est:
        e = drag.relative_error(vv, V.w_veh.values, V.dir_veh.values, V[c].values, V[dcol].values, V.heading.values)
        errs.append(np.abs(e[np.isfinite(e)]) * 100)
    ax[2].boxplot(errs, showfliers=False, widths=0.6, medianprops=dict(color=INK))
    ax[2].set_xticks(range(1, 5)); ax[2].set_xticklabels(["nearest\nMET", "nearest\nroad st.", "MET\nNordic", "this\nstudy"])
    ax[2].set_ylabel("|air-drag error| at 80 km h$^{-1}$ [%]"); ax[2].set_title("(c) tractor-semitrailer drag", loc="left")
    fig.tight_layout(); save(fig, "fig09_vehicle_validation")


if __name__ == "__main__":
    fig_map_norway()
    if (OUT / "vehicle_validation_rows.parquet").exists():
        fig_vehicle()
