"""Graphical abstract (JWEIA: encouraged; >= 1328 x 531 px, readable at 13 x 5 cm) -> <repo>/figures/graphical_abstract.pdf

Numbers are read from sections/generated/numbers.tex (15_paper_numbers.py), so the picture stays in step with the paper.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon

REPO = Path(__file__).resolve().parents[2]
N = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (REPO / "sections/generated/numbers.tex").read_text()))
BLUE, ORANGE, GREEN, INK, MUTED, PAPER = "#2a78d6", "#eb6834", "#1baf7a", "#1d1d1b", "#6b6a65", "#f6f5f1"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

# ---- left: reference station in open terrain
ax.add_patch(Rectangle((1, 9), 15, 0.5, color=MUTED, lw=0))
ax.plot([8.5, 8.5], [9.5, 27], color=INK, lw=2.2)
ax.plot([6.5, 10.5], [27, 27], color=INK, lw=2.2)
for x in (6.5, 10.5):
    ax.add_patch(plt.Circle((x, 27.6), 0.6, color=INK))
ax.annotate("", xy=(15.5, 30.5), xytext=(1.5, 30.5), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=BLUE))
ax.text(8.5, 33.3, "reference station\n10 m, open terrain", ha="center", va="bottom", fontsize=13, color=INK)

# ---- middle: correction chain
steps = [("height", "log profile"), ("roughness", "$z_0$ from laser\nscanning"), ("site factor", "one per\nstation"),
         ("roadside\nexposure", "trees ≤ 100 m +\nforest in 1 km"), ("combine", "all stations,\n$1/d^{1.5}$")]
x0, w, gap, y0, h = 19, 10.6, 1.6, 16, 12.5
for i, (t, s) in enumerate(steps):
    x = x0 + i * (w + gap)
    hi = i in (3, 4)
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=(GREEN + "22") if hi else PAPER, ec=GREEN if hi else MUTED, lw=1.6 if hi else 1.0))
    ax.text(x + w / 2, y0 + h - 2.2, t, ha="center", va="top", fontsize=14, weight="bold", color=INK)
    ax.text(x + w / 2, y0 + 3.4, s, ha="center", va="center", fontsize=11.5, color=MUTED)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.15, y0 + h / 2), (x + w + gap - 0.15, y0 + h / 2), arrowstyle="-|>",
                                     mutation_scale=14, color=INK, lw=1.2))
ax.text(x0 + (5 * w + 4 * gap) / 2, y0 + h + 2.2, "open data only: weather stations, road weather stations, laser scanning, land cover",
        ha="center", fontsize=12, color=MUTED)

# ---- right: road segment with forest and truck
xr = x0 + 5 * (w + gap) + 0.5
ax.add_patch(FancyArrowPatch((xr - gap + 0.2, y0 + h / 2), (xr + 1.0, y0 + h / 2), arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.4))
ax.add_patch(Rectangle((xr + 1.5, 9), 98.5 - xr - 1.5, 0.5, color=MUTED, lw=0))
for xt in (xr + 2.3, xr + 4.3, xr + 15.3, xr + 17.3):
    ax.add_patch(Polygon([[xt - 1.3, 9.5], [xt + 1.3, 9.5], [xt, 19]], color=GREEN, lw=0))
ax.add_patch(Rectangle((xr + 6.4, 10.0), 5.2, 3.2, fc=BLUE, ec="none"))
ax.add_patch(Rectangle((xr + 11.7, 10.0), 1.5, 2.4, fc=INK, ec="none"))
for xw in (xr + 7.4, xr + 10.6, xr + 12.6):
    ax.add_patch(plt.Circle((xw, 9.9), 0.55, color=INK))
ax.annotate("", xy=(xr + 9.5, 15.6), xytext=(xr + 9.5, 22.5), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=BLUE))
ax.text(xr + 9.0, 23.2, "wind and air drag\non a 200–300 m\nroad segment", ha="center", va="bottom", fontsize=13, color=INK)

# ---- bottom: results
res = (f"median roadside wind-speed error (out of sample):   Norway {N['noCvRaw']} → {N['noCvMed']} m/s   ·   "
       f"Sweden {N['seCvRaw']} → {N['seCvMed']} m/s\n"
       f"median air-drag error, tractor–semitrailer at 80 km/h:   Norway {N['noDragRaw']} → {N['noDragModel']} %   ·   "
       f"Sweden {N['seDragRaw']} → {N['seDragModel']} %")
ax.text(50, 3.6, res, ha="center", va="center", fontsize=13.5, color=INK, linespacing=1.6)

out = REPO / "figures" / "graphical_abstract.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
print("wrote", out)
