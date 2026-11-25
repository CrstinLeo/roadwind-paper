"""Sensitivity scans scored by leave-one-time-block-out (4 blocks) x leave-one-TrV-station-out."""
import dataclasses, os, pickle
import numpy as np
import pandas as pd
from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import Engine, Spec, cv

cfg = Settings(max_pair_km=150.0)
V = pd.read_parquet(cfg.paths.root / "features_variants.parquet")
E = Engine(cfg, V, t_split="2026-11-19 00:00", t_end=os.environ.get("SWW_END", "2026-11-23 15:00"))
print("hours", len(E.H), "TrV", len(E.TM), "pairs", len(E.pj))
res = {}

def add(group, label, value, **kw):
    sp = dataclasses.replace(Spec(), **kw)
    res[(group, label)] = dict(value=value, spec=sp, r=cv(E, sp))

scans = {
    "z_blend": [(f"{v} m", v, dict(z_blend=v)) for v in (30, 40, 60, 80, 100, 150)],
    "c_tree": [(f"{v}", v, dict(c_tree=v)) for v in (0.05, 0.075, 0.1, 0.125, 0.15)],
    "L_smhi": [(f"{v} m", v, dict(L_smhi=v)) for v in (25, 50, 100, 200, 400, 1000, 2000)],
    "L_trv": [(f"{v} m", v, dict(L_trv=v)) for v in (25, 50, 100, 200, 400, 1000, 2000)],
    "H_z0": [(f"±{v:g}°", v, dict(H_smhi=v, H_trv=v)) for v in (22.5, 45.0)],
    "site": [(v, i, dict(site=v)) for i, v in enumerate(("none", "mean", "sector"))],
    "site_nsec": [(f"{v}", v, dict(site_nsec=v)) for v in (4, 8, 16)],
    "site_min_net": [(f"{v} m/s", v, dict(site_min_net=v)) for v in (1.0, 2.0, 3.0, 4.0, 5.0)],
    "tree_R": [(f"{R} m", R, dict(tree=f"tree_R{R}_H22.5")) for R in (50, 100, 150, 200, 300, 500, 1000)],
    "tree_H": [(f"±{h:g}°", h, dict(tree=f"tree_R200_H{h:g}")) for h in (11.25, 22.5, 45.0, 90.0, 180.0)],
    "tree_soft": [(f"L={L} m", L, dict(tree=f"treew_L{L}_H22.5")) for L in (25, 50, 100, 200, 400)],
    "road_window": [(f"±{S} m", S, dict(tree=f"along:{S}", along_z0=S)) for S in (0, 100, 250, 500, 750, 1000)],
    "fit_min": [(f"{v} m/s", v, dict(fit_min=v)) for v in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)],
    "km": [(f"{v} km", v, dict(km=v, min_n=1)) for v in (10, 20, 30, 40, 60, 80, 100, 150)],
    "min_n": [(f"{v}", v, dict(min_n=v)) for v in (1, 2, 3, 4, 6)],
    "power": [(f"{v}", v, dict(power=v, km=150, min_n=1)) for v in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)],
    "dir_km": [(f"{v} km", v, dict(dir_km=v)) for v in (20, 40, 60, 80, 150)],
    "structure": [(l, i, kw) for i, (l, kw) in enumerate([
        ("baseline", {}), ("no ln(x/h)", dict(use_lnxh=False)), ("no lnU×tree", dict(interaction=False)),
        ("+ cross-road", dict(extra=("cross",))), ("+ cross-road×tree", dict(extra=("cross", "cross_tree"))),
        ("+ stability", dict(extra=("stab",))), ("+ stability×tree", dict(extra=("stab", "stab_tree"))),
        ("+ night×tree", dict(extra=("night", "night_tree"))), ("+ Sx, TPI", dict(extra=("sx", "tpi"))),
        ("+ h_mid", dict(extra=("hmid",))), ("lnU of the pair", dict(extra=("lnU_pair",))),
        ("skill weights", dict(weights="skill"))])],
}
for g, items in scans.items():
    for label, v, kw in items:
        add(g, label, v, **kw)
    print("done", g, flush=True)
cfg.paths.out.mkdir(parents=True, exist_ok=True)
pickle.dump(dict(res=res, trv=list(E.TM.index)), open(cfg.paths.out / "sensitivity_cv.pkl", "wb"))
