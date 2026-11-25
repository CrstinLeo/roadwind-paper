"""Sensitivity scans and structural variants. Tuning period A, holdout B, LOSO over TrV stations."""
import dataclasses, os, pickle, time
import numpy as np
import pandas as pd
from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import Engine, Spec, summarise

cfg = Settings()
V = pd.read_parquet(cfg.paths.root / "features_variants.parquet")
E = Engine(cfg, V, t_split=os.environ.get("SWW_SPLIT", "2026-11-19 00:00"), t_end=os.environ.get("SWW_END", "2026-11-23 15:00"))
print("hours A/B", E.A.sum(), (~E.A).sum(), "TrV", len(E.TM), "pairs", len(E.pj))
base = E.run(Spec())
rows, keep = [], {}

def add(group, label, value, **kw):
    sp = dataclasses.replace(Spec(), **kw)
    r = E.run(sp)
    s = summarise(r, base); s.update(group=group, label=label, value=value, n_fit=r["n_fit"])
    rows.append(s); keep[(group, label)] = {p: dict(mae=r[p]["mae"], dir=r[p]["dir"]) for p in ("A", "B")}
    keep[(group, label)]["b"] = r["b"]

add("baseline", "baseline", 0)
# --- step 2 physics
for v in (30, 40, 60, 80, 100, 150): add("z_blend", f"{v} m", v, z_blend=v)
for v in (0.05, 0.075, 0.1, 0.125, 0.15): add("c_tree", f"{v}", v, c_tree=v)
for v in (25, 50, 100, 200, 400, 1000, 2000): add("L_smhi", f"{v} m", v, L_smhi=v)
for v in (25, 50, 100, 200, 400, 1000, 2000): add("L_trv", f"{v} m", v, L_trv=v)
for v in (22.5, 45.0): add("H_z0", f"±{v:g}°", v, H_smhi=v, H_trv=v)
# --- step 3 site factors
for v in ("none", "mean", "sector"): add("site", v, {"none": 0, "mean": 1, "sector": 2}[v], site=v)
for v in (4, 8, 16): add("site_nsec", f"{v}", v, site_nsec=v)
for v in (1.0, 2.0, 3.0, 4.0, 5.0): add("site_min_net", f"{v} m/s", v, site_min_net=v)
# --- step 4 tree feature: radius, sector width, soft radius, road window
for R in (50, 100, 150, 200, 300, 500, 1000): add("tree_R", f"{R} m", R, tree=f"tree_R{R}_H22.5")
for Hh in (11.25, 22.5, 45.0, 90.0, 180.0): add("tree_H", f"±{Hh:g}°", Hh, tree=f"tree_R200_H{Hh:g}")
for L in (25, 50, 100, 200, 400): add("tree_soft", f"L={L} m", L, tree=f"treew_L{L}_H22.5")
for S in (0, 100, 250, 500, 750, 1000): add("road_window", f"±{S} m", S, tree=f"along:{S}", along_z0=S)
for v in (0.0, 1.0, 2.0, 3.0, 4.0): add("fit_min", f"{v} m/s", v, fit_min=v)
# --- step 6 combination
for v in (5, 10, 20, 30, 40, 60, 80): add("km", f"{v} km", v, km=v, min_n=1)
for v in (1, 2, 3, 4): add("min_n", f"{v}", v, min_n=v)
for v in (0.0, 0.5, 1.0, 2.0, 3.0): add("power", f"{v}", v, power=v)
for v in (20, 40, 60, 80): add("dir_km", f"{v} km", v, dir_km=v)
add("weights", "skill (1/MSE, tuned on A)", 1, weights="skill")
# --- structure
add("structure", "no ln(x/h)", 1, use_lnxh=False)
add("structure", "no lnU interaction", 2, interaction=False)
add("structure", "+ cross-road |sin α|", 3, extra=("cross",))
add("structure", "+ cross-road × tree", 4, extra=("cross", "cross_tree"))
add("structure", "+ stability (T_surf−T_air)", 5, extra=("stab",))
add("structure", "+ stability × tree", 6, extra=("stab", "stab_tree"))
add("structure", "+ night × tree", 7, extra=("night", "night_tree"))
add("structure", "+ terrain Sx, TPI", 8, extra=("sx", "tpi"))
add("structure", "+ h_mid", 9, extra=("hmid",))
add("structure", "lnU of the pair (not network)", 10, extra=("lnU_pair",))
R = pd.DataFrame(rows)
cfg.paths.out.mkdir(parents=True, exist_ok=True)
R.to_csv(cfg.paths.out / "sensitivity.csv", index=False)
pickle.dump(dict(keep=keep, base=base["B"], hours=E.H, A=E.A, trv=list(E.TM.index)), open(cfg.paths.out / "sensitivity_detail.pkl", "wb"))
pd.set_option("display.width", 250)
print(R[["group", "label", "A_mae_mean", "A_dmae", "A_dmae_se", "B_mae_mean", "B_dmae", "B_dmae_se", "B_mae_med", "B_dir_med", "B_bias_log"]].round(4).to_string())
