"""Numbers and tables of the two-country manuscript -> <repo>/sections/generated/

Inputs: SWW_NO_OUT/norway_results.pkl, SWW_NO_OUT/vehicle_validation_rows.parquet (Norway, scripts 11, 12),
        SWW_SE_OUT/cv_full.pkl, SWW_SE_OUT/result.pkl (Sweden, scripts 14, 04).
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import drag, evaluate
from smhi_trv_wind.geo import wrap180

NO = Path(os.environ.get("SWW_NO_OUT", "/tmp/nout2"))
SE = Path(os.environ.get("SWW_SE_OUT", "/tmp/sww8"))
GEN = Path(os.environ.get("SWW_GEN", Path(__file__).resolve().parents[2] / "sections" / "generated"))
GEN.mkdir(parents=True, exist_ok=True)
R = pickle.load(open(NO / "norway_results.pkl", "rb"))
SEcv = pickle.load(open(SE / "cv_full.pkl", "rb"))
SEres = pickle.load(open(SE / "result.pkl", "rb"))
N = {}


def paired(a, b):
    d = a - b
    ok = np.isfinite(d)
    return d[ok].mean(), d[ok].std(ddof=1) / np.sqrt(ok.sum())


f2 = lambda x: f"{x:.2f}"
pc = lambda x: f"{100 * x:.0f}"

# ---------------- data
m = R["meta"]
N.update(noHours=f"{m['hours']:,}".replace(",", "\\,"), noTrv=m["n_trv"], noRef=m["n_ref"], noPairs=m["n_pairs"],
         noStart=m["start"][:10], noEnd=m["end"][:10], seHours=SEcv["hours"], seTrv=SEcv["n_trv"], seRef=SEres.smhi.shape[0])

# ---------------- ladders (pairs <= 80 km)
L = R["ladder"]; L = L[L.dist <= 80]
N["noPairsEighty"] = len(L)
for k, s in (("Raw", "U0"), ("Rough", "U2"), ("Site", "U3"), ("Expo", "U4")):
    N[f"noLad{k}"] = f2(L[f"mae_{s}"].median()); N[f"noLad{k}Bias"] = f"{100 * L[f'bias_{s}'].median():+.0f}"
lad = evaluate.ladder(SEres.P)
for i, k in enumerate(["Raw", "Height", "Rough", "Site", "Expo"]):
    N[f"seLad{k}"] = f2(lad.mae.iloc[i]); N[f"seLad{k}Bias"] = f"{100 * lad.bias.iloc[i]:+.0f}"
N["sePairsEighty"] = int((SEres.P.dist <= 80).sum())

# ---------------- cross-validation
c, c0 = R["cv_final"], R["cv_ref6"]
d, s = paired(c["mae"], c0["mae"])
N.update(noCvMed=f2(np.nanmedian(c["mae"])), noCvMean=f2(np.nanmean(c["mae"])), noCvRaw=f2(np.nanmedian(c["mae_raw"])),
         noCvRed=pc(1 - np.nanmedian(c["mae"]) / np.nanmedian(c["mae_raw"])), noCvBetter=int(np.sum(c["mae"] < c["mae_raw"])),
         noCvDir=f"{np.nanmedian(c['dir']):.0f}", noCvDirRaw=f"{np.nanmedian(c['dir_raw']):.0f}",
         noCvBias=f"{100 * (np.exp(np.nanmedian(c['bias_log'])) - 1):.0f}", noRefSixDiff=f"{-d:.3f}", noRefSixSE=f"{s:.3f}")
for g, k in (("leaf-on (May-Sep)", "On"), ("leaf-off (Nov-Mar)", "Off")):
    N[f"no{k}Med"] = f2(np.nanmedian(c[g]["mae"])); N[f"no{k}Raw"] = f2(np.nanmedian(c[g]["mae_raw"]))
    N[f"no{k}Red"] = pc(1 - np.nanmedian(c[g]["mae"]) / np.nanmedian(c[g]["mae_raw"]))
cs, cs0 = SEcv["cv"], SEcv["cv0"]
d, s = paired(cs["mae"], cs0["mae"])
N.update(seCvMed=f2(np.nanmedian(cs["mae"])), seCvMean=f2(np.nanmean(cs["mae"])), seCvRaw=f2(np.nanmedian(cs["mae_raw"])),
         seCvRed=pc(1 - np.nanmedian(cs["mae"]) / np.nanmedian(cs["mae_raw"])), seCvBetter=int(np.sum(cs["mae"] < cs["mae_raw"])),
         seCvDir=f"{np.nanmedian(cs['dir']):.0f}", seCvDirRaw=f"{np.nanmedian(cs['dir_raw']):.0f}",
         seCvBias=f"{100 * (np.exp(np.nanmedian(cs['bias_log'])) - 1):.0f}", seRefSixDiff=f"{-d:.3f}", seRefSixSE=f"{s:.3f}")

# ---------------- site factors (Norway)
sf = np.exp(R["site_c"].mean(1))
N.update(noSfMin=f2(sf.min()), noSfMax=f2(sf.max()))

# ---------------- scans (Norway, relative to the final model)
base = c["mae"]
def scan(g, l):
    d, s = paired(R["scans"][(g, l)]["mae"], base)
    return f"{d:+.3f}", f"{s:.3f}"
for key, (g, l) in {"KmTwenty": ("km", "20 km"), "KmForty": ("km", "40 km"), "KmEighty": ("km", "80 km"), "FarNone": ("far", "none"),
                    "FarUp": ("far", "1 km upwind"), "FarFive": ("far", "500 m"), "SiteNone": ("site", "none"), "SiteSector": ("site", "sector"),
                    "NoRough": ("variants", "no roughness conversion"), "NoXh": ("variants", "no ln(x/h)"), "Cross": ("variants", "+ cross-road x tree"),
                    "Inter": ("variants", "+ lnU x tree"), "TreeTwoHundred": ("tree_R", "200 m"), "TreeFifty": ("tree_R", "50 m"),
                    "PowOne": ("power", "1.0"), "PowTwo": ("power", "2.0")}.items():
    N[f"no{key}"], N[f"no{key}SE"] = scan(g, l)

# ---------------- transfer
N.update(seWithNo=f2(np.nanmedian(R["se_with_no_coef"]["mae"])), noWithSe=f2(np.nanmedian((R["no_with_se_coef"]["B"] if "B" in R["no_with_se_coef"] else R["no_with_se_coef"])["mae"])))

# ---------------- drag at stations
pn = R["pred_final"]; ob = R["obs"]
dm, _ = drag.station_errors(ob["ws"], ob["wd"], pn["U"], pn["wdm"], ob["road"])
dr, _ = drag.station_errors(ob["ws"], ob["wd"], pn["U0n"], pn["wd0n"], ob["road"])
N.update(noDragModel=pc(np.nanmedian(dm)), noDragRaw=pc(np.nanmedian(dr)),
         seDragModel=pc(np.nanmedian(SEcv["drag"]["model_station"])), seDragRaw=pc(np.nanmedian(SEcv["drag"]["raw_station"])))

# ---------------- vehicle validation (Norway)
V = pd.read_parquet(NO / "vehicle_validation_rows.parquet")
est = [("model", "model_dir", "This study (station-based)"), ("metnordic", "metnordic_dir", "MET Nordic analysis"),
       ("ref_raw", "ref_dir", "Nearest MET/Avinor station"), ("road_raw", "road_dir", "Nearest road weather station")]
rows = []
for c_, dcol, lab in est:
    out = [lab]
    for site in ("dovre", "froya", None):
        g = V if site is None else V[V.site == site]
        ok = np.isfinite(g[c_])
        out.append(f"{np.mean(np.abs(g[c_] - g.w_veh)[ok]):.2f}")
    e = drag.relative_error(80 / 3.6, V.w_veh.values, V.dir_veh.values, V[c_].values, V[dcol].values, V.heading.values)
    ddir = np.abs(wrap180(V[dcol] - V.dir_veh))[V.w_veh >= 2]
    out += [f"{np.nanmean(ddir):.0f}", f"{100 * np.nanmedian(np.abs(e)):.0f}"]
    rows.append(out)
    N[f"veh{c_.replace('_', '').title()}"] = out[3]
    N[f"vehDrag{c_.replace('_', '').title()}"] = out[5]
N.update(vehN=len(V), vehNDovre=int((V.site == "dovre").sum()), vehNFroya=int((V.site == "froya").sum()))
with open(GEN / "tab_vehicle.tex", "w") as f:
    f.write("\\begin{tabular}{p{0.34\\textwidth}ccccc}\n\\toprule\n & \\multicolumn{3}{c}{wind-speed MAE [m\\,s$^{-1}$]} & direction & drag error \\\\\n")
    f.write("Estimate & Dovre & Frøya & all & MAE [deg] & median [\\%] \\\\\n\\midrule\n")
    for r in rows:
        f.write(" & ".join(r) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- coefficient table (both countries)
names = ["intercept $b_0$", "near trees $T_{100}$", "$\\ln(x/h)$", "$\\ln U_\\mathrm{net}$", "forest fraction $F_{1000}$",
         "direction $d_0$ [deg]", "direction $d_1$ [deg]"]
no_v = np.r_[R["coef"]["b"], R["coef"]["d"]]; no_f = R["coef_folds"]
se_v = np.r_[SEcv["coef"]["b"], SEcv["coef"]["d"]]; se_f = np.c_[SEcv["folds_b"], SEcv["folds_d"]]
with open(GEN / "tab_coefs.tex", "w") as f:
    f.write("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Norway} & \\multicolumn{2}{c}{Sweden} \\\\\n")
    f.write("Term & estimate & block range & estimate & block range \\\\\n\\midrule\n")
    for i, n_ in enumerate(names):
        fmt = (lambda x: f"{x:+.1f}") if i >= 5 else (lambda x: f"{x:+.3f}")
        f.write(f"{n_} & {fmt(no_v[i])} & {fmt(no_f[:, i].min())}\\,…\\,{fmt(no_f[:, i].max())} & {fmt(se_v[i])} & {fmt(se_f[:, i].min())}\\,…\\,{fmt(se_f[:, i].max())} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
N.update(noSmear=f2(R["coef"]["smear"]), seSmear=f2(SEcv["coef"]["smear"]))

# ---------------- skill table (both countries)
with open(GEN / "tab_skill.tex", "w") as f:
    f.write("\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{3}{c}{Norway} & \\multicolumn{3}{c}{Sweden} \\\\\n")
    f.write(" & raw & model & reduction & raw & model & reduction \\\\\n\\midrule\n")
    f.write(f"Speed MAE, median station [m\\,s$^{{-1}}$] & {N['noCvRaw']} & {N['noCvMed']} & {N['noCvRed']}\\,\\% & {N['seCvRaw']} & {N['seCvMed']} & {N['seCvRed']}\\,\\% \\\\\n")
    f.write(f"Direction MAE, median station [deg] & {N['noCvDirRaw']} & {N['noCvDir']} & & {N['seCvDirRaw']} & {N['seCvDir']} & \\\\\n")
    f.write(f"Stations improved & & {N['noCvBetter']}/{N['noTrv']} & & & {N['seCvBetter']}/{N['seTrv']} & \\\\\n")
    f.write(f"Drag error at 80 km\\,h$^{{-1}}$, median station [\\%] & {N['noDragRaw']} & {N['noDragModel']} & & {N['seDragRaw']} & {N['seDragModel']} & \\\\\n")
    f.write(f"Leaf-on (May--Sep) speed MAE [m\\,s$^{{-1}}$] & {N['noOnRaw']} & {N['noOnMed']} & {N['noOnRed']}\\,\\% & & & \\\\\n")
    f.write(f"Leaf-off (Nov--Mar) speed MAE [m\\,s$^{{-1}}$] & {N['noOffRaw']} & {N['noOffMed']} & {N['noOffRed']}\\,\\% & & & \\\\\n")
    f.write(f"Coefficients from the other country [m\\,s$^{{-1}}$] & & {N['noWithSe']} & & & {N['seWithNo']} & \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(GEN / "numbers.tex", "w") as f:
    f.write("% generated by code/scripts/15_paper_numbers.py -- do not edit by hand\n")
    for k, v in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in N.items()))
