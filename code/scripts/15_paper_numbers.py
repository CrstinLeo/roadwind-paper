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

# share of Norwegian hourly speeds removed by the quality control (all Frost stations of the region, raw download)
import glob
from smhi_trv_wind.data import frost
_raw = pd.concat([pd.read_parquet(p_) for p_ in sorted(glob.glob(str(Path(os.environ.get("SWW_NO_DATA", "/home/claude/no")) / "raw" / "*.parquet")))])
_raw = _raw.rename(columns={"wind_speed": "ws", "wind_from_direction": "wd"}).drop_duplicates(["id", "time"]).sort_values(["id", "time"])
_n0 = int(_raw.ws.notna().sum()); _q = frost.qc(_raw)
N["noQcPct"] = f"{100 * (1 - _q.ws.notna().sum() / _n0):.1f}"

# ---------------- ladders (pairs <= 80 km)
X = pickle.load(open(NO / "norway_extra.pkl", "rb"))          # 16_norway_ladder_terrain.py
L = X["ladder"]; L = L[L.dist <= 80]
N["noPairsEighty"] = len(L)
for k, s in (("Raw", "U0"), ("Rough", "U2"), ("Site", "U3"), ("Expo", "U4")):
    N[f"noLad{k}"] = f2(L[f"mae_{s}"].median()); N[f"noLad{k}Bias"] = f"{100 * L[f'rbias_{s}'].median():+.0f}"
lad = evaluate.ladder(SEres.P)
for i, k in enumerate(["Raw", "Height", "Rough", "Site", "Expo"]):
    N[f"seLad{k}"] = f2(lad.mae.iloc[i]); N[f"seLad{k}Bias"] = f"{100 * lad.rbias.iloc[i]:+.0f}"
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

# unconditional bias of the cross-validated estimate: median over stations of mean(U)/mean(U_road) - 1, and the
# conditional ratio mean(U_road)/mean(U) - 1 for hours with an estimate >= 8 m/s (pooled)
def rbias(U, ws):
    m = np.isfinite(U) & np.isfinite(ws)
    st = np.nansum(np.where(m, U, np.nan), 0) / np.nansum(np.where(m, ws, np.nan), 0) - 1
    h = m & (U >= 8)
    return np.nanmedian(st), ws[h].sum() / U[h].sum() - 1
for pre, P_, O_ in (("no", R["pred_final"], R["obs"]), ("se", SEcv["pred"], SEcv["obs"])):
    b, hb = rbias(P_["U"].astype(float), O_["ws"].astype(float))
    assert hb < 0, "text says 'lower than estimated'"
    N[f"{pre}CvRbias"] = f"{100 * b:+.0f}"; N[f"{pre}CvHighObs"] = f"{-100 * hb:.0f}"

# roadside stations where the corrected estimate is worse than the nearest station (Norway)
_S = pd.read_csv(Path(os.environ.get("SWW_NO_DATA", "/home/claude/no")) / "stations_region.csv"); _S["id"] = _S.id.astype(str)
_nm = _S.set_index("id").name.reindex(R["meta"]["trv"]).str.title().str.replace("Fv", "Fv~", regex=False).values
_w = np.where(c["mae"] > c["mae_raw"])[0]
N["noWorseList"] = "; ".join(f"{_nm[i].replace('Fv~', 'Fv ')} ({c['mae_raw'][i]:.2f}\\,$\\rightarrow$\\,{c['mae'][i]:.2f})" for i in _w[np.argsort(-(c["mae"] - c["mae_raw"])[_w])])
N["noWorseN"] = len(_w)

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

N["noTreeFiveHundred"], N["noTreeFiveHundredSE"] = scan("tree_R", "500 m")
N["noKmOneTwenty"], N["noKmOneTwentySE"] = scan("km", "120 km")

# ---------------- terrain indices (Norway, final model, 16_norway_ladder_terrain.py)
T = X["terrain"]; dts = [paired(v["mae"], T["final"]["mae"]) for k, v in T.items() if k != "final"]
N.update(noTerrMin=f"{min(d for d, _ in dts):+.3f}", noTerrMax=f"{max(d for d, _ in dts):+.3f}",
         noTerrSEmax=f"{max(s for _, s in dts):.3f}", noTerrN=len(dts))

# ---------------- Sweden scans (relative to the round-6 reference configuration, 08_sensitivity)
SS = pickle.load(open(SE / "sensitivity_cv.pkl", "rb"))["res"]
from smhi_trv_wind.sensitivity import Spec
def se_d(g, l):
    base_ = [v for k, v in SS.items() if k[0] == g and v["spec"] == Spec()]
    b_ = base_[0]["r"]["mae"] if base_ else SS[(g, "±0 m")]["r"]["mae"]
    return paired(SS[(g, l)]["r"]["mae"], b_)
phys = [abs(se_d(g, l)[0]) for g in ("z_blend", "c_tree", "L_smhi", "L_trv") for (gg, l) in SS if gg == g]
N["sePhysMax"] = f"{max(phys):.3f}"
d_, s_ = se_d("site", "none"); N["seSiteNone"], N["seSiteNoneSE"] = f"{d_:+.3f}", f"{s_:.3f}"
for lab, key in (("±100 m", "Two"), ("±250 m", "Five"), ("±500 m", "Thousand")):
    d_, s_ = se_d("road_window", lab); N[f"seSeg{key}"], N[f"seSeg{key}SE"] = f"{d_:+.3f}", f"{s_:.3f}"

# ---------------- along-road correlation of the exposure features (both countries)
offs = (100, 250, 1000)
cors = {o: [] for o in offs}
for path in (Path(os.environ.get("SWW_SE_DATA", "/home/claude/se")) / "features_variants.parquet",
             Path(os.environ.get("SWW_NO_DATA", "/home/claude/no")) / "features_variants.parquet"):
    Vf = pd.read_parquet(path); Vf = Vf[Vf.src == "trv"]
    for col in ("tree", "lnz0"):
        for o in offs:
            for sg in (o, -o):
                a, b = Vf[f"along_{col}_s0"].values, Vf[f"along_{col}_s{sg}"].values
                ok = np.isfinite(a) & np.isfinite(b)
                cors[o].append(np.corrcoef(a[ok], b[ok])[0, 1])
for o, k in zip(offs, ("Hundred", "TwoFifty", "Km")):
    N[f"along{k}Min"], N[f"along{k}Max"] = f"{min(cors[o]):.2f}", f"{max(cors[o]):.2f}"

# ---------------- direction variant with an untreed channelling term (20_direction_variant.py)
for pre, path in (("no", NO), ("se", SE)):
    Dv = pickle.load(open(path / f"dir_open_{pre.upper()}.pkl", "rb"))
    d_, s_ = paired(Dv["open"]["dir"], Dv["base"]["dir"])
    N[f"{pre}DirOpen"], N[f"{pre}DirOpenSE"] = f"{d_:+.1f}", f"{s_:.1f}"


# ---------------- air-drag energy (22_energy.py): VECTO crosswind correction, C_dA(0) = 5.8 m2, 80 km/h
EN = pickle.load(open(NO / "energy.pkl", "rb"))
BET_TOTAL = 96.0          # kWh/100 km, mean real-world consumption of battery-electric trucks (Le Corguille et al. 2025)
N["enEzero"] = f"{EN[('Norway', 80, 'vecto')]['E0']:.1f}"
N["enEzeroSixty"], N["enEzeroNinety"] = (f"{EN[('Norway', k, 'vecto')]['E0']:.0f}" for k in (60, 90))
N["enBetTotal"] = f"{BET_TOTAL:.0f}"
for c, pre in (("Norway", "No"), ("Sweden", "Se")):
    for model, suf in (("vecto", ""), ("model1", "MOne")):
        o = EN[(c, 80, model)]; E0 = o["E0"]; m = o["metrics"]
        N[f"enAdd{pre}{suf}"] = f"{100 * (np.mean(o['true_one']) / E0 - 1):.1f}"
        N[f"enSs{pre}Cor{suf}"] = f"{m['corrected']['ss_one']:.2f}"; N[f"enSs{pre}Raw{suf}"] = f"{m['nearest station']['ss_one']:.2f}"
        N[f"enBias{pre}Raw{suf}"] = f"{100 * m['nearest station']['mean_bias_med']:+.1f}"
        if model != "vecto":
            continue
        N[f"enAddKwh{pre}"] = f"{np.mean(o['true_one']) - E0:.1f}"
        N[f"enAddShareTotal{pre}"] = f"{100 * (np.mean(o['true_one']) - E0) / BET_TOTAL:.0f}"
        N[f"enHead{pre}"] = f"{100 * (np.mean(o['head_only_one']) / E0 - 1):.1f}"
        N[f"enCross{pre}"] = f"{100 * (np.mean(o['cross_only']) / E0 - 1):.1f}"
        q = np.percentile(100 * (o["true_one"] / E0 - 1), [5, 95]); N[f"enHourQfive{pre}"], N[f"enHourQninetyfive{pre}"] = f"{q[0]:.0f}", f"{q[1]:.0f}"
        qr = np.percentile(100 * (o["true_rt"] / E0 - 1), 95); N[f"enRtQninetyfive{pre}"] = f"{qr:.0f}"
        for k, nm in (("no wind", "None"), ("nearest station", "Raw"), ("corrected", "Cor")):
            N[f"enMae{pre}{nm}"] = f"{m[k]['mae_one']:.1f}"; N[f"enMaeRt{pre}{nm}"] = f"{m[k]['mae_rt']:.1f}"
            N[f"enBias{pre}{nm}"] = f"{100 * m[k]['mean_bias_med']:+.1f}"; N[f"enStrong{pre}{nm}"] = f"{m[k]['strong_mae']:.1f}"
        # long-term wind penalty of each road (mean over hours and both directions), true vs estimated
        df = pd.DataFrame(dict(s=m["corrected"]["station"], t=o["true_one"], c=m["corrected"]["one_est"], r=m["nearest station"]["one_est"]))
        pen = 100 * (df.groupby("s").mean() / E0 - 1)
        qq = np.percentile(pen.t, [0, 50, 90, 100])
        N.update({f"enRoadMin{pre}": f"{qq[0]:.0f}", f"enRoadMed{pre}": f"{qq[1]:.0f}", f"enRoadPninety{pre}": f"{qq[2]:.0f}",
                  f"enRoadMax{pre}": f"{qq[3]:.0f}", f"enRoadPninetyKwh{pre}": f"{qq[2] / 100 * E0:.0f}"})
        for k in ("c", "r"):
            nm = "Cor" if k == "c" else "Raw"
            N[f"enRoadCorr{pre}{nm}"] = f"{np.corrcoef(pen.t, pen[k])[0, 1]:.2f}"
            N[f"enRoadMae{pre}{nm}"] = f"{np.mean(np.abs(pen[k] - pen.t)):.1f}"
            N[f"enRoadMean{pre}{nm}"] = f"{pen[k].mean():.1f}"
        N[f"enRoadMean{pre}True"] = f"{pen.t.mean():.1f}"
    for v_kmh, w in ((60, "Sixty"), (90, "Ninety")):
        o = EN[(c, v_kmh, "vecto")]
        N[f"enAdd{pre}{w}"] = f"{100 * (np.mean(o['true_one']) / o['E0'] - 1):.0f}"
        N[f"enSs{pre}Cor{w}"] = f"{o['metrics']['corrected']['ss_one']:.2f}"
for model, suf in (("vecto", ""), ("model1", "MOne")):
    VE = EN[("vehicle", model)]
    for k, nm in (("no wind", "None"), ("nearest MET/Avinor station", "Ref"), ("nearest road weather station", "Road"), ("MET Nordic", "Mn"), ("corrected", "Cor")):
        N[f"enVeh{nm}{suf}"] = f"{VE[k]['mae_hour']:.1f}"; N[f"enVehBias{nm}{suf}"] = f"{100 * VE[k]['bias']:+.0f}"
    N[f"enVehTrue{suf}"] = f"{VE['true_mean']:.1f}"

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
N.update(vehN=len(V), vehNDovre=int((V.site == "dovre").sum()), vehNFroya=int((V.site == "froya").sum()),
         vehHours=int(V.t.dt.floor("h").nunique()), vehHoursDovre=int(V[V.site == "dovre"].t.dt.floor("h").nunique()),
         vehHoursFroya=int(V[V.site == "froya"].t.dt.floor("h").nunique()),
         vehFroyaKmMin=f"{min(V[V.site == 'froya'].ref_km.min(), V[V.site == 'froya'].road_km.min()):.0f}")
# speed vs direction: drag error with one of the two components replaced by the measurement or by MET Nordic
for key, u, d in (("ModelSpeedMnDir", "model", "metnordic_dir"), ("ModelSpeedTrueDir", "model", "dir_veh"),
                  ("MnSpeedTrueDir", "metnordic", "dir_veh"), ("TrueSpeedModelDir", "w_veh", "model_dir"), ("MnSpeedModelDir", "metnordic", "model_dir")):
    e = drag.relative_error(80 / 3.6, V.w_veh.values, V.dir_veh.values, V[u].values, V[d].values, V.heading.values)
    N[f"vehDrag{key}"] = f"{100 * np.nanmedian(np.abs(e)):.0f}"
for c_ in ("model", "metnordic", "ref_raw", "road_raw"):          # overall speed bias over the hours of Dovre, 31 Oct evening
    g = V[(V.site == "dovre") & (V.t.dt.day == 31)]
    N[f"vehDovreEve{c_.replace('_', '').title()}"] = f"{100 * (g[c_].mean() / g.w_veh.mean() - 1):+.0f}"
with open(GEN / "tab_vehicle.tex", "w") as f:
    f.write("\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.3\\textwidth}ccccc}\n\\toprule\n & \\multicolumn{3}{c}{wind-speed MAE [m\\,s$^{-1}$]} & direction & drag error \\\\\n")
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
    f.write("Term & estimate & range & estimate & range \\\\\n\\midrule\n")
    for i, n_ in enumerate(names):
        fmt = (lambda x: f"${x:+.1f}$") if i >= 5 else (lambda x: f"${x:+.3f}$")
        f.write(f"{n_} & {fmt(no_v[i])} & {fmt(no_f[:, i].min())}\\,…\\,{fmt(no_f[:, i].max())} & {fmt(se_v[i])} & {fmt(se_f[:, i].min())}\\,…\\,{fmt(se_f[:, i].max())} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
N.update(noSmear=f2(R["coef"]["smear"]), seSmear=f2(SEcv["coef"]["smear"]),
         noFarFortyPct=f"{100 * (1 - np.exp(0.4 * R['coef']['b'][4])):.0f}", noFarRatio=f"{R['coef']['b'][4] / SEcv['coef']['b'][4]:.1f}",
         noLadRawBiasAbs=f"{100 * L['rbias_U0'].median():.0f}", seLadRawBiasAbs=f"{100 * lad.rbias.iloc[0]:.0f}")

# ---------------- skill table (both countries)
with open(GEN / "tab_skill.tex", "w") as f:
    f.write("\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.3\\textwidth}cccccc}\n\\toprule\n & \\multicolumn{3}{c}{Norway} & \\multicolumn{3}{c}{Sweden} \\\\\n")
    f.write(" & raw & model & reduction & raw & model & reduction \\\\\n\\midrule\n")
    f.write(f"Speed MAE [m\\,s$^{{-1}}$] & {N['noCvRaw']} & {N['noCvMed']} & {N['noCvRed']}\\,\\% & {N['seCvRaw']} & {N['seCvMed']} & {N['seCvRed']}\\,\\% \\\\\n")
    f.write(f"Direction MAE [deg] & {N['noCvDirRaw']} & {N['noCvDir']} & & {N['seCvDirRaw']} & {N['seCvDir']} & \\\\\n")
    f.write(f"Stations improved & & {N['noCvBetter']}/{N['noTrv']} & & & {N['seCvBetter']}/{N['seTrv']} & \\\\\n")
    f.write(f"Drag error [\\%] & {N['noDragRaw']} & {N['noDragModel']} & & {N['seDragRaw']} & {N['seDragModel']} & \\\\\n")
    f.write(f"Leaf-on speed MAE [m\\,s$^{{-1}}$] & {N['noOnRaw']} & {N['noOnMed']} & {N['noOnRed']}\\,\\% & & & \\\\\n")
    f.write(f"Leaf-off speed MAE [m\\,s$^{{-1}}$] & {N['noOffRaw']} & {N['noOffMed']} & {N['noOffRed']}\\,\\% & & & \\\\\n")
    f.write(f"Other country's coefficients [m\\,s$^{{-1}}$] & & {N['noWithSe']} & & & {N['seWithNo']} & \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(GEN / "numbers.tex", "w") as f:
    f.write("% generated by code/scripts/15_paper_numbers.py -- do not edit by hand\n")
    for k, v in N.items():
        v = str(v)
        if v in ("-0.000", "+0.000", "-0", "+0"):
            v = v[1:]
        if v.startswith("-"):
            v = "$-$" + v[1:]                     # typographic minus in text mode
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in N.items()))
