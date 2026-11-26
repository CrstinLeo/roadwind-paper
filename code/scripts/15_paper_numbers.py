"""Numbers and tables of the two-country manuscript -> <repo>/sections/generated/

Inputs: SWW_NO_OUT/ norway_results.pkl (11), norway_extra.pkl (16), vehicle_validation_rows.parquet (12),
        review_NO.pkl (23), baselines.pkl (24), energy.pkl (22), misc_NO.pkl (27, optional);
        SWW_SE_OUT/ cv_full.pkl (14), result.pkl (04), review_SE.pkl (26).
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from smhi_trv_wind import drag, evaluate
from smhi_trv_wind.geo import wrap180

NO = Path(os.environ.get("SWW_NO_OUT", "output/no"))
SE = Path(os.environ.get("SWW_SE_OUT", "output/se"))
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
_raw = pd.concat([pd.read_parquet(p_) for p_ in sorted(glob.glob(str(Path(os.environ.get("SWW_NO_DATA", "data/no")) / "raw" / "*.parquet")))])
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
_S = pd.read_csv(Path(os.environ.get("SWW_NO_DATA", "data/no")) / "stations_region.csv"); _S["id"] = _S.id.astype(str)
_nm = _S.set_index("id").name.reindex(R["meta"]["trv"]).str.title().str.replace("Fv", "Fv~", regex=False).values
_w = np.where(c["mae"] > c["mae_raw"])[0]
N["noWorseList"] = "; ".join(f"{_nm[i].replace('Fv~', 'Fv ')} ({c['mae_raw'][i]:.2f}\\,$\\rightarrow$\\,{c['mae'][i]:.2f})" for i in _w[np.argsort(-(c["mae"] - c["mae_raw"])[_w])])
N["noWorseN"] = len(_w)

# ---------------- road axis snapping (OSRM, features/road.py)
_rb = pd.read_csv(Path(os.environ.get("SWW_NO_DATA", "data/no")) / "road_bearing.csv", dtype={"id": str}).set_index("id").reindex(R["meta"]["trv"])
N.update(noSnapMed=f"{_rb.snap_m.median():.0f}", noSnapMax=f"{_rb.snap_m.max():.0f}", noSnapOverThirty=int((_rb.snap_m > 30).sum()))

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

for key, (g, l) in {"NoTree": ("tree_R", "none"), "TreeThreeHundred": ("tree_R", "300 m"), "FarThree": ("far", "300 m"),
                    "Ring": ("far", "1 km + ring 300 m")}.items():
    if (g, l) in R["scans"]:
        N[f"no{key}"], N[f"no{key}SE"] = scan(g, l)

# ---------------- honest hold-out, selection on 2024, coefficient uncertainty (23_norway_review.py)
RV = pickle.load(open(NO / "review_NO.pkl", "rb"))
h = RV["holdout2025"]["B"]
N.update(noHoldMed=f2(np.nanmedian(h["mae"])), noHoldRaw=f2(np.nanmedian(h["mae_raw"])),
         noHoldRed=pc(1 - np.nanmedian(h["mae"]) / np.nanmedian(h["mae_raw"])))
S24 = RV["select2024"]; f24 = S24["final"]["mae"]
dd = {k: paired(v["mae"], f24) for k, v in S24.items() if k != "final"}
best = min(dd, key=lambda k: dd[k][0])
N["noSelBest"] = best; N["noSelBestD"], N["noSelBestSE"] = f"{dd[best][0]:+.3f}", f"{dd[best][1]:.3f}"
N["noSelWithinSE"] = "yes" if all(d > -s_ for d, s_ in dd.values()) else "no"
rw = RV["rows"]; rng_ = np.random.default_rng(0); nst = len(rw["XtX"])
Bb = np.array([np.linalg.solve(rw["XtX"][i].sum(0), rw["Xty"][i].sum(0)) for i in (rng_.integers(0, nst, nst) for _ in range(2000))])
COEF_CI = {"Norway": np.percentile(Bb, [2.5, 97.5], 0)}
N["noVifMax"] = f"{rw['vif'].max():.1f}"; N["noCorrTreeXh"] = f2(rw["corr"][0, 1]); N["noCorrTreeFar"] = f2(rw["corr"][0, 3])
_S0 = pd.read_csv(Path(os.environ.get("SWW_NO_DATA", "data/no")) / "stations_region.csv"); _S0["id"] = _S0.id.astype(str); _S0 = _S0.set_index("id")
_nmr = _S0.name.reindex(rw["ids"]).str.lower().fillna("")
import re as _re
bridge = np.array([bool(_re.search(r"bru(a)?\b|atlanterhav", n_)) for n_ in _nmr])          # bridges and the sea causeway
keep_ = ~bridge
b_nb = np.linalg.solve(rw["XtX"][keep_].sum(0), rw["Xty"][keep_].sum(0))
N["noBridgeN"] = int(bridge.sum()); N["noBridgeList"] = ", ".join(_S0.name.reindex(np.array(rw["ids"])[bridge]).str.title())
for i_, k_ in enumerate(("Const", "Tree", "Xh", "Unet", "Far")):
    N[f"noB{k_}"] = f"{rw['b'][i_]:+.2f}"; N[f"noB{k_}NoBridge"] = f"{b_nb[i_]:+.2f}"
    N[f"noB{k_}Lo"], N[f"noB{k_}Hi"] = (f"{x:+.2f}" for x in COEF_CI["Norway"][:, i_])
if (SE / "rows_SE.pkl").exists():
    rs = pickle.load(open(SE / "rows_SE.pkl", "rb")); nse = len(rs["XtX"])
    Bs = np.array([np.linalg.solve(rs["XtX"][i].sum(0), rs["Xty"][i].sum(0)) for i in (rng_.integers(0, nse, nse) for _ in range(2000))])
    COEF_CI["Sweden"] = np.percentile(Bs, [2.5, 97.5], 0); N["seVifMax"] = f"{rs['vif'].max():.1f}"
    for i_, k_ in enumerate(("Const", "Tree", "Xh", "Unet", "Far")):
        N[f"seB{k_}Lo"], N[f"seB{k_}Hi"] = (f"{x:+.2f}" for x in COEF_CI["Sweden"][:, i_])

# ---------------- smaller checks (27_norway_misc_checks.py)
if (NO / "misc_NO.pkl").exists():
    MI = pickle.load(open(NO / "misc_NO.pkl", "rb"))
    _pd = pd.DataFrame({k: MI["persistence"][k] for k in ("c2023", "c2024", "c2025")}).dropna().corr()
    N["noSfPersistA"], N["noSfPersistB"] = f2(_pd.loc["c2023", "c2024"]), f2(_pd.loc["c2023", "c2025"])
    tn = MI["turn"]
    N.update(noTurnAll=f"{tn['all open']['mean']:+.0f}", noTurnDay=f"{tn['summer day (10-14 UTC)']['mean']:+.0f}",
             noTurnNight=f"{tn['summer night (22-02 UTC)']['mean']:+.0f}", noTurnLowU=f"{tn['U0 3-6']['mean']:+.0f}",
             noTurnHighU=f"{tn['U0 >= 10']['mean']:+.0f}", noTurnOpen=f"{tn['F1000 < 0.15']['mean']:+.0f}",
             noTurnForest=f"{tn['F1000 >= 0.3']['mean']:+.0f}")
    dn = MI["daynight"]
    N.update(noUnetDay=f2(dn["summer day"]["b"][3]), noUnetNight=f2(dn["summer night"]["b"][3]), noUnetWinter=f2(dn["winter"]["b"][3]))
    tr_ = MI["terrain"]
    N.update(noFarWithTerrain=f2(tr_["terrain + F1000"][-1]), noFarFinal=f2(tr_["final"][4]))

# ---------------- terrain indices (Norway, final model, 16_norway_ladder_terrain.py)
T = X["terrain"]; dts = [paired(v["mae"], T["final"]["mae"]) for k, v in T.items() if k != "final"]
N.update(noTerrMin=f"{min(d for d, _ in dts):+.3f}", noTerrMax=f"{max(d for d, _ in dts):+.3f}",
         noTerrSEmax=f"{max(s for _, s in dts):.3f}", noTerrN=len(dts))

# ---------------- Sweden scans around the FINAL model on the full record (26_sweden_review.py)
RS = pickle.load(open(SE / "review_SE.pkl", "rb"))
def se_d(g, l):
    return paired(RS["scans"][(g, l)]["mae"], RS["final"]["mae"])
phys = [se_d(g, l) for (g, l) in RS["scans"] if g in ("z_blend", "c_tree", "L_smhi", "L_trv")]
N["sePhysMax"] = f"{max(abs(d) for d, _ in phys):.3f}"
for key, (g, l) in {"SiteNone": ("site", "none"), "SiteSector": ("site", "sector"), "NoTree": ("tree_R", "none"),
                    "TreeTwoHundred": ("tree_R", "200 m"), "TreeThreeHundred": ("tree_R", "300 m"), "TreeFifty": ("tree_R", "50 m"),
                    "FarNone": ("far", "none"), "FarThree": ("far", "300 m all dir."), "FarFive": ("far", "500 m all dir."),
                    "FarUp": ("far", "1 km upwind"), "KmForty": ("km", "40 km"), "KmEighty": ("km", "80 km"),
                    "SegTwo": ("road_window", "+-100 m"), "SegFive": ("road_window", "+-250 m"), "SegThousand": ("road_window", "+-500 m"),
                    "SegZero": ("road_window", "+-0 m"), "Inter": ("structure", "+ lnU x tree"), "Stab": ("structure", "+ stability x tree"),
                    "CtreeLow": ("c_tree", "0.05")}.items():
    d_, s_ = se_d(g, l); N[f"se{key}"], N[f"se{key}SE"] = f"{d_:+.3f}", f"{s_:.3f}"
N["seFinalMean"] = f2(np.nanmean(RS["final"]["mae"]))
for lab, key in (("+-100 m", "Two"), ("+-250 m", "Five"), ("+-500 m", "Thousand")):     # segment averages vs point features
    d_, s_ = paired(RS["scans"][("road_window", lab)]["mae"], RS["scans"][("road_window", "+-0 m")]["mae"])
    N[f"seSeg{key}"], N[f"seSeg{key}SE"] = f"{d_:+.3f}", f"{s_:.3f}"

# ---------------- along-road correlation of the exposure features (both countries)
offs = (100, 250, 1000)
cors = {o: [] for o in offs}
for path in (Path(os.environ.get("SWW_SE_DATA", "data/se")) / "features_variants.parquet",
             Path(os.environ.get("SWW_NO_DATA", "data/no")) / "features_variants.parquet"):
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


# ---------------- air-drag energy (22_energy.py): VECTO, C_dA(0) = 5.8 m2, 80 km/h, stations with direction coverage
EN = pickle.load(open(NO / "energy.pkl", "rb"))
BET_TOTAL = 96.0          # kWh/100 km, mean real-world battery-side consumption of battery-electric trucks (Le Corguille et al. 2025)
ETA_DRIVE = 0.85          # battery-to-wheel efficiency used to compare wheel energy with battery-side consumption
f1 = lambda x: f"{x:.1f}"
NM = {"no wind": "None", "nearest station": "Raw", "rescaled nearest": "Resc", "nearest road station": "Road",
      "corrected": "Cor", "calibrated": "Cal", "MET Nordic": "Mn"}
N["enEzero"] = f1(EN["Norway"][("vecto", 80)]["sums"]["E0"]); N["enBetTotal"] = f"{BET_TOTAL:.0f}"; N["enEta"] = f"{ETA_DRIVE:.2f}"
for c, pre in (("Norway", "No"), ("Sweden", "Se")):
    o = EN[c]; B = o[("vecto", 80)]; pt, ci = B["point"], B["ci"]; E0 = B["sums"]["E0"]
    N[f"en{pre}Stations"] = o["n_station"]
    N[f"enAdd{pre}"] = f1(pt["pen_mean"]); N[f"enAdd{pre}Lo"], N[f"enAdd{pre}Hi"] = (f1(x) for x in ci["pen_mean"])
    N[f"enAddKwh{pre}"] = f1(pt["pen_mean"] / 100 * E0)
    N[f"enAddShareTotal{pre}"] = f"{pt['pen_mean'] / 100 * E0 / ETA_DRIVE / BET_TOTAL * 100:.0f}"
    N[f"enHead{pre}"], N[f"enCross{pre}"] = f1(pt["pen_head"]), f1(pt["pen_cross"])
    N[f"enStrongFrac{pre}"] = f"{100 * pt['strong_frac']:.0f}"; N[f"enLargeFrac{pre}"] = f"{100 * pt['large_frac']:.0f}"
    q = np.percentile(pt["pen_station"], [0, 50, 90, 100])
    N.update({f"enRoadMin{pre}": f1(q[0]), f"enRoadMed{pre}": f"{q[1]:.0f}", f"enRoadPninety{pre}": f"{q[2]:.0f}",
              f"enRoadMax{pre}": f"{q[3]:.0f}", f"enRoadPninetyKwh{pre}": f"{q[2] / 100 * E0:.0f}"})
    N[f"enRoadMean{pre}True"] = f1(np.mean(pt["pen_station"]))
    for k, nm in NM.items():
        if k not in pt:
            continue
        m, cc = pt[k], ci[k]
        N[f"enMae{pre}{nm}"] = f1(m["mae"]); N[f"enMae{pre}{nm}Lo"], N[f"enMae{pre}{nm}Hi"] = (f1(x) for x in cc["mae"])
        N[f"enMaeRt{pre}{nm}"] = f1(m["mae_rt"])
        N[f"enSs{pre}{nm}"] = f2(m["skill"]); N[f"enSs{pre}{nm}Lo"], N[f"enSs{pre}{nm}Hi"] = (f2(x) for x in cc["skill"])
        N[f"enBias{pre}{nm}"] = f"{m['bias_med']:+.1f}"; N[f"enStrong{pre}{nm}"] = f1(m["strong_mae"])
        N[f"enRoadMean{pre}{nm}"] = f1(np.mean(m["pen_station"])); N[f"enRoadMae{pre}{nm}"] = f1(m["pen_mae"])
        if k != "no wind":
            N[f"enRoadCorr{pre}{nm}"] = f2(m["corr"]); N[f"enRoadCorr{pre}{nm}Lo"], N[f"enRoadCorr{pre}{nm}Hi"] = (f2(x) for x in cc["corr"])
    N[f"enMaeGain{pre}"] = f"{100 * (1 - pt['corrected']['mae'] / pt['no wind']['mae']):.0f}"
    N[f"enMaeGainRaw{pre}"] = f"{100 * (1 - pt['corrected']['mae'] / pt['nearest station']['mae']):.0f}"
    N[f"enStrongGain{pre}"] = f"{100 * (1 - pt['corrected']['strong_mae'] / pt['no wind']['strong_mae']):.0f}"
    for (a_, b_, kk), (d_, cci) in B["diff"].items():
        key = f"enD{pre}{NM[a_]}{NM[b_]}{kk.title().replace('_', '')}"
        N[key] = f2(d_) if kk in ("corr", "skill") else f1(d_)
        N[key + "Lo"], N[key + "Hi"] = ((f2 if kk in ("corr", "skill") else f1)(x) for x in cci)
    if "metnordic_hours" in o:
        M = o["metnordic_hours"]
        for k in ("MET Nordic", "corrected", "no wind", "nearest road station", "nearest station"):
            nm = NM[k]
            N[f"enMnh{pre}Mae{nm}"] = f1(M["point"][k]["mae"]); N[f"enMnh{pre}Ss{nm}"] = f2(M["point"][k]["skill"])
            N[f"enMnh{pre}Ss{nm}Lo"], N[f"enMnh{pre}Ss{nm}Hi"] = (f2(x) for x in M["ci"][k]["skill"])
            if k != "no wind":
                N[f"enMnh{pre}Corr{nm}"] = f2(M["point"][k]["corr"])
        N[f"enMnh{pre}Hours"] = f"{o['metnordic_n_hours']:,}".replace(",", "\\,")
    band = o["band"]
    hb = [v for k, v in band.items() if k not in ("model1",) and not k.startswith("TI")]
    N[f"enBand{pre}Lo"], N[f"enBand{pre}Hi"] = f1(min(hb)), f1(max(hb))
    N[f"enBand{pre}Disp"], N[f"enBand{pre}NoRed"] = f1(band["z0 0.3 d 2"]), f1(band["no reduction"])
    N[f"enBand{pre}ZHigh"] = f1(band["z0 0.3"]); N[f"enBand{pre}ZLow"] = f1(band["z0 0.01"])
    if "sensor 6 m" in band:
        N[f"enBand{pre}Six"] = f1(band["sensor 6 m"])
    N[f"enBand{pre}TiLow"], N[f"enBand{pre}TiHigh"] = f1(band["TI 0.15"]), f1(band["TI 0.3"])
    N[f"enAdd{pre}MOne"] = f1(band["model1"])
    N[f"enCapPct{pre}"] = f"{100 * o['beta_cap_frac']:.0f}" if o["beta_cap_frac"] >= 0.01 else f"{100 * o['beta_cap_frac']:.1f}"
    M1 = o[("model1", 80)]
    N[f"enSs{pre}CorMOne"], N[f"enSs{pre}RawMOne"] = f2(M1["point"]["corrected"]["skill"]), f2(M1["point"]["nearest station"]["skill"])
    N[f"enRoadCorr{pre}CorMOne"] = f2(M1["point"]["corrected"]["corr"])
    for v_kmh, w in ((60, "Sixty"), (90, "Ninety")):
        p_ = o[("vecto", v_kmh)]["point"]
        N[f"enAdd{pre}{w}"] = f"{p_['pen_mean']:.0f}"; N[f"enSs{pre}Cor{w}"] = f2(p_["corrected"]["skill"])
        N[f"enEzero{w}"] = f"{o[('vecto', v_kmh)]['sums']['E0']:.0f}"
    cov = o["info"]["coverage"]
    N[f"{pre.lower()}DirOkN"] = int(o["info"]["dir_ok"].sum()); N[f"{pre.lower()}DirExclN"] = int((~o["info"]["dir_ok"]).sum())
    N[f"{pre.lower()}StuckPct"] = f"{100 * o['info']['stuck_frac']:.1f}"
    N[f"{pre.lower()}RescConst"] = f2(o["info"]["rescale"])
    if "road_km_med" in o["info"]:
        N[f"{pre.lower()}RoadKm"] = f"{o['info']['road_km_med']:.0f}"
N["noDirExclList"] = "; ".join(f"{str(nm).title()} ({int(a)}/{int(b)}\\,h)" for nm, a, b, ok in
                               zip(EN["Norway"]["names"], EN["Norway"]["info"]["coverage"].n_dir, EN["Norway"]["info"]["coverage"].n_speed,
                                   EN["Norway"]["info"]["dir_ok"]) if not ok)

# vehicle (NOR-XWIND), by reference hour
VE = EN["vehicle"]
N.update(vehHours=VE["n_hours"], vehHoursDovre=VE["hours_by_site"]["dovre"], vehHoursFroya=VE["hours_by_site"]["froya"])
VN = {"no wind": "None", "nearest station": "Ref", "nearest road station": "Road", "MET Nordic": "Mn", "corrected": "Cor",
      "corrected, local z0": "CorLoc"}
for model, suf in (("vecto", ""), ("model1", "MOne")):
    for k, nm in VN.items():
        if k not in VE[model]:
            continue
        r = VE[model][k]
        N[f"enVeh{nm}{suf}"] = f1(r["mae"]); N[f"enVeh{nm}{suf}Lo"], N[f"enVeh{nm}{suf}Hi"] = (f1(x) for x in r["mae_ci"])
        N[f"enVehBias{nm}{suf}"] = f"{r['bias']:+.0f}"
        if k not in ("no wind",):
            N[f"vehSpdBias{nm}{suf}"] = f"{r['spd_bias']:+.0f}"
    N[f"enVehTrue{suf}"] = f1(VE[model + "_true"]["mean"]); N[f"enVehPen{suf}"] = f"{VE[model + '_true']['pen']:.0f}"
ns = VE["near_station"]
N.update(vehNearSeg=ns["n_seg"], vehNearHours=ns["n_hours"], vehNearRatio=f2(ns["ratio"]), vehNearFactor=f2(ns["implied_factor"]))

# baselines for wind speed (24_baselines.py)
BL = pickle.load(open(NO / "baselines.pkl", "rb"))
for c, pre in (("Norway", "no"), ("Sweden", "se")):
    t = BL[c]["speed_all"].set_index("est")
    for k, nm in NM.items():
        if k in t.index:
            N[f"{pre}Spd{nm}"] = f2(t.loc[k, "mae_med"]); N[f"{pre}Spd{nm}Lo"], N[f"{pre}Spd{nm}Hi"] = f2(t.loc[k, "mae_lo"]), f2(t.loc[k, "mae_hi"])
            N[f"{pre}SpdBias{nm}"] = f"{t.loc[k, 'rbias_med']:+.0f}"; N[f"{pre}SpdBiasAbs{nm}"] = f"{abs(t.loc[k, 'rbias_med']):.0f}"
    for k, nm in NM.items():
        if k in BL[c]["dir"]:
            N[f"{pre}Dir{nm}"] = f"{BL[c]['dir'][k]:.0f}"
    if "speed_mn" in BL[c]:
        tm = BL[c]["speed_mn"].set_index("est")
        for k in ("MET Nordic", "corrected", "nearest road station", "nearest station"):
            N[f"{pre}SpdMnh{NM[k]}"] = f2(tm.loc[k, "mae_med"]); N[f"{pre}SpdBiasMnh{NM[k]}"] = f"{tm.loc[k, 'rbias_med']:+.0f}"
        dm = BL[c]["dir_mn"]
        N[f"{pre}DirMnhMn"], N[f"{pre}DirMnhCor"] = f"{dm['MET Nordic']:.0f}", f"{dm['corrected']:.0f}"
    rel = BL[c]["reliability"]
    rc = rel["corrected"]
    N[f"{pre}RelLow"] = f2(rc.ratio.iloc[0]); N[f"{pre}RelHigh"] = f2(rc[rc.n >= 100].ratio.iloc[-1])
    N[f"{pre}RelHighLo"] = f"{rc[rc.n >= 100].lo.iloc[-1]:.0f}"
    hi = rc[rc.lo >= 8]; N[f"{pre}RelEight"] = f"{100 * (1 - (hi.obs * hi.n).sum() / (hi.est * hi.n).sum()):.0f}"
    rk = rel["calibrated"]; rk = rk[rk.n >= 100]
    N[f"{pre}RelCalMin"], N[f"{pre}RelCalMax"] = f2(rk.ratio.iloc[1:].min()), f2(rk.ratio.iloc[1:].max())
    rn = rel["nearest station"]; rn = rn[rn.n >= 100]
    N[f"{pre}RelRawHigh"] = f2(rn.ratio.iloc[-1])
    b = BL[c]["by_obs"]
    for k in ("corrected", "nearest station", "nearest road station", "rescaled nearest", "MET Nordic"):
        g_ = b[(b.est == k) & (b.lo == 15)]
        if len(g_) and g_.n.iloc[0] >= 50:
            N[f"{pre}StrongObs{NM[k]}"] = f2(g_.mae.iloc[0]); N[f"{pre}StrongObsMe{NM[k]}"] = f"{g_.me.iloc[0]:+.1f}"
            N[f"{pre}StrongObsN"] = int(g_.n.iloc[0])

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
         vehFroyaKmMin=f"{min(V[V.site == 'froya'].ref_km.min(), V[V.site == 'froya'].road_km.min()):.0f}")
# speed vs direction: drag error with one of the two components replaced by the measurement or by MET Nordic
for key, u, d in (("ModelSpeedMnDir", "model", "metnordic_dir"), ("ModelSpeedTrueDir", "model", "dir_veh"),
                  ("MnSpeedTrueDir", "metnordic", "dir_veh"), ("TrueSpeedModelDir", "w_veh", "model_dir"), ("MnSpeedModelDir", "metnordic", "model_dir")):
    e = drag.relative_error(80 / 3.6, V.w_veh.values, V.dir_veh.values, V[u].values, V[d].values, V.heading.values)
    N[f"vehDrag{key}"] = f"{100 * np.nanmedian(np.abs(e)):.0f}"
from smhi_trv_wind import assess as _as
_Vh = pd.to_datetime(V.t, utc=True).dt.round("1h"); _Et = _as.energy(V.w_veh.values, V.dir_veh.values, V.heading.values)
for key, u, d in (("CorCor", "model", "model_dir"), ("CorMn", "model", "metnordic_dir"), ("MnCor", "metnordic", "model_dir"),
                  ("MnMn", "metnordic", "metnordic_dir"), ("CorTrue", "model", "dir_veh"), ("MnTrue", "metnordic", "dir_veh")):
    _g = pd.DataFrame(dict(h=_Vh.values, t=_Et, e=_as.energy(V[u].values, V[d].values, V.heading.values))).groupby("h").mean()
    N[f"vehE{key}"] = f"{np.mean(np.abs(_g.e - _g.t)):.1f}"          # hourly energy MAE, speed from u, direction from d
for c_ in ("model", "metnordic", "ref_raw", "road_raw"):          # overall speed bias over the hours of Dovre, 31 Oct evening
    g = V[(V.site == "dovre") & (V.t.dt.day == 31)]
    N[f"vehDovreEve{c_.replace('_', '').title()}"] = f"{100 * (g[c_].mean() / g.w_veh.mean() - 1):+.0f}"
# ---------------- coefficient table (both countries)
names = ["$b_0$ intercept", "$b_1$ $T_{100}$", "$b_2$ $\\ln(x/h)$", "$b_3$ $\\ln U_\\mathrm{net}$", "$b_4$ $F_{1000}$",
         "direction $d_0$ [deg]", "direction $d_1$ [deg]"]
no_v = np.r_[R["coef"]["b"], R["coef"]["d"]]; no_f = R["coef_folds"]
se_v = np.r_[SEcv["coef"]["b"], SEcv["coef"]["d"]]; se_f = np.c_[SEcv["folds_b"], SEcv["folds_d"]]
def _ci(country, i):
    c_ = COEF_CI.get(country)
    return f"$[{c_[0, i]:+.2f}, {c_[1, i]:+.2f}]$" if c_ is not None and i < 5 else "--"
with open(GEN / "tab_coefs.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{Norway} & \\multicolumn{3}{c}{Sweden} \\\\\n")
    f.write("Term & estimate & blocks & stations & estimate & blocks & stations \\\\\n\\midrule\n")
    for i, n_ in enumerate(names):
        fmt = (lambda x: f"${x:+.1f}$") if i >= 5 else (lambda x: f"${x:+.2f}$")
        f.write(f"{n_} & {fmt(no_v[i])} & {fmt(no_f[:, i].min())}\\,…\\,{fmt(no_f[:, i].max())} & {_ci('Norway', i)} & "
                f"{fmt(se_v[i])} & {fmt(se_f[:, i].min())}\\,…\\,{fmt(se_f[:, i].max())} & {_ci('Sweden', i)} \\\\\n")
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

# ---------------- table: estimators of the roadside wind (Norway)
EB = EN["Norway"][("vecto", 80)]; EM = EN["Norway"].get("metnordic_hours")
tS = BL["Norway"]["speed_all"].set_index("est"); tSm = BL["Norway"].get("speed_mn")
rows_ = [("no wind", "No wind"), ("nearest station", "Nearest reference station"), ("rescaled nearest", "Nearest station, rescaled"),
         ("nearest road station", "Nearest other road station"), ("MET Nordic", "MET Nordic analysis$^{*}$"),
         ("corrected", "Corrected (this study)"), ("calibrated", "Corrected, recalibrated")]
with open(GEN / "tab_estimators.tex", "w") as f:
    f.write("\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.2\\textwidth}cccccc}\n\\toprule\n")
    f.write(" & \\multicolumn{3}{c}{wind} & \\multicolumn{3}{c}{air-drag energy} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
    f.write("Estimator & speed MAE & bias [\\%] & dir.\\ [deg] & MAE & skill & $r$ \\\\\n\\midrule\n")
    for k, lab in rows_:
        if k == "MET Nordic":
            if tSm is None or EM is None:
                continue
            sp = tSm.set_index("est").loc[k]; en, ec = EM["point"][k], EM["ci"][k]; dr_ = BL["Norway"]["dir_mn"].get(k)
        else:
            sp = tS.loc[k] if k in tS.index else None; en, ec = EB["point"][k], EB["ci"][k]; dr_ = BL["Norway"]["dir"].get(k)
        spd = f"{sp.mae_med:.2f} [{sp.mae_lo:.2f}, {sp.mae_hi:.2f}]" if sp is not None else "--"
        bia = f"${sp.rbias_med:+.0f}$" if sp is not None else "--"
        dd_ = f"{dr_:.0f}" if dr_ is not None and k != "no wind" else "--"
        sk = f"${en['skill']:+.2f}$ [${ec['skill'][0]:+.2f}$, ${ec['skill'][1]:+.2f}$]" if k != "no wind" else "0"
        rr = f"{en['corr']:.2f}" if k != "no wind" else "--"
        f.write(f"{lab} & {spd} & {bia} & {dd_} & {en['mae']:.1f} & {sk} & {rr} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- table: vehicle height, by reference hour
V["h"] = pd.to_datetime(V.t, utc=True).dt.round("1h")
vrows = [(None, None, "No wind", "no wind"), ("ref_raw", "ref_dir", "Nearest reference station", "nearest station"),
         ("road_raw", "road_dir", "Nearest road weather station", "nearest road station"), ("metnordic", "metnordic_dir", "MET Nordic analysis", "MET Nordic"),
         ("model", "model_dir", "Corrected (this study)", "corrected"), ("model_loc", "model_dir", "Corrected, local-$z_0$ profile", "corrected, local z0")]
with open(GEN / "tab_vehicle.tex", "w") as f:
    f.write("\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.27\\textwidth}cccccc}\n\\toprule\n")
    f.write(" & \\multicolumn{3}{c}{speed MAE [m\\,s$^{-1}$]} & direction & \\multicolumn{2}{c}{energy [kWh/100\\,km]} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\n")
    f.write("Estimate & Dovre & Frøya & all & MAE [deg] & MAE [95\\,\\% CI] & bias [\\%] \\\\\n\\midrule\n")
    for c_, dcol, lab, ek in vrows:
        if c_ is not None and c_ not in V:
            continue
        r = EN["vehicle"]["vecto"][ek]
        if c_ is None:
            sp_ = ["--"] * 3; dmae = "--"
        else:
            hh = V.groupby(["site", "h"]).agg(w=("w_veh", "mean"), e=(c_, "mean")).reset_index()
            sp_ = [f"{np.mean(np.abs(g.e - g.w)):.2f}" for g in (hh[hh.site == "dovre"], hh[hh.site == "froya"], hh)]
            dmae = f"{np.nanmean(np.abs(wrap180(V[dcol] - V.dir_veh))[V.w_veh >= 2]):.0f}"
        f.write(f"{lab} & {' & '.join(sp_)} & {dmae} & {r['mae']:.1f} [{r['mae_ci'][0]:.1f}, {r['mae_ci'][1]:.1f}] & ${r['bias']:+.0f}$ \\\\\n")
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
