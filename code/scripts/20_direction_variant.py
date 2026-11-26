"""Direction model variant: add an untreed channelling term d2 sin(2 alpha) (dir_open=True) and compare with the final
model in blocked-time x LOSO CV. Usage: SWW_DATA=... SWW_OUT=... python 20_direction_variant.py NO|SE -> dir_open_<c>.pkl"""
import os, sys, dataclasses, pickle, numpy as np, pandas as pd
from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine, cv
c = sys.argv[1]
cfg = Settings.norway(max_pair_km=120.0) if c == "NO" else Settings(max_pair_km=150.0)
E = Engine(cfg, pd.read_parquet(cfg.paths.root / "features_variants.parquet"), t_split=cfg.test_end if c=="SE" else "2025-01-01", t_end=cfg.test_end)
NB = 8 if c == "NO" else 4
a = cv(E, RECOMMENDED, NB); b = cv(E, dataclasses.replace(RECOMMENDED, dir_open=True), NB)
d = b["dir"] - a["dir"]; ok = np.isfinite(d)
print(c, "dir median", np.nanmedian(a["dir"]).round(2), "->", np.nanmedian(b["dir"]).round(2), "| paired mean", d[ok].mean().round(2), "+-", (d[ok].std(ddof=1)/np.sqrt(ok.sum())).round(2), "| speed MAE unchanged", np.nanmean(b["mae"]-a["mae"]).round(4), flush=True)
pickle.dump(dict(base=a, open=b), open(os.environ["SWW_OUT"] + f"/dir_open_{c}.pkl", "wb"))
