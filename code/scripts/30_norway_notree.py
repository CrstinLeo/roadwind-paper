"""Norway: final model without the near-tree term (T100 is not determined in Norway), cross-validated with predictions,
so that its effect on the energy results can be reported -> SWW_OUT/notree_NO.pkl"""
import dataclasses
import os
import pickle
from pathlib import Path

import pandas as pd

from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import RECOMMENDED, Engine, cv

cfg = Settings.norway(max_pair_km=120.0)
OUT = Path(os.environ.get("SWW_OUT", "./output"))
E = Engine(cfg, pd.read_parquet(cfg.paths.root / "features_variants.parquet"), t_split="2025-01-01 00:00", t_end=cfg.test_end)
c = cv(E, dataclasses.replace(RECOMMENDED, use_tree=False), 8, keep_pred=True)
pickle.dump(dict(pred=c.pop("pred"), cv=c), open(OUT / "notree_NO.pkl", "wb"))
print("done")
