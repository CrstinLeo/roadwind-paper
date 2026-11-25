"""Steps 1-6 on the test period, metrics, outputs and the interactive map."""
import json
import os

import numpy as np

from smhi_trv_wind import Settings, pipeline, evaluate
from smhi_trv_wind.viz import station_map

cfg = Settings()
res = pipeline.run(cfg)
pipeline.save(res, cfg)
print(evaluate.ladder(res.P).round(3).to_string(index=False))
print(json.dumps(evaluate.summary(res.S), indent=1))
print(json.dumps(res.exposure.coefficients(), indent=1))
print(res.V[["name", "top1", "raw_top1", "quality_eff", "quality_distance_only"]].round(3).to_string())

route = None
if os.environ.get("SWW_ROUTE"):                       # OSRM route JSON, e.g. RV40
    route = np.array(json.load(open(os.environ["SWW_ROUTE"]))["routes"][0]["geometry"]["coordinates"])
print("map:", station_map.build(res, cfg, cfg.paths.out / "SMHI_TrV_station_map.html", route))
