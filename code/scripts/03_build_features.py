"""Station exposure features (NMD object height + land cover) and road bearing (OSRM).
Needs the NMD rasters under SWW_DATA (see README, 'GIS data'). Takes ~10-20 min for 170 stations."""
from smhi_trv_wind import Settings
from smhi_trv_wind.data import smhi, trafikverket as trv
from smhi_trv_wind.features import gis, road

cfg = Settings()
T, S = trv.stations(cfg), smhi.stations(cfg)
T = T[T.wind_height.notna()]
gis.build_all(cfg, T, S)
road.build(T).to_csv(cfg.paths.road_bearing, index=False)
print("features written")
