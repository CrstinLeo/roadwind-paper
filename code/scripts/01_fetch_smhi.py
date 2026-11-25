"""Download SMHI wind stations within 120 km of RV40 and their wind speed / direction / gust series.
Re-run to refresh the 'latest-months' files."""
from smhi_trv_wind import Settings
from smhi_trv_wind.data import smhi

cfg = Settings()
cfg.paths.smhi_dir.mkdir(parents=True, exist_ok=True)
meta = smhi.fetch_stations()
meta.to_csv(cfg.paths.smhi_stations, index=False)
print(meta.to_string())
for sid in meta.id:
    smhi.fetch_csv(sid, cfg.paths.smhi_dir)
print("done")
