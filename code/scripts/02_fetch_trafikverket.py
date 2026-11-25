"""Download Trafikverket station list and the observations the API still holds (~7 days), append to the archive.
Needs TRV_API_KEY (or TRV_API_KEY_FILE). Schedule e.g. every 3 days to build a longer series."""
from smhi_trv_wind import Settings
from smhi_trv_wind.data import trafikverket as trv

cfg = Settings()
cfg.paths.trv_dir.mkdir(parents=True, exist_ok=True)
st = trv.fetch_stations()
st.to_csv(cfg.paths.trv_stations, index=False)
near = st[(st.dist_km <= 120) & st.wind_height.notna()]
obs = trv.fetch_observations(list(near.id))
d = trv.append_observations(obs, cfg.paths.trv_obs)
print(d.shape, d.id.nunique(), d.time.min(), d.time.max())
