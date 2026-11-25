"""Exposure features at the NOR-XWIND road segments (Dovre E6, Frøya), same definitions as for stations."""
import numpy as np, pandas as pd
from smhi_trv_wind import Settings
from smhi_trv_wind.features import norway
cfg = Settings()
rows = []
for site, f in (('dovre', '/home/claude/nx/data/bins_dovre.parquet'), ('froya', '/home/claude/nx/data/bins_froya.parquet')):
    b = pd.read_parquet(f)
    g = b.groupby('seg').agg(lat=('lat', 'mean'), lon=('lon', 'mean'), heading=('heading', lambda h: np.degrees(np.angle(np.mean(np.exp(2j * np.radians(h))))) / 2 % 180))
    for seg, s in g.iterrows():
        Hm, water, dtm = norway.station_arrays(s.lat, s.lon)
        sid = f"{site}_{seg}"
        V = pd.DataFrame(norway.variants_from_arrays(cfg, 'veh', sid, Hm, water, s.heading))
        E = pd.DataFrame(norway.edge_from_array(cfg, 'veh', sid, Hm))
        V = V.merge(E[['dir', 'x_edge', 'h_edge', 'xh_edge']], on='dir')
        V['site'], V['seg'], V['lat'], V['lon'], V['road_bearing'] = site, seg, s.lat, s.lon, s.heading
        rows.append(V); print(sid, flush=True)
pd.concat(rows).to_parquet('vehicle_segment_features.parquet'); print('SEGMENTS DONE')
