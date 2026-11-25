"""Terrain features (Sx, TPI, relief) for Norwegian stations and NOR-XWIND segments from the Kartverket DTM."""
import numpy as np, pandas as pd
from smhi_trv_wind.features import norway
from smhi_trv_wind.features.terrain import terrain_features
S = pd.read_csv('stations_region.csv'); S['id'] = S.id.astype(str)
rows = []
for _, s in S.iterrows():
    x, y = norway.TO_UTM33.transform(s.lon, s.lat)
    rows += terrain_features(norway._wcs('dtm', x, y), 'trv' if s.grp == 'SVV' else 'smhi', s.id, z_sensor=10.0)
    print('terrain', s.id, flush=True)
seg = pd.read_parquet('vehicle_segment_features.parquet')[['id', 'lat', 'lon']].drop_duplicates('id')
for _, s in seg.iterrows():
    x, y = norway.TO_UTM33.transform(s.lon, s.lat)
    rows += terrain_features(norway._wcs('dtm', x, y), 'veh', s.id, z_sensor=3.4)
pd.DataFrame(rows).to_parquet('features_terrain.parquet'); print('TERRAIN DONE')
