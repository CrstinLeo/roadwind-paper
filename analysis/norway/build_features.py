import pandas as pd, numpy as np
from smhi_trv_wind import Settings
from smhi_trv_wind.features import norway, road
cfg = Settings()
S = pd.read_csv('stations_region.csv'); S['id'] = S.id.astype(str); S['lat'] = S.lat.astype(float); S['lon'] = S.lon.astype(float)
S['src'] = np.where(S.grp == 'SVV', 'trv', 'smhi')
S = S.set_index('id')
rb = road.build(S[S.src == 'trv'][['lat', 'lon']])
rb.to_csv('road_bearing.csv', index=False)
print(rb.describe(), flush=True)
norway.build(cfg, S, rb.set_index('id').road_bearing, 'features_variants.parquet', 'features2_16dir.parquet', 'station_meta_gis.parquet')
print('FEATURES DONE')
