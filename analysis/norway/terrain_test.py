import numpy as np, pandas as pd, dataclasses, pickle
from smhi_trv_wind import Settings
from smhi_trv_wind.sensitivity import Engine, RECOMMENDED, cv
cfg=Settings.norway(max_pair_km=120.0)
E=Engine(cfg, pd.read_parquet('/home/claude/no/features_variants.parquet'), t_split='2025-01-01', t_end=cfg.test_end)
base=cv(E,RECOMMENDED,8); out={'final':base}
far=("v:tree_R1000_H22.5",)
for lab,ex in [('+sx_300',far+('t:sx_300',)),('+sx_1000',far+('t:sx_1000',)),('+tpi_2000',far+('t:tpi_2000',)),('+tpi_500',far+('t:tpi_500',)),('+sx_300+tpi_2000',far+('t:sx_300','t:tpi_2000')),('+sx_1000+tpi_2000+relief',far+('t:sx_1000','t:tpi_2000','t:relief_2000'))]:
    r=cv(E,dataclasses.replace(RECOMMENDED,extra=ex),8); out[lab]=r
    d=r['mae']-base['mae']; ok=np.isfinite(d)
    print(lab,'mean',np.nanmean(r['mae']).round(4),'median',np.nanmedian(r['mae']).round(4),'d',d[ok].mean().round(4),'+-',(d[ok].std(ddof=1)/np.sqrt(ok.sum())).round(4),flush=True)
pickle.dump(out,open('/tmp/nveh/terrain_cv.pkl','wb'))
