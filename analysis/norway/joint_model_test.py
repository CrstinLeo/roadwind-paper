import numpy as np, pandas as pd, pickle, dataclasses, sys
from pathlib import Path
from smhi_trv_wind import Settings, Paths
from smhi_trv_wind.sensitivity import Engine, RECOMMENDED, cv
M9 = dataclasses.replace(RECOMMENDED, site="mean", extra=("v:tree_R1000_H180",))
M9b = dataclasses.replace(RECOMMENDED, site="mean")
M9c = dataclasses.replace(RECOMMENDED, extra=("v:tree_R1000_H180",))
country=sys.argv[1]
if country=='SE':
    cfg=Settings(max_pair_km=150.0, paths=Paths(root=Path('/home/claude/se'))); V=pd.read_parquet('/home/claude/se/features_variants.parquet'); nb=4
    E=Engine(cfg,V,t_split='2026-11-19 00:00',t_end=cfg.test_end)
else:
    cfg=Settings.norway(max_pair_km=120.0); V=pd.read_parquet('/home/claude/no/features_variants.parquet'); nb=8
    E=Engine(cfg,V,t_split='2025-01-01',t_end=cfg.test_end)
out={}
base=cv(E,RECOMMENDED,nb); out['final']=base
for lab,s in [('site mean + forest all-dir',M9),('site mean',M9b),('forest all-dir',M9c)]:
    r=cv(E,s,nb); out[lab]=r; d=r['mae']-base['mae']; ok=np.isfinite(d)
    print(country,lab,'mean',np.nanmean(r['mae']).round(4),'median',np.nanmedian(r['mae']).round(4),'d',d[ok].mean().round(4),'+-',(d[ok].std(ddof=1)/np.sqrt(ok.sum())).round(4),'dir',np.nanmedian(r['dir']).round(1),flush=True)
pickle.dump(out,open(f'/tmp/nout/m9_{country}.pkl','wb'))
