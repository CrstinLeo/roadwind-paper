import numpy as np, pandas as pd, pickle
from pathlib import Path
from smhi_trv_wind import Settings, Paths
from smhi_trv_wind.sensitivity import Engine, RECOMMENDED, cv
R=pickle.load(open('/tmp/nout/norway_results.pkl','rb'))
cse=Settings(max_pair_km=150.0, paths=Paths(root=Path('/home/claude/se')))
ES=Engine(cse, pd.read_parquet('/home/claude/se/features_variants.parquet'), t_split='2026-11-19 00:00', t_end=cse.test_end)
ES.A=np.ones(len(ES.H),bool); f=ES.run(RECOMMENDED); R['coef_se']=dict(b=f['b'],d=f['d'],smear=f['smear'])
R['se_own_cv']=cv(ES,RECOMMENDED,4)
ES.A=np.zeros(len(ES.H),bool); R['se_with_no_coef']=ES.run(RECOMMENDED,fixed=R['coef'])['B']
del ES
cfg=Settings.norway(max_pair_km=120.0)
E=Engine(cfg, pd.read_parquet('/home/claude/no/features_variants.parquet'), t_split='2025-01-01', t_end=cfg.test_end)
E.A=np.zeros(len(E.H),bool); R['no_with_se_coef']=E.run(RECOMMENDED,fixed=R['coef_se'])['B']
pickle.dump(R,open('/tmp/nout/norway_results.pkl','wb'))
m=lambda x: (np.nanmean(x['mae']).round(3), np.nanmedian(x['mae']).round(3))
print('SE own CV', m(R['se_own_cv']), '| SE with NO coef', m(R['se_with_no_coef']), '| NO own CV', m(R['cv_final']), '| NO with SE coef', m(R['no_with_se_coef']), '| raw NO', np.nanmedian(R['cv_final']['mae_raw']).round(3), 'raw SE', np.nanmedian(R['se_own_cv']['mae_raw']).round(3))
print('coef NO', np.round(R['coef']['b'],3), 'SE', np.round(R['coef_se']['b'],3))
