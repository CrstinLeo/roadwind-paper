"""Download 10-min (SVV) / hourly (MET, Avinor) wind from Frost for the Trøndelag-Dovre region, month by month."""
import time, sys
from pathlib import Path
import numpy as np, pandas as pd, requests
cid = open('/home/claude/nx/frost_id.txt').read().strip()
A = pd.read_parquet('frost_available_wind.parquet'); D = pd.read_parquet('frost_wind_sources.parquet'); D['dist'] = D.dist.astype(float)
reg = D[D.dist <= 170].copy()
reg['grp'] = np.where(reg.holder.str.contains('VEGVESEN'), 'SVV', np.where(reg.holder.str.contains('MET.NO'), 'MET', np.where(reg.holder.str.contains('AVINOR'), 'AVINOR', 'OTHER')))
reg = reg[reg.grp != 'OTHER']
ws = A[(A.elementId == 'wind_speed') & A.sid.isin(reg.id)]
res = ws.groupby('sid').timeResolution.agg(lambda s: 'PT1H' if 'PT1H' in set(s) else ('PT10M' if 'PT10M' in set(s) else None))
reg['res'] = reg.id.map(res)
reg = reg[reg.res.notna()]
reg.to_csv('stations_region.csv', index=False)
out = Path('raw'); out.mkdir(exist_ok=True)
months = pd.date_range('2023-01-01', '2025-12-01', freq='MS')
end = pd.Timestamp('2025-11-25')
for m in months:
    if m > end: break
    t0, t1 = m, min(m + pd.offsets.MonthBegin(1), end)
    for r_, G in reg.groupby('res'):
        ids = list(G.id); step = 10 if r_ == 'PT10M' else 40
        for i in range(0, len(ids), step):
            f = out / f"{m:%Y%m}_{r_}_{i:03d}.parquet"
            if f.exists(): continue
            p = dict(sources=','.join(ids[i:i + step]), elements='wind_speed,wind_from_direction', referencetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}",
                     timeresolutions=r_, fields='sourceId,referenceTime,elementId,value,level', levels='default')
            for a in range(4):
                try:
                    rr = requests.get('https://frost.met.no/observations/v0.jsonld', params=p, auth=(cid, ''), timeout=300)
                    if rr.status_code == 404: rows = []; break
                    rr.raise_for_status(); data = rr.json()['data']
                    rows = [dict(id=o['sourceId'].split(':')[0], time=o['referenceTime'], el=x['elementId'], v=x['value']) for o in data for x in o['observations']]
                    break
                except Exception as e:
                    print('retry', f.name, e, flush=True); time.sleep(10 * (a + 1)); rows = None
            if rows is None: continue
            d = pd.DataFrame(rows)
            if len(d):
                d['time'] = pd.to_datetime(d.time, utc=True)
                d = d[d.time.dt.minute == 0]
                d = d.pivot_table(index=['id', 'time'], columns='el', values='v').reset_index()
            d.to_parquet(f)
    print('month done', m.date(), flush=True)
print('ALL DONE')
