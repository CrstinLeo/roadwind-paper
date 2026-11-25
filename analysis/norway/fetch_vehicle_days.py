"""Frost wind for the NOR-XWIND test days (2020-10-28 to 2020-11-06) at all study stations."""
import pandas as pd, requests, time
cid = open('/home/claude/nx/frost_id.txt').read().strip()
S = pd.read_csv('stations_region.csv'); rows = []
for res, G in S.groupby('res'):
    ids = list(G.id); step = 10 if res == 'PT10M' else 40
    for i in range(0, len(ids), step):
        p = dict(sources=','.join(ids[i:i + step]), elements='wind_speed,wind_from_direction', referencetime='2020-10-28/2020-11-07',
                 timeresolutions=res, fields='sourceId,referenceTime,elementId,value', levels='default')
        r = requests.get('https://frost.met.no/observations/v0.jsonld', params=p, auth=(cid, ''), timeout=300)
        if r.status_code == 404: continue
        r.raise_for_status()
        rows += [dict(id=o['sourceId'].split(':')[0], time=o['referenceTime'], el=x['elementId'], v=x['value']) for o in r.json()['data'] for x in o['observations']]
d = pd.DataFrame(rows); d['time'] = pd.to_datetime(d.time, utc=True); d = d[d.time.dt.minute == 0]
d = d.pivot_table(index=['id', 'time'], columns='el', values='v').reset_index().rename(columns={'wind_speed': 'ws', 'wind_from_direction': 'wd'})
d.to_parquet('wind_vehicle_days.parquet'); print(d.shape, d.id.nunique(), d.time.min(), d.time.max())
