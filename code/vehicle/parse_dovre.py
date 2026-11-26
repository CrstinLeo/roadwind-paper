import glob, os, pandas as pd
from vbo import read_vbo, to_1hz
rows = []
for f in sorted(glob.glob("raw/**/*.vbo", recursive=True)):
    date = "2020-10-29" if "day_201029" in f else "2020-10-31"
    d = to_1hz(read_vbo(f, date))
    d["session"] = os.path.basename(f).replace(".vbo", "")
    d["day"] = date
    rows.append(d)
    print(f, len(d), d.time.min(), d.time.max(), round(d.v.mean(), 1), flush=True)
pd.concat(rows, ignore_index=True).to_parquet("dovre_1hz.parquet")
