"""Extract selected NOR-XWIND serial .vbo files, convert each to 1 Hz parquet, delete raw."""
import sys, os, re, io, shutil
import numpy as np, pandas as pd
from remotezip import RemoteZip

URL = "https://zenodo.org/api/records/18483903/files/NOR-XWIND.zip/content"
NX = os.environ.get("NORXWIND_DIR", "./norxwind")
OUT = os.path.join(NX, "p1hz")
TMP = os.path.join(NX, "tmpraw")
EXT = os.path.join(NX, "ext")

# day -> (analogue-input config, session filter)
DAYS = {
    "201105": ("AI2", lambda n: (m := re.search(r"Session (\d+)", n)) and (14 <= int(m.group(1)) <= 23 or 31 <= int(m.group(1)) <= 44)),
    "231011": ("AI3", lambda n: True),
    "241021": ("AI5", lambda n: True),
    "201121": ("AI2", lambda n: True),
    "241012": ("AI4", lambda n: "north_01" in n or "south_01" in n),
}


def read_vbo(path, date, ai):
    with open(path, "r", errors="ignore") as f:
        lines = f.read().splitlines()
    cols = lines[lines.index("[column names]") + 1].split()
    dat = lines[lines.index("[data]") + 1:]
    use = [c for c in ["time", "lat", "long", "velocity", "heading", "sats",
                       "VB3i_AD2", "VB3i_AD3", "VB3i_AD4"] if c in cols]
    df = pd.read_csv(io.StringIO("\n".join(dat)), sep=r"\s+", names=cols, header=None,
                     usecols=use, on_bad_lines="skip", dtype=str)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    t = df.time.values
    sec = (t // 10000) * 3600 + ((t // 100) % 100) * 60 + (t % 100)
    out = pd.DataFrame(dict(
        t=pd.Timestamp(date, tz="UTC") + pd.to_timedelta(sec, unit="s"),
        lat=df.lat.values / 60.0, lon=-df.long.values / 60.0,
        v=df.velocity.values / 3.6, heading=df.heading.values, sats=df.sats.values))
    if ai == "AI2":
        out["U"] = df.VB3i_AD2.values * 12 - 30
        out["V"] = df.VB3i_AD3.values * 12 - 30
    elif ai == "AI3":                         # polar output: magnitude, direction
        M = df.VB3i_AD2.values * 12
        D = np.radians(df.VB3i_AD3.values * 72)
        out["U"] = M * np.cos(D)
        out["V"] = M * np.sin(D)
    elif ai == "AI5":
        out["U"] = df.VB3i_AD3.values * 12 - 30
        out["V"] = df.VB3i_AD4.values * 12 - 30
    else:                                     # AI4: no vehicle anemometer
        out["U"] = np.nan
        out["V"] = np.nan
    return out[(out.sats > 3) & out.lat.between(50, 72)].reset_index(drop=True)


def to_1hz(df):
    h = np.radians(df.heading.values)
    tmp = df.assign(hs=np.sin(h), hc=np.cos(h))
    a = tmp.groupby(df.t.dt.floor("1s")).agg(
        lat=("lat", "mean"), lon=("lon", "mean"), v=("v", "mean"), hs=("hs", "mean"),
        hc=("hc", "mean"), U=("U", "mean"), V=("V", "mean"), n=("v", "size"))
    a["heading"] = np.degrees(np.arctan2(a.hs, a.hc)) % 360
    return a.drop(columns=["hs", "hc"]).reset_index().rename(columns={"t": "time"})


if __name__ == "__main__":
    days = sys.argv[1:] or list(DAYS)
    with RemoteZip(URL) as z:
        infos = z.infolist()
        for day in days:
            ai, filt = DAYS[day]
            date = pd.Timestamp("20" + day).strftime("%Y-%m-%d")
            os.makedirs(f"{OUT}/{day}", exist_ok=True)
            for i in infos:
                f = i.filename
                if f"day_{day}/" not in f:
                    continue
                if day == "241012" and "03_external_anemometer" in f and f.endswith(".txt"):
                    z.extract(i, EXT); print("ext", f, flush=True); continue
                if "/01_serial/" not in f or not f.lower().endswith(".vbo") or not filt(f):
                    continue
                name = os.path.splitext(os.path.basename(f))[0].replace(" ", "_")
                dst = f"{OUT}/{day}/{name}.parquet"
                if os.path.exists(dst):
                    continue
                p = z.extract(i, TMP)
                try:
                    to_1hz(read_vbo(p, date, ai)).to_parquet(dst)
                    print("ok", day, name, flush=True)
                except Exception as e:
                    print("FAIL", day, name, e, flush=True)
                os.remove(p)
    shutil.rmtree(TMP, ignore_errors=True)
    print("DONE", flush=True)
