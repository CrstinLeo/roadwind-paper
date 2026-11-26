import numpy as np, pandas as pd, io

def read_vbo(path, date):
    """读取 VBOX .vbo 文本 → 100 Hz DataFrame（UTC 时间、经纬度°、车速 m/s、航向°、风速仪 U/V m/s）"""
    with open(path, "r", errors="ignore") as f:
        lines = f.read().splitlines()
    i_col = lines.index("[column names]") + 1
    i_dat = lines.index("[data]") + 1
    cols = lines[i_col].split()
    df = pd.read_csv(io.StringIO("\n".join(lines[i_dat:])), sep=r"\s+", names=cols, header=None,
                     usecols=["time", "lat", "long", "velocity", "heading", "VB3i_AD2", "VB3i_AD3", "sats"],
                     on_bad_lines="skip", dtype=str)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    t = df.time.values
    sec = (t // 10000) * 3600 + ((t // 100) % 100) * 60 + (t % 100)
    out = pd.DataFrame(dict(
        t=pd.Timestamp(date, tz="UTC") + pd.to_timedelta(sec, unit="s"),
        lat=df.lat.values / 60.0,
        lon=-df.long.values / 60.0,            # VBOX 约定：经度以西为正
        v=df.velocity.values / 3.6,
        heading=df.heading.values,
        U=df.VB3i_AD2.values * 12 - 30,        # AI2 配置：纵向分量（风速仪坐标）
        V=df.VB3i_AD3.values * 12 - 30,        # AI2 配置：横向分量
        sats=df.sats.values))
    return out[(out.sats > 3) & out.lat.between(50, 72)].reset_index(drop=True)

def to_1hz(df):
    """1 Hz 平均：风和速度按矢量平均，航向按圆平均"""
    g = df.groupby(df.t.dt.floor("1s"))
    h = np.radians(df.heading.values)
    tmp = df.assign(hs=np.sin(h), hc=np.cos(h))
    a = tmp.groupby(df.t.dt.floor("1s")).agg(lat=("lat", "mean"), lon=("lon", "mean"), v=("v", "mean"),
                                             hs=("hs", "mean"), hc=("hc", "mean"), U=("U", "mean"),
                                             V=("V", "mean"), n=("v", "size"))
    a["heading"] = np.degrees(np.arctan2(a.hs, a.hc)) % 360
    return a.drop(columns=["hs", "hc"]).reset_index().rename(columns={"t": "time"})
