import numpy as np, pandas as pd, json
from pyproj import Transformer
from shapely.geometry import LineString, Point
from scipy.optimize import minimize

def load_passes():
    d = pd.read_parquet("dovre_1hz.parquet")
    d = d[~d.session.isin(["fokstugu2oppdal", "oppdal2fokstugu"]) & (d.v > 8)].copy()
    js = json.load(open("e6_osrm.json"))
    tr = Transformer.from_crs(4326, 25832, always_xy=True)
    xy = np.array(tr.transform(*np.array(js["routes"][0]["geometry"]["coordinates"]).T)).T
    line = LineString(xy)
    d["x"], d["y"] = tr.transform(d.lon.values, d.lat.values)
    d["s"] = [line.project(Point(a, b)) for a, b in zip(d.x, d.y)]
    d["off"] = [line.distance(Point(a, b)) for a, b in zip(d.x, d.y)]
    d = d[d.off < 30]
    d = d[d.groupby(["day", "session"]).s.transform("size") > 100]
    d["pass_id"] = d.groupby(["day", "session"]).ngroup()
    d["t0"] = d.groupby("pass_id").time.transform("min")
    d["dirE"] = d.groupby("pass_id").s.transform(lambda s: np.polyfit(np.arange(len(s)), s.values, 1)[0] > 0)
    return d.sort_values(["t0", "time"]).reset_index(drop=True), line

def true_wind(d, alpha, gx, gy, sgn):
    """风速仪坐标 (U, sgn·V) → 旋转 alpha → 车体坐标（x 前、y 左）→ 乘增益 → 加上车速 → 地面坐标真实风"""
    a = np.radians(alpha)
    mx, my = d.U.values, sgn * d.V.values
    rx = gx * (mx * np.cos(a) - my * np.sin(a))
    ry = gy * (mx * np.sin(a) + my * np.cos(a))
    h = np.radians(d.heading.values)
    fe, fn = np.sin(h), np.cos(h)          # 前向单位矢量（东、北）
    le, ln = -np.cos(h), np.sin(h)         # 左向单位矢量
    we = d.v.values * fe + rx * fe + ry * le
    wn = d.v.values * fn + rx * fn + ry * ln
    return we, wn                          # 空气运动矢量（东、北分量）

def reciprocal_cost(params, d, pairs, gy_fixed=None):
    alpha, gx = params[0], params[1]
    gy = gx if gy_fixed is None else gy_fixed
    best = np.inf
    for sgn in (1, -1):
        we, wn = true_wind(d, alpha, gx, gy, sgn)
        tmp = pd.DataFrame(dict(p=d.pass_id.values, b=(d.s.values // 1000).astype(int), we=we, wn=wn))
        m = tmp.groupby(["p", "b"])[["we", "wn"]].mean()
        c = 0.0; n = 0
        for p1, p2 in pairs:
            if p1 in m.index.get_level_values(0) and p2 in m.index.get_level_values(0):
                diff = (m.loc[p1] - m.loc[p2]).dropna()
                c += (diff ** 2).sum().sum(); n += len(diff)
        best = min(best, c / max(n, 1))
    return best

def make_pairs(d):
    """相邻两趟、方向相反、覆盖同一路段 → 一对"""
    info = d.groupby("pass_id").agg(t0=("t0", "first"), dirE=("dirE", "first"), smin=("s", "min"),
                                    day=("day", "first")).sort_values("t0")
    ids = info.index.values; pairs = []
    for a, b in zip(ids[:-1], ids[1:]):
        A, B = info.loc[a], info.loc[b]
        if A.day == B.day and A.dirE != B.dirE and abs(A.smin - B.smin) < 2000:
            pairs.append((a, b))
    return pairs
