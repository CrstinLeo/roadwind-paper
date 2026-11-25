"""Norwegian observations from the MET Norway Frost API (https://frost.met.no, free client id, CC BY 4.0).

Roadside network   Statens vegvesen road weather stations (station holder VEGVESEN), 10-min wind,
                   on-the-hour samples; Frost reports the sensor level as 10 m above ground.
Reference network  MET Norway and Avinor stations, hourly wind at 10 m (WMO standard exposure where possible).

Layout under SWW_DATA (scripts in analysis/norway/):
  stations_region.csv           station list with grp in {SVV, MET, AVINOR}, lat, lon, masl
  raw/YYYYMM_<res>_<i>.parquet  long table id, time, wind_speed, wind_from_direction
  wind_hourly.parquet           all raw files merged (built on first use)
  road_bearing.csv              OSRM road axis at the SVV stations
The functions mirror data.smhi (reference) and data.trafikverket (roadside), so the pipeline is country-agnostic.
"""
from __future__ import annotations

import glob
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings


def _stations(cfg: Settings) -> pd.DataFrame:
    s = pd.read_csv(cfg.paths.root / "stations_region.csv")
    s["id"] = s.id.astype(str)
    return s.set_index("id")


@lru_cache(maxsize=2)
def _hourly(root: str) -> pd.DataFrame:
    root = Path(root)
    f = root / "wind_hourly.parquet"
    if not f.exists():
        parts = [pd.read_parquet(p) for p in sorted(glob.glob(str(root / "raw" / "*.parquet")))]
        d = pd.concat([p for p in parts if len(p)], ignore_index=True)
        d = d.rename(columns={"wind_speed": "ws", "wind_from_direction": "wd"})
        d = d.drop_duplicates(["id", "time"]).sort_values(["id", "time"])
        d = qc(d)
        d.to_parquet(f)
    return pd.read_parquet(f)


def qc(d: pd.DataFrame, max_ws: float = 60.0, run_nonzero: int = 6, run_zero: int = 12) -> pd.DataFrame:
    """remove implausible speeds and stuck sensors: the same value repeated in >= run_nonzero consecutive hours
    (>= run_zero hours for exactly 0, which also occurs with icing); direction is removed with the speed"""
    d = d.copy()
    d.loc[(d.ws < 0) | (d.ws > max_ws), "ws"] = np.nan
    new_run = (d.id != d.id.shift()) | (d.ws != d.ws.shift()) | (d.time - d.time.shift() != pd.Timedelta("1h"))
    run_id = new_run.cumsum()
    run_len = d.groupby(run_id).ws.transform("size")
    stuck = ((d.ws > 0) & (run_len >= run_nonzero)) | ((d.ws == 0) & (run_len >= run_zero))
    d.loc[stuck, ["ws", "wd"]] = np.nan
    d.attrs["qc_removed"] = int(stuck.sum())
    return d


def _matrices(cfg: Settings, index: pd.DatetimeIndex, ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = _hourly(str(cfg.paths.root))
    d = d[d.id.isin(list(ids))]
    ws = d.pivot_table(index="time", columns="id", values="ws").reindex(index=index, columns=list(ids))
    wd = d.pivot_table(index="time", columns="id", values="wd").reindex(index=index, columns=list(ids))
    return ws.astype("float32"), wd.astype("float32")


class _Reference:
    """MET Norway + Avinor stations: plays the role of SMHI"""

    @staticmethod
    def stations(cfg: Settings) -> pd.DataFrame:
        s = _stations(cfg)
        return s[s.grp.isin(["MET", "AVINOR"])]

    @classmethod
    def wind_matrices(cls, cfg: Settings, index, ids=None):
        return _matrices(cfg, index, cls.stations(cfg).index if ids is None else ids)

    @classmethod
    def history(cls, cfg: Settings):
        idx = pd.date_range(pd.Timestamp(cfg.train_start, tz="UTC"), pd.Timestamp(cfg.train_end, tz="UTC"), freq="1h")
        return cls.wind_matrices(cfg, idx)


class _Roadside:
    """Statens vegvesen road weather stations: plays the role of Trafikverket"""

    @staticmethod
    def stations(cfg: Settings, max_km: float | None = None) -> pd.DataFrame:
        s = _stations(cfg)
        return s[s.grp == "SVV"]

    @classmethod
    def wind_matrices(cls, cfg: Settings, index, ids=None):
        return _matrices(cfg, index, cls.stations(cfg).index if ids is None else ids)

    @staticmethod
    def road_bearing(cfg: Settings) -> pd.Series:
        return pd.read_csv(cfg.paths.root / "road_bearing.csv", dtype={"id": str}).set_index("id").road_bearing

    @staticmethod
    def temperature_difference(cfg: Settings, index, ids=None) -> pd.DataFrame:
        return pd.DataFrame(np.nan, index=index, columns=list(ids) if ids is not None else [])


reference = _Reference()
roadside = _Roadside()
