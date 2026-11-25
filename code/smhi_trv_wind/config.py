"""All paths, periods, physical constants and tunable choices in one place.

Every number used by the model is defined here, with where it comes from:
  [phys]  fixed physical constant / literature value, not fitted
  [data]  property of a data source
  [choice] our own modelling choice -> needs a sensitivity analysis in the paper
Fitted parameters (site factors c_s(k), exposure coefficients b0..b4, d0, d1) are NOT here:
they are estimated by model/site_factors.py and model/exposure.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Paths:
    """Data layout on disk. Set the environment variable SWW_DATA to the data root."""
    root: Path = Path(os.environ.get("SWW_DATA", "./data"))

    # --- SMHI open data (data/fetch_smhi.py)
    @property
    def smhi_dir(self) -> Path: return self.root / "smhi"
    @property
    def smhi_stations(self) -> Path: return self.smhi_dir / "stations_wind.csv"

    # --- Trafikverket Open API (data/fetch_trafikverket.py)
    @property
    def trv_dir(self) -> Path: return self.root / "trv"
    @property
    def trv_stations(self) -> Path: return self.trv_dir / "measurepoints.csv"
    @property
    def trv_obs(self) -> Path: return self.trv_dir / "obs_7d_120km.parquet"
    @property
    def road_bearing(self) -> Path: return self.trv_dir / "road_bearing.csv"

    # --- GIS rasters (features/gis.py)
    @property
    def nmd_object_height(self) -> Path:   # NMD2023 additional layer: object height 5-45 m, 10 m, laser scanning
        return self.root / "nmd_ext/NMD2023_Tillaggsskikt_Objekthojd_objekttackning_v1_1/NMD2023_Objekt_hojd_intervall_5_till_45_v1_1.tif"
    @property
    def nmd_base_glob(self) -> list[str]:  # NMD2018 base land cover per county (O, N, F), 10 m
        return [str(self.root / "nmd_base/*/*/*.tif"), str(self.root / "nmd_O/*/*.tif")]

    # --- derived station features, 16 wind-from sectors (features/gis.py)
    @property
    def feat_tree(self) -> Path: return self.root / "features_16dir.parquet"          # tree_near
    @property
    def feat_edge(self) -> Path: return self.root / "features2_16dir.parquet"         # x_edge, h_edge, xh_edge
    @property
    def feat_z0(self) -> Path: return self.root / "features_water_16dir.parquet"      # z0w_fp_L, dist_coast_km
    @property
    def feat_variants(self) -> Path: return self.root / "features_variants.parquet"   # tree_R{R}_H{H}, lnz0_L{L}_H{H}, along-road

    # --- optional: RV40 route for the map (OSRM geometry JSON)
    route_json: Path | None = None

    out: Path = Path(os.environ.get("SWW_OUT", "./output"))


@dataclass
class Settings:
    # ---------------- periods
    train_start: str = "2022-01-01"                 # [choice] SMHI history for site factors
    train_end: str = "2026-11-13 19:00"             # [choice] must end before the test period
    test_start: str = "2026-11-13 20:00"            # [data]   TrV Open API keeps ~7 days only
    test_end: str = os.environ.get("SWW_TEST_END", "2026-11-25 18:00")

    # ---------------- physics (step 1-2)
    z_smhi: float = 10.0                            # [data] SMHI anemometer height
    z_trv: float = 6.0                              # [data] TrV WMT700 height (Trafikverket metadata)
    z0_open: float = 0.03                           # [phys] WMO open terrain, Wieringa class
    z_blend: float = 60.0                           # [phys] blending height (Wieringa 1986: 50-60 m)
    z0_tree_factor: float = 0.1                     # [phys] z0 = 0.1 h over vegetation (rule of thumb)
    z0_water: float = 0.0002                        # [phys] open water
    tree_min_height: float = 5.0                    # [data] NMD object-height layer starts at 5 m
    footprint_L_smhi: float = 100.0                 # [choice] exp(-r/L) footprint; round 7: 25-400 m equivalent (round 6: 1000)
    footprint_L_trv: float = 100.0                  # [choice] 6 m roadside sensor (best L in round 2: 50-100 m)
    n_sectors: int = 16                             # [choice] 16 sector centres, 22.5 deg apart
    sector_half_angle: float = 22.5                 # [choice] features use +-22.5 deg around each centre (overlapping)

    # ---------------- site factors (step 3)
    site_min_network_ws: float = 3.0                # [choice] hours with network median >= 3 m/s
    site_min_ws: float = 0.5                        # [choice] ignore calm readings in ln U
    site_min_stations: int = 12                     # [choice] hours with >= 12 of 17 stations reporting
    site_iters: int = 15
    site_nsec: int = 8                              # [choice] direction sectors of the site factors; round 7: 4-16 equivalent (round 6: 16)

    # ---------------- exposure model (step 4-5)
    tree_near_radius: int = 100                     # [choice] near trees within R m upwind; round 7: 50-100 m (round 6: 200)
    tree_far_radius: int | None = 1000              # [choice] forest fraction within 1 km upwind; None = off (round 6)
    tree_dir_radius: int = 200                      # [choice] tree share used by the direction (channelling) model
    use_stability: bool = False                     # dT = T_surface - T_air at the TrV station; off: CV gain ~0.002 m/s and
                                                    # its coefficient changes sign between time blocks (round 7, 287 h)
    lnU_tree_interaction: bool = False              # round 6: on; round 7: removing it improves CV error
    fit_min_ref_ws: float = 2.0                     # [choice] speed model: SMHI raw >= 2 m/s
    fit_min_ref_ws_dir: float = 3.0                 # [choice] direction model: SMHI >= 3 m/s ...
    fit_min_obs_ws_dir: float = 1.5                 # [choice] ... and TrV >= 1.5 m/s
    xh_clip: tuple = (0.2, 200.0)                   # [choice] bounds for x/h before the log

    # ---------------- pairing and combination (step 6)
    max_pair_km: float = 150.0                      # [choice] candidate SMHI stations (round 6: 80)
    speed_km: float = 150.0                         # [choice] speed: all stations (round 7: >= 80 km best; round 6: 60)
    speed_min_n: int = 1
    speed_power: float = 1.5                        # [choice] w = 1/d^p; round 7: 1-2 (round 6: 1)
    combine_km: float = 60.0                        # [choice] direction: stations within 60 km ...
    combine_min_n: int = 2                          # [choice] ... at least the 2 nearest
    dir_power: float = 1.0                          # [choice] direction weights 1/d
    combine_min_dist: float = 3.0                   # [choice] floor for 1/d weights
    min_trv_hours: int = 100                        # [choice] TrV stations with >= 100 valid hours

    paths: Paths = field(default_factory=Paths)

    @classmethod
    def round6(cls, **kw) -> "Settings":
        """the round-6 model (reproduces the round-6 numbers on the round-6 period)"""
        base = dict(footprint_L_smhi=1000.0, site_nsec=16, tree_near_radius=200, tree_far_radius=None,
                    use_stability=False, lnU_tree_interaction=True, max_pair_km=80.0, speed_km=60.0,
                    speed_min_n=2, speed_power=1.0)
        base.update(kw)
        return cls(**base)

    @property
    def hours(self) -> pd.DatetimeIndex:
        return pd.date_range(pd.Timestamp(self.test_start, tz="UTC"), pd.Timestamp(self.test_end, tz="UTC"), freq="1h")
