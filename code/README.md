# smhi_trv_wind — station-to-roadside wind correction and air-drag energy

Python package and scripts behind every number, table and figure of the paper. No data are stored in the repository.

## Environment

| Variable | Meaning |
|---|---|
| `SWW_DATA` | data root of the country being processed (Norway: Frost download, features; Sweden: SMHI/Trafikverket download, features) |
| `SWW_OUT` | output folder of the country being processed |
| `SWW_NO_DATA`, `SWW_SE_DATA` | data roots, for scripts that use both countries |
| `SWW_NO_OUT`, `SWW_SE_OUT` | output folders, for scripts that use both countries |
| `SWW_TEST_END` | end of the Swedish record (default `2026-11-25 18:00`, 287 h) |
| `NORXWIND_DIR` | NOR-XWIND vehicle data (see `vehicle/README.md`) |
| `TRV_API_KEY` / `TRV_API_KEY_FILE`, Frost client id | API credentials, read from the environment or a file outside the repository; never commit them |

Defaults are relative (`data/no`, `output/no`, ...).

## Package

| Module | Content |
|---|---|
| `config.py` | settings (`Settings()` = Sweden, `Settings.norway()`), paths |
| `data/` | SMHI, Trafikverket and Frost readers and quality control |
| `features/` | GIS features: object heights, roughness footprints, edge distance, forest fractions, terrain indices, road axis (OSRM) |
| `model/physics.py` | log-law height factor, Wieringa roughness conversion, network direction |
| `model/site_factors.py` | two-way fixed-effects site factors |
| `sensitivity.py` | `Engine` (all station pairs as arrays) and `cv` (blocked time × leave-one-station-out); `RECOMMENDED` = final model |
| `assess.py` | direction coverage and QC, comparison estimators, cross-fitted recalibration, energy sums, station-cluster bootstrap |
| `drag.py` | VECTO crosswind correction and Askerdal Model 1, drag force and energy |

## Run order

Sweden (development region; `SWW_DATA`, `SWW_OUT` Swedish):
`01_fetch_smhi.py`, `02_fetch_trafikverket.py` (collect every few days), `03_build_features.py`, `04_run_model.py`,
`14_final_cv.py` (`SWW_COUNTRY=SE`), `26_sweden_review.py` (scans around the final model), `29_sweden_rows.py`,
`18_mechanism_data.py` (`SWW_COUNTRY=SE`), `20_direction_variant.py SE`, `09_paper_figures.py map site along`.

Norway (main test; `SWW_DATA`, `SWW_OUT` Norwegian; `SWW_SE_DATA` for the transfer test):
`11_norway.py` (ladder, site factors, final CV, scans, transfer) → `23_norway_review.py` (selection on 2024 and 2025
hold-out, per-station normal equations) → `16_norway_ladder_terrain.py` → `27_norway_misc_checks.py` →
`12_vehicle_norway.py` (needs `vehicle/` output) → `25_metnordic_roadside.py` (MET Nordic at the stations, every third
hour) → `18_mechanism_data.py` (`SWW_COUNTRY=NO`) → `20_direction_variant.py NO` → `13_norway_figures.py`.

Both countries (`SWW_NO_*`, `SWW_SE_*`): `24_baselines.py` (wind-speed comparison of estimators, reliability) →
`22_energy.py` (air-drag energy, uncertainty band, vehicle by reference hour) → `15_paper_numbers.py` (writes
`sections/generated/`) → `19_figures_v2.py`, `28_figures_r1.py`, `17_graphical_abstract.py`.

Scripts `05`–`08`, `10`, `21` belong to earlier rounds and are kept for provenance; their results are not used in the paper.

## Tests

`python -m pytest tests` — physics identities, parameter recovery on synthetic data, drag and heading conventions.
