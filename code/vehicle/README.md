# NOR-XWIND vehicle pre-processing

Builds the segment tables (`bins_<site>.parquet`) used by `scripts/12_vehicle_norway.py` from the open NOR-XWIND
record (Zenodo 18483903): van-mounted ultrasonic anemometer at 3.4 m, 1 Hz GNSS.

Data live outside git. Set `NORXWIND_DIR` (default `./norxwind`) with sub-folders `p1hz/`, `data/`, and
`NORXWIND_DOVRE` (default `$NORXWIND_DIR/dovre`) for the Dovre E6 days.

| Step | Script | Output |
|---|---|---|
| Dovre raw .vbo (2020-10-29/31) | `fetch_dovre.py`, `parse_dovre.py` (`vbo.py`) | `dovre_1hz.parquet` |
| Other site-days, 1 Hz | `fetch_days.py <day>` | `p1hz/<day>/*.parquet` |
| MET Nordic / MEPS at the passes | `metnordic.py`, `meps.py` (thredds.met.no) | `mn_<site>.nc`, `meps_<site>.nc` |
| Anemometer calibration | `calib.py` (Dovre), `nxsite.calibrate` (others) | `cal_<site>.json` |
| Segment bins | `build_all.py dovre froya` | `data/bins_<site>.parquet` |

**Anemometer calibration.** Rotation `alpha` and gain `g` are fitted so that the ground-fixed wind from consecutive
passes in opposite directions agrees (reciprocal-pass consistency). The mirror ambiguity of the lateral axis (`sgn`)
is resolved per measurement day by the smaller summed reciprocal cost over all sessions of that day (`nxsite.day_sign`);
MET Nordic is not used for the choice (it is recorded in `cal_*.json` as a diagnostic only). A mirror about the
vehicle axis leaves the head/tail and the absolute crosswind component unchanged, so the measured air drag does not
depend on `sgn`; only the measured wind direction does (Froya: mean direction 255 deg with +1, 241 deg with -1).
