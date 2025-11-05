# nyc-taxi-trips
Projet Data E4 DSIA: Théo Phan - Melchior Laurens

## User Guide
- Install dependencies: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Prepare data and launch dashboard: `python3 main.py` (downloads January 2025 yellow taxi data and required shape/lookup files on first run).
- Cleaned outputs land in `data/cleaned/`; rerunning reuses existing files.

## Data
- Yellow Taxi trips from NYC TLC S3 (`yellow_tripdata_{year}-{month}.parquet`).
- Taxi zones geometry (`taxi_zones.zip`) and lookup metadata (`taxi_zone_lookup.csv`) from NYC TLC.
- Default selection processes January 2025; adjust in `src/utils/paths.py` or by calling `download_months`.

## Developer Guide
- Paths and defaults: `src/utils/paths.py`.
- Data acquisition: `src/utils/get_data.py`; cleaning routines in `src/utils/clean_data.py`.
- Shared geo helpers: `src/utils/common_functions.py`; visual components in `src/components/`.
- Tests live under `tests/`; run `python3 -m pytest`.

## Analysis Highlights
- Dashboard currently surfaces pickup density by zone and log-scale histograms for distance, fare, and tip amount.
- Borough filter and metric selector enable quick spatial comparisons; extendable via new callbacks/pages.

## Copyright
I declare that, unless otherwise noted below, the code in this repository was produced by the project authors. No external code snippets requiring attribution are included at this time.
