from __future__ import annotations

from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "cleaned"
TAXI_ZONES_DIR = PROJECT_ROOT / "taxi_zones"
TAXI_ZONES_SHP = TAXI_ZONES_DIR / "taxi_zones.shp"
RAW_TAXI_ZONE_LOOKUP_CSV = RAW_DATA_DIR / "taxi_zone_lookup.csv"

# Defaults for downloads / processing
DEFAULT_YEAR = 2025
DEFAULT_MONTH = 1

# Standard output locations
CLEAN_TAXI_ZONES_GEOJSON = CLEAN_DATA_DIR / "taxi_zones_wgs84.geojson"
CLEAN_YELLOW_PARQUET = CLEAN_DATA_DIR / "yellow_clean.parquet"
RAW_TAXI_ZONES_ZIP = RAW_DATA_DIR / "taxi_zones.zip"
CLEAN_TAXI_ZONE_LOOKUP_CSV = CLEAN_DATA_DIR / "taxi_zone_lookup.csv"


def yellow_tripdata_path(year: int, month: int) -> Path:
    """Return the expected parquet path for a given yellow tripdata month."""
    return RAW_DATA_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"
