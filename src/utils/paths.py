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
# Explicit list of (year, month) periods used by downloads and UI filtering.
# Supports cross-year ranges like [(2024, 12), (2025, 1)].
# Default mirrors Jan–Mar 2025; adjust as needed.


DEFAULT_PERIODS = [(DEFAULT_YEAR, month) for month in range(1, 10)]

# Standard output locations
CLEAN_TAXI_ZONES_GEOJSON = CLEAN_DATA_DIR / "taxi_zones_wgs84.geojson"
RAW_TAXI_ZONES_ZIP = RAW_DATA_DIR / "taxi_zones.zip"
CLEAN_TAXI_ZONE_LOOKUP_CSV = CLEAN_DATA_DIR / "taxi_zone_lookup.csv"


def yellow_tripdata_path(year: int, month: int) -> Path:
    """Return the expected parquet path for a given yellow tripdata month."""
    return RAW_DATA_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"

# Directory for monthly cleaned tripdata
CLEAN_YELLOW_MONTHLY_DIR = CLEAN_DATA_DIR / "yellow_monthly"

def clean_yellow_parquet_path(year: int, month: int) -> Path:
    """Return the path for a monthly cleaned parquet (per month)."""
    return CLEAN_YELLOW_MONTHLY_DIR / f"yellow_clean_{year}-{month:02d}.parquet"

# Background image configuration (dashboard theming)
BACKGROUND_DIR = PROJECT_ROOT / "background"
BACKGROUND_IMAGE_NAME = "nyc_background_1.jpg"  # place your image there
BACKGROUND_IMAGE_PATH = BACKGROUND_DIR / BACKGROUND_IMAGE_NAME
