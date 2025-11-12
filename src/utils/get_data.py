from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, List
import zipfile

import requests

from src.utils.paths import (
    RAW_TAXI_ZONES_ZIP,
    TAXI_ZONES_DIR,
    RAW_TAXI_ZONE_LOOKUP_CSV,
    yellow_tripdata_path,
    SQLITE_DB_PATH,
)
from src.database.sqlite_ops import (
    load_multiple_parquets_to_sqlite,
    get_table_info,
)

TRIPDATA_URL_TEMPLATE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
TAXI_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def _download_file(url: str, destination: Path) -> Path:
    """Download url into destination if it is missing."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with destination.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                fh.write(chunk)
    return destination


def download_months(periods: Iterable[Tuple[int, int]], use_sqlite: bool = True) -> List[Path]:
    """
    Download one or several yellow taxi trip months.

    Pass an iterable of (year, month). Example:
        download_months([(2025, 1), (2025, 2)])

    Args:
        periods: Iterable of (year, month) tuples to download
        use_sqlite: If True, load downloaded parquets into SQLite database

    Returns:
        List of paths to downloaded parquet files
    """
    destinations: List[Path] = []
    for year, month in periods:
        url = TRIPDATA_URL_TEMPLATE.format(year=year, month=month)
        dest = yellow_tripdata_path(year, month)
        destinations.append(_download_file(url, dest))

    # SQLite round-trip for demo purposes
    if use_sqlite and destinations:
        sync_parquets_to_sqlite(destinations)

    return destinations
# Note: callers should pass explicit periods, e.g. DEFAULT_PERIODS


def sync_parquets_to_sqlite(parquet_paths: List[Path], force_reload: bool = False) -> None:
    """
    Load parquet files into SQLite database.

    This creates a demonstration round-trip where parquet data is stored in SQLite
    before being exported back to pandas DataFrames for processing.

    Args:
        parquet_paths: List of parquet files to load into SQLite
        force_reload: If True, replace existing data; if False, skip if DB exists
    """
    if not parquet_paths:
        return

    # Check if database already exists and has data
    db_info = get_table_info(SQLITE_DB_PATH)

    if db_info.get("exists") and not force_reload:
        print(f"[SQLite] Database already exists with {db_info.get('row_count', 0):,} rows")
        print("[SQLite] Skipping reload (use force_reload=True to rebuild)")
        return

    print("[SQLite] Starting SQLite round-trip demonstration...")
    print(f"[SQLite] Step 1/2: Loading {len(parquet_paths)} parquet files into SQLite database")

    # Load all parquets into SQLite
    load_multiple_parquets_to_sqlite(
        parquet_paths=parquet_paths,
        db_path=SQLITE_DB_PATH,
        table_name="yellow_tripdata",
        replace_existing=True,
    )

    print("[SQLite] Step 2/2: Data now ready to be exported from SQLite to pandas")
    print(f"[SQLite] Database location: {SQLITE_DB_PATH}")
    print("[SQLite] Round-trip complete!")


def download_assets() -> Path:
    """
    Download the taxi zones zip into the raw folder and extract it locally.

    Returns the directory containing the shapefile parts.
    """
    zip_path = _download_file(TAXI_ZONES_URL, RAW_TAXI_ZONES_ZIP)
    TAXI_ZONES_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(TAXI_ZONES_DIR)
    _download_file(TAXI_ZONE_LOOKUP_URL, RAW_TAXI_ZONE_LOOKUP_CSV)
    return TAXI_ZONES_DIR
