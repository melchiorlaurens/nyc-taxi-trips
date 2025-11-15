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


def download_months(periods: Iterable[Tuple[int, int]]) -> List[Path]:
    """
    Download one or several yellow taxi trip months.

    Pass an iterable of (year, month). Example:
        download_months([(2025, 1), (2025, 2)])
    """
    destinations: List[Path] = []
    periods_list = list(periods)
    print(f"Starting download of {len(periods_list)} months of taxi data...")
    for i, (year, month) in enumerate(periods_list, 1):
        url = TRIPDATA_URL_TEMPLATE.format(year=year, month=month)
        dest = yellow_tripdata_path(year, month)
        print(f"[{i}/{len(periods_list)}] Downloading {year}-{month:02d}...")
        destinations.append(_download_file(url, dest))
        print(f"[{i}/{len(periods_list)}] Saved to {dest.name}")
    print(f"Download complete: {len(destinations)} parquet files ready")
    return destinations
# Note: callers should pass explicit periods, e.g. DEFAULT_PERIODS


def download_month(year: int, month: int) -> Path:
    """Convenience wrapper to download a single month."""
    results = download_months([(year, month)])
    return results[0]


def download_assets() -> Path:
    """
    Download the taxi zones zip into the raw folder and extract it locally.

    Returns the directory containing the shapefile parts.
    """
    print("Downloading taxi zones shapefile...")
    zip_path = _download_file(TAXI_ZONES_URL, RAW_TAXI_ZONES_ZIP)
    print(f"Extracting shapefile to {TAXI_ZONES_DIR}...")
    TAXI_ZONES_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(TAXI_ZONES_DIR)
    print("Downloading taxi zone lookup table...")
    _download_file(TAXI_ZONE_LOOKUP_URL, RAW_TAXI_ZONE_LOOKUP_CSV)
    print(f"Assets ready in {RAW_TAXI_ZONE_LOOKUP_CSV.parent}")
    return TAXI_ZONES_DIR
