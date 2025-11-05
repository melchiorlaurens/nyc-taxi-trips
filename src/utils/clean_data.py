from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.common_functions import load_shapefile, to_geojson_wgs84
from src.utils.paths import (
    TAXI_ZONES_SHP,
    CLEAN_TAXI_ZONES_GEOJSON,
    CLEAN_YELLOW_PARQUET,
    RAW_TAXI_ZONE_LOOKUP_CSV,
    CLEAN_TAXI_ZONE_LOOKUP_CSV,
)


def make_geojson(raw_dir: Path, clean_dir: Path) -> Path:
    """Convert the taxi zones shapefile to a WGS84 GeoJSON used by the app."""
    gdf = load_shapefile(TAXI_ZONES_SHP)
    return to_geojson_wgs84(gdf, CLEAN_TAXI_ZONES_GEOJSON)


def make_yellow_clean(raw_dir: Path, clean_dir: Path) -> Path:
    """
    Load the yellow trip parquet files found in raw_dir and write the cleaned parquet.

    The cleaning stays minimal for now: concatenate all matching files, keep the
    columns used by the dashboard, and drop rows missing the pickup zone.
    """
    parquet_files = sorted(raw_dir.glob("yellow_tripdata_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Aucun fichier yellow_tripdata_*.parquet dans {raw_dir}")

    frames = [pd.read_parquet(path) for path in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    required_columns: Iterable[str] = [
        "tpep_pickup_datetime",
        "PULocationID",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "passenger_count",
    ]
    existing_cols = [c for c in required_columns if c in df.columns]
    if existing_cols:
        df = df[existing_cols]

    if "tpep_pickup_datetime" in df.columns:
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")

    df = df.dropna(subset=["PULocationID"]).copy()
    df["PULocationID"] = df["PULocationID"].astype("Int64")

    CLEAN_YELLOW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_YELLOW_PARQUET, index=False)
    return CLEAN_YELLOW_PARQUET


def make_zone_lookup(raw_dir: Path, clean_dir: Path) -> Path:
    """Clean the taxi zone lookup CSV and store the standardized version."""
    src = RAW_TAXI_ZONE_LOOKUP_CSV
    if not src.exists():
        raise FileNotFoundError(f"Fichier introuvable : {src}")

    df = pd.read_csv(src)

    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key == "locationid":
            rename_map[col] = "location_id"
        elif key in {"borough", "zone", "service_zone"}:
            rename_map[col] = key
    if rename_map:
        df = df.rename(columns=rename_map)

    if "location_id" in df.columns:
        df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce").astype("Int64")
        df = df.sort_values("location_id").reset_index(drop=True)

    CLEAN_TAXI_ZONE_LOOKUP_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_TAXI_ZONE_LOOKUP_CSV, index=False)
    return CLEAN_TAXI_ZONE_LOOKUP_CSV
