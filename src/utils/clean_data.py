from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.common_functions import load_shapefile, to_geojson_wgs84
from src.utils.paths import (
    TAXI_ZONES_SHP,
    CLEAN_TAXI_ZONES_GEOJSON,
    CLEAN_YELLOW_MONTHLY_DIR,
    clean_yellow_parquet_path,
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

    # Assure le répertoire des nettoyés mensuels
    CLEAN_YELLOW_MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    monthly_frames = []
    for path in parquet_files:
        # Déduire (year, month) à partir du nom de fichier
        name = path.name
        try:
            stem = name.replace(".parquet", "")
            ym = stem.split("_")[-1]  # '2025-03'
            year = int(ym.split("-")[0])
            month = int(ym.split("-")[1])
        except Exception:
            # Si parsing échoue, continuer mais sans sortie mensuelle dédiée
            year = month = None

        dfm = pd.read_parquet(path)

        required_columns: Iterable[str] = [
            "tpep_pickup_datetime",
            "PULocationID",
            "trip_distance",
            "fare_amount",
            "tip_amount",
            "passenger_count",
        ]
        existing_cols = [c for c in required_columns if c in dfm.columns]
        if existing_cols:
            dfm = dfm[existing_cols]

        if "tpep_pickup_datetime" in dfm.columns:
            dfm["tpep_pickup_datetime"] = pd.to_datetime(dfm["tpep_pickup_datetime"], errors="coerce")
            dfm["year_month"] = dfm["tpep_pickup_datetime"].dt.to_period("M").astype(str)

        dfm = dfm.dropna(subset=["PULocationID"]).copy()
        dfm["PULocationID"] = dfm["PULocationID"].astype("Int64")

        # Écrit un parquet nettoyé mensuel si (year, month) reconnu
        if year is not None and month is not None:
            monthly_path = clean_yellow_parquet_path(year, month)
            dfm.to_parquet(monthly_path, index=False)

        monthly_frames.append(dfm)

    # Concatène et écrit le parquet global combiné pour la simplicité du dashboard
    df = pd.concat(monthly_frames, ignore_index=True)

    # Retourne le répertoire contenant les nettoyés mensuels
    return CLEAN_YELLOW_MONTHLY_DIR


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
