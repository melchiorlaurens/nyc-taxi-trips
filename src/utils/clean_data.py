from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import BaseModel, Field

from src.utils.common_functions import load_shapefile, to_geojson_wgs84
from src.utils.paths import (
    TAXI_ZONES_SHP,
    CLEAN_TAXI_ZONES_GEOJSON,
    CLEAN_YELLOW_MONTHLY_DIR,
    clean_yellow_parquet_path,
    RAW_TAXI_ZONE_LOOKUP_CSV,
    CLEAN_TAXI_ZONE_LOOKUP_CSV,
    SQLITE_DB_PATH,
)
from src.database.sqlite_ops import export_sqlite_to_pandas, get_table_info


class OutlierThresholds(BaseModel):
    """Defines the threshold values for outlier filtering."""

    max_trip_distance: float = Field(
        default=5000.0,
        description="Maximum trip distance in miles. Trips above this are discarded."
    )
    max_fare_amount: float = Field(
        default=10000.0,
        description="Maximum fare amount in dollars. Fares above this are discarded."
    )
    max_tip_amount: float = Field(
        default=10000.0,
        description="Maximum tip amount in dollars. Tips above this are discarded."
    )


def filter_outliers(df: pd.DataFrame, thresholds: OutlierThresholds) -> pd.DataFrame:
    """
    Filter out outlier records based on defined thresholds.

    Args:
        df: DataFrame to filter
        thresholds: OutlierThresholds instance with filtering rules

    Returns:
        Filtered DataFrame with outliers removed
    """
    initial_count = len(df)

    # Filter trip distance (only upper bound)
    if "trip_distance" in df.columns:
        df = df[df["trip_distance"] <= thresholds.max_trip_distance]

    # Filter fare amount (only upper bound)
    if "fare_amount" in df.columns:
        df = df[df["fare_amount"] <= thresholds.max_fare_amount]

    # Filter tip amount (only upper bound)
    if "tip_amount" in df.columns:
        df = df[df["tip_amount"] <= thresholds.max_tip_amount]

    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"Filtered {filtered_count:,} outlier records ({filtered_count/initial_count*100:.2f}%)")

    return df


def make_geojson(raw_dir: Path, clean_dir: Path) -> Path:
    """Convert the taxi zones shapefile to a WGS84 GeoJSON used by the app."""
    gdf = load_shapefile(TAXI_ZONES_SHP)
    return to_geojson_wgs84(gdf, CLEAN_TAXI_ZONES_GEOJSON)


def make_yellow_clean(raw_dir: Path, clean_dir: Path, use_sqlite: bool = True) -> Path:
    """
    Load the yellow trip data and write the cleaned parquet files.

    The data is loaded from SQLite database (if use_sqlite=True) or from parquet files.
    The cleaning includes: concatenate all matching files, keep the columns used by
    the dashboard, drop rows missing the pickup zone, and filter outliers.

    Args:
        raw_dir: Directory containing raw parquet files (used as fallback)
        clean_dir: Directory where cleaned files will be written
        use_sqlite: If True, read from SQLite database; if False, read from parquet files

    Returns:
        Path to the directory containing cleaned monthly parquet files
    """
    # Assure le répertoire des nettoyés mensuels
    CLEAN_YELLOW_MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize outlier thresholds
    thresholds = OutlierThresholds()

    # Determine data source
    if use_sqlite and SQLITE_DB_PATH.exists():
        db_info = get_table_info(SQLITE_DB_PATH)
        if db_info.get("exists"):
            print("[Cleaning] Reading data from SQLite database...")
            return _make_yellow_clean_from_sqlite(thresholds)

    # Fallback to parquet files
    print("[Cleaning] Reading data from parquet files...")
    return _make_yellow_clean_from_parquet(raw_dir, thresholds)


def _make_yellow_clean_from_sqlite(thresholds: OutlierThresholds) -> Path:
    """
    Load data from SQLite and create cleaned monthly parquet files.

    Args:
        thresholds: Outlier filtering thresholds

    Returns:
        Path to the directory containing cleaned monthly parquet files
    """
    # Export all data from SQLite
    df_all = export_sqlite_to_pandas(SQLITE_DB_PATH, table_name="yellow_tripdata")

    if df_all.empty:
        print("[Cleaning] No data found in SQLite database")
        return CLEAN_YELLOW_MONTHLY_DIR

    # Process columns
    required_columns: Iterable[str] = [
        "tpep_pickup_datetime",
        "PULocationID",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "passenger_count",
    ]
    existing_cols = [c for c in required_columns if c in df_all.columns]
    if existing_cols:
        df_all = df_all[existing_cols]

    if "tpep_pickup_datetime" in df_all.columns:
        df_all["tpep_pickup_datetime"] = pd.to_datetime(df_all["tpep_pickup_datetime"], errors="coerce")
        df_all["year_month"] = df_all["tpep_pickup_datetime"].dt.to_period("M").astype(str)

    df_all = df_all.dropna(subset=["PULocationID"]).copy()
    df_all["PULocationID"] = df_all["PULocationID"].astype("Int64")

    # Filter outliers on the entire dataset
    df_all = filter_outliers(df_all, thresholds)

    # Split by month and save individual monthly files
    if "year_month" in df_all.columns:
        for ym in df_all["year_month"].unique():
            if pd.isna(ym):
                continue
            try:
                year = int(str(ym).split("-")[0])
                month = int(str(ym).split("-")[1])
                dfm = df_all[df_all["year_month"] == ym]
                monthly_path = clean_yellow_parquet_path(year, month)
                dfm.to_parquet(monthly_path, index=False)
                print(f"[Cleaning] Saved {len(dfm):,} cleaned rows for {ym}")
            except Exception as e:
                print(f"[Cleaning] Error processing {ym}: {e}")
                continue

    return CLEAN_YELLOW_MONTHLY_DIR


def _make_yellow_clean_from_parquet(raw_dir: Path, thresholds: OutlierThresholds) -> Path:
    """
    Load data from parquet files and create cleaned monthly parquet files.

    Args:
        raw_dir: Directory containing raw parquet files
        thresholds: Outlier filtering thresholds

    Returns:
        Path to the directory containing cleaned monthly parquet files
    """
    parquet_files = sorted(raw_dir.glob("yellow_tripdata_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Aucun fichier yellow_tripdata_*.parquet dans {raw_dir}")

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

        # Filter outliers
        dfm = filter_outliers(dfm, thresholds)

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
