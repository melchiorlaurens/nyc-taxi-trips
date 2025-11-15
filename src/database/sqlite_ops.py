from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from src.utils.paths import DATABASE_DIR, SQLITE_DB_PATH

TABLE_NAME = "yellow_tripdata_raw"


def sync_sqlite_database(parquet_paths: Iterable[Path], db_path: Optional[Path] = None) -> Path:
    """
    Ensure the SQLite database mirrors the provided parquet files.

    Rebuilds the database when it is missing or when any parquet file is newer.
    """
    db_path = Path(db_path or SQLITE_DB_PATH)
    parquet_list: List[Path] = [Path(p) for p in parquet_paths if Path(p).exists()]
    if not parquet_list:
        raise FileNotFoundError("No parquet files found to populate the SQLite database.")

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    rebuild = True
    if db_path.exists():
        db_mtime = db_path.stat().st_mtime
        rebuild = any(p.stat().st_mtime > db_mtime for p in parquet_list)

    if not rebuild:
        return db_path

    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        for parquet_path in parquet_list:
            df = pd.read_parquet(parquet_path)
            if df.empty:
                continue
            df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

    return db_path


def read_month_from_sqlite(
    year: int,
    month: int,
    columns: Optional[Sequence[str]] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return a DataFrame for a given (year, month) pulled from SQLite."""
    db_path = Path(db_path or SQLITE_DB_PATH)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    month_str = f"{month:02d}"
    column_clause = "*"
    if columns:
        safe_cols = [c for c in columns if isinstance(c, str) and c.strip()]
        if safe_cols:
            quoted = ", ".join(f'"{c}"' for c in safe_cols)
            column_clause = quoted

    query = f"""
        SELECT {column_clause}
        FROM {TABLE_NAME}
        WHERE strftime('%Y', tpep_pickup_datetime) = ?
          AND strftime('%m', tpep_pickup_datetime) = ?
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(str(year), month_str))

    return df
