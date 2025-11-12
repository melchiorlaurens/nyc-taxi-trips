"""SQLite operations for NYC taxi data.

This module provides functions for a demonstration round-trip:
  Parquet files → SQLite database → Pandas DataFrames

The SQLite database serves as an intermediate storage layer for demo purposes.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd


def get_connection(db_path: Path) -> sqlite3.Connection:
    """
    Create and return a connection to the SQLite database.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        SQLite connection object
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    return conn


def load_parquet_to_sqlite(
    parquet_path: Path,
    db_path: Path,
    table_name: str = "yellow_tripdata",
    if_exists: str = "append",
) -> None:
    """
    Load a parquet file into a SQLite database table.

    Args:
        parquet_path: Path to the parquet file to load
        db_path: Path to the SQLite database file
        table_name: Name of the table to create/append to
        if_exists: How to behave if table exists ('fail', 'replace', 'append')
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    print(f"[SQLite] Loading {parquet_path.name} into database...")

    # Read parquet file
    df = pd.read_parquet(parquet_path)

    # Connect to SQLite and write data
    conn = get_connection(db_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        print(f"[SQLite] Loaded {len(df):,} rows from {parquet_path.name}")
    finally:
        conn.close()


def load_multiple_parquets_to_sqlite(
    parquet_paths: List[Path],
    db_path: Path,
    table_name: str = "yellow_tripdata",
    replace_existing: bool = True,
) -> None:
    """
    Load multiple parquet files into a single SQLite database table.

    Args:
        parquet_paths: List of paths to parquet files
        db_path: Path to the SQLite database file
        table_name: Name of the table to create/append to
        replace_existing: If True, replace existing table; if False, append
    """
    if not parquet_paths:
        print("[SQLite] No parquet files to load")
        return

    print(f"[SQLite] Loading {len(parquet_paths)} parquet files into {db_path.name}...")

    if_exists = "replace" if replace_existing else "append"

    for i, parquet_path in enumerate(parquet_paths):
        # Use 'replace' for first file if replacing, 'append' for rest
        current_if_exists = if_exists if i == 0 else "append"
        load_parquet_to_sqlite(parquet_path, db_path, table_name, current_if_exists)

    # Show total count
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        print(f"[SQLite] Database now contains {total_rows:,} total rows in '{table_name}' table")
    finally:
        conn.close()


def export_sqlite_to_pandas(
    db_path: Path,
    table_name: str = "yellow_tripdata",
    query: Optional[str] = None,
    year_month: Optional[str] = None,
) -> pd.DataFrame:
    """
    Export data from SQLite database to a pandas DataFrame.

    Args:
        db_path: Path to the SQLite database file
        table_name: Name of the table to export from
        query: Optional custom SQL query (overrides table_name)
        year_month: Optional year-month filter (e.g., "2025-01")

    Returns:
        DataFrame containing the exported data
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = get_connection(db_path)
    try:
        if query:
            df = pd.read_sql_query(query, conn)
        elif year_month:
            # Filter by year_month column if provided
            query = f"""
                SELECT * FROM {table_name}
                WHERE year_month = ?
            """
            df = pd.read_sql_query(query, conn, params=(year_month,))
            print(f"[SQLite] Exported {len(df):,} rows for {year_month}")
        else:
            # Export entire table
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(f"[SQLite] Exported {len(df):,} rows from '{table_name}' table")

        return df
    finally:
        conn.close()


def export_sqlite_to_parquet(
    db_path: Path,
    output_path: Path,
    table_name: str = "yellow_tripdata",
    year_month: Optional[str] = None,
) -> Path:
    """
    Export data from SQLite database directly to a parquet file.

    Args:
        db_path: Path to the SQLite database file
        output_path: Path where the parquet file will be saved
        table_name: Name of the table to export from
        year_month: Optional year-month filter (e.g., "2025-01")

    Returns:
        Path to the created parquet file
    """
    df = export_sqlite_to_pandas(db_path, table_name, year_month=year_month)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"[SQLite] Saved {len(df):,} rows to {output_path}")
    return output_path


def get_table_info(db_path: Path, table_name: str = "yellow_tripdata") -> dict:
    """
    Get information about a table in the SQLite database.

    Args:
        db_path: Path to the SQLite database file
        table_name: Name of the table to inspect

    Returns:
        Dictionary with table information (row count, columns, etc.)
    """
    if not db_path.exists():
        return {"exists": False}

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        if not cursor.fetchone():
            return {"exists": False}

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]

        return {
            "exists": True,
            "row_count": row_count,
            "columns": columns,
        }
    finally:
        conn.close()
