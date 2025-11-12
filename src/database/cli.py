"""CLI utility for managing the NYC Taxi SQLite database."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.paths import RAW_DATA_DIR, SQLITE_DB_PATH
from src.database.sqlite_ops import (
    get_table_info,
    load_multiple_parquets_to_sqlite,
    export_sqlite_to_pandas,
)


def info_command():
    """Show information about the database."""
    print("=" * 60)
    print("NYC Taxi SQLite Database Info")
    print("=" * 60)
    print(f"\nDatabase path: {SQLITE_DB_PATH}")

    if not SQLITE_DB_PATH.exists():
        print("Status: Database file does not exist")
        return

    db_info = get_table_info(SQLITE_DB_PATH)

    if not db_info.get("exists"):
        print("Status: Database exists but table 'yellow_tripdata' not found")
        return

    print(f"Status: Active")
    print(f"Total rows: {db_info.get('row_count', 0):,}")
    print(f"Columns: {len(db_info.get('columns', []))}")
    print("\nColumn names:")
    for col in db_info.get("columns", []):
        print(f"  - {col}")


def load_command(force: bool = False):
    """Load parquet files into the database."""
    parquet_files = sorted(RAW_DATA_DIR.glob("yellow_tripdata_*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {RAW_DATA_DIR}")
        print("Run 'python main.py' first to download the data")
        return

    print(f"Found {len(parquet_files)} parquet files")

    if not force:
        db_info = get_table_info(SQLITE_DB_PATH)
        if db_info.get("exists"):
            print(f"Database already exists with {db_info.get('row_count', 0):,} rows")
            print("Use --force to reload the database")
            return

    load_multiple_parquets_to_sqlite(
        parquet_paths=parquet_files,
        db_path=SQLITE_DB_PATH,
        table_name="yellow_tripdata",
        replace_existing=True,
    )
    print("Load complete!")


def export_command(output_file: str):
    """Export database to a parquet file."""
    output_path = Path(output_file)

    if not SQLITE_DB_PATH.exists():
        print("Database does not exist. Run 'load' command first.")
        return

    print(f"Exporting database to {output_path}...")
    df = export_sqlite_to_pandas(SQLITE_DB_PATH)
    df.to_parquet(output_path, index=False)
    print(f"Exported {len(df):,} rows to {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage NYC Taxi SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Info command
    subparsers.add_parser("info", help="Show database information")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load parquet files into database")
    load_parser.add_argument(
        "--force", action="store_true", help="Force reload even if database exists"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export database to parquet")
    export_parser.add_argument("output", help="Output parquet file path")

    args = parser.parse_args()

    if args.command == "info":
        info_command()
    elif args.command == "load":
        load_command(force=args.force)
    elif args.command == "export":
        export_command(args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
