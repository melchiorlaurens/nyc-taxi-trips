# SQLite Database Integration

This module provides SQLite database functionality for the NYC Taxi Trips project. The SQLite database serves as an intermediate storage layer for demonstration purposes, creating a round-trip data flow:

```
Raw Parquet Files → SQLite Database → Pandas DataFrames → Cleaned Parquet Files → Dashboard
```

## Architecture

### Directory Structure

```
src/database/
├── __init__.py           # Module exports
├── sqlite_ops.py         # Core SQLite operations
├── cli.py               # Command-line interface
└── README.md            # This file

data/database/
└── nyc_taxi.db          # SQLite database file (created automatically)
```

### Data Flow

1. **Download Phase** ([get_data.py](../utils/get_data.py))
   - Downloads raw parquet files from NYC Open Data
   - Automatically loads them into SQLite database

2. **Storage Phase** (SQLite Database)
   - All raw data stored in `yellow_tripdata` table
   - Database location: `data/database/nyc_taxi.db`

3. **Export Phase** ([clean_data.py](../utils/clean_data.py))
   - Reads data from SQLite database
   - Applies cleaning and filtering
   - Saves cleaned monthly parquet files

4. **Dashboard Phase** ([main.py](../../main.py))
   - Reads cleaned parquet files
   - Displays interactive visualizations

## Usage

### Automatic (Recommended)

The SQLite round-trip happens automatically when you run:

```bash
python main.py
```

This will:
1. Download raw parquet files (if not already present)
2. Load them into SQLite database
3. Export from SQLite and create cleaned files
4. Launch the dashboard

### Manual Management

You can also manage the database manually using the CLI tool:

#### Check Database Status

```bash
python -m src.database.cli info
```

Output:
```
============================================================
NYC Taxi SQLite Database Info
============================================================

Database path: /Users/melchior/nyc-taxi-trips/data/database/nyc_taxi.db
Status: Active
Total rows: 27,982,347
Columns: 20

Column names:
  - VendorID
  - tpep_pickup_datetime
  - ...
```

#### Load Parquet Files into Database

```bash
python -m src.database.cli load
```

Force reload (replace existing data):
```bash
python -m src.database.cli load --force
```

#### Export Database to Parquet

```bash
python -m src.database.cli export output.parquet
```

### Testing

Test the full round-trip workflow:

```bash
python test_full_roundtrip.py
```

Expected output:
```
======================================================================
Testing Full SQLite Round-Trip Workflow
======================================================================

[Step 1] Checking SQLite database...
✓ Database exists with 27,982,347 rows
✓ Columns: VendorID, tpep_pickup_datetime, ...

[Step 2] Reading from SQLite and creating cleaned monthly files...
[Cleaning] Reading data from SQLite database...
[SQLite] Exported 27,982,347 rows from 'yellow_tripdata' table
Filtered 1,308 outlier records (0.00%)
[Cleaning] Saved 3,475,126 cleaned rows for 2025-01
...

SUCCESS: Full round-trip workflow completed!
```

## API Reference

### Core Functions

#### `load_parquet_to_sqlite(parquet_path, db_path, table_name='yellow_tripdata')`

Load a single parquet file into SQLite database.

**Parameters:**
- `parquet_path` (Path): Path to the parquet file
- `db_path` (Path): Path to the SQLite database
- `table_name` (str): Name of the table to create/append to
- `if_exists` (str): 'fail', 'replace', or 'append'

#### `load_multiple_parquets_to_sqlite(parquet_paths, db_path, table_name='yellow_tripdata')`

Load multiple parquet files into a single SQLite table.

**Parameters:**
- `parquet_paths` (List[Path]): List of parquet file paths
- `db_path` (Path): Path to the SQLite database
- `table_name` (str): Name of the table
- `replace_existing` (bool): Replace existing table if True

#### `export_sqlite_to_pandas(db_path, table_name='yellow_tripdata', year_month=None)`

Export data from SQLite to a pandas DataFrame.

**Parameters:**
- `db_path` (Path): Path to the SQLite database
- `table_name` (str): Name of the table to export
- `query` (str, optional): Custom SQL query
- `year_month` (str, optional): Filter by year-month (e.g., "2025-01")

**Returns:**
- DataFrame with the exported data

#### `get_table_info(db_path, table_name='yellow_tripdata')`

Get information about a table in the database.

**Returns:**
- Dictionary with table info (row count, columns, etc.)

### Example Usage

```python
from pathlib import Path
from src.database.sqlite_ops import (
    load_multiple_parquets_to_sqlite,
    export_sqlite_to_pandas,
    get_table_info
)
from src.utils.paths import SQLITE_DB_PATH, RAW_DATA_DIR

# Load parquet files into SQLite
parquet_files = list(RAW_DATA_DIR.glob("yellow_tripdata_*.parquet"))
load_multiple_parquets_to_sqlite(parquet_files, SQLITE_DB_PATH)

# Check database info
info = get_table_info(SQLITE_DB_PATH)
print(f"Database has {info['row_count']:,} rows")

# Export to pandas
df = export_sqlite_to_pandas(SQLITE_DB_PATH)
print(f"Loaded {len(df):,} rows")

# Export specific month
df_jan = export_sqlite_to_pandas(SQLITE_DB_PATH, year_month="2025-01")
print(f"January 2025: {len(df_jan):,} rows")
```

## Configuration

### Disable SQLite Round-Trip

If you want to skip the SQLite round-trip and use parquet files directly:

In [get_data.py](../utils/get_data.py):
```python
download_months(DEFAULT_PERIODS, use_sqlite=False)
```

In [clean_data.py](../utils/clean_data.py):
```python
make_yellow_clean(RAW_DATA_DIR, CLEAN_DATA_DIR, use_sqlite=False)
```

### Database Location

The database location is configured in [paths.py](../utils/paths.py):

```python
DB_DIR = DATA_DIR / "database"
SQLITE_DB_PATH = DB_DIR / "nyc_taxi.db"
```

## Performance Notes

- **Loading**: Loading ~28 million rows from parquet to SQLite takes ~1-2 minutes
- **Exporting**: Exporting from SQLite to pandas takes ~30-60 seconds
- **Database Size**: ~2-3 GB for 9 months of taxi data
- **Parquet Advantage**: Parquet files are faster for read-heavy workloads, but SQLite adds SQL query capabilities

## Troubleshooting

### Database Locked Error

If you get a "database is locked" error:
1. Make sure no other Python process is accessing the database
2. Close any SQLite browser tools
3. Try again

### Force Rebuild

To force rebuild the database:
```bash
python -m src.database.cli load --force
```

### Check Data Integrity

After loading, verify the data:
```bash
python -m src.database.cli info
```

## Future Enhancements

Possible improvements:
- Add indexing for faster queries
- Support for incremental updates
- Query optimization with EXPLAIN ANALYZE
- Connection pooling for concurrent access
- Migration to PostgreSQL for production use
