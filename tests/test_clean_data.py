from __future__ import annotations

import pandas as pd
import pytest

from src.utils.clean_data import OutlierThresholds, filter_outliers


def test_outlier_thresholds_defaults():
    """Test that OutlierThresholds has correct default values."""
    thresholds = OutlierThresholds()
    assert thresholds.max_trip_distance == 5000.0
    assert thresholds.max_fare_amount == 10000.0
    assert thresholds.max_tip_amount == 10000.0


def test_outlier_thresholds_custom_values():
    """Test that OutlierThresholds accepts custom values."""
    thresholds = OutlierThresholds(
        max_trip_distance=1000.0,
        max_fare_amount=5000.0,
        max_tip_amount=2000.0
    )
    assert thresholds.max_trip_distance == 1000.0
    assert thresholds.max_fare_amount == 5000.0
    assert thresholds.max_tip_amount == 2000.0


def test_filter_outliers_trip_distance():
    """Test filtering of trip_distance outliers (upper bound only)."""
    df = pd.DataFrame({
        "trip_distance": [0.5, 10.0, 100.0, 5000.0, 5001.0, 10000.0],
        "PULocationID": [1, 2, 3, 4, 5, 6]
    })
    thresholds = OutlierThresholds()

    result = filter_outliers(df, thresholds)

    # Should keep trips <= 5000 miles (including very small values)
    assert len(result) == 4
    assert result["trip_distance"].max() == 5000.0
    assert result["trip_distance"].min() == 0.5  # No lower bound


def test_filter_outliers_fare_amount():
    """Test filtering of fare_amount outliers (upper bound only)."""
    df = pd.DataFrame({
        "fare_amount": [0.01, 5.0, 100.0, 10000.0, 10001.0, 50000.0],
        "PULocationID": [1, 2, 3, 4, 5, 6]
    })
    thresholds = OutlierThresholds()

    result = filter_outliers(df, thresholds)

    # Should keep fares <= $10,000 (including very small values)
    assert len(result) == 4
    assert result["fare_amount"].max() == 10000.0
    assert result["fare_amount"].min() == 0.01  # No lower bound


def test_filter_outliers_tip_amount():
    """Test filtering of tip_amount outliers (upper bound only)."""
    df = pd.DataFrame({
        "tip_amount": [0.0, 2.0, 500.0, 10000.0, 10001.0, 25000.0],
        "PULocationID": [1, 2, 3, 4, 5, 6]
    })
    thresholds = OutlierThresholds()

    result = filter_outliers(df, thresholds)

    # Should keep tips <= $10,000 (including zero values)
    assert len(result) == 4
    assert result["tip_amount"].max() == 10000.0
    assert result["tip_amount"].min() == 0.0  # No lower bound


def test_filter_outliers_all_columns():
    """Test filtering when all three columns have outliers."""
    df = pd.DataFrame({
        "trip_distance": [1.0, 2.0, 6000.0, 4.0, 5.0],
        "fare_amount": [10.0, 20.0, 30.0, 15000.0, 50.0],
        "tip_amount": [2.0, 3.0, 4.0, 5.0, 12000.0],
        "PULocationID": [1, 2, 3, 4, 5]
    })
    thresholds = OutlierThresholds()

    result = filter_outliers(df, thresholds)

    # Should filter rows 2, 3, and 4 (indices with outliers)
    # Row 0: trip_distance=6000 > 5000 -> removed
    # Row 1: fare_amount=15000 > 10000 -> removed
    # Row 2: tip_amount=12000 > 10000 -> removed
    assert len(result) == 2
    assert list(result["PULocationID"]) == [1, 2]


def test_filter_outliers_no_outliers():
    """Test that no rows are removed when there are no outliers."""
    df = pd.DataFrame({
        "trip_distance": [1.0, 2.0, 3.0],
        "fare_amount": [10.0, 20.0, 30.0],
        "tip_amount": [2.0, 3.0, 4.0],
        "PULocationID": [1, 2, 3]
    })
    thresholds = OutlierThresholds()

    result = filter_outliers(df, thresholds)

    assert len(result) == 3
    assert len(result) == len(df)


def test_filter_outliers_empty_dataframe():
    """Test that filtering works on an empty DataFrame."""
    df = pd.DataFrame({
        "trip_distance": [],
        "fare_amount": [],
        "tip_amount": [],
        "PULocationID": []
    })
    thresholds = OutlierThresholds()

    result = filter_outliers(df, thresholds)

    assert len(result) == 0
    assert isinstance(result, pd.DataFrame)


def test_filter_outliers_missing_columns():
    """Test that filtering works when some columns are missing."""
    df = pd.DataFrame({
        "trip_distance": [1.0, 6000.0, 3.0],
        "PULocationID": [1, 2, 3]
    })
    thresholds = OutlierThresholds()

    result = filter_outliers(df, thresholds)

    # Should only filter based on trip_distance
    assert len(result) == 2
    assert 6000.0 not in result["trip_distance"].values


def test_filter_outliers_preserves_other_columns():
    """Test that filtering preserves columns that are not being filtered."""
    df = pd.DataFrame({
        "trip_distance": [1.0, 6000.0, 3.0],
        "fare_amount": [10.0, 20.0, 30.0],
        "PULocationID": [100, 200, 300],
        "passenger_count": [1, 2, 3],
        "extra_column": ["a", "b", "c"]
    })
    thresholds = OutlierThresholds()

    result = filter_outliers(df, thresholds)

    # Should have all original columns
    assert set(result.columns) == set(df.columns)
    # Row with trip_distance=6000 should be removed
    assert len(result) == 2
    assert list(result["PULocationID"]) == [100, 300]
    assert list(result["extra_column"]) == ["a", "c"]


def test_filter_outliers_boundary_values():
    """Test that boundary values are handled correctly (inclusive)."""
    df = pd.DataFrame({
        "trip_distance": [4999.0, 5000.0, 5001.0],
        "fare_amount": [9999.0, 10000.0, 10001.0],
        "tip_amount": [9999.0, 10000.0, 10001.0],
        "PULocationID": [1, 2, 3]
    })
    thresholds = OutlierThresholds()

    result = filter_outliers(df, thresholds)

    # Should keep rows 0 and 1 (values <= threshold), row 2 exceeds all thresholds
    assert len(result) == 2
    assert list(result["PULocationID"]) == [1, 2]
    # Boundary values (exactly at threshold) should be kept
    assert result.iloc[1]["trip_distance"] == 5000.0
    assert result.iloc[1]["fare_amount"] == 10000.0
    assert result.iloc[1]["tip_amount"] == 10000.0


def test_filter_outliers_with_custom_thresholds():
    """Test filtering with custom threshold values."""
    df = pd.DataFrame({
        "trip_distance": [10.0, 50.0, 100.0, 200.0],
        "fare_amount": [10.0, 50.0, 100.0, 200.0],
        "PULocationID": [1, 2, 3, 4]
    })
    thresholds = OutlierThresholds(
        max_trip_distance=100.0,
        max_fare_amount=100.0
    )

    result = filter_outliers(df, thresholds)

    # Should keep only rows where both values <= 100
    assert len(result) == 3
    assert result["trip_distance"].max() == 100.0
    assert result["fare_amount"].max() == 100.0
