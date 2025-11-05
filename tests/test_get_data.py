from __future__ import annotations

import io
from pathlib import Path
import zipfile

import pytest

from src.utils import get_data


class DummyResponse:
    """Minimal Response stub for requests.get."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 1024) -> bytes:
        for start in range(0, len(self._payload), chunk_size):
            yield self._payload[start : start + chunk_size]


def test_download_month_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "yellow_2024-03.parquet"
    content = b"parquet-binary"

    def fake_get(url: str, stream: bool = True, timeout: int = 60) -> DummyResponse:
        return DummyResponse(content)

    monkeypatch.setattr(get_data.requests, "get", fake_get)
    monkeypatch.setattr(get_data, "yellow_tripdata_path", lambda y, m: target)

    result_path = get_data.download_month(2024, 3)

    assert result_path == target
    assert result_path.exists()
    assert result_path.read_bytes() == content


def test_download_assets_extracts_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zones_zip = tmp_path / "raw" / "taxi_zones.zip"
    extract_dir = tmp_path / "taxi_zones"
    lookup_csv = tmp_path / "raw" / "taxi_zone_lookup.csv"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as zf:
        zf.writestr("taxi_zones.shp", b"dummy shp data")
    payload = buffer.getvalue()
    lookup_payload = b"LocationID,Borough,Zone,service_zone\n1,Test,Zone A,Yellow\n"

    def fake_get(url: str, stream: bool = True, timeout: int = 60) -> DummyResponse:
        if url.endswith(".zip"):
            return DummyResponse(payload)
        if url.endswith(".csv"):
            return DummyResponse(lookup_payload)
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(get_data.requests, "get", fake_get)
    monkeypatch.setattr(get_data, "RAW_TAXI_ZONES_ZIP", zones_zip)
    monkeypatch.setattr(get_data, "TAXI_ZONES_DIR", extract_dir)
    monkeypatch.setattr(get_data, "RAW_TAXI_ZONE_LOOKUP_CSV", lookup_csv)

    out_dir = get_data.download_assets()

    assert out_dir == extract_dir
    shp_path = extract_dir / "taxi_zones.shp"
    assert shp_path.exists()
    assert shp_path.read_bytes() == b"dummy shp data"
    assert lookup_csv.exists()
    assert lookup_csv.read_bytes() == lookup_payload
