from __future__ import annotations

from pathlib import Path
import geopandas as gpd


def load_shapefile(shapefile_path: Path) -> gpd.GeoDataFrame:
    """Return the GeoDataFrame for the provided shapefile."""
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile missing: {shapefile_path}")
    return gpd.read_file(shapefile_path)


def to_geojson_wgs84(gdf: gpd.GeoDataFrame, output_path: Path) -> Path:
    """Project the GeoDataFrame to WGS84 and export as GeoJSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_crs(4326).to_file(output_path, driver="GeoJSON")
    return output_path
