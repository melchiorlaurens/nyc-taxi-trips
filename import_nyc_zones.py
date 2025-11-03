from pathlib import Path
import geopandas as gpd
from pyproj import CRS



# ★ 1) Dossier réel du script (fiable même lancé depuis ailleurs)
source = Path(__file__).resolve().parent

# Chemin vers le dossier DÉJÀ DÉZIPPE contenant .shp/.shx/.dbf/.prj
SRC_PATH = source / "taxi_zones"
# Chemin de sortie du GeoJSON (WGS84)
OUT_PATH = source / "taxi_zones_wgs84.geojson"


def find_shp(src: Path) -> Path:
    """Retourne le chemin du .shp à partir d'un dossier ou d'un .shp direct."""
    if src.is_file() and src.suffix.lower() == ".shp":
        return src
    if src.is_dir():
        # ★ 2) cherche d'abord à la racine, puis récursivement
        shp = next(src.glob("*.shp"), None) or next(src.rglob("*.shp"), None)
        if shp:
            return shp
    raise FileNotFoundError(
        f"Aucun .shp trouvé. Donne un dossier contenant un .shp ou un chemin .shp.\n"
        f"SRC_PATH actuel: {src}\n"
    )


def read_with_crs(shp_path: Path) -> gpd.GeoDataFrame:
    """Lit le shapefile et gère le cas CRS manquant en lisant le .prj à côté."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs:
        return gdf  # CRS déjà connu

    # CRS manquant → tenter .prj à côté du .shp
    prj_path = shp_path.with_suffix(".prj")
    if prj_path.exists():
        try:
            crs = CRS.from_wkt(prj_path.read_text(errors="ignore"))
            return gpd.read_file(shp_path).set_crs(crs)
        except Exception as e:
            raise RuntimeError(f"Impossible de définir le CRS depuis {prj_path.name}: {e}")
    raise RuntimeError(
        "CRS manquant et aucun .prj trouvé. Place .shp/.shx/.dbf/.prj ensemble ou indique un EPSG connu."
    )

def main():

    # Aide debug rapide
    print("Script dir :", source)
    print("SRC_PATH   :", SRC_PATH.resolve())

    shp = find_shp(SRC_PATH)
    print("Shapefile trouvé :", shp)

    gdf = read_with_crs(shp)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Reprojeter en WGS84 (EPSG:4326) et exporter en GeoJSON (lon/lat)
    gdf.to_crs(4326).to_file(OUT_PATH, driver="GeoJSON")

    print(f"Écrit : {OUT_PATH.resolve()}")
    print("Colonnes :", list(gdf.columns))
    if "LocationID" in gdf.columns:
        print("OK : 'LocationID' = PU/DOLocationID (Yellow).")

if __name__ == "__main__":
    main()