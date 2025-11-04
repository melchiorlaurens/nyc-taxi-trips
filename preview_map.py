# preview_map.py
import json, pandas as pd, geopandas as gpd, plotly.express as px

# 1) fond de carte déjà généré
zones = gpd.read_file("taxi_zones_wgs84.geojson")  # à changer s'il a été déplacé
geojson = json.loads(zones.to_json())

# 2) agrégat rapide sur les yellow trips
df = pd.read_parquet("data/yellow_tripdata_2025-01.parquet", columns=["PULocationID"])
agg = df.value_counts("PULocationID").rename("n_trips").reset_index()
agg = agg.rename(columns={"PULocationID": "LocationID"})

# 3) jointure + carte
gdf = zones.merge(agg, on="LocationID", how="left").fillna({"n_trips": 0})
fig = px.choropleth(
    gdf,
    geojson=geojson,
    locations="LocationID",
    featureidkey="properties.LocationID",
    color="n_trips",
    hover_name="zone",
    hover_data={"borough": True, "LocationID": True, "n_trips": True},
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title="Pickups par Taxi Zone (Yellow)")
fig.show()
