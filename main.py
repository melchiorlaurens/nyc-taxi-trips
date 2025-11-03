# main.py — Dashboard Dash (carte choroplèthe + histogramme)
# Hypothèses :
#  - taxi_zones_wgs84.geojson est à la racine
#  - il existe EXACTEMENT un fichier yellow_tripdata_*.parquet dans le dossier data

from pathlib import Path
import json
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import numpy as np

ROOT = Path(__file__).resolve().parent

# --- GeoJSON à la racine ---
GEOJSON_FILE = ROOT / "taxi_zones_wgs84.geojson"
if not GEOJSON_FILE.exists():
    raise FileNotFoundError("Introuvable : taxi_zones_wgs84.geojson (attendu à la racine).")

# --- Parquet : il doit y en avoir exactement un ---
matches = list((ROOT / "data").glob("yellow_tripdata_*.parquet"))
if len(matches) == 0:
    raise FileNotFoundError("Aucun data/yellow_tripdata_*.parquet trouvé.")
if len(matches) > 1:
    names = ", ".join(m.name for m in matches)
    raise RuntimeError(f"Plus d'un parquet trouvé dans data/: {names}. Laisse-en un seul.")
PARQUET_FILE = matches[0]

# --- Chargement des zones ---
zones_gdf = gpd.read_file(GEOJSON_FILE)
#zones_json = json.loads(zones_gdf.to_json())
boroughs = sorted(zones_gdf["borough"].dropna().unique())

# --- Lecture parquet : colonnes utiles (+ datetime si dispo) ---
schema_names = set(pq.ParquetFile(PARQUET_FILE).schema.names)
base_cols = ["PULocationID", "trip_distance", "fare_amount", "tip_amount"]
dt_candidates = ["tpep_pickup_datetime", "pickup_datetime", "lpep_pickup_datetime"]
dt_col = next((c for c in dt_candidates if c in schema_names), None)
cols_to_read = [c for c in base_cols if c in schema_names] + ([dt_col] if dt_col else [])

if "PULocationID" not in cols_to_read:
    raise ValueError(f"Le parquet {PARQUET_FILE.name} ne contient pas 'PULocationID' (requis).")

df = pd.read_parquet(PARQUET_FILE, columns=cols_to_read)
df["PULocationID"] = df["PULocationID"].astype("Int64")

# enrichissement : relier borough/zone + features temporelles
df_enriched = df.merge(
    zones_gdf[["LocationID", "borough", "zone"]],
    left_on="PULocationID", right_on="LocationID", how="left"
)
if dt_col:
    df_enriched[dt_col] = pd.to_datetime(df_enriched[dt_col], errors="coerce")
    df_enriched["hour"] = df_enriched[dt_col].dt.hour
    df_enriched["dow"]  = df_enriched[dt_col].dt.dayofweek  # 0=lun … 6=dim
else:
    df_enriched["hour"] = pd.NA
    df_enriched["dow"]  = pd.NA

# --- Pré-agrégats pour la carte ---
def make_agg(metric: str) -> pd.DataFrame:
    if metric == "count":
        out = df.value_counts("PULocationID").rename("value").reset_index()
    elif metric in df.columns:
        out = df.groupby("PULocationID", dropna=False)[metric].mean().rename("value").reset_index()
    else:
        out = df.value_counts("PULocationID").rename("value").reset_index()
    out = out.rename(columns={"PULocationID": "LocationID"})
    out["LocationID"] = out["LocationID"].astype("Int64")
    return out

aggs_cache = {m: make_agg(m) for m in ["count", "trip_distance", "fare_amount", "tip_amount"] if (m=="count" or m in df.columns)}

# --- App Dash ---
app = Dash(__name__)
app.title = "NYC Yellow Taxi — Dashboard"

numeric_hist_cols = [c for c in ["trip_distance", "fare_amount", "tip_amount"] if c in df.columns]

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto", "padding":"8px 12px"},
    children=[
        html.H1("NYC Yellow Taxi — Dashboard", style={"margin":"8px 0 4px"}),
        html.Div(f"Fichiers : data/{PARQUET_FILE.name} / taxi_zones_wgs84.geojson • {len(df):,} lignes".replace(",", " ")),
        html.Hr(),

        # --- Contrôles carte + carte ---
        html.Div([
            html.Div([
                html.Label("Boroughs (filtre carte)"),
                dcc.Checklist(
                    id="borough-filter",
                    options=[{"label": b, "value": b} for b in boroughs],
                    value=boroughs, inline=True
                ),
            ], style={"marginBottom":"8px"}),

            html.Div([
                html.Label("Métrique (carte)"),
                dcc.Dropdown(
                    id="metric",
                    options=(
                        [{"label":"Pickups (count)", "value":"count"}] +
                        ([{"label":"Distance moyenne (mi)", "value":"trip_distance"}] if "trip_distance" in df.columns else []) +
                        ([{"label":"Montant moyen ($)", "value":"fare_amount"}] if "fare_amount" in df.columns else []) +
                        ([{"label":"Pourboire moyen ($)", "value":"tip_amount"}] if "tip_amount" in df.columns else [])
                    ),
                    value="count", clearable=False, style={"width":"320px"}
                ),
            ]),
        ]),

        dcc.Graph(id="map", style={"height":"68vh", "marginTop":"8px"}),

        html.Hr(),

        # ----- Barres catégorielles -----
        html.H3("Barres catégorielles — agrégat d'une variable continue"),
        html.Div([
            html.Div([
                html.Label("Variable continue"),
                dcc.Dropdown(
                    id="bar-metric",
                    options=[{"label": "Distance (trip_distance)", "value": "trip_distance"},
                             {"label": "Montant ($) (fare_amount)", "value": "fare_amount"},
                             {"label": "Pourboire ($) (tip_amount)", "value": "tip_amount"}],
                    value=next((c for c in ["trip_distance", "fare_amount", "tip_amount"] if c in df.columns), None),
                    clearable=False, style={"width": "340px"}
                ),
            ], style={"marginRight": "12px"}),
            html.Div([
                html.Label("Catégorie"),
                dcc.Dropdown(
                    id="bar-category",
                    options=[{"label": "Borough", "value": "borough"},
                             {"label": "Heure (0-23)", "value": "hour"},
                             {"label": "Jour semaine (0=Lun)", "value": "dow"}],
                    value="borough", clearable=False, style={"width": "220px"}
                ),
            ], style={"marginRight": "12px"}),
            html.Div([
                html.Label("Agrégat"),
                dcc.Dropdown(
                    id="bar-agg",
                    options=[{"label": "Moyenne", "value": "mean"},
                             {"label": "Médiane", "value": "median"},
                             {"label": "Somme", "value": "sum"},
                             {"label": "Nombre", "value": "count"}],
                    value="mean", clearable=False, style={"width": "160px"}
                ),
            ]),
        ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "end", "gap": "8px", "marginTop": "8px"}),
        dcc.Graph(id="barcat", style={"height": "55vh"}),
    ]
)

# --- Callbacks ---
@app.callback(Output("map","figure"), 
              Input("metric","value"), 
              Input("borough-filter","value"))
def update_map(metric, borough_selected):
    z = zones_gdf if not borough_selected else zones_gdf[zones_gdf["borough"].isin(borough_selected)]
    agg = aggs_cache.get(metric, aggs_cache["count"])
    g = z.merge(agg, on="LocationID", how="left").fillna({"value": 0})

    label = {
        "count": "Pickups",
        "trip_distance": "Distance moyenne (mi)",
        "fare_amount": "Montant moyen ($)",
        "tip_amount": "Pourboire moyen ($)",
    }.get(metric, "Valeur")

    fig = px.choropleth(
        g,
        geojson=json.loads(g.to_json()),
        locations="LocationID",
        featureidkey="properties.LocationID",
        color="value",
        hover_name="zone",
        hover_data={"borough": True, "LocationID": True, "value": True},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), title=f"Carte — {label}")
    return fig

@app.callback(Output("barcat", "figure"),
              Input("bar-metric", "value"),
              Input("bar-category", "value"),
              Input("bar-agg", "value"))
def update_barcat(metric, category, agg):
    # colonnes dispo
    if category not in df_enriched.columns:
        return px.bar(title="Catégorie indisponible")
    if agg != "count" and metric not in df_enriched.columns:
        return px.bar(title="Variable continue indisponible")

    d = df_enriched[[category] + ([metric] if agg != "count" else [])].dropna(subset=[category])

    if agg == "count":
        g = d.groupby(category, dropna=False).size().rename("value").reset_index()
        ylab = "count"
    else:
        d = d[pd.to_numeric(d[metric], errors="coerce").notna()]
        func = {"mean": "mean", "median": "median", "sum": "sum"}[agg]
        g = d.groupby(category, dropna=False)[metric].agg(func).rename("value").reset_index()
        ylab = f"{agg} de {metric}"

    # jolies étiquettes pour le jour
    if category == "dow":
        mapping = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Jeu", 4: "Ven", 5: "Sam", 6: "Dim"}
        g["__x__"] = g["dow"].map(mapping)
        xcol = "__x__"
    else:
        xcol = category

    g = g.sort_values("value", ascending=False)
    fig = px.bar(g, x=xcol, y="value", text="value",
                 labels={xcol: category, "value": ylab})
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0),
                      title=f"{ylab} par {category}")
    return fig

# --- Run ---
if __name__ == "__main__":
    import webbrowser
    url = "http://127.0.0.1:8050/"
    print(f"Ouvre {url} dans ton navigateur si ça ne s’ouvre pas tout seul.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    print("GeoJSON :", GEOJSON_FILE, GEOJSON_FILE.exists())
    print("Parquet :", PARQUET_FILE, PARQUET_FILE.exists())
    print("Rows parquet :", len(df))
    # Dépendances suggérées : dash plotly pandas pyarrow geopandas
    app.run(debug=True)
