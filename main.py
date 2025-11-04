# main.py — Tout-en-un minimal : prépare (si besoin) + lance la carte & l'histogramme
from pathlib import Path
import pandas as pd
import geopandas as gpd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from src.utils.get_data import download_month, download_assets
from src.utils.clean_data import make_geojson, make_yellow_clean
from src.components.figures import make_map_figure, make_hist_figure

# ---------------- Config ----------------
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "cleaned"

DEFAULT_YEAR = 2025
DEFAULT_MONTH = 1

CLEAN_GEOJSON = CLEAN / "taxi_zones_wgs84.geojson"
CLEAN_YELLOW = CLEAN / "yellow_clean.parquet"
# ---------------------------------------

def ensure_clean():
    """Télécharge si besoin les bruts et génère les fichiers cleaned attendus par le dashboard."""
    if CLEAN_GEOJSON.exists() and CLEAN_YELLOW.exists():
        print("[info] Données prêtes en data/cleaned.")
        return
    print("[info] Préparation des données…")
    RAW.mkdir(parents=True, exist_ok=True)
    download_month(DEFAULT_YEAR, DEFAULT_MONTH, RAW)
    download_assets(RAW)
    CLEAN.mkdir(parents=True, exist_ok=True)
    make_geojson(RAW, CLEAN)
    make_yellow_clean(RAW, CLEAN)
    print("[info] Données prêtes.")

def build_app():
    zones_gdf = gpd.read_file(CLEAN_GEOJSON)
    df = pd.read_parquet(CLEAN_YELLOW)

    boroughs = sorted(zones_gdf["borough"].dropna().unique())
    numeric_hist_cols = [c for c in ["trip_distance", "fare_amount", "tip_amount"] if c in df.columns]

    # Pré-agrégats pour la carte
    def agg_map(metric: str) -> pd.DataFrame:
        if metric == "count":
            out = df.value_counts("PULocationID").rename("value").reset_index()
        elif metric in df.columns:
            out = df.groupby("PULocationID", dropna=False)[metric].mean().rename("value").reset_index()
        else:
            out = df.value_counts("PULocationID").rename("value").reset_index()
        return out.rename(columns={"PULocationID": "LocationID"})

    aggs_cache = {m: agg_map(m) for m in ["count", "trip_distance", "fare_amount", "tip_amount"] if (m == "count" or m in df.columns)}

    app = Dash(__name__)
    app.title = "NYC Yellow Taxi — Dashboard (minimal)"

    app.layout = html.Div(style={"fontFamily":"Inter, system-ui", "padding":"8px 12px"}, children=[
        html.H1("NYC Yellow Taxi — Dashboard (minimal)", style={"margin":"8px 0 4px"}),
        html.Div("Exécution : python main.py — données locales en cache (data/cleaned)"),
        html.Hr(),

        # Carte
        html.Div([
            html.Div([
                html.Label("Boroughs (filtre carte)"),
                dcc.Checklist(id="borough-filter",
                    options=[{"label":b,"value":b} for b in boroughs],
                    value=boroughs, inline=True),
            ], style={"marginBottom":"8px"}),

            html.Div([
                html.Label("Métrique (carte)"),
                dcc.Dropdown(id="metric", value="count", clearable=False, style={"width":"320px"},
                    options=(
                        [{"label":"Pickups (count)","value":"count"}] +
                        ([{"label":"Distance moyenne (mi)","value":"trip_distance"}] if "trip_distance" in df.columns else []) +
                        ([{"label":"Montant moyen ($)","value":"fare_amount"}] if "fare_amount" in df.columns else []) +
                        ([{"label":"Pourboire moyen ($)","value":"tip_amount"}] if "tip_amount" in df.columns else [])
                    )),
            ]),
        ]),
        dcc.Graph(id="map", style={"height":"62vh","marginTop":"8px"}),

        html.Hr(),

        # Histogramme
        html.Div([
            html.Label("Histogramme — variable (échelle log)"),
            dcc.Dropdown(
                id="hist-col",
                options=[{"label": c, "value": c} for c in numeric_hist_cols] or [{"label":"(aucune)","value":"_none"}],
                value=("trip_distance" if "trip_distance" in numeric_hist_cols else (numeric_hist_cols[0] if numeric_hist_cols else "_none")),
                clearable=False,
                style={"width":"320px"}
            ),
            html.Div(dcc.Slider(id="bins", min=10, max=120, step=5, value=40,
                                marks=None, tooltip={"placement":"bottom","always_visible":True}),
                     style={"width":"320px","marginTop":"8px"}),
        ], style={"marginTop":"6px"}),
        dcc.Graph(id="hist", style={"height":"42vh"}),
    ])

    # Callbacks
    @app.callback(Output("map","figure"),
                  Input("metric","value"),
                  Input("borough-filter","value"))
    def _map(metric, borough_filter):
        label = {
            "count":"Pickups",
            "trip_distance":"Distance moyenne (mi)",
            "fare_amount":"Montant moyen ($)",
            "tip_amount":"Pourboire moyen ($)",
        }.get(metric, "Valeur")
        return make_map_figure(zones_gdf, aggs_cache.get(metric, aggs_cache["count"]), borough_filter, label)

    @app.callback(Output("hist","figure"),
                  Input("hist-col","value"), Input("bins","value"))
    def _hist(col, bins):
        return make_hist_figure(df, col, int(bins))

    return app

if __name__ == "__main__":
    ensure_clean()
    app = build_app()
    app.run(debug=True)
