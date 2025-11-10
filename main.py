# main.py — Tout-en-un minimal : prépare (si besoin) + lance la carte & l'histogramme
import pandas as pd
import geopandas as gpd
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px

from src.utils.get_data import download_months, download_assets
from src.utils.clean_data import make_geojson, make_yellow_clean, make_zone_lookup
from src.components.figures import make_map_figure, make_hist_figure
from src.utils.paths import (
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
    DEFAULT_PERIODS,
    CLEAN_TAXI_ZONES_GEOJSON,
    CLEAN_TAXI_ZONE_LOOKUP_CSV,
    CLEAN_YELLOW_MONTHLY_DIR,
    clean_yellow_parquet_path,
)

# ---------------- Config ----------------
# ---------------------------------------

def ensure_clean():
    """Prépare les données si besoin et génère les fichiers cleaned attendus par le dashboard."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Télécharge selon la config (si manquants)
    download_months(DEFAULT_PERIODS)
    download_assets()

    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Shapefile/lookup: générer si manquants
    if not CLEAN_TAXI_ZONES_GEOJSON.exists():
        make_geojson(RAW_DATA_DIR, CLEAN_DATA_DIR)
    if not CLEAN_TAXI_ZONE_LOOKUP_CSV.exists():
        make_zone_lookup(RAW_DATA_DIR, CLEAN_DATA_DIR)
    # Nettoyés mensuels: reconstruire si des bruts plus récents existent ou si des sorties manquent
    from pathlib import Path as _P
    raw_files = sorted(_P(RAW_DATA_DIR).glob("yellow_tripdata_*.parquet"))
    needs_yellow = not CLEAN_YELLOW_MONTHLY_DIR.exists()
    if not needs_yellow:
        for p in raw_files:
            ym = p.name.replace('.parquet','').split('_')[-1]
            try:
                y, m = map(int, ym.split('-'))
            except Exception:
                continue
            monthly_target = clean_yellow_parquet_path(y, m)
            if not monthly_target.exists() or monthly_target.stat().st_mtime < p.stat().st_mtime:
                needs_yellow = True
                break
    if needs_yellow:
        print("[info] (Re)nettoyage des parquets mensuels…")
        make_yellow_clean(RAW_DATA_DIR, CLEAN_DATA_DIR)
    else:
        print("[info] Données nettoyées déjà à jour.")


def build_app():
    zones_gdf = gpd.read_file(CLEAN_TAXI_ZONES_GEOJSON)
    # Déterminer les mois disponibles à partir des fichiers nettoyés mensuels
    monthly_paths = sorted(CLEAN_YELLOW_MONTHLY_DIR.glob("yellow_clean_*.parquet"))
    months_all: list[str] = []
    for p in monthly_paths:
        ym = p.name.replace(".parquet", "").split("_")[-1]
        months_all.append(ym)
    months_all = sorted(set(months_all))
    configured = [f"{y}-{m:02d}" for y, m in DEFAULT_PERIODS]
    months = [m for m in months_all if m in configured] or months_all or ["(indéfini)"]

    # Lire un fichier mensuel pour déterminer les colonnes dispo
    if monthly_paths:
        sample_df = pd.read_parquet(monthly_paths[-1])
        cols_available = set(sample_df.columns)
        numeric_hist_cols = [c for c in ["trip_distance", "fare_amount", "tip_amount"] if c in cols_available]
    else:
        cols_available = set()
        numeric_hist_cols = []

    boroughs = sorted(zones_gdf["borough"].dropna().unique())

    app = Dash(__name__)
    app.title = "Dashboard NYC Yellow Taxi"

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
                html.Label("Mois"),
                dcc.Slider(
                    id="month-index",
                    min=0,
                    max=len(months)-1,
                    step=1,
                    value=len(months)-1,
                    marks={i: m for i, m in enumerate(months)},
                    included=False,
                    updatemode="drag",
                ),
                html.Div(id="info", style={"marginTop":"6px", "color":"#6b7280"}),
                html.Div([
                    html.Button("Play", id="play", n_clicks=0, style={"marginRight":"6px"}),
                    html.Button("Pause", id="pause", n_clicks=0),
                    dcc.Interval(id="timer", interval=4000, n_intervals=0, disabled=True),      # intervalle à 4s car long à charger
                ], style={"marginTop":"6px"}),
            ], style={"marginBottom":"8px"}),

            html.Div([
                html.Label("Métrique (carte)"),
                dcc.Dropdown(id="metric", value="count", clearable=False, style={"width":"320px"},
                    options=(
                        [{"label":"Pickups (count)","value":"count"}] +
                        ([{"label":"Distance moyenne (mi)","value":"trip_distance"}] if "trip_distance" in cols_available else []) +
                        ([{"label":"Montant moyen ($)","value":"fare_amount"}] if "fare_amount" in cols_available else []) +
                        ([{"label":"Pourboire moyen ($)","value":"tip_amount"}] if "tip_amount" in cols_available else [])
                    )),
            ]),
        ]),
        dcc.Graph(id="map", style={"height":"62vh","marginTop":"8px"}),

        html.Hr(),

        # Histogramme
        html.Div([
            html.Label("Veuillez choisir la variable à afficher sur l'histogramme : distance (en miles), le montant ($), ou le pourboire ($)"),
            dcc.Dropdown(
                id="hist-col",
                options=[{"label": c, "value": c} for c in numeric_hist_cols] or [{"label":"(aucune)","value":"_none"}],
                value=("trip_distance" if "trip_distance" in numeric_hist_cols else (numeric_hist_cols[0] if numeric_hist_cols else "_none")),
                clearable=False,
                style={"width":"320px"}
            )
        ]),
        dcc.Graph(id="hist", style={"height":"100vh"})
    ])

    # Callbacks
    @app.callback(Output("map","figure"),
                  Input("metric","value"),
                  Input("borough-filter","value"),
                  Input("month-index","value"))
    def _map(metric, borough_filter, month_idx):
        label = {
            "count":"Pickups",
            "trip_distance":"Distance moyenne (mi)",
            "fare_amount":"Montant moyen ($)",
            "tip_amount":"Pourboire moyen ($)",
        }.get(metric, "Valeur")
        current_month = months[month_idx] if 0 <= month_idx < len(months) else months[-1]
        try:
            y, m = map(int, current_month.split('-'))
            df_month = pd.read_parquet(clean_yellow_parquet_path(y, m))
        except Exception:
            df_month = pd.DataFrame(columns=["PULocationID"])  # vide
        if metric == "count":
            agg_month = df_month.value_counts("PULocationID").rename("value").reset_index()
        elif metric in df_month.columns:
            agg_month = df_month.groupby("PULocationID", dropna=False)[metric].mean().rename("value").reset_index()
        else:
            agg_month = df_month.value_counts("PULocationID").rename("value").reset_index()
        agg_month = agg_month.rename(columns={"PULocationID": "LocationID"})
        return make_map_figure(zones_gdf, agg_month, borough_filter, f"{label} — {current_month}")

    @app.callback(Output("hist","figure"),
                  Input("hist-col","value"),
                  Input("month-index","value"))
    def _hist(col, month_idx):
        current_month = months[month_idx] if 0 <= month_idx < len(months) else months[-1]
        try:
            y, m = map(int, current_month.split('-'))
            df_month = pd.read_parquet(clean_yellow_parquet_path(y, m))
        except Exception:
            df_month = pd.DataFrame(columns=[col])
        fig = make_hist_figure(df_month, col)
        fig.update_layout(title=(fig.layout.title.text or "Histogramme") + f" — {current_month}")
        return fig

    # Lecture automatique du slider
    @app.callback(Output("timer", "disabled"),
                  Input("play", "n_clicks"),
                  Input("pause", "n_clicks"))
    def _toggle_timer(n_play, n_pause):
        n_play = n_play or 0
        n_pause = n_pause or 0
        return not (n_play > n_pause)

    @app.callback(Output("month-index", "value"),
                  Input("timer", "n_intervals"),
                  State("month-index", "value"))
    def _advance_slider(_n, idx):
        if not months:
            return 0
        return (idx + 1) % len(months)

    @app.callback(Output("info", "children"),
                  Input("month-index", "value"))
    def _info_msg(month_idx):
        current_month = months[month_idx] if 0 <= month_idx < len(months) else months[-1]
        try:
            y, m = map(int, current_month.split('-'))
            p = clean_yellow_parquet_path(y, m)
            if not p.exists() or pd.read_parquet(p).empty:
                return f"Aucune donnée disponible pour {current_month}."
        except Exception:
            return f"Aucune donnée disponible pour {current_month}."
        return ""

    return app

if __name__ == "__main__":
    ensure_clean()
    app = build_app()
    app.run(debug=True)
