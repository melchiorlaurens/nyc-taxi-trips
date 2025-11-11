# main.py — Tout-en-un minimal : prépare (si besoin) + lance la carte & l'histogramme
import pandas as pd
import numpy as np
from functools import lru_cache
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
        print("[info] Nettoyage des fichiers parquets mensuels…")
        make_yellow_clean(RAW_DATA_DIR, CLEAN_DATA_DIR)
    print("[info] Données nettoyées et prêtes à être utilisées.")


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

    # Pré-calcul des bornes globales (x_min/x_max) et du y_max pour l'histogramme par variable
    hist_bounds = {}
    NBINS = 40

    def _clean_series(df_in: pd.DataFrame, col_in: str) -> pd.Series:
        s = pd.to_numeric(df_in.get(col_in), errors="coerce")
        if s is None:
            return pd.Series(dtype=float)
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        s = s[s > 0].astype(float)
        return s

    @lru_cache(maxsize=24)
    def _load_month_df(ym_str: str) -> pd.DataFrame:
        try:
            y, m = map(int, ym_str.split('-'))
            return pd.read_parquet(clean_yellow_parquet_path(y, m))
        except Exception:
            return pd.DataFrame()

    for col in numeric_hist_cols:
        gmin = None
        gmax = None
        # Première passe: bornes x
        for ym in months:
            dfm = _load_month_df(ym)
            if col not in dfm.columns:
                continue
            s = _clean_series(dfm, col)
            if s.empty:
                continue
            vmin = float(s.min())
            vmax = float(s.max())
            gmin = vmin if gmin is None else min(gmin, vmin)
            gmax = vmax if gmax is None else max(gmax, vmax)
        if gmin is None or gmax is None or gmin <= 0:
            continue
        xmin_dec = int(np.floor(np.log10(gmin)))
        xmax_dec = int(np.ceil(np.log10(gmax)))
        edges = np.logspace(xmin_dec, xmax_dec, NBINS + 1)
        # Deuxième passe: y_max avec ces edges
        ymax = 0
        for ym in months:
            dfm = _load_month_df(ym)
            if col not in dfm.columns:
                continue
            s = _clean_series(dfm, col)
            if s.empty:
                continue
            counts, _ = np.histogram(s.values, bins=edges)
            if counts.size:
                ymax = max(ymax, int(counts.max()))
        hist_bounds[col] = dict(x_min=float(10 ** xmin_dec), x_max=float(10 ** xmax_dec), y_max=float(ymax))

    app = Dash(__name__)
    app.title = "Dashboard NYC Yellow Taxi"

    app.layout = html.Div(style={"fontFamily":"Inter, system-ui", 
                                 "padding":"8px 12px"
                                 }, children=[
        html.H1("Dashboard : NYC Yellow Taxi", 
                style={"margin":"8px 0 4px"}
                ),
        html.Div("Exécution : python main.py"),
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
                html.Label("Métrique (carte)", 
                           style={"fontWeight":"bold", 
                                  "marginBottom":"6px"
                                  }
                           ),
                dcc.Dropdown(id="metric", 
                             value="count", 
                             clearable=False, 
                             style={"width":"320px"},
                            options=(
                                [{"label":"Pickups (count)","value":"count"}] +
                                ([{"label":"Distance moyenne (mi)","value":"trip_distance"}] if "trip_distance" in cols_available else []) +
                                ([{"label":"Montant moyen ($)","value":"fare_amount"}] if "fare_amount" in cols_available else []) +
                                ([{"label":"Pourboire moyen ($)","value":"tip_amount"}] if "tip_amount" in cols_available else [])
                            )),
            ], style={
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "center",
        "marginTop": "20px",
        "marginBottom": "10px"
    }),
        ]),
        dcc.Graph(id="map", style={"height":"62vh",
                                   "marginTop":"8px"
                                   }),

        html.Hr(),

        # Histogramme
        html.Div([
            html.Label("Veuillez choisir la variable à afficher sur l'histogramme : distance (en miles), le montant ($), ou le pourboire ($)", 
                       style={"fontWeight":"bold", 
                              "marginBottom":"6px"
                              }),
            dcc.Dropdown(
                id="hist-col",
                options=[{"label": c, "value": c} for c in numeric_hist_cols] or [{"label":"(aucune)","value":"_none"}],
                value=("trip_distance" if "trip_distance" in numeric_hist_cols else (numeric_hist_cols[0] if numeric_hist_cols else "_none")),
                clearable=False,
                style={"width":"320px"}
            )
        ], style={
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "center",
        "marginTop": "20px",
        "marginBottom": "10px"
        }),
        dcc.Graph(id="hist", 
                  style={"height":"80vh", 
                         "marginBottom":"70px"
                         }),

        # Overlay fixe pour garder le slider visible pendant le scroll
        html.Div([
            html.Label("Mois", 
                       style={"fontWeight":"bold", 
                              "textAlign":"center", 
                              "display":"block"
                              }),
            dcc.Slider(
                id="month-index",
                min=0,
                max=len(months)-1,
                step=1,
                value=len(months)-1,
                marks={i: {"label": m, 
                           "style": {"whiteSpace": "nowrap", 
                                     "fontSize": "12px"
                                     }} for i, m in enumerate(months)},
                included=False,
                updatemode="drag",
            ),
            html.Div(id="info", 
                     style={"marginTop":"6px", 
                            "color":"#6b7280"
                            }),
        ], style={
            "position": "fixed",
            "left": "12px",
            "right": "12px",
            "bottom": "12px",
            "zIndex": 1000,
            "background": "rgba(255,255,255,0.96)",
            "backdropFilter": "blur(2px)",
            "padding": "0px 14px",
            "border": "1px solid #e5e7eb",
            "borderRadius": "10px",
            "boxShadow": "0 6px 18px rgba(0,0,0,0.08)",
        })
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
        df_month = _load_month_df(current_month)
        if df_month.empty:
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
        df_month = _load_month_df(current_month)
        if df_month.empty:
            df_month = pd.DataFrame(columns=[col])
        bounds = hist_bounds.get(col)
        if bounds:
            fig = make_hist_figure(df_month, col, x_min=bounds["x_min"], x_max=bounds["x_max"], y_max=bounds["y_max"])
        else:
            fig = make_hist_figure(df_month, col)
        fig.update_layout(title=(fig.layout.title.text or "Histogramme") + f" — {current_month}")
        return fig

    @app.callback(Output("info", "children"),
                  Input("month-index", "value"))
    def _info_msg(month_idx):
        current_month = months[month_idx] if 0 <= month_idx < len(months) else months[-1]
        dfm = _load_month_df(current_month)
        if dfm.empty:
            return f"Aucune donnée disponible pour {current_month}."
        return ""

    return app

if __name__ == "__main__":
    ensure_clean()
    app = build_app()
    app.run(debug=True)
