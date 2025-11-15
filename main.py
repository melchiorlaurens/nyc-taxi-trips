# main.py — Tout-en-un minimal : prépare (si besoin) + lance la carte & l'histogramme
import pandas as pd
import numpy as np
from functools import lru_cache
import geopandas as gpd
from dash import Dash, dcc, html, Input, Output

from src.utils.get_data import download_months, download_assets
from src.utils.clean_data import make_geojson, make_yellow_clean, make_zone_lookup
from src.components.figures import make_map_figure, make_hist_figure, make_box_figure
from src.database import sync_sqlite_database
from src.utils.paths import (
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
    DEFAULT_PERIODS,
    CLEAN_TAXI_ZONES_GEOJSON,
    CLEAN_TAXI_ZONE_LOOKUP_CSV,
    CLEAN_YELLOW_MONTHLY_DIR,
    clean_yellow_parquet_path,
    BACKGROUND_IMAGE_PATH,
)

# ---------------- Config ----------------
# ---------------------------------------

def ensure_clean():
    """Prépare les données si besoin et génère les fichiers cleaned attendus par le dashboard."""
    print("[info] Analyse des données…")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Télécharge selon la config (si manquants)
    download_months(DEFAULT_PERIODS)
    download_assets()

    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    from pathlib import Path as _P
    raw_files = sorted(_P(RAW_DATA_DIR).glob("yellow_tripdata_*.parquet"))
    if raw_files:
        print("[info] Synchronisation de la base SQLite…")
        sync_sqlite_database(raw_files)

    # Shapefile/lookup: générer si manquants
    if not CLEAN_TAXI_ZONES_GEOJSON.exists():
        make_geojson(RAW_DATA_DIR, CLEAN_DATA_DIR)
    if not CLEAN_TAXI_ZONE_LOOKUP_CSV.exists():
        make_zone_lookup(RAW_DATA_DIR, CLEAN_DATA_DIR)
    # Nettoyés mensuels: reconstruire si des bruts plus récents existent ou si des sorties manquent
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
    months_all = sorted({p.name.replace(".parquet", "").split("_")[-1] for p in monthly_paths})
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
    # Lookup borough ← LocationID, typé pour des maps rapides
    _bdf = zones_gdf[["LocationID", "borough"]].copy()
    _bdf["LocationID"] = pd.to_numeric(_bdf["LocationID"], errors="coerce").astype("Int64")
    borough_lookup = (
        _bdf.dropna(subset=["LocationID", "borough"])  # assure des clés/valeurs valides
            .drop_duplicates(subset=["LocationID"], keep="first")  # index unique
            .set_index("LocationID")["borough"]
    )
    borough_map = borough_lookup.to_dict()

    # Pré-calcul des bornes globales (x_min/x_max) pour l'histogramme par variable
    hist_bounds = {}

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
        hist_bounds[col] = dict(x_min=float(gmin), x_max=float(gmax))

    # Pré-calcul des bornes de couleurs pour la carte, par métrique (fixes à travers les mois)
    map_color_ranges = {}

    # Count (toujours présent)
    max_count = 0
    for ym in months:
        dfm = _load_month_df(ym)
        if "PULocationID" not in dfm.columns or dfm.empty:
            continue
        s = dfm["PULocationID"].value_counts()
        if not s.empty:
            max_count = max(max_count, int(s.max()))
    if max_count > 0:
        map_color_ranges["count"] = (0.0, float(max_count))

    # Moyennes pour autres colonnes numériques
    for metric in ["trip_distance", "fare_amount", "tip_amount"]:
        if metric not in cols_available:
            continue
        gmin, gmax = None, None
        for ym in months:
            dfm = _load_month_df(ym)
            if metric not in dfm.columns or "PULocationID" not in dfm.columns or dfm.empty:
                continue
            grp = dfm.groupby("PULocationID", dropna=False)[metric].mean()
            if grp.empty:
                continue
            vmin = float(grp.min())
            vmax = float(grp.max())
            gmin = vmin if gmin is None else min(gmin, vmin)
            gmax = vmax if gmax is None else max(gmax, vmax)
        if gmin is not None and gmax is not None and gmax > gmin:
            map_color_ranges[metric] = (gmin, gmax)

    app = Dash(__name__)
    app.title = "Dashboard NYC Yellow Taxi"

    # Background image (if present) and dark theme base colors
    import base64
    from pathlib import Path as _Path
    bg_style = {
        "minHeight": "100vh",
        "backgroundColor": "#0b1220",
        "backgroundSize": "cover",
        "backgroundAttachment": "fixed",
        "backgroundPosition": "center center",
        "color": "#e5e7eb",

    }
    try:
        if BACKGROUND_IMAGE_PATH.exists():
            mime = "image/" + BACKGROUND_IMAGE_PATH.suffix.lstrip(".").lower()
            data = base64.b64encode(_Path(BACKGROUND_IMAGE_PATH).read_bytes()).decode("ascii")
            bg_style["backgroundImage"] = (
                f"linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)), url('data:{mime};base64,{data}')"
            )
    except Exception:
        pass

    app.layout = html.Div(
        style={
            "fontFamily":"Rubik-Variable, system-ui, sans-serif",
            "padding":"0",
            **bg_style,
            }, 
        children=[
            html.H1("Dashboard : NYC Yellow Taxi",
                    style={
                        "margin":"8px 0 4px",
                        "fontWeight":"800",
                        "fontSize":"28px",
                        "textAlign":"center",
                        "color":"#f3f4f6",
                        }
                    ),
            html.Div("Exécution : python main.py",
                     style={
                         
                         "textAlign":"center",
                         "color":"#9ca3af",
                         "fontSize":"14px"
                         }
                    ),
            html.Hr(),

            # Carte
            html.Div([
                html.Div([
                    html.Label("Métrique (carte)",
                               style={
                                   "fontWeight":"bold",
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
                                 )
                                 ),
                    ], 
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "marginTop": "20px",
                        "marginBottom": "10px"
                        }
                    ),
                
                html.Div([
                    html.Label("Arrondissements de New York :"),
                    dcc.Checklist(id="borough-filter",
                                  options=[{"label":b,"value":b} for b in boroughs],
                                  value=boroughs, inline=True)
                    ], 
                    style={
                        "margin":"10px 0",
                        "textAlign":"center"
                        }
                    ),
            ]),
            dcc.Graph(id="map", 
                      style={
                          "height":"62vh",
                          "marginTop":"8px"
                          }
                      ),

            html.Hr(),

            # Histogramme
            html.Div([
                html.Label("Veuillez choisir la variable à afficher sur l'histogramme : distance (en miles), le montant ($), ou le pourboire ($)",
                           style={
                               "fontWeight":"bold",
                               "marginBottom":"6px"
                               }
                           ),
                dcc.Dropdown(id="hist-col",
                             options=(
                                 ([{"label":"Distance moyenne (mi)","value":"trip_distance"}] if "trip_distance" in numeric_hist_cols else []) +
                                 ([{"label":"Montant moyen ($)","value":"fare_amount"}] if "fare_amount" in numeric_hist_cols else []) +
                                 ([{"label":"Pourboire moyen ($)","value":"tip_amount"}] if "tip_amount" in numeric_hist_cols else [])
                             ) or [{"label":"(aucune)","value":"_none"}],
                             value=("trip_distance" if "trip_distance" in numeric_hist_cols else (numeric_hist_cols[0] if numeric_hist_cols else "_none")),
                             clearable=False,
                             style={"width":"320px"}
                             )
            ], 
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "marginTop": "20px",
                "marginBottom": "10px"
                }
            ),

            # Scale controls for histogram
            html.Div([
                html.Div([
                    html.Label("Échelle de l'axe X"),
                    dcc.RadioItems(
                        id="hist-scale-type",
                        options=[{"label": "Logarithmique", "value": "log"},
                                 {"label": "Linéaire", "value": "linear"}
                                 ],
                        value="log",
                        inline=True
                    )
                ], 
                style={
                    "marginBottom":"8px",
                    "textAlign":"center"
                    }
                ),
                html.Div([
                    html.Label("X axis range (min-max) - slider in log scale"),
                    dcc.RangeSlider(id="hist-range-slider",
                                    min=0,
                                    max=2,
                                    step=0.01,
                                    value=[0, 2],
                                    marks={},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                        "template": "{value}"
                                    },
                                    allowCross=False
                                    ),
                    html.Div(id="slider-display", style={"fontSize": "11px", "color": "#9ca3af", "marginTop": "4px", "textAlign": "center"})
                ],
                style={"marginBottom":"8px"}
                )
            ],
            style={"marginBottom":"10px"}
            ),

            dcc.Graph(id="hist",
                      style={
                          "height":"80vh",
                          "marginBottom":"90px"
                          }
                      ),

            # Box plot — même variable et filtres que l'histogramme
            dcc.Graph(id="box",
                      style={
                          "height":"60vh",
                          "marginBottom":"90px"
                          }
                      ),

            html.Div(style={"height":"60px"}), # permet de décaler le contenu au-dessus du slider fixe

            # Overlay fixe pour garder le slider visible pendant le scroll
            html.Div([
                html.Label("Mois",
                           style={
                               "fontWeight":"bold",
                               "textAlign":"center",
                               "display":"block"
                               }
                           ),
                dcc.Slider(id="month-index",
                           min=0,
                           max=len(months)-1,
                           step=1,
                           value=len(months)-1,
                           marks={i: {"label": m,
                                      "style": {
                                          "whiteSpace": "nowrap",
                                          "fontSize": "12px"
                                          }
                                      } for i, m in enumerate(months)},
                           included=False,
                           updatemode="drag",
                ),
            html.Div(id="info",
                     style={
                         "marginTop":"6px",
                         "color":"#6b7280"
                         }
                     ),
            ], 
            style={
                "position": "fixed",
                "left": "12px",
                "right": "12px",
                "bottom": "12px",
                "zIndex": 1000,
                "background": "rgba(17,24,39,0.80)",
                "backdropFilter": "blur(2px)",
                "padding": "8px 14px",
                "border": "0",
                "borderRadius": "10px",
                "boxShadow": "0 6px 18px rgba(0,0,0,0.35)",
                }
            )
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
            "tip_amount":"Pourboire moyen ($)"
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
        return make_map_figure(
            zones_gdf,
            agg_month,
            borough_filter,
            f"{label} — {current_month}",
            color_range=map_color_ranges.get(metric),
        )

    @app.callback([Output("hist-range-slider", "min"),
                   Output("hist-range-slider", "max"),
                   Output("hist-range-slider", "value"),
                   Output("hist-range-slider", "marks")],
                  Input("hist-col", "value"))
    def _update_range_slider(col):
        # Calcule les bornes globales pour cette colonne (tous les mois)
        bounds = hist_bounds.get(col)
        if bounds:
            data_min = bounds["x_min"]
            data_max = bounds["x_max"]
            # Use log scale for slider
            log_min = np.log10(max(data_min, 1e-12))
            log_max = np.log10(data_max)
            # Create marks at nice round numbers in log space
            marks = {}
            marks[log_min] = f"{data_min:.2g}"
            marks[log_max] = f"{data_max:.2g}"
            # Add intermediate marks at powers of 10
            for i in range(int(np.floor(log_min)), int(np.ceil(log_max)) + 1):
                if log_min < i < log_max:
                    marks[i] = f"{10**i:.2g}"
            return log_min, log_max, [log_min, log_max], marks
        return 0, 2, [0, 2], {}

    # Libellés lisibles pour histogramme
    hist_label_for = {
        "trip_distance": "Distance moyenne (mi)",
        "fare_amount": "Montant moyen ($)",
        "tip_amount": "Pourboire moyen ($)",
    }

    @app.callback(Output("hist","figure"),
                  Input("hist-col","value"),
                  Input("month-index","value"),
                  Input("hist-range-slider","value"),
                  Input("hist-scale-type","value"))
    def _hist(col, month_idx, range_val, scale_type):
        current_month = months[month_idx] if 0 <= month_idx < len(months) else months[-1]
        df_month = _load_month_df(current_month)
        if df_month.empty:
            df_month = pd.DataFrame(columns=[col])

        # Convert from log scale (slider values) back to linear scale
        xmin_filter = 10**range_val[0] if range_val[0] is not None else None
        xmax_filter = 10**range_val[1] if range_val[1] is not None else None

        # Apply minimum threshold
        if xmin_filter is not None and xmin_filter > 0:
            xmin_filter = xmin_filter
        else:
            xmin_filter = None

        if xmax_filter is not None and xmax_filter < float('inf'):
            xmax_filter = xmax_filter
        else:
            xmax_filter = None

        display_label = hist_label_for.get(col, col)
        fig = make_hist_figure(df_month, col, xmin_filter, xmax_filter, scale_type, display_label=display_label)
        fig.update_layout(title=(fig.layout.title.text or "Histogramme") + f" — {current_month}")
        return fig

    @app.callback(Output("box", "figure"),
                  Input("hist-col","value"),
                  Input("month-index","value"))
    def _box(col, month_idx):
        current_month = months[month_idx] if 0 <= month_idx < len(months) else months[-1]
        df_month = _load_month_df(current_month)
        if df_month.empty:
            df_month = pd.DataFrame(columns=[col])

        # Ajoute le borough comme groupe pour des box plots multiples (horizontaux)
        df_month = df_month.copy()
        # Cherche la colonne pickup zone la plus probable
        pickup_candidates = [
            "PULocationID", "PUlocationID", "puLocationID",
            "pickup_location_id", "pickup_zone_id"
        ]
        pickup_col = next((c for c in pickup_candidates if c in df_month.columns), None)
        if pickup_col is not None:
            pu = pd.to_numeric(df_month[pickup_col], errors="coerce").astype("Int64")
            df_month["borough"] = pu.map(borough_map).astype("string").str.strip().fillna("Unknown")
        else:
            df_month["borough"] = "Unknown"

        display_label = hist_label_for.get(col, col)
        # Ordre fixe des boroughs pour l'affichage
        fixed_order = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        fig = make_box_figure(
            df_month,
            col,
            display_label=display_label,
            group_col="borough",
            max_groups=len(fixed_order),
            y_order=fixed_order,
        )
        fig.update_layout(title=(fig.layout.title.text or "Box plot") + f" — {current_month}")
        return fig

    @app.callback(Output("slider-display", "children"),
                  Input("hist-range-slider", "value"))
    def _slider_display(range_val):
        if range_val is None or len(range_val) != 2:
            return ""
        # Convert from log scale to linear scale
        min_val = 10**range_val[0]
        max_val = 10**range_val[1]
        return f"Valeurs: {min_val:.3g} à {max_val:.3g}"

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
