# main.py — Minimal all-in-one: prepares data (if needed) and launches the dashboard
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
    """Prepares data if needed and generates cleaned files for the dashboard."""
    print("[info] Preparing data…")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Download raw data (if missing)
    downloaded_files = download_months(DEFAULT_PERIODS)
    print(f"[info] Downloaded {len(downloaded_files)} parquet files")
    download_assets()
    print("[info] Downloaded supporting assets (taxi zones, lookup table)")

    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    from pathlib import Path as _P
    raw_files = sorted(_P(RAW_DATA_DIR).glob("yellow_tripdata_*.parquet"))
    if raw_files:
        print("[info] Syncing SQLite database…")
        sync_sqlite_database(raw_files)

    # Create geojson and lookup if missing
    if not CLEAN_TAXI_ZONES_GEOJSON.exists():
        make_geojson(RAW_DATA_DIR, CLEAN_DATA_DIR)
    if not CLEAN_TAXI_ZONE_LOOKUP_CSV.exists():
        make_zone_lookup(RAW_DATA_DIR, CLEAN_DATA_DIR)
    # Rebuild monthly cleaned files if raw data is newer or if outputs are missing
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
        print("[info] Cleaning and transforming monthly parquet files…")
        make_yellow_clean(RAW_DATA_DIR, CLEAN_DATA_DIR)
        cleaned_files = sorted(CLEAN_YELLOW_MONTHLY_DIR.glob("yellow_clean_*.parquet"))
        print(f"[info] Cleaned {len(cleaned_files)} months of data")
    print("[info] Data preparation complete. Ready to launch dashboard. Thanks for waiting !")


def build_app():
    zones_gdf = gpd.read_file(CLEAN_TAXI_ZONES_GEOJSON)
    # Determine available months from cleaned monthly files
    monthly_paths = sorted(CLEAN_YELLOW_MONTHLY_DIR.glob("yellow_clean_*.parquet"))
    months_all = sorted({p.name.replace(".parquet", "").split("_")[-1] for p in monthly_paths})
    configured = [f"{y}-{m:02d}" for y, m in DEFAULT_PERIODS]
    months = [m for m in months_all if m in configured] or months_all or ["(undefined)"]

    # Read a monthly file to determine available columns
    if monthly_paths:
        sample_df = pd.read_parquet(monthly_paths[-1])
        cols_available = set(sample_df.columns)
        numeric_hist_cols = [c for c in ["trip_distance", "fare_amount", "tip_amount"] if c in cols_available]
    else:
        cols_available = set()
        numeric_hist_cols = []

    boroughs = sorted(zones_gdf["borough"].dropna().unique())
    # Borough lookup ← LocationID, typed for fast map lookups
    _bdf = zones_gdf[["LocationID", "borough"]].copy()
    _bdf["LocationID"] = pd.to_numeric(_bdf["LocationID"], errors="coerce").astype("Int64")
    borough_lookup = (
        _bdf.dropna(subset=["LocationID", "borough"])  # ensures valid keys/values
            .drop_duplicates(subset=["LocationID"], keep="first")  # unique index
            .set_index("LocationID")["borough"]
    )
    borough_map = borough_lookup.to_dict()

    # Pre-calculate global bounds (x_min/x_max) for histogram scaling by variable
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
        # First pass: calculate x bounds
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

    # Pre-calculate color bounds for map by metric (fixed across all months)
    map_color_ranges = {}

    # Count (always present)
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

    # Averages for other numeric columns
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

    # Load background image (if present) and apply dark theme
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

    # Dashboard layout
    app.layout = html.Div(
        style={
            "fontFamily":"Rubik-Variable, system-ui, sans-serif",
            "padding":"0",
            **bg_style,
            }, 
        children=[
            html.Div(style={"height":"60px"}),
            html.H1("Dashboard : NYC Yellow Taxi",
                    style={
                        "margin":"0 0 4px",
                        "fontWeight":"800",
                        "fontSize":"30px",
                        "textAlign":"center",
                        "color":"#f3f4f6",
                        }
                    ),
            html.Div("Run: python main.py",
                     style={
                         
                         "textAlign":"center",
                         "color":"#9ca3af",
                         "fontSize":"14px"
                         }
                    ),
            # Overview text — edit below
            html.Div([
                html.H2("Context", style={"margin":"20px 0 0 0"}),
                html.P([
                    "This dashboard explores New York City yellow taxi activity using the monthly trip records published by the ",
                    html.A("NYC Taxi & Limousine Commission (TLC)",
                           href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
                           target="_blank",
                           rel="noopener noreferrer",
                           style={"textDecoration": "underline",
                                  "color": "#2563eb"
                                  }),
                    ". Each trip contains at least a pickup and drop-off TLC zone, the distance in miles, the fare in dollars and, "
                    "when applicable, the tip.",
                ],
                    style={"marginBottom":"8px"}
                ),
                html.P(
                    "The objective is to give both a spatial and statistical overview: where activity concentrates across the city and "
                    "how distances, fares and tips are distributed. Use the month slider at the bottom to focus on any month between "
                    "January and September 2025.",
                )
            ], style={
                "maxWidth":"920px",
                "margin":"0 auto 16px",
                "color":"#f3f4f6",
                "lineHeight":"1.6"
            }),

            html.Div(style={"height":"60px"}),

            # Map
            html.Div([
                html.Div([
                    html.H3("Choropleth map", style={"textAlign":"center",
                                                          "fontSize":"22px"
                                                          }),
                    html.P(
                        "Pick a metric to color the TLC pickup zones: number of pickups, average trip distance, average fare amount or average tip amount.",
                    ),
                    html.P(
                        "Use the borough checklist to limit the display to specific areas.",
                        ),
                    html.P(
                        "Hovering a zone reveals its name, borough and the exact metric value.",
                    )
                ], style={"maxWidth":"900px",
                          "margin":"0 auto 40px",
                          "color":"#d1d5db",
                          "textAlign":"center"
                          }
                ),
                html.Div([
                    html.Label("Metric (map)",
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
                                     ([{"label":"Average distance (mi)","value":"trip_distance"}] if "trip_distance" in cols_available else []) +
                                     ([{"label":"Average fare ($)","value":"fare_amount"}] if "fare_amount" in cols_available else []) +
                                     ([{"label":"Average tip ($)","value":"tip_amount"}] if "tip_amount" in cols_available else [])
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
                    html.Label("New York boroughs :"),
                    dcc.Checklist(id="borough-filter",
                                  options=[{"label":b,"value":b} for b in boroughs],
                                  value=boroughs, inline=True)
                    ], 
                    style={
                        "margin":"20px 0",
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

            html.Div(style={"height":"120px"}),

            # Histogram
            html.Div([
                html.Div([
                    html.H3("Histogram and box plot", style={"textAlign":"center",
                                                              "fontSize":"22px"
                                                              }),
                    html.P(
                        "The histogram displays the overall distribution of the selected continuous variable (distance, fare or tip). "
                        "Use the slider right underneath to adjust the X-axis range: it works in log space and acts as a zoom into the values of interest."
                    ),
                    html.P(
                        "The box plot, which is deliberately static, complements the histogram by summarizing the distribution per borough using min, max, median and quartiles."
                        ),
                ], style={"maxWidth":"900px",
                          "margin":"0 auto 40px",
                          "color":"#d1d5db",
                          "textAlign":"center"
                          }
                ),
                html.Label("Select the variable to display: trip distance (mi), fare ($), or tip ($)",
                           style={
                               "fontWeight":"bold",
                               "marginBottom":"6px"
                               }
                           ),
                dcc.Dropdown(id="hist-col",
                             options=(
                                 ([{"label":"Average distance (mi)","value":"trip_distance"}] if "trip_distance" in numeric_hist_cols else []) +
                                 ([{"label":"Average fare ($)","value":"fare_amount"}] if "fare_amount" in numeric_hist_cols else []) +
                                 ([{"label":"Average tip ($)","value":"tip_amount"}] if "tip_amount" in numeric_hist_cols else [])
                             ) or [{"label":"(none)","value":"_none"}],
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
                    html.Label("X-axis scale"),
                    dcc.RadioItems(
                        id="hist-scale-type",
                        options=[{"label": "Logarithmic", "value": "log"},
                                 {"label": "Linear", "value": "linear"}
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
                    html.Label(
                        "X axis range (min-max) - slider in log scale"
                    ),
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

            # Box plot — same metric (static view, not linked to the slider)
            dcc.Graph(id="box",
                      style={
                          "height":"60vh",
                          "marginBottom":"90px"
                          }
                      ),

            html.Div(style={"height":"60px"}),  # spacer so content stays above the fixed slider

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

    # Dashboard callbacks
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
            df_month = pd.DataFrame(columns=["PULocationID"])  # empty
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
        # Calculate global bounds for this column (all months)
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

    # Readable labels for histogram
    hist_label_for = {
        "trip_distance": "Average distance (mi)",
        "fare_amount": "Average fare ($)",
        "tip_amount": "Average tip ($)",
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
        fig.update_layout(title=(fig.layout.title.text or "Histogram") + f" — {current_month}")
        return fig

    @app.callback(Output("box", "figure"),
                  Input("hist-col","value"),
                  Input("month-index","value"))
    def _box(col, month_idx):
        current_month = months[month_idx] if 0 <= month_idx < len(months) else months[-1]
        df_month = _load_month_df(current_month)
        if df_month.empty:
            df_month = pd.DataFrame(columns=[col])

        # Add borough as grouping variable for multiple box plots (horizontal layout)
        df_month = df_month.copy()
        # Find the pickup location column
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
        # Fixed order of boroughs for display
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
        return f"Values: {min_val:.3g} to {max_val:.3g}"

    @app.callback(Output("info", "children"),
                  Input("month-index", "value"))
    def _info_msg(month_idx):
        current_month = months[month_idx] if 0 <= month_idx < len(months) else months[-1]
        dfm = _load_month_df(current_month)
        if dfm.empty:
            return f"No data available for {current_month}."
        return ""

    return app

if __name__ == "__main__":
    ensure_clean()
    app = build_app()
    app.run(debug=True, use_reloader=False)
