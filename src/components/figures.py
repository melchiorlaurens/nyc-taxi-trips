# src/components/figures.py
from typing import Iterable, Optional
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px

HOVER_ZONE_MAX = 28
HOVER_BOROUGH_MAX = 24

def _format_number(value: float, decimals: int = 0) -> str:
    fmt = f"{{:,.{decimals}f}}".format(value)
    if decimals == 0:
        fmt = fmt.split(".")[0]
    return fmt.replace(",", " ")

def _trim_label(text: str, max_chars: int, fallback: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return fallback
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"

def _format_hover_value(value: float, metric_label: str) -> str:
    if pd.isna(value):
        return "No data"
    label = metric_label.lower()
    if "pickup" in label:
        return f"{_format_number(round(value), 0)} pickups"
    if "distance" in label:
        return f"{_format_number(value, 2)} miles"
    if "montant" in label or "pourboire" in label:
        return f"${_format_number(value, 2)}"
    return f"{_format_number(value, 2)} {metric_label.lower()}"

def _build_hovertemplate() -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "<span style='color:#b7b7b7;'>%{customdata[1]}</span><br>"
        "%{customdata[2]}"
        "<extra></extra>"
    )

def make_map_figure(
    zones_gdf: gpd.GeoDataFrame,
    agg_df: pd.DataFrame,
    borough_filter: Optional[Iterable[str]],
    metric_label: str,
):
    z = zones_gdf if not borough_filter else zones_gdf[zones_gdf["borough"].isin(list(borough_filter))]
    g = z.merge(agg_df, on="LocationID", how="left")
    g["value"] = g["value"].fillna(0)
    g["zone"] = g["zone"].fillna("Zone inconnue")
    g["borough"] = g["borough"].fillna("Borough inconnu")
    g["value_display"] = g["value"].apply(lambda val: _format_hover_value(val, metric_label))
    g["zone_display"] = g["zone"].apply(lambda txt: _trim_label(txt, HOVER_ZONE_MAX, "NYC Zone"))
    g["borough_display"] = g["borough"].apply(lambda txt: _trim_label(txt, HOVER_BOROUGH_MAX, "Borough"))
    fig = px.choropleth(
        g,
        geojson=json.loads(g.to_json()),
        locations="LocationID",
        featureidkey="properties.LocationID",
        color="value",
        hover_data=None,
        custom_data=["zone_display", "borough_display", "value_display"],
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_traces(
        hovertemplate=_build_hovertemplate(),
        hoverlabel=dict(
            bgcolor="#111",
            bordercolor="#444",
            align="left",
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#fff"),
            namelength=0,
        ),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title=f"Carte — {metric_label}")
    return fig

def make_hist_figure(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return px.histogram()

    # 1) nettoyer
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0].astype(float)
    if s.empty:
        return px.histogram()

    # régler les barres
    NBINS = 40
    BARGAP = 0.03

    # axe x en échelle LOG
    xmin = int(np.floor(np.log10(s.min())))
    xmax = int(np.ceil(np.log10(s.max())))

    edges = np.logspace(xmin, xmax, NBINS + 1)      # [lo0, lo1, ... hiN]
    counts, _ = np.histogram(s.values, bins=edges)

    widths = edges[1:] - edges[:-1]
    mid = np.sqrt(edges[:-1] * edges[1:])

    # bar chart + hover en valeurs brutes (intervalle)
    fig = px.bar(x=mid, y=counts, labels={"x": col, "y": "Nombre de trajets"})
    fig.update_traces(
        customdata=np.c_[edges[:-1], edges[1:]],
        hovertemplate=(
            "value ∈ [%{customdata[0]:.3g}, %{customdata[1]:.3g}]<br>"
            "count = %{y:.0f}<extra></extra>"
        ),
        width=widths,
        marker_line_width=0,
        opacity=0.95,
    )

    tickvals = [10**k for k in range(xmin, xmax + 1)]

    def format_num(x: float) -> str:                      # 10**k -> "0.1", "1", "10", "1000", …
        if x >= 1000:
            return f"{x:,.0f}".replace(",", " ")       # séparateur de milliers
        if x >= 1:
            return f"{x:.0f}"
        return f"{x:.3g}"

    ticktext = [format_num(v) for v in tickvals]

    fig.update_xaxes(
        type="log",
        title=col + " (échelle log)",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        ticks="outside", 
        ticklen=6, 
        tickwidth=1
    )
    fig.update_yaxes(
        title="Nombre de trajets",
        ticks="outside", 
        ticklen=6, 
        tickwidth=1,
        tickformat=".0f",
        separatethousands=True
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        bargap=BARGAP,
        bargroupgap=0,
        title=f"Répartition du nombre de trajets en fonction de {col} sous forme d'histogramme"
    )
    return fig
