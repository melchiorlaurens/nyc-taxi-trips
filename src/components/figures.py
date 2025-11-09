# src/components/figures.py
from typing import Iterable, Optional
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px

def make_map_figure(
    zones_gdf: gpd.GeoDataFrame,
    agg_df: pd.DataFrame,
    borough_filter: Optional[Iterable[str]],
    metric_label: str,
):
    z = zones_gdf if not borough_filter else zones_gdf[zones_gdf["borough"].isin(list(borough_filter))]
    g = z.merge(agg_df, on="LocationID", how="left").fillna({"value": 0})
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
