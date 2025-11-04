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

def make_hist_figure(df: pd.DataFrame, col: str, bins: int):
    if col not in df.columns:
        return px.histogram()

    # 1) nettoyer
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0].astype(float)
    if s.empty:
        return px.histogram()

    # 2) passer en log10 pour un binning stable
    s_log = np.log10(s.values)

    # 3) tracer en linéaire sur s_log
    d = pd.DataFrame({"log10_value": s_log})
    fig = px.histogram(d, x="log10_value", nbins=int(bins))

    # 4) replacer des ticks lisibles (10^k)
    kmin = int(np.floor(s_log.min()))
    kmax = int(np.ceil(s_log.max()))
    tickvals = list(range(kmin, kmax + 1))
    ticktext = [f"10^{k}" if k != 0 else "1" for k in tickvals]

    fig.update_xaxes(
        title=col,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext
    )
    fig.update_yaxes(title="count")
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        bargap=0.02,
        title=f"Histogramme — {col} (échelle log, n={len(s):,})".replace(",", " ")
    )
    return fig
