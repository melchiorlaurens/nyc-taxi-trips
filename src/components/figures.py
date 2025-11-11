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
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="#0f172a")
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
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Carte — {metric_label}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def make_hist_figure(
    df: pd.DataFrame,
    col: str,
    xmin_filter: Optional[float] = None,
    xmax_filter: Optional[float] = None,
    scale_type: str = "log",
):
    NBINS = 40
    BARGAP = 0.03

    if col not in df.columns:
        return px.histogram()

    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0].astype(float)
    if s.empty:
        return px.histogram()

    slider_min = None
    slider_max = None
    if xmin_filter is not None:
        slider_min = float(xmin_filter)
        if scale_type == "log":
            slider_min = max(slider_min, 1e-12)
    if xmax_filter is not None:
        slider_max = float(xmax_filter)

    s_filtered = s.copy()
    if slider_min is not None:
        s_filtered = s_filtered[s_filtered >= slider_min]
    if slider_max is not None:
        s_filtered = s_filtered[s_filtered <= slider_max]

    if scale_type == "log":
        range_min = slider_min if slider_min is not None else float(s.min())
        range_max = slider_max if slider_max is not None else float(s.max())
        range_min = max(range_min, 1e-12)
        if range_max <= range_min:
            range_max = range_min * (1 + 1e-9)

        edges = np.logspace(np.log10(range_min), np.log10(range_max), NBINS + 1)
        data_for_hist = s_filtered.values if not s_filtered.empty else np.array([])
        counts, _ = np.histogram(data_for_hist, bins=edges)

        widths = edges[1:] - edges[:-1]
        mid = np.sqrt(edges[:-1] * edges[1:])

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

        tick_lo = int(np.floor(np.log10(range_min)))
        tick_hi = int(np.ceil(np.log10(range_max)))
        tickvals = [10**k for k in range(tick_lo, tick_hi + 1)]

        def format_num(x: float) -> str:
            if x >= 1000:
                return f"{x:,.0f}".replace(",", " ")
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
            tickwidth=1,
            range=[np.log10(range_min), np.log10(range_max)]
        )
    else:
        range_min = slider_min if slider_min is not None else float(s.min())
        range_max = slider_max if slider_max is not None else float(s.max())
        if range_max <= range_min:
            range_max = range_min + 1e-9

        edges = np.linspace(range_min, range_max, NBINS + 1)
        data_for_hist = s_filtered.values if not s_filtered.empty else np.array([])
        counts, _ = np.histogram(data_for_hist, bins=edges)

        widths = edges[1:] - edges[:-1]
        mid = (edges[:-1] + edges[1:]) / 2

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

        fig.update_xaxes(
            type="linear",
            title=col + " (échelle linéaire)",
            ticks="outside",
            ticklen=6,
            tickwidth=1,
            range=[range_min, range_max]
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
        title=f"Répartition du nombre de trajets en fonction de {col} sous forme d'histogramme",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
