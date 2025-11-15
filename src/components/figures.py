# src/components/figures.py
from typing import Iterable, Optional, List, Tuple
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

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
    # Display zone (line 1), borough (line 2 gray), value (line 3)
    return (
        "<b>%{customdata[0]}</b><br>"
        "<span style='color:#9ca3af;'>%{customdata[1]}</span><br>"
        "%{customdata[2]}"
        "<extra></extra>"
    )

def _format_tick_value(x: float) -> str:
    """Tick formatter aligned with _format_number rules, with compact < 1."""
    if x >= 1000:
        return _format_number(x, 0)
    if x >= 1:
        return f"{x:.0f}"
    return f"{x:.3g}"

def _log_ticks(lo: float, hi: float) -> Tuple[List[float], List[str]]:
    lo_dec = int(np.floor(np.log10(lo)))
    hi_dec = int(np.ceil(np.log10(hi)))
    vals = [10 ** k for k in range(lo_dec, hi_dec + 1)]
    txt = [_format_tick_value(v) for v in vals]
    return vals, txt

def make_map_figure(
    zones_gdf: gpd.GeoDataFrame,
    agg_df: pd.DataFrame,
    borough_filter: Optional[Iterable[str]],
    metric_label: str,
    color_range: Optional[tuple] = None,
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
        range_color=color_range,
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
        # Place the colorbar just to the right of the map
        coloraxis_colorbar=dict(
            title=metric_label,
            x=0.80,             # closer to the center than 1.0
            xanchor="left",
            y=0.5,
            len=0.8,
            thickness=30,
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)"
        ),
    )
    return fig

def make_hist_figure(
    df: pd.DataFrame,
    col: str,
    xmin_filter: Optional[float] = None,
    xmax_filter: Optional[float] = None,
    scale_type: str = "log",
    display_label: Optional[str] = None,
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

    label_txt = display_label if display_label else col

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

        fig = px.bar(x=mid, y=counts, labels={"x": label_txt, "y": "Nombre de trajets"})
        # Prepare shared parameters for a single trace update
        _customdata = np.c_[edges[:-1], edges[1:]]
        _widths = widths
        _counts = counts
        trace_kwargs = dict(
            customdata=_customdata,
            hovertemplate=(
                "value ∈ [%{customdata[0]:.3g}, %{customdata[1]:.3g}]<br>"
                "count = %{y:.0f}<extra></extra>"
            ),
            width=_widths,
            marker_line_width=0,
            opacity=0.95,
        )

        tickvals, ticktext = _log_ticks(range_min, range_max)

        fig.update_xaxes(
            type="log",
            title=label_txt + " (échelle log)",
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

        fig = px.bar(x=mid, y=counts, labels={"x": label_txt, "y": "Nombre de trajets"})
        # Prepare shared parameters for a single trace update
        _customdata = np.c_[edges[:-1], edges[1:]]
        _widths = widths
        _counts = counts
        trace_kwargs = dict(
            customdata=_customdata,
            hovertemplate=(
                "value ∈ [%{customdata[0]:.3g}, %{customdata[1]:.3g}]<br>"
                "count = %{y:.0f}<extra></extra>"
            ),
            width=_widths,
            marker_line_width=0,
            opacity=0.95,
        )

        fig.update_xaxes(
            type="linear",
            title=label_txt + " (échelle linéaire)",
            ticks="outside",
            ticklen=6,
            tickwidth=1,
            range=[range_min, range_max]
        )

    # Apply shared parameters to the bars once
    try:
        fig.update_traces(**trace_kwargs)
    except Exception:
        pass

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
        title=f"Répartition du nombre de trajets en fonction de {label_txt} sous forme d'histogramme",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def make_box_figure(
    df: pd.DataFrame,
    col: str,
    display_label: Optional[str] = None,
    group_col: Optional[str] = None,
    max_groups: int = 6,
    exclude_groups: Optional[Iterable[str]] = ("Unknown",),
    y_order: Optional[Iterable[str]] = None,
):
    """Static box plot (log-scale X axis) for the same variable as the histogram."""
    if col not in df.columns:
        return px.box()

    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0].astype(float)
    if s.empty:
        return px.box()

    label_txt = display_label if display_label else col

    # Prepare filtered DataFrame (including optional group column)
    df_plot = df.copy()
    df_plot[col] = s  # cleaned values

    # If no rows remain after filtering, return an empty figure
    if df_plot.empty:
        return px.box()

    # Horizontal: values on X axis, groups on Y (aggregated version for performance)
    if group_col and group_col in df_plot.columns:
        data = df_plot[[group_col, col]].dropna()
        # Strictly exclude Unknown (required behavior)
        data = data[data[group_col] != "Unknown"]
        # Determine order: prefer y_order when overlapping, otherwise top N present
        present = list(data[group_col].unique())
        if y_order:
            order = [c for c in y_order if c in present]
        else:
            order = []
        if not order:
            order = (
                data[group_col]
                .value_counts()
                .head(max_groups)
                .index
                .tolist()
            )
        # Build a trace per selected borough
        fig = go.Figure()
        for name in order:
            vals = pd.to_numeric(data.loc[data[group_col] == name, col], errors="coerce").dropna()
            vals = vals[vals > 0]
            if vals.empty:
                continue
            q1 = np.quantile(vals, 0.25)
            med = np.quantile(vals, 0.5)
            q3 = np.quantile(vals, 0.75)
            low = float(vals.min())
            high = float(vals.max())
            # Box trace without hover (overlay a scatter for clean hover without city name)
            fig.add_trace(go.Box(
                name="",
                y0=str(name),
                orientation="h",
                q1=[q1], median=[med], q3=[q3],
                lowerfence=[low], upperfence=[high],
                boxpoints=False,
                hoverinfo="skip",
            ))
            def _fmt(v: float) -> str:
                return _format_number(v, 0) if v >= 1000 else (f"{v:.2f}" if v >= 1 else f"{v:.3g}")
            hover_txt = (
                f"min = {_fmt(low)}<br>"
                f"q1 = {_fmt(q1)}<br>"
                f"median = {_fmt(med)}<br>"
                f"q3 = {_fmt(q3)}<br>"
                f"max = {_fmt(high)}"
            )
            fig.add_trace(go.Scatter(
                x=[med], y=[str(name)], mode="markers",
                marker=dict(size=12, color="rgba(0,0,0,0)"),
                hovertemplate="%{text}<extra></extra>", text=[hover_txt],
                showlegend=False,
            ))
        # If no trace was added (all categories excluded), return an annotated empty figure
        if not fig.data:
            empty = go.Figure()
            empty.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                title=f"Box plot — {label_txt}",
                showlegend=False,
            )
            empty.add_annotation(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text="Aucune catégorie disponible",
                showarrow=False,
                font=dict(size=13, color="#e5e7eb"),
                bgcolor="rgba(17,24,39,0.55)",
            )
            return empty
        fig.update_yaxes(categoryorder="array", categoryarray=order, title=group_col)
    else:
        vals = df_plot[col].dropna()
        vals = vals[vals > 0]
        if vals.empty:
            return go.Figure()
        q1 = np.quantile(vals, 0.25)
        med = np.quantile(vals, 0.5)
        q3 = np.quantile(vals, 0.75)
        low = float(vals.min())
        high = float(vals.max())
        # Single box trace without hover (overlay scatter for custom hover)
        fig = go.Figure(go.Box(
            name="",
            orientation="h",
            q1=[q1], median=[med], q3=[q3],
            lowerfence=[low], upperfence=[high],
            boxpoints=False,
            hoverinfo="skip",
        ))
        def _fmt(v: float) -> str:
            return _format_number(v, 0) if v >= 1000 else (f"{v:.2f}" if v >= 1 else f"{v:.3g}")
        hover_txt = (
            f"min = {_fmt(low)}<br>"
            f"q1 = {_fmt(q1)}<br>"
            f"median = {_fmt(med)}<br>"
            f"q3 = {_fmt(q3)}<br>"
            f"max = {_fmt(high)}"
        )
        fig.add_trace(go.Scatter(
            x=[med], y=[""], mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)"),
            hovertemplate="%{text}<extra></extra>", text=[hover_txt],
            showlegend=False,
        ))

    # X-axis bounds (values) — static in log scale
    data_min = max(float(df_plot[col].min()), 1e-12)
    data_max = float(df_plot[col].max())
    x_lo = data_min
    x_hi = max(data_max, x_lo * (1 + 1e-9))
    tickvals, ticktext = _log_ticks(x_lo, x_hi)
    fig.update_xaxes(
        type="log",
        title=label_txt + " (échelle log)",
        range=[np.log10(x_lo), np.log10(x_hi)],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        title=f"Box plot — {label_txt}",
        showlegend=False,
    )

    # Subtle, non-rotated hover label (left-aligned)
    fig.update_traces(
        hoverlabel=dict(
            bgcolor="#111",
            bordercolor="#444",
            font=dict(size=12, color="#fff"),
            align="left",
            namelength=0
        )
    )

    return fig
