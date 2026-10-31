"""Plotly figures for the dashboard. Pure display: no risk logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui.theme import TIER_COLORS, TIER_TITLES

TIER_SEQUENCE = ["critical", "needs_review", "elevated", "safe"]
_MARGIN = dict(l=10, r=10, t=40, b=10)


def _empty(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(size=14))
    fig.update_layout(title=title, xaxis_visible=False, yaxis_visible=False, height=320, margin=_MARGIN)
    return fig


def _pc_text(series: pd.Series) -> list[str]:
    return [f"{v:.2e}" if np.isfinite(v) else "unavailable" for v in series.astype(float)]


def tier_breakdown(latest: pd.DataFrame) -> go.Figure:
    if latest.empty:
        return _empty("Encounters by risk tier", "No encounters match the filters")
    counts = latest["risk_tier"].value_counts().reindex(TIER_SEQUENCE, fill_value=0)
    counts = counts[counts > 0]
    fig = go.Figure(go.Bar(
        x=counts.values, y=[TIER_TITLES[t] for t in counts.index], orientation="h",
        marker_color=[TIER_COLORS[t] for t in counts.index], text=counts.values, textposition="auto",
        hovertemplate="%{y}: %{x} encounters<extra></extra>",
    ))
    fig.update_layout(title="Encounters by risk tier (latest message per event)", height=320,
                      margin=_MARGIN, yaxis=dict(autorange="reversed"), xaxis_title="Encounters")
    return fig


def risk_map(latest: pd.DataFrame, thr: dict[str, float]) -> go.Figure:
    data = latest.dropna(subset=["predicted_log10_pc", "miss_distance_km"])
    data = data[data["miss_distance_km"] > 0]
    if data.empty:
        return _empty("Estimated Pc vs miss distance", "No encounters with a Pc estimate")
    fig = go.Figure()
    for tier in TIER_SEQUENCE:
        part = data[data["risk_tier"] == tier]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["miss_distance_km"], y=part["predicted_log10_pc"], mode="markers",
            name=TIER_TITLES[tier], marker=dict(color=TIER_COLORS[tier], size=10, line=dict(width=0.5)),
            customdata=np.stack([part["event_id"], _pc_text(part["predicted_pc"]), part["tca_hours"],
                                 part["c_object_type"]], axis=1),
            hovertemplate=("Event %{customdata[0]} (%{customdata[3]})<br>Estimated Pc %{customdata[1]}"
                           "<br>Miss distance %{x:.2f} km<br>%{customdata[2]:.1f} h to TCA<extra></extra>"),
        ))
    for y, label in [(thr["LOG10_PC_CRITICAL"], "Pc 1e-4"), (thr["LOG10_PC_ELEVATED"], "Pc 1e-6")]:
        fig.add_hline(y=y, line_dash="dash", line_color="gray", annotation_text=label,
                      annotation_position="top left")
    for x, label in [(thr["MISS_DISTANCE_CRITICAL_M"] / 1000, "1 km"),
                     (thr["MISS_DISTANCE_ELEVATED_M"] / 1000, "5 km")]:
        fig.add_vline(x=x, line_dash="dot", line_color="gray", annotation_text=label,
                      annotation_position="top right")
    fig.update_layout(title="Estimated Pc vs miss distance", height=380, margin=_MARGIN,
                      xaxis=dict(title="Miss distance (km, log scale)", type="log"),
                      yaxis=dict(title="Estimated log10(Pc)"), legend=dict(orientation="h", y=-0.2))
    return fig


def tca_timeline(latest: pd.DataFrame) -> go.Figure:
    if latest.empty:
        return _empty("Time to closest approach", "No encounters match the filters")
    max_h = max(float(latest["tca_hours"].max()), 24.0)
    edges = np.arange(0, np.ceil(max_h / 6) * 6 + 6, 6)
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=24, fillcolor="#D64545", opacity=0.08, line_width=0,
                  annotation_text="Next 24 h", annotation_position="top left")
    for tier in reversed(TIER_SEQUENCE):
        part = latest[latest["risk_tier"] == tier]
        if part.empty:
            continue
        fig.add_trace(go.Histogram(
            x=part["tca_hours"], xbins=dict(start=0, end=edges[-1], size=6), name=TIER_TITLES[tier],
            marker_color=TIER_COLORS[tier], hovertemplate="%{y} encounters<extra>" + TIER_TITLES[tier] + "</extra>",
        ))
    fig.update_layout(barmode="stack", title="Encounters by hours to TCA (6 h bins)", height=320,
                      margin=_MARGIN, xaxis_title="Hours to TCA", yaxis_title="Encounters",
                      legend=dict(orientation="h", y=-0.25))
    return fig


def object_types(latest: pd.DataFrame) -> go.Figure:
    if latest.empty:
        return _empty("Object types", "No encounters match the filters")
    table = latest.pivot_table(index="c_object_type", columns="risk_tier", values="event_id",
                               aggfunc="count", fill_value=0)
    fig = go.Figure()
    for tier in reversed(TIER_SEQUENCE):
        if tier in table.columns:
            fig.add_trace(go.Bar(x=table.index, y=table[tier], name=TIER_TITLES[tier],
                                 marker_color=TIER_COLORS[tier]))
    fig.update_layout(barmode="stack", title="Encounters by secondary object type", height=320,
                      margin=_MARGIN, yaxis_title="Encounters", legend=dict(orientation="h", y=-0.25))
    return fig


def _history_base(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values(["time_to_tca_days", "row_id"], ascending=[False, True])


def risk_history(rows: pd.DataFrame, thr: dict[str, float]) -> go.Figure:
    rows = _history_base(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rows["tca_hours"], y=rows["predicted_log10_pc"], mode="lines+markers", name="Estimated log10(Pc)",
        line=dict(color="#4A6FA5"), marker=dict(size=10, color=[TIER_COLORS[t] for t in rows["risk_tier"]],
                                                line=dict(width=1, color="white")),
        customdata=np.stack([_pc_text(rows["predicted_pc"]), rows["risk_tier"]], axis=1),
        hovertemplate="%{x:.1f} h to TCA<br>Estimated Pc %{customdata[0]}<br>Tier: %{customdata[1]}<extra></extra>",
    ))
    fig.add_hline(y=thr["LOG10_PC_CRITICAL"], line_dash="dash", line_color=TIER_COLORS["critical"],
                  annotation_text="Critical: Pc 1e-4", annotation_position="top left")
    fig.add_hline(y=thr["LOG10_PC_ELEVATED"], line_dash="dash", line_color=TIER_COLORS["elevated"],
                  annotation_text="Elevated: Pc 1e-6", annotation_position="bottom left")
    fig.update_layout(title="Risk estimate history (model estimates)", height=320, margin=_MARGIN,
                      xaxis=dict(title="Hours to TCA (TCA on the right)", autorange="reversed"),
                      yaxis_title="Estimated log10(Pc)", showlegend=False)
    return fig


def miss_history(rows: pd.DataFrame, thr: dict[str, float]) -> go.Figure:
    rows = _history_base(rows)
    fig = go.Figure(go.Scatter(
        x=rows["tca_hours"], y=rows["miss_distance_km"], mode="lines+markers", line=dict(color="#6B7A8F"),
        marker=dict(size=9), hovertemplate="%{x:.1f} h to TCA<br>Miss distance %{y:.2f} km<extra></extra>",
    ))
    fig.add_hline(y=thr["MISS_DISTANCE_CRITICAL_M"] / 1000, line_dash="dash", line_color=TIER_COLORS["critical"],
                  annotation_text="1 km", annotation_position="top left")
    fig.add_hline(y=thr["MISS_DISTANCE_ELEVATED_M"] / 1000, line_dash="dash", line_color=TIER_COLORS["elevated"],
                  annotation_text="5 km", annotation_position="top left")
    fig.update_layout(title="Miss distance history", height=320, margin=_MARGIN,
                      xaxis=dict(title="Hours to TCA (TCA on the right)", autorange="reversed"),
                      yaxis=dict(title="Miss distance (km)", rangemode="tozero"), showlegend=False)
    return fig


def rtn_geometry(rows: pd.DataFrame) -> go.Figure:
    rows = _history_base(rows).dropna(subset=["relative_position_r", "relative_position_t", "relative_position_n"])
    if rows.empty:
        return _empty("Encounter geometry at TCA (R/T/N)", "No relative-position data")
    r, t, n = (rows[f"relative_position_{a}"] / 1000 for a in "rtn")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["Primary"],
                               marker=dict(size=6, color="#2B6CB0", symbol="diamond"), name="Primary object",
                               hoverinfo="name"))
    fig.add_trace(go.Scatter3d(
        x=t, y=n, z=r, mode="lines+markers", name="Successive estimates",
        line=dict(color="#8899AA", width=3),
        marker=dict(size=4, color=[TIER_COLORS[x] for x in rows["risk_tier"]]),
        customdata=rows["tca_hours"],
        hovertemplate="Estimate made %{customdata:.1f} h before TCA<br>T %{x:.2f} km, N %{y:.2f} km, "
                      "R %{z:.2f} km<extra></extra>",
    ))
    last = rows.iloc[-1]
    fig.add_trace(go.Scatter3d(
        x=[last.relative_position_t / 1000], y=[last.relative_position_n / 1000],
        z=[last.relative_position_r / 1000], mode="markers", name="Latest estimate",
        marker=dict(size=8, color=TIER_COLORS[last.risk_tier], line=dict(width=2, color="black")),
        hoverinfo="name",
    ))
    fig.update_layout(
        title="Successive estimates of encounter geometry at TCA (not flight paths)", height=420,
        margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.05),
        scene=dict(xaxis_title="T (km)", yaxis_title="N (km)", zaxis_title="R (km)", aspectmode="cube"),
    )
    return fig
