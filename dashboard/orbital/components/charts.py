"""Plotly figures. Display only: tiers and values come from the data layer."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from orbital.data.contract import LOG10_PC_FLOOR, MIN_INPUT_TCA_DAYS

CYAN, MUTED, GRID = "#38bdf8", "#7d91b6", "rgba(56,189,248,0.08)"
TIER_COLORS = {"critical": "#ef4444", "elevated": "#f59e0b", "safe": "#22c55e", "needs_review": "#a78bfa"}


def _base(title: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        meta=title, height=height, margin=dict(l=56, r=18, t=18, b=44), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, system-ui, sans-serif", size=11, color="#9fb3d6"),
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10.5), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0b1a33", bordercolor=CYAN, font=dict(family="JetBrains Mono, monospace", size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def empty(title: str, message: str, height: int = 300) -> go.Figure:
    fig = _base(title, height)
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(size=12, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _time_axis(fig: go.Figure, max_t: float) -> None:
    fig.update_xaxes(title="time_to_tca (days) — closest approach at 0", range=[max(max_t, 2.5) + 0.2, -0.15])
    fig.add_vrect(x0=MIN_INPUT_TCA_DAYS, x1=-0.15, fillcolor="rgba(125,145,182,0.06)", line_width=0)
    fig.add_vline(x=MIN_INPUT_TCA_DAYS, line=dict(color=CYAN, width=1.5, dash="dash"),
                  annotation=dict(text="2-day prediction cutoff", font=dict(color=CYAN, size=10.5)),
                  annotation_position="top left")


def risk_evolution(hist: pd.DataFrame, ev: dict) -> go.Figure:
    fig = _base("Risk evolution — observed log10(Pc) vs time_to_tca", 320)
    h = hist.dropna(subset=["observed_log10_pc"]).sort_values("time_to_tca_days", ascending=False)
    if h.empty and not math.isfinite(ev.get("predicted_final_log10_pc") or float("nan")):
        return empty("", "No observed risk values or prediction available", 320)
    used, after = h[h["used_for_prediction"]], h[~h["used_for_prediction"]]
    hover = "time_to_tca %{x:.2f} d<br>log10(Pc) %{y:.2f}<extra>%{fullData.name}</extra>"
    fig.add_trace(go.Scatter(x=used["time_to_tca_days"], y=used["observed_log10_pc"], name="Observed CDMs (model input, ≥ 2 d)",
                             mode="lines+markers", line=dict(color=CYAN, width=2), marker=dict(size=7), hovertemplate=hover))
    if not after.empty:
        bridge = pd.concat([used.tail(1), after])
        fig.add_trace(go.Scatter(x=bridge["time_to_tca_days"], y=bridge["observed_log10_pc"], mode="lines+markers",
                                 name="Observed after cutoff (held-out truth, not input)",
                                 line=dict(color=MUTED, width=1.5, dash="dot"),
                                 marker=dict(size=7, symbol="circle-open"), hovertemplate=hover))
    pred = ev.get("predicted_final_log10_pc")
    if pred is not None and math.isfinite(pred):
        lo, hi = ev.get("predicted_final_log10_pc_lower"), ev.get("predicted_final_log10_pc_upper")
        err = None
        if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
            err = dict(type="data", symmetric=False, array=[hi - pred], arrayminus=[pred - lo], color="#e0f2fe", width=6)
        color = TIER_COLORS.get(ev.get("predicted_final_tier"), CYAN)
        fig.add_trace(go.Scatter(x=[0], y=[pred], mode="markers", name="Model prediction: final log10(Pc)",
                                 marker=dict(size=15, symbol="star", color=color, line=dict(color="white", width=1)),
                                 error_y=err, hovertemplate="Predicted FINAL log10(Pc) %{y:.2f}<extra>model</extra>"))
    for y, label, color in [(-4, "Pc 1e-4 (critical)", TIER_COLORS["critical"]), (-6, "Pc 1e-6 (elevated)", TIER_COLORS["elevated"])]:
        fig.add_hline(y=y, line=dict(color=color, width=1, dash="dot"), annotation=dict(text=label, font=dict(size=10, color=color)),
                      annotation_position="bottom left")
    values = pd.concat([h["observed_log10_pc"], pd.Series([pred], dtype=float)]).dropna()
    floor = (values <= LOG10_PC_FLOOR + 1e-9).any()
    ymin = LOG10_PC_FLOOR - 1 if floor else min(values.min() - 1, -7) if len(values) else -10
    fig.update_yaxes(title="log10(Pc)", range=[ymin, 0.3])
    if floor:
        fig.add_hline(y=LOG10_PC_FLOOR, line=dict(color=MUTED, width=1), annotation=dict(
            text="risk −30 floor ≈ 0 probability", font=dict(size=10, color=MUTED)), annotation_position="top right")
    _time_axis(fig, float(hist["time_to_tca_days"].max()) if len(hist) else 7)
    return fig


def miss_evolution(hist: pd.DataFrame, ev: dict) -> go.Figure:
    fig = _base("Miss-distance evolution (metres) vs time_to_tca", 320)
    h = hist.dropna(subset=["miss_distance_m"]).sort_values("time_to_tca_days", ascending=False)
    if h.empty:
        return empty("", "No miss-distance history available", 320)
    used, after = h[h["used_for_prediction"]], h[~h["used_for_prediction"]]
    hover = "time_to_tca %{x:.2f} d<br>miss %{y:,.0f} m<extra>%{fullData.name}</extra>"
    fig.add_trace(go.Scatter(x=used["time_to_tca_days"], y=used["miss_distance_m"], name="Observed (model input, ≥ 2 d)",
                             mode="lines+markers", line=dict(color=CYAN, width=2), marker=dict(size=7), hovertemplate=hover))
    if not after.empty:
        bridge = pd.concat([used.tail(1), after])
        fig.add_trace(go.Scatter(x=bridge["time_to_tca_days"], y=bridge["miss_distance_m"], mode="lines+markers",
                                 name="Observed after cutoff (held-out truth)", line=dict(color=MUTED, dash="dot"),
                                 marker=dict(size=7, symbol="circle-open"), hovertemplate=hover))
    pm = ev.get("predicted_final_miss_distance_m")
    if ev.get("final_miss_distance_basis") == "predicted" and pm is not None and math.isfinite(pm):
        fig.add_trace(go.Scatter(x=[0], y=[pm], mode="markers", name="Model prediction: final miss distance",
                                 marker=dict(size=14, symbol="star", color=TIER_COLORS.get(ev.get("predicted_final_tier"), CYAN),
                                             line=dict(color="white", width=1)),
                                 hovertemplate="Predicted FINAL miss %{y:,.0f} m<extra>model</extra>"))
    for y, label, color in [(1000, "1,000 m", TIER_COLORS["critical"]), (5000, "5,000 m", TIER_COLORS["elevated"])]:
        fig.add_hline(y=y, line=dict(color=color, width=1, dash="dot"), annotation=dict(text=label, font=dict(size=10, color=color)),
                      annotation_position="top left")
    fig.update_yaxes(title="miss distance (m)", rangemode="tozero", tickformat=",")
    _time_axis(fig, float(h["time_to_tca_days"].max()))
    return fig


def rtn_geometry(hist: pd.DataFrame) -> go.Figure:
    title = "Relative encounter geometry at TCA — R/T/N (km), not an orbital trajectory"
    cols = ["relative_position_r_m", "relative_position_t_m", "relative_position_n_m"]
    h = hist.dropna(subset=cols).sort_values("time_to_tca_days", ascending=False)
    if h.empty:
        return empty(title, "No relative-position data", 380)
    fig = _base(title, 380)
    r, t, n = (h[c] / 1000 for c in cols)
    colors = np.where(h["used_for_prediction"], CYAN, MUTED)
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="Primary (origin)",
                               marker=dict(size=6, color="#e0f2fe", symbol="diamond")))
    fig.add_trace(go.Scatter3d(
        x=t, y=n, z=r, mode="lines+markers", name="Successive CDM estimates of secondary at TCA",
        line=dict(color="rgba(56,189,248,0.45)", width=3), marker=dict(size=4, color=colors),
        customdata=np.stack([h["time_to_tca_days"], h["miss_distance_m"]], axis=1),
        hovertemplate="CDM at T−%{customdata[0]:.2f} d<br>T %{x:.2f} km · N %{y:.2f} km · R %{z:.2f} km"
                      "<br>miss %{customdata[1]:,.0f} m<extra></extra>"))
    axis = dict(gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)", color="#9fb3d6", zerolinecolor="rgba(56,189,248,0.3)")
    fig.update_layout(scene=dict(xaxis=dict(title="T (km)", **axis), yaxis=dict(title="N (km)", **axis),
                                 zaxis=dict(title="R (km)", **axis), aspectmode="cube", bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=0, r=0, t=10, b=10), legend=dict(y=-0.02))
    return fig


def window_chart(windows: list[dict]) -> go.Figure:
    fig = _base("Predicted final tier by warning window (time_to_tca at last eligible CDM)", 280)
    labels = [w["label"] for w in windows]
    for key, name, color in [("critical", "Critical", TIER_COLORS["critical"]), ("elevated", "Elevated", TIER_COLORS["elevated"]),
                             ("review", "Needs review", TIER_COLORS["needs_review"]), ("safe", "Safe", TIER_COLORS["safe"])]:
        fig.add_trace(go.Bar(x=labels, y=[w[key] for w in windows], name=name, marker_color=color,
                             hovertemplate="%{x}: %{y} events<extra>" + name + "</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="encounters")
    return fig
