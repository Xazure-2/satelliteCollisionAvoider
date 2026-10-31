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


def risk_evolution(hist: pd.DataFrame, ev: dict, height: int = 320) -> go.Figure:
    fig = _base("Risk evolution — observed log10(Pc) vs time_to_tca", height)
    fig.update_layout(margin=dict(l=52, r=12, t=14, b=86), legend=dict(y=-0.36, font=dict(size=10)))
    h = hist.dropna(subset=["observed_log10_pc"]).sort_values("time_to_tca_days", ascending=False)
    if h.empty and not math.isfinite(ev.get("predicted_final_log10_pc") or float("nan")):
        return empty("", "No observed risk values or prediction available", height)
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


def miss_evolution(hist: pd.DataFrame, ev: dict, height: int = 320) -> go.Figure:
    fig = _base("Miss-distance evolution (metres) vs time_to_tca", height)
    fig.update_layout(margin=dict(l=58, r=12, t=14, b=86), legend=dict(y=-0.36, font=dict(size=10)))
    h = hist.dropna(subset=["miss_distance_m"]).sort_values("time_to_tca_days", ascending=False)
    if h.empty:
        return empty("", "No miss-distance history available", height)
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
    fig.update_yaxes(title="miss distance (m, log scale)", type="log", tickformat="~s")
    _time_axis(fig, float(h["time_to_tca_days"].max()))
    return fig


RTN_COLS = ["relative_position_r_m", "relative_position_t_m", "relative_position_n_m"]


def _sphere(radius_km: float, color: str, name: str, hover: str, opacity: float) -> go.Surface:
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 18))
    return go.Surface(x=radius_km * np.cos(u) * np.sin(v), y=radius_km * np.sin(u) * np.sin(v), z=radius_km * np.cos(v),
                      colorscale=[[0, color], [1, color]], showscale=False, opacity=opacity, name=name,
                      showlegend=True, hoverinfo="skip", contours=dict(
                          x=dict(show=True, color=color, width=1), y=dict(show=True, color=color, width=1)))


def encounter_geometry(hist: pd.DataFrame, ev: dict, reset: int = 0, height: int = 680) -> go.Figure:
    """Relative R/T/N geometry centred on the primary. Not an orbit and not an Earth map."""
    h = hist.dropna(subset=RTN_COLS).sort_values("time_to_tca_days", ascending=False).reset_index(drop=True)
    if h.empty:
        return empty("", "No relative-position telemetry for this encounter", height)
    fig = _base("", height)
    h = h.assign(n_cdm=np.arange(1, len(h) + 1), r_km=h[RTN_COLS[0]] / 1000, t_km=h[RTN_COLS[1]] / 1000,
                 n_km=h[RTN_COLS[2]] / 1000,
                 pc=[f"{10 ** v:.1e}" if np.isfinite(v) and v > -30 else ("≈ 0" if np.isfinite(v) else "—")
                     for v in h["observed_log10_pc"].astype(float)])
    hover = ("CDM %{customdata[0]} · T−%{customdata[1]:.2f} d<br>R %{z:.3f} km · T %{x:.3f} km · N %{y:.3f} km"
             "<br>miss %{customdata[2]:,.0f} m · observed Pc %{customdata[3]}<extra>%{fullData.name}</extra>")

    def custom(frame):
        return np.stack([frame["n_cdm"], frame["time_to_tca_days"], frame["miss_distance_m"], frame["pc"]], axis=1)

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["Primary"], textposition="top center",
                               name="Primary object (origin)", marker=dict(size=7, color="#e0f2fe", symbol="diamond"),
                               textfont=dict(color="#e0f2fe", size=11), hovertemplate="Primary object at origin<extra></extra>"))
    used, after = h[h["used_for_prediction"]], h[~h["used_for_prediction"]]
    if not used.empty:
        fig.add_trace(go.Scatter3d(x=used["t_km"], y=used["n_km"], z=used["r_km"], mode="lines+markers",
                                   name="Input CDMs (≥ 2 d before TCA)", line=dict(color="rgba(56,189,248,0.55)", width=4),
                                   marker=dict(size=5, color=CYAN), customdata=custom(used), hovertemplate=hover))
    if not after.empty:
        bridge = pd.concat([used.tail(1), after]) if not used.empty else after
        fig.add_trace(go.Scatter3d(x=bridge["t_km"], y=bridge["n_km"], z=bridge["r_km"], mode="lines+markers",
                                   name="After cutoff: held-out truth, not a model input",
                                   line=dict(color="rgba(125,145,182,0.6)", width=3, dash="dot"),
                                   marker=dict(size=4, color=MUTED, symbol="circle-open"), customdata=custom(bridge),
                                   hovertemplate=hover))

    cutoff_id = ev.get("cutoff_row_id")
    latest = h[h["row_id"] == cutoff_id] if cutoff_id is not None and np.isfinite(cutoff_id) else used.tail(1)
    if latest.empty:
        latest = used.tail(1)
    if not latest.empty:
        p = latest.iloc[0]
        color = TIER_COLORS.get(ev.get("observed_tier"), CYAN)
        fig.add_trace(go.Scatter3d(x=[0, p.t_km], y=[0, p.n_km], z=[0, p.r_km], mode="lines", name="Observed miss vector",
                                   line=dict(color=color, width=3, dash="dash"),
                                   hovertemplate=f"Observed miss distance {p.miss_distance_m:,.0f} m<extra></extra>"))
        fig.add_trace(go.Scatter3d(x=[p.t_km], y=[p.n_km], z=[p.r_km], mode="markers+text", text=["Latest eligible CDM"],
                                   textposition="top center", textfont=dict(color=color, size=11),
                                   name="Latest eligible CDM (prediction cutoff)",
                                   marker=dict(size=11, color=color, line=dict(color="white", width=2)),
                                   customdata=custom(latest.iloc[[0]]), hovertemplate=hover))

    miss = ev.get("predicted_final_miss_distance_m")
    if ev.get("final_miss_distance_basis") == "predicted" and miss is not None and np.isfinite(miss):
        fig.add_trace(_sphere(miss / 1000, CYAN, "Predicted final miss distance (radius only)",
                              f"Predicted final miss distance {miss:,.0f} m<br>Radius only: the model does not "
                              "predict direction", 0.09))
    fig.add_trace(_sphere(1.0, TIER_COLORS["critical"], "1 km critical threshold", "1,000 m critical threshold", 0.18))

    axis = dict(gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)", color="#9fb3d6", zerolinecolor="rgba(56,189,248,0.35)",
                showspikes=False)
    fig.update_layout(
        scene=dict(xaxis=dict(title="T · along-track (km)", **axis), yaxis=dict(title="N · cross-track (km)", **axis),
                   zaxis=dict(title="R · radial (km)", **axis), aspectmode="data", bgcolor="rgba(0,0,0,0)",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=0.9))),
        margin=dict(l=0, r=0, t=6, b=0), legend=dict(orientation="v", x=0.0, y=0.02, yanchor="bottom",
                                                     bgcolor="rgba(4,9,20,0.7)", font=dict(size=10.5)),
        uirevision=f"{ev.get('source')}|{ev.get('event_id')}|{reset}",
    )
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
