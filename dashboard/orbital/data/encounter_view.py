"""Data for the Encounter Analysis visualisations (no UI code).

The illustrative view places each CDM at its MEASURED relative position
(R/T/N, metres) projected onto the 2-D plane that shows the most of the
geometry. Only the decoration (Earth backdrop, icons, faint approach arcs) is
illustrative. Nothing here invents a trajectory, an absolute position, a
predicted final position or a predicted miss distance.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .analytics import fmt_countdown, fmt_log, fmt_m, fmt_pc

AXES = {"R": "relative_position_r_m", "T": "relative_position_t_m", "N": "relative_position_n_m"}
AXIS_NAMES = {"R": "radial", "T": "along-track", "N": "cross-track"}


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def ordered_history(hist: pd.DataFrame) -> pd.DataFrame:
    """CDMs from oldest (largest time_to_tca) to newest, numbered 1..n."""
    if hist is None or hist.empty:
        return pd.DataFrame()
    h = hist.sort_values(["time_to_tca_days", "row_id"], ascending=[False, True]).reset_index(drop=True)
    return h.assign(n_cdm=np.arange(1, len(h) + 1))


def latest_eligible_row(ev: dict, hist: pd.DataFrame) -> int | None:
    """row_id of the last CDM the model was allowed to use."""
    cutoff = _num(ev.get("cutoff_row_id"))
    if cutoff is not None and not hist.empty and (hist["row_id"] == int(cutoff)).any():
        return int(cutoff)
    used = hist[hist["used_for_prediction"]] if not hist.empty else hist
    if used.empty:
        return None
    return int(used.sort_values("time_to_tca_days").iloc[0]["row_id"])


def cdm_details(row: dict, latest_row: int | None) -> dict[str, str]:
    """Readable measurements of one CDM for tooltips and the inspector."""
    def comp(key):
        v = _num(row.get(AXES[key]))
        return f"{v:,.0f} m" if v is not None else "—"

    log_pc = _num(row.get("observed_log10_pc"))
    speed = _num(row.get("relative_speed_mps"))
    used = bool(row.get("used_for_prediction"))
    return {
        "row_id": str(int(row["row_id"])),
        "n": str(int(row.get("n_cdm", 0))),
        "t": f"{row['time_to_tca_days']:.2f} d",
        "countdown": fmt_countdown(row["time_to_tca_days"]),
        "miss": fmt_m(_num(row.get("miss_distance_m"))),
        "r": comp("R"), "tt": comp("T"), "nn": comp("N"),
        "pc": fmt_pc(log_pc), "log": fmt_log(log_pc),
        "speed": f"{speed:,.0f} m/s" if speed is not None else "—",
        "role": "Latest eligible CDM (model input)" if latest_row is not None and int(row["row_id"]) == latest_row
        else ("Model input (≥ 2 d)" if used else "After cutoff: held-out truth, not a model input"),
        "flags": ", ".join(row.get("quality_flags") or []) or "none",
    }


def scene(ev: dict, hist: pd.DataFrame) -> dict:
    """Projected measured geometry for the illustrative encounter view."""
    h = ordered_history(hist)
    out = {"available": False, "points": [], "h_axis": "", "v_axis": "", "dropped_axis": "",
           "plane_note": "No relative-position telemetry for this encounter.", "predicted_miss_km": None,
           "predicted_miss_label": "", "latest_row": None, "separation_label": ""}
    if h.empty:
        return out
    h = h.dropna(subset=list(AXES.values()))
    if h.empty:
        return out
    spread = {k: float(np.sqrt(np.mean(np.square(h[c].astype(float))))) for k, c in AXES.items()}
    h_axis, v_axis = sorted(spread, key=spread.get, reverse=True)[:2]
    dropped = next(k for k in AXES if k not in (h_axis, v_axis))
    latest = latest_eligible_row(ev, h)
    points = []
    for rec in h.to_dict("records"):
        d = cdm_details(rec, latest)
        points.append({
            "id": d["row_id"], "x": float(rec[AXES[h_axis]]) / 1000, "y": float(rec[AXES[v_axis]]) / 1000,
            "used": bool(rec["used_for_prediction"]), "latest": latest is not None and int(rec["row_id"]) == latest,
            "label": f"CDM {d['n']} · {d['countdown']}", "miss": d["miss"], "pc": d["pc"], "role": d["role"],
            "rtn": f"R {d['r']} · T {d['tt']} · N {d['nn']}",
        })
    latest_rec = h[h["row_id"] == latest].iloc[0] if latest is not None else None
    miss = _num(ev.get("predicted_final_miss_distance_m"))
    has_pred_miss = ev.get("final_miss_distance_basis") == "predicted" and miss is not None
    out.update(
        available=True, points=points, h_axis=h_axis, v_axis=v_axis, dropped_axis=dropped,
        plane_note=(f"Measured positions projected onto the {h_axis}–{v_axis} plane "
                    f"({AXIS_NAMES[h_axis]} × {AXIS_NAMES[v_axis]}); the {dropped} component is in each tooltip."),
        latest_row=latest,
        separation_label=(f"{fmt_m(_num(latest_rec['miss_distance_m']))} (observed)" if latest_rec is not None else ""),
        predicted_miss_km=(miss / 1000) if has_pred_miss else None,
        predicted_miss_label=(f"Predicted final miss distance {fmt_m(miss)} (distance only)" if has_pred_miss else ""),
    )
    return out


def nearest_cdm(hist: pd.DataFrame, point: dict) -> int | None:
    """Map a clicked 3-D plot point (km; x=T, y=N, z=R) back to its CDM row_id."""
    x, y, z = (_num(point.get(k)) for k in ("x", "y", "z"))
    if hist is None or hist.empty or None in (x, y, z):
        return None
    h = hist.dropna(subset=list(AXES.values()))
    if h.empty:
        return None
    d = ((h[AXES["T"]] / 1000 - x) ** 2 + (h[AXES["N"]] / 1000 - y) ** 2 + (h[AXES["R"]] / 1000 - z) ** 2)
    i = d.idxmin()
    return int(h.loc[i, "row_id"]) if d.loc[i] < 1e-6 else None


def uncertainty(ev: dict) -> dict:
    """Class probabilities / Pc range exactly as supplied; nothing estimated here."""
    probs = []
    for tier in ("critical", "elevated", "safe"):
        p = _num(ev.get(f"p_{tier}"))
        if p is not None:
            probs.append({"tier": tier, "label": tier.upper(), "value": f"{p:.0%}", "width": f"{max(p, 0) * 100:.1f}%"})
    lo, hi = _num(ev.get("predicted_final_log10_pc_lower")), _num(ev.get("predicted_final_log10_pc_upper"))
    interval = f"{fmt_pc(lo)} – {fmt_pc(hi)}" if lo is not None and hi is not None else ""
    return {"probs": probs, "interval": interval, "provided": bool(probs or interval)}
