"""Aggregations and display formatting over a validated Dataset (no UI code).

Nothing here assigns a risk tier. Tiers are read from the data; this module
only counts, filters, orders and formats them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contract import HELD_OUT_SPLITS, LOG10_PC_FLOOR

TIER_RANK = {"safe": 0, "elevated": 1, "critical": 2}
QUEUE_ORDER = {"critical": 0, "needs_review": 1, "elevated": 2, "safe": 3}
TIER_LABEL = {"critical": "CRITICAL", "elevated": "ELEVATED", "safe": "SAFE", "needs_review": "REVIEW"}
WINDOWS = [("2–3 d", 2.0, 3.0), ("3–5 d", 3.0, 5.0), ("5–7 d", 5.0, math.inf)]
SORTS = {
    "priority": "Priority",
    "pc": "Predicted final Pc",
    "tca": "Time to TCA",
    "miss": "Observed miss distance",
    "event": "Event ID",
}


# ------------------------------------------------------------------ formatting

def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and math.isfinite(x)


def fmt_pc(log10_pc, at_floor: bool = False) -> str:
    if not _finite(log10_pc):
        return "—"
    if at_floor or log10_pc <= LOG10_PC_FLOOR + 1e-9:
        return "≈ 0"
    return f"{10 ** log10_pc:.1e}"


def fmt_log(log10_pc) -> str:
    return f"{log10_pc:.2f}" if _finite(log10_pc) else "—"


def fmt_m(metres) -> str:
    return f"{metres:,.0f} m" if _finite(metres) else "—"


def fmt_countdown(days) -> str:
    if not _finite(days):
        return "—"
    total_h = int(math.floor(days * 24 + 1e-9))  # a countdown never rounds up
    return f"T−{total_h // 24} d {total_h % 24:02d} h"


def fmt_prob(p) -> str:
    return f"{p:.0%}" if _finite(p) else "—"


def tier_label(tier) -> str:
    return TIER_LABEL.get(tier, "N/A") if tier else "N/A"


def transition(observed, predicted) -> str:
    if observed not in TIER_RANK or predicted not in TIER_RANK:
        return "unknown"
    diff = TIER_RANK[predicted] - TIER_RANK[observed]
    return "escalation" if diff > 0 else ("de-escalation" if diff < 0 else "unchanged")


# ------------------------------------------------------------------ filtering

@dataclass
class Filters:
    tiers: tuple[str, ...] = ()          # predicted final tiers; empty = all
    object_types: tuple[str, ...] = ()   # empty = all
    tca_min: float = 2.0
    tca_max: float = 7.0
    search: str = ""
    scope: str = "test"                  # "test", "held_out" or "all"


def scope_frame(events: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "test":
        return events[events["split"] == "test"]
    if scope == "held_out":
        return events[events["split"].isin(HELD_OUT_SPLITS)]
    return events


def apply_filters(events: pd.DataFrame, f: Filters) -> pd.DataFrame:
    df = scope_frame(events, f.scope)
    if f.tiers:
        df = df[df["predicted_final_tier"].isin(f.tiers)]
    if f.object_types:
        df = df[df["c_object_type"].isin(f.object_types)]
    upper = math.inf if f.tca_max >= 7.0 else f.tca_max
    df = df[(df["cutoff_time_to_tca_days"] >= f.tca_min) & (df["cutoff_time_to_tca_days"] <= upper)]
    text = f.search.strip()
    if text:
        df = df[df["event_id"].astype(str).str.contains(text, regex=False)]
    return df


def tier_counts(events: pd.DataFrame) -> dict[str, int]:
    counts = events["predicted_final_tier"].value_counts()
    out = {t: int(counts.get(t, 0)) for t in QUEUE_ORDER}
    out["total"] = int(len(events))
    return out


def transition_counts(events: pd.DataFrame) -> dict[str, int]:
    kinds = [transition(o, p) for o, p in zip(events["observed_tier"], events["predicted_final_tier"])]
    series = pd.Series(kinds, dtype=object)
    return {k: int((series == k).sum()) for k in ["escalation", "unchanged", "de-escalation", "unknown"]}


def warning_windows(events: pd.DataFrame) -> list[dict]:
    rows = []
    t = events["cutoff_time_to_tca_days"]
    for label, lo, hi in WINDOWS:
        part = events[(t >= lo) & (t < hi)]
        tiers = part["predicted_final_tier"]
        rows.append({"label": label, "critical": int((tiers == "critical").sum()),
                     "elevated": int((tiers == "elevated").sum()), "safe": int((tiers == "safe").sum()),
                     "review": int((tiers == "needs_review").sum()), "total": int(len(part))})
    return rows


def sort_queue(events: pd.DataFrame, key: str, descending: bool) -> pd.DataFrame:
    df = events.assign(
        _rank=events["predicted_final_tier"].map(QUEUE_ORDER),
        _esc=[0 if transition(o, p) == "escalation" else 1
              for o, p in zip(events["observed_tier"], events["predicted_final_tier"])],
        _pc=events["predicted_final_log10_pc"].fillna(-np.inf),
    )
    if key == "priority":
        df = df.sort_values(["_rank", "_esc", "cutoff_time_to_tca_days", "_pc"], ascending=[True, True, True, False])
        if descending:
            df = df.iloc[::-1]
    else:
        column = {"pc": "_pc", "tca": "cutoff_time_to_tca_days", "miss": "observed_miss_distance_m",
                  "event": "event_id"}[key]
        df = df.sort_values(column, ascending=not descending, na_position="last", kind="stable")
    return df.drop(columns=["_rank", "_esc", "_pc"])


def uncertainty_text(row) -> str:
    lo, hi = row.get("predicted_final_log10_pc_lower"), row.get("predicted_final_log10_pc_upper")
    if _finite(lo) and _finite(hi):
        return f"Pc {fmt_pc(lo)} – {fmt_pc(hi)}"
    if _finite(row.get("p_critical")):
        return f"P(critical) {fmt_prob(row.get('p_critical'))}"
    return "not provided"


def queue_rows(events: pd.DataFrame) -> list[dict[str, str]]:
    rows = []
    for rec in events.to_dict("records"):
        kind = transition(rec["observed_tier"], rec["predicted_final_tier"])
        rows.append({
            "key": f"{rec['source']}|{rec['event_id']}",
            "event_id": str(rec["event_id"]),
            "object_type": rec["c_object_type"],
            "observed_tier": rec["observed_tier"] or "none",
            "observed_label": tier_label(rec["observed_tier"]),
            "predicted_tier": rec["predicted_final_tier"],
            "predicted_label": tier_label(rec["predicted_final_tier"]),
            "transition": kind,
            "pred_pc": fmt_pc(rec["predicted_final_log10_pc"], rec["predicted_final_at_floor"]),
            "obs_miss": fmt_m(rec["observed_miss_distance_m"]),
            "countdown": fmt_countdown(rec["cutoff_time_to_tca_days"]),
            "uncertainty": uncertainty_text(rec),
            "action": rec["recommended_action"] or "—",
            "split": rec["split"],
            "actual_tier": rec.get("actual_final_tier") or "none",
            "actual_label": tier_label(rec.get("actual_final_tier")),
        })
    return rows


# ------------------------------------------------------------------ validation

def validation_metrics(events: pd.DataFrame) -> dict:
    held = events[events["split"].isin(HELD_OUT_SPLITS) & events["actual_final_tier"].notna()]
    labels_true = ["critical", "elevated", "safe"]
    labels_pred = ["critical", "elevated", "safe", "needs_review"]
    matrix = [[int(((held["actual_final_tier"] == a) & (held["predicted_final_tier"] == p)).sum())
               for p in labels_pred] for a in labels_true]
    n = len(held)
    correct = int((held["actual_final_tier"] == held["predicted_final_tier"]).sum())
    actual_crit = held["actual_final_tier"] == "critical"
    pred_crit = held["predicted_final_tier"] == "critical"
    tp = int((actual_crit & pred_crit).sum())
    fn = int((actual_crit & ~pred_crit).sum())
    fp = int((~actual_crit & pred_crit).sum())
    missed = held[actual_crit & ~pred_crit]
    return {
        "n": n,
        "splits": ", ".join(sorted(held["split"].unique())) if n else "—",
        "accuracy": correct / n if n else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "tp": tp, "fn": fn, "fp": fp, "critical_actual": int(actual_crit.sum()),
        "labels_true": labels_true, "labels_pred": labels_pred, "matrix": matrix,
        "missed": missed,
    }


# ------------------------------------------------------------------ encounter decision support
# These helpers EXPLAIN a tier that the risk pipeline already assigned. They never
# change it: a disagreement between the data's tier and this check is reported.

PC_CRITICAL_LOG, PC_ELEVATED_LOG = -4.0, -6.0
MISS_CRITICAL_M, MISS_ELEVATED_M = 1000.0, 5000.0
_SEVERITY = {"none": 0, "elevated": 1, "critical": 2}


def threshold_checks(ev: dict) -> dict:
    """Compare the predicted final values against the published thresholds."""
    rows = []
    log_pc = ev.get("predicted_final_log10_pc")
    if _finite(log_pc):
        level = "critical" if log_pc >= PC_CRITICAL_LOG else "elevated" if log_pc >= PC_ELEVATED_LOG else "none"
        rows.append({"criterion": "Predicted final Pc", "value": fmt_pc(log_pc, ev.get("predicted_final_at_floor", False)),
                     "rule": "≥ 1e-4 critical · ≥ 1e-6 elevated", "level": level})
    else:
        rows.append({"criterion": "Predicted final Pc", "value": "not provided", "rule": "≥ 1e-4 critical · ≥ 1e-6 elevated",
                     "level": "unknown"})

    basis = ev.get("final_miss_distance_basis")
    miss = ev.get("predicted_final_miss_distance_m")
    label = "Predicted final miss distance" if basis == "predicted" else (
        "Miss distance (observed at cutoff)" if basis == "observed_at_cutoff" else "Final miss distance")
    if _finite(miss):
        level = "critical" if miss < MISS_CRITICAL_M else "elevated" if miss <= MISS_ELEVATED_M else "none"
        rows.append({"criterion": label, "value": fmt_m(miss), "rule": "< 1,000 m critical · ≤ 5,000 m elevated",
                     "level": level})
    else:
        rows.append({"criterion": label, "value": "not provided", "rule": "< 1,000 m critical · ≤ 5,000 m elevated",
                     "level": "unknown"})

    tier = ev.get("predicted_final_tier")
    known = [r for r in rows if r["level"] != "unknown"]
    implied = max((r["level"] for r in known), key=_SEVERITY.get, default=None)
    for r in rows:
        r["triggered"] = "yes" if tier in ("critical", "elevated") and r["level"] == tier else "no"

    if tier == "needs_review":
        summary = "No tier could be predicted: the model output is missing or unusable."
    elif tier in ("critical", "elevated"):
        hits = [r for r in rows if r["triggered"] == "yes"]
        summary = (f"{tier_label(tier)} because " + " and ".join(
            f"{r['criterion'][0].lower()}{r['criterion'][1:]} {r['value']} meets the {tier} threshold" for r in hits) + ".") if hits else (
            f"{tier_label(tier)} as supplied by the risk pipeline; the values shown do not reproduce it.")
    else:
        summary = "SAFE: predicted Pc is below 1e-6 and the miss distance is above 5,000 m."
    consistent = tier == "needs_review" or not known or len(known) < 2 or (implied or "none") == (
        "none" if tier == "safe" else tier)
    return {"rows": rows, "summary": summary, "consistent": consistent}


def reliability(ev: dict, used_flag_count: int = 0) -> dict:
    """Summarise uncertainty from fields the model actually supplied. Never invents values."""
    tier = ev.get("predicted_final_tier")
    notes: list[str] = []
    prob = {"critical": ev.get("p_critical"), "elevated": ev.get("p_elevated"), "safe": ev.get("p_safe")}.get(tier)
    lo, hi = ev.get("predicted_final_log10_pc_lower"), ev.get("predicted_final_log10_pc_upper")
    has_interval = _finite(lo) and _finite(hi)

    if tier == "needs_review":
        level = "unavailable"
    elif _finite(prob):
        level = "high" if prob >= 0.8 else "moderate" if prob >= 0.5 else "low"
        notes.append(f"Model probability for the predicted tier: {prob:.0%}.")
    elif has_interval:
        level = "interval only"
    else:
        level = "not assessed"
        notes.append("The model supplied no probability or interval for this encounter.")

    crosses = []
    if has_interval:
        notes.append(f"Predicted Pc range {fmt_pc(lo)} – {fmt_pc(hi)}.")
        for value, name in ((PC_CRITICAL_LOG, "1e-4"), (PC_ELEVATED_LOG, "1e-6")):
            if lo < value <= hi:
                crosses.append(name)
        if crosses:
            notes.append("The range spans the " + " and ".join(crosses) + " threshold"
                         + ("s" if len(crosses) > 1 else "") + ", so the tier could differ.")
            if level == "high":
                level = "moderate"
    flags = list(ev.get("quality_flags") or [])
    if flags or used_flag_count:
        notes.append(f"Data-quality flags: {len(flags)} on the prediction, {used_flag_count} on input CDMs.")
        if level == "high":
            level = "moderate"
    if ev.get("predicted_final_at_floor"):
        notes.append("Prediction is at the risk −30 floor (≈ 0 probability).")
    return {"level": level, "notes": notes, "crosses": crosses, "probability": fmt_prob(prob) if _finite(prob) else "—"}


NEXT_STEPS = {
    "refer": "Refer for manoeuvre assessment",
    "request": "Request additional data",
    "monitor": "Continue monitoring",
}


def next_step(ev: dict, rel: dict) -> dict:
    """Operator triage suggestion. Decision support only; never recommends a manoeuvre."""
    tier = ev.get("predicted_final_tier")
    reasons: list[str] = []
    if tier == "critical":
        key = "refer"
        reasons.append("Predicted final tier is CRITICAL.")
        reasons.append("Specialists decide whether any manoeuvre is needed; this system does not.")
    elif tier == "needs_review":
        key = "request"
        reasons.append("No usable prediction; check input CDMs and model output.")
    elif tier == "elevated":
        key = "request"
        reasons.append("Predicted final tier is ELEVATED; updated CDMs would firm up the assessment before TCA.")
    else:
        key = "monitor"
        reasons.append("Predicted final tier is SAFE.")
        if rel["level"] == "low" or "1e-6" in rel["crosses"] or ev.get("quality_flags"):
            key = "request"
            reasons.append("Low confidence, a range spanning 1e-6, or data-quality flags: verify with more data.")
    if transition(ev.get("observed_tier"), tier) == "escalation":
        reasons.append("Risk is predicted to rise above the currently observed tier.")
    return {"key": key, "label": NEXT_STEPS[key], "reasons": reasons}
