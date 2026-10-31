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
