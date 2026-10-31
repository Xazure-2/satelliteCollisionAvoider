"""Event-level selection, 24-hour summaries, action queue, insights and export.

All functions consume the scored-row table produced by
:func:`src.risk.score_rows` (ROADMAP.md section 5). List columns may be Python
lists or JSON strings (as read back from CSV).

Design rules
------------
* An *event* is the pair ``(source, event_id)``; equal ``event_id`` values from
  different sources never collide. A row with a missing ``event_id`` is
  treated as its own event so it is never merged with others or dropped.
* ``time_to_tca_days`` is a message's lead time before TCA. Smaller
  nonnegative values are newer. Equal lead times are broken by the highest
  ``row_id`` (the later CSV row is newer), then by later position. Negative
  (post-TCA) messages are never selected as latest; they stay historical.
* Nothing here invents observation timestamps. Replay output is labelled
  ``historical_replay`` and keeps the original lead time next to the
  hypothetical replay time to TCA.
* Input risk labels are never read. Tiers come from ``risk_tier`` as assigned
  by :func:`src.risk.classify_risk`; probabilities are model-estimated Pc.
* Ordering is fully deterministic (stable sorts, explicit tie-breakers).

Dashboard entry points (``ui/backend.py`` EVENTS_API)
-----------------------------------------------------
``select_latest_messages``, ``summarize_upcoming_24h``, ``build_action_queue``,
``build_insights`` and ``select_replay``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .risk import (
    DIST_CRITICAL_M,
    DIST_ELEVATED_M,
    LIST_COLUMNS,
    PC_CRITICAL_LOG10,
    PC_ELEVATED_LOG10,
    SCORED_COLUMNS,
    TIER_CRITICAL,
    TIER_ELEVATED,
    TIER_NEEDS_REVIEW,
    TIER_SAFE,
    describe_alerts,
)

__all__ = [
    "EVENT_KEY",
    "TIER_ORDER",
    "WINDOW_HOURS",
    "DATA_MODE_REPLAY",
    "REPLAY_LABEL",
    "REPLAY_COLUMNS",
    "QUEUE_COLUMNS",
    "SUMMARY_COUNT_KEYS",
    "TIME_BASIS_ORIGINAL",
    "TIME_BASIS_REPLAY",
    "select_latest_messages",
    "select_replay",
    "summarize_upcoming_24h",
    "build_action_queue",
    "compute_risk_changes",
    "build_insights",
    "export_scored_csv",
    "read_scored_csv",
    # Backward-compatible aliases
    "select_training_replay",
    "summarize_24h",
    "prioritize_events",
    "generate_insights",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EVENT_KEY: List[str] = ["source", "event_id"]
# needs_review ranks above elevated: missing data may hide a critical event.
TIER_ORDER: Dict[str, int] = {
    TIER_CRITICAL: 0,
    TIER_NEEDS_REVIEW: 1,
    TIER_ELEVATED: 2,
    TIER_SAFE: 3,
}
UNKNOWN_TIER_RANK: int = TIER_ORDER[TIER_NEEDS_REVIEW]  # unknown tiers stay visible
WINDOW_HOURS: float = 24.0

DATA_MODE_REPLAY: str = "historical_replay"
REPLAY_LABEL: str = (
    "Historical replay, events aligned to TCA (hypothetical common TCA; "
    "not a real-time fleet forecast)"
)
# Columns added by select_replay, in order.
REPLAY_COLUMNS: List[str] = [
    "data_mode",
    "replay_label",
    "replay_horizon_hours",
    "replay_tca_hours",
    "message_staleness_hours",
]
# Columns added by build_action_queue, in order.
QUEUE_COLUMNS: List[str] = ["priority_rank", "reason_text"]

TIME_BASIS_AUTO: str = "auto"
TIME_BASIS_ORIGINAL: str = "original"  # window on each message's tca_hours
TIME_BASIS_REPLAY: str = "replay"      # window on replay_tca_hours

SUMMARY_COUNT_KEYS: List[str] = [
    "total_encounters",        # unique events with 0 <= hours <= 24
    "distance_dangerous",      # miss_distance_m < 1000
    "critical",                # risk_tier == critical
    "elevated",                # risk_tier == elevated
    "needs_review",            # risk_tier == needs_review (or unknown)
    "safe",                    # risk_tier == safe
    "pc_critical",             # estimated log10 Pc >= -4
    "distance_only_critical",  # miss < 1000 m without estimated Pc >= -4
    "pc_elevated",             # elevated tier with -6 <= estimated log10 Pc < -4
    "distance_only_elevated",  # elevated tier triggered by distance alone
]

_HELPERS = ["_pos", "_tca", "_rid", "_src", "_evt"]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _require_dataframe(df: Any) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")


def _require_event_columns(df: pd.DataFrame) -> None:
    missing = [c for c in EVENT_KEY if c not in df.columns]
    if missing:
        raise ValueError(f"scored rows must contain {EVENT_KEY}; missing {missing}")


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def _tca_days(df: pd.DataFrame) -> pd.Series:
    """Numeric time to TCA in days (NaN if missing or invalid)."""
    for col in ("time_to_tca_days", "time_to_tca"):
        if col in df.columns:
            return _numeric(df, col)
    if "tca_hours" in df.columns:
        return _numeric(df, "tca_hours") / 24.0
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _tca_hours(df: pd.DataFrame) -> pd.Series:
    if "tca_hours" in df.columns:
        return _numeric(df, "tca_hours")
    return _tca_days(df) * 24.0


def _tiers(df: pd.DataFrame) -> pd.Series:
    if "risk_tier" not in df.columns:
        return pd.Series(TIER_NEEDS_REVIEW, index=df.index, dtype="string")
    return df["risk_tier"].astype("string").str.strip().str.lower().fillna(TIER_NEEDS_REVIEW)


def _with_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """Copy ``df`` and add position, lead time, row id and event-key helpers."""
    work = df.copy()
    work["_pos"] = np.arange(len(work))
    work["_tca"] = _tca_days(work)
    work["_rid"] = _numeric(work, "row_id")
    work["_src"] = work["source"].astype("string").fillna("<missing>").astype(str)
    missing_id = work["event_id"].isna().to_numpy()
    evt = work["event_id"].astype(object).to_numpy(copy=True)
    evt[missing_id] = [f"<missing event_id, row {p}>" for p in np.flatnonzero(missing_id)]
    work["_evt"] = evt
    return work


def _eligible(work: pd.DataFrame, cutoff_days: Optional[float]) -> pd.Series:
    tca = work["_tca"]
    ok = np.isfinite(tca) & (tca >= 0)
    if cutoff_days is not None:
        ok &= tca >= cutoff_days
    return ok


def _validated_nonnegative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite nonnegative number") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _as_list(value: Any) -> List[Any]:
    """Normalise a list-like / JSON string / scalar / NaN into a Python list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                return list(parsed) if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                return [text]
        return [text]
    if isinstance(value, float) and math.isnan(value):
        return []
    return [value]


def _fmt(value: Any, digits: int = 2, unit: str = "") -> str:
    """Deterministic number formatting with explicit 'n/a' for missing."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}{unit}"


def _fmt_pc(log10_pc: Any) -> str:
    try:
        number = float(log10_pc)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{10.0 ** number:.1e} (log10 {number:.2f})"


def _event_label(src: Any, evt: Any) -> str:
    return f"{src}/{evt}"


# --------------------------------------------------------------------------- #
# 1. Latest message selection
# --------------------------------------------------------------------------- #
def select_latest_messages(
    df: pd.DataFrame,
    replay_cutoff_days: Optional[float] = None,
) -> pd.DataFrame:
    """Select one message per ``(source, event_id)``: the newest eligible one.

    Eligibility:
        * ``time_to_tca_days`` is present, finite and ``>= 0``.
        * If ``replay_cutoff_days`` is given, ``time_to_tca_days >=
          replay_cutoff_days``: messages issued after the replay point are
          never used.

    Within an event the smallest eligible ``time_to_tca_days`` wins. Ties are
    broken by the highest ``row_id``, then by the later position in ``df``.
    Events without an eligible message are omitted.

    Returns:
        A DataFrame with the same columns as ``df``, one row per event, in the
        original row order of the selected messages, with a fresh index.
    """
    _require_dataframe(df)
    cutoff = None
    if replay_cutoff_days is not None:
        cutoff = _validated_nonnegative(replay_cutoff_days, "replay_cutoff_days")
    if df.empty:
        return df.iloc[0:0].reset_index(drop=True)
    _require_event_columns(df)

    work = _with_helpers(df)
    work = work.loc[_eligible(work, cutoff)]
    work["_rid_desc"] = -work["_rid"].fillna(-1.0)
    work = work.sort_values(["_tca", "_rid_desc", "_pos"], ascending=[True, True, False],
                            kind="mergesort")
    latest = work.drop_duplicates(subset=["_src", "_evt"], keep="first")
    latest = latest.sort_values("_pos", kind="mergesort")
    return latest.drop(columns=_HELPERS + ["_rid_desc"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Training replay selection
# --------------------------------------------------------------------------- #
def select_replay(df: pd.DataFrame, horizon_hours: float) -> pd.DataFrame:
    """Select, per event, the newest message at least ``horizon_hours`` before TCA.

    Each event's newest nonnegative message with ``time_to_tca_days >=
    horizon_hours / 24`` is chosen (ROADMAP.md section 7), using the same
    tie-breaking as :func:`select_latest_messages`. This aligns unrelated
    events to a common hypothetical TCA.

    The original ``time_to_tca_days`` and ``tca_hours`` are left unchanged.
    These columns are appended (see :data:`REPLAY_COLUMNS`):

        data_mode                'historical_replay'
        replay_label             :data:`REPLAY_LABEL`
        replay_horizon_hours     horizon_hours
        replay_tca_hours         horizon_hours: the displayed hypothetical time
                                 to TCA. Use this, not tca_hours, for replay
                                 24-hour summaries.
        message_staleness_hours  tca_hours - horizon_hours: how much older the
                                 selected message is than the replay point.

    No timestamps are synthesised and nothing is presented as real-time.
    """
    _require_dataframe(df)
    horizon = _validated_nonnegative(horizon_hours, "horizon_hours")
    selected = select_latest_messages(df, replay_cutoff_days=horizon / 24.0)
    selected["data_mode"] = DATA_MODE_REPLAY
    selected["replay_label"] = REPLAY_LABEL
    selected["replay_horizon_hours"] = horizon
    selected["replay_tca_hours"] = horizon
    selected["message_staleness_hours"] = _tca_hours(selected) - horizon
    return selected


# --------------------------------------------------------------------------- #
# 3. 24-hour summary
# --------------------------------------------------------------------------- #
def summarize_upcoming_24h(
    df: pd.DataFrame,
    *,
    time_basis: str = TIME_BASIS_AUTO,
) -> Dict[str, Any]:
    """Count unique events expected within 24 hours of TCA.

    Args:
        df: Either all scored rows (``time_basis='original'``) or the output
            of :func:`select_replay` (``time_basis='replay'``).
        time_basis:
            * ``'original'``: each event's latest pre-TCA message is used and
              the window is ``0 <= tca_hours <= 24``.
            * ``'replay'``: requires ``replay_tca_hours``; the window is
              ``0 <= replay_tca_hours <= 24``. Counts are historical replay,
              events aligned to TCA.
            * ``'auto'`` (default): ``'replay'`` if ``replay_tca_hours`` is a
              column, else ``'original'``.

    Returns:
        Integer counts for every key in :data:`SUMMARY_COUNT_KEYS`, plus
        ``time_basis`` (``'original'`` or ``'replay'``), ``data_mode``
        (``'latest_message'`` or ``'historical_replay'``) and
        ``window_hours``. Each event is counted at most once. Distance-only
        and Pc-driven counters are kept separate.
    """
    _require_dataframe(df)
    if time_basis not in (TIME_BASIS_AUTO, TIME_BASIS_ORIGINAL, TIME_BASIS_REPLAY):
        raise ValueError("time_basis must be 'auto', 'original' or 'replay'")
    if time_basis == TIME_BASIS_AUTO:
        time_basis = TIME_BASIS_REPLAY if "replay_tca_hours" in df.columns else TIME_BASIS_ORIGINAL
    if time_basis == TIME_BASIS_REPLAY and "replay_tca_hours" not in df.columns:
        raise ValueError("time_basis='replay' needs select_replay() output (replay_tca_hours)")

    result: Dict[str, Any] = {key: 0 for key in SUMMARY_COUNT_KEYS}
    result["time_basis"] = time_basis
    result["data_mode"] = DATA_MODE_REPLAY if time_basis == TIME_BASIS_REPLAY else "latest_message"
    result["window_hours"] = WINDOW_HOURS
    if df.empty:
        return result

    latest = select_latest_messages(df)  # one row per event, never double counted
    hours = _numeric(latest, "replay_tca_hours") if time_basis == TIME_BASIS_REPLAY \
        else _tca_hours(latest)
    window = latest.loc[np.isfinite(hours) & (hours >= 0) & (hours <= WINDOW_HOURS)]

    tier = _tiers(window)
    log_pc = _numeric(window, "predicted_log10_pc")
    miss = _numeric(window, "miss_distance_m")
    pc_critical = log_pc >= PC_CRITICAL_LOG10
    pc_elevated = (log_pc >= PC_ELEVATED_LOG10) & ~pc_critical
    dist_critical = (miss >= 0) & (miss < DIST_CRITICAL_M)
    dist_elevated = (miss >= DIST_CRITICAL_M) & (miss <= DIST_ELEVATED_M)
    is_elevated = tier == TIER_ELEVATED

    result.update({
        "total_encounters": int(len(window)),
        "distance_dangerous": int(dist_critical.sum()),
        "critical": int((tier == TIER_CRITICAL).sum()),
        "elevated": int(is_elevated.sum()),
        "needs_review": int((~tier.isin([TIER_CRITICAL, TIER_ELEVATED, TIER_SAFE])).sum()),
        "safe": int((tier == TIER_SAFE).sum()),
        "pc_critical": int(pc_critical.sum()),
        "distance_only_critical": int((dist_critical & ~pc_critical).sum()),
        "pc_elevated": int((is_elevated & pc_elevated).sum()),
        "distance_only_elevated": int((is_elevated & dist_elevated & ~pc_elevated).sum()),
    })
    return result


# --------------------------------------------------------------------------- #
# 4. Action queue
# --------------------------------------------------------------------------- #
def _reason_text(row: pd.Series, tier: str) -> str:
    details = describe_alerts(row.get("predicted_log10_pc"), row.get("miss_distance_m"))
    flags = [str(f) for f in _as_list(row.get("quality_flags"))]
    parts: List[str] = []
    if tier == TIER_NEEDS_REVIEW or tier not in TIER_ORDER:
        parts.append("Needs manual review: safety cannot be determined from the available inputs.")
    parts.extend(details)
    if tier == TIER_SAFE and not details:
        parts.append("No trigger: estimated Pc below 1e-6 and miss distance above 5 km.")
    if flags:
        parts.append("Data quality: " + ", ".join(flags) + ".")
    return " ".join(parts)


def build_action_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Order events for operator attention.

    Pass one row per event (e.g. :func:`select_latest_messages` or
    :func:`select_replay` output). Sort order (stable, deterministic):

        1. tier: critical, needs_review, elevated, safe (unknown tiers rank
           with needs_review so they stay visible);
        2. shortest nonnegative ``tca_hours`` (negative or missing last);
        3. highest estimated ``predicted_log10_pc`` (missing last);
        4. lowest ``row_id``, then original position.

    Returns:
        ``df`` reordered with a fresh index plus ``priority_rank`` (1-based)
        and ``reason_text``, a readable explanation such as
        ``"Miss distance 0.72 km is below 1 km."``. Rows are never dropped.
    """
    _require_dataframe(df)
    out = df.copy()
    if out.empty:
        out["priority_rank"] = pd.Series(dtype="int64")
        out["reason_text"] = pd.Series(dtype=object)
        return out.reset_index(drop=True)

    tier = _tiers(out)
    hours = _tca_hours(out)
    log_pc = _numeric(out, "predicted_log10_pc")
    keys = pd.DataFrame({
        "_tier": tier.map(TIER_ORDER).astype("Float64").fillna(UNKNOWN_TIER_RANK).astype(int),
        "_hours": hours.where(np.isfinite(hours) & (hours >= 0), np.inf),
        "_neg_pc": (-log_pc).fillna(np.inf),
        "_rid": _numeric(out, "row_id").fillna(np.inf),
        "_pos": np.arange(len(out)),
    }, index=out.index)
    order = keys.sort_values(list(keys.columns), kind="mergesort").index
    out = out.loc[order]
    out["reason_text"] = [_reason_text(row, t) for (_, row), t in zip(out.iterrows(), tier.loc[order])]
    out.insert(len(out.columns) - 1, "priority_rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. Risk changes and insights
# --------------------------------------------------------------------------- #
def compute_risk_changes(
    df: pd.DataFrame,
    replay_cutoff_days: Optional[float] = None,
) -> pd.DataFrame:
    """Change in estimated log10 Pc between consecutive messages of each event.

    Only pre-TCA messages (``time_to_tca_days >= 0``, and ``>= cutoff`` when
    given) are used, in chronological order: decreasing lead time, then
    increasing ``row_id`` and position (the inverse of the latest-message rule).

    Returns:
        One row per message that has a predecessor, with columns ``source``,
        ``event_id``, ``row_id``, ``previous_row_id``, ``time_to_tca_days``,
        ``previous_time_to_tca_days``, ``predicted_log10_pc``,
        ``previous_predicted_log10_pc``, ``delta_log10_pc``, ``trend``
        (``rising``/``falling``/``unchanged``/``unknown``), ``risk_tier``,
        ``previous_risk_tier`` and ``tier_changed``. A rising estimate does
        not by itself mean a tier change.
    """
    _require_dataframe(df)
    columns = [
        "source", "event_id", "row_id", "previous_row_id", "time_to_tca_days",
        "previous_time_to_tca_days", "predicted_log10_pc", "previous_predicted_log10_pc",
        "delta_log10_pc", "trend", "risk_tier", "previous_risk_tier", "tier_changed",
    ]
    cutoff = None
    if replay_cutoff_days is not None:
        cutoff = _validated_nonnegative(replay_cutoff_days, "replay_cutoff_days")
    if df.empty:
        return pd.DataFrame(columns=columns)
    _require_event_columns(df)

    work = _with_helpers(df)
    work = work.loc[_eligible(work, cutoff)].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["_grp"] = work.groupby(["_src", "_evt"], sort=False).ngroup()
    work["_tier"] = _tiers(work)
    work["_lpc"] = _numeric(work, "predicted_log10_pc")
    work = work.sort_values(["_grp", "_tca", "_rid", "_pos"],
                            ascending=[True, False, True, True], kind="mergesort")
    grouped = work.groupby("_grp", sort=False)
    prev_rid = grouped["_rid"].shift(1)
    prev_tca = grouped["_tca"].shift(1)
    prev_lpc = grouped["_lpc"].shift(1)
    prev_tier = grouped["_tier"].shift(1)
    has_prev = grouped.cumcount() > 0

    delta = work["_lpc"] - prev_lpc
    trend = np.select([delta > 0, delta < 0, delta == 0], ["rising", "falling", "unchanged"],
                      default="unknown")
    out = pd.DataFrame({
        "source": work["source"],
        "event_id": work["event_id"],
        "row_id": work["_rid"],
        "previous_row_id": prev_rid,
        "time_to_tca_days": work["_tca"],
        "previous_time_to_tca_days": prev_tca,
        "predicted_log10_pc": work["_lpc"],
        "previous_predicted_log10_pc": prev_lpc,
        "delta_log10_pc": delta,
        "trend": trend,
        "risk_tier": work["_tier"].astype(object),
        "previous_risk_tier": prev_tier.astype(object),
        "tier_changed": work["_tier"].astype(object) != prev_tier.astype(object),
    }, columns=columns)
    return out.loc[has_prev].reset_index(drop=True)


def build_insights(
    df: pd.DataFrame,
    *,
    horizon_hours: Optional[float] = None,
    max_trend_items: int = 3,
) -> List[str]:
    """Deterministic insight sentences for scored rows.

    Covers the highest estimated risk, the closest predicted encounter,
    unique-event counts by object type and tier, and the change in estimated
    log10 Pc between each event's last two messages. With ``horizon_hours``
    the same replay selection as :func:`select_replay` is used and the text is
    labelled historical replay. Probabilities are always called estimated Pc.
    """
    _require_dataframe(df)
    if max_trend_items < 0:
        raise ValueError("max_trend_items must be >= 0")

    cutoff = None
    if horizon_hours is None:
        lines = ["Latest pre-TCA message per event. Probabilities are model-estimated Pc, "
                 "not physically computed collision probabilities."]
    else:
        horizon = _validated_nonnegative(horizon_hours, "horizon_hours")
        cutoff = horizon / 24.0
        lines = [f"{REPLAY_LABEL}, {horizon:g} h before TCA. Probabilities are model-estimated "
                 "Pc, not physically computed collision probabilities."]
    if df.empty:
        lines.append("No scored encounters available.")
        return lines

    _require_event_columns(df)
    latest = select_latest_messages(df, replay_cutoff_days=cutoff)
    all_events = _with_helpers(df)[["_src", "_evt"]].drop_duplicates()
    excluded = len(all_events) - len(latest)
    if latest.empty:
        lines.append("No event has an eligible pre-TCA message; nothing to rank.")
        return lines

    log_pc = _numeric(latest, "predicted_log10_pc")
    if np.isfinite(log_pc).any():
        r = latest.loc[log_pc.idxmax()]
        lines.append(
            f"Highest estimated risk: event {_event_label(r['source'], r['event_id'])} "
            f"({r.get('c_object_type', 'UNKNOWN')}), estimated Pc {_fmt_pc(r['predicted_log10_pc'])}, "
            f"message {_fmt(r.get('tca_hours'), 1, ' h')} before TCA, tier {r.get('risk_tier', 'n/a')}."
        )
    else:
        lines.append("Highest estimated risk: no valid estimated Pc available.")

    miss = _numeric(latest, "miss_distance_m")
    if np.isfinite(miss).any():
        r = latest.loc[miss.idxmin()]
        lines.append(
            f"Closest predicted encounter: event {_event_label(r['source'], r['event_id'])} "
            f"({r.get('c_object_type', 'UNKNOWN')}), miss distance "
            f"{_fmt(float(r['miss_distance_m']) / 1000.0, 3, ' km')}, tier {r.get('risk_tier', 'n/a')}."
        )
    else:
        lines.append("Closest predicted encounter: no valid miss distance available.")

    if "c_object_type" in latest.columns:
        types = latest["c_object_type"].astype("string").fillna("UNKNOWN")
        ordered = sorted(types.value_counts().items(), key=lambda kv: (-kv[1], str(kv[0])))
        lines.append("Events by object type: " + ", ".join(f"{k} {v}" for k, v in ordered) + ".")

    tiers = _tiers(latest)
    ordered_t = sorted(tiers.value_counts().items(),
                       key=lambda kv: (TIER_ORDER.get(kv[0], UNKNOWN_TIER_RANK), kv[0]))
    lines.append("Events by risk tier: " + ", ".join(f"{k} {v}" for k, v in ordered_t) + ".")

    changes = compute_risk_changes(df, replay_cutoff_days=cutoff)
    if not changes.empty:
        last = changes.groupby(["source", "event_id"], sort=False, dropna=False).tail(1)
        counts = last["trend"].value_counts()
        lines.append(
            "Estimated log10 Pc between each event's last two messages: "
            f"rising {int(counts.get('rising', 0))}, falling {int(counts.get('falling', 0))}, "
            f"unchanged {int(counts.get('unchanged', 0))}."
        )
        rises = last.loc[last["trend"] == "rising"].copy()
        rises["_label"] = [_event_label(s, e) for s, e in zip(rises["source"], rises["event_id"])]
        rises = rises.sort_values(["delta_log10_pc", "_label"], ascending=[False, True],
                                  kind="mergesort")
        for _, r in rises.head(max_trend_items).iterrows():
            tier_note = (f"tier {r['previous_risk_tier']} -> {r['risk_tier']}." if r["tier_changed"]
                         else f"tier unchanged ({r['risk_tier']}); a rising estimated Pc is "
                              "not by itself a tier change.")
            lines.append(
                f"Rising estimate: event {r['_label']}, log10 Pc {r['previous_predicted_log10_pc']:.2f} "
                f"-> {r['predicted_log10_pc']:.2f} ({r['delta_log10_pc']:+.2f}); {tier_note}"
            )
    else:
        lines.append("No event has two eligible messages, so no risk trend is available.")

    if excluded > 0:
        lines.append(f"{excluded} event(s) have no eligible pre-TCA message and are excluded "
                     "from this view (post-TCA messages are historical only).")
    return lines


# --------------------------------------------------------------------------- #
# 6. CSV export
# --------------------------------------------------------------------------- #
def export_scored_csv(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write scored rows to CSV with list fields serialised as JSON strings.

    Contract columns come first in contract order, followed by any other
    columns in their existing order. The index is not written.
    """
    _require_dataframe(df)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [c for c in SCORED_COLUMNS if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    out = df[ordered].copy()
    for col in LIST_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(lambda v: json.dumps([str(x) for x in _as_list(v)]))
    out.to_csv(out_path, index=False)
    return out_path


def read_scored_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a CSV written by :func:`export_scored_csv`, parsing list fields back."""
    frame = pd.read_csv(path)
    for col in LIST_COLUMNS:
        if col in frame.columns:
            frame[col] = frame[col].map(_as_list)
    return frame


# --------------------------------------------------------------------------- #
# Backward-compatible aliases
# --------------------------------------------------------------------------- #
select_training_replay = select_replay
summarize_24h = summarize_upcoming_24h
prioritize_events = build_action_queue


def generate_insights(df: pd.DataFrame, **kwargs: Any) -> str:
    """Insight text as one newline-joined string (see :func:`build_insights`)."""
    return "\n".join(build_insights(df, **kwargs))
