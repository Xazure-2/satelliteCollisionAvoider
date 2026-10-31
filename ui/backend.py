"""Connects the dashboard to teammates' modules without duplicating their logic.

* ``src/model.py`` (Teammate 1) provides ``predict_log10_pc(frame)``.
* ``src/risk.py`` (Teammate 2) provides ``score_rows(frame, predicted_log10_pc)``.
* ``src/events.py`` (Teammate 2) provides event selection, 24-hour summaries,
  the action queue, insight text and replay selection.

ROADMAP.md does not name the ``src/events.py`` functions, so the names the
dashboard looks for are listed in EVENTS_API below. If Teammate 2 picks
different names, change only that mapping.

Until ``src/events.py`` exists, the dashboard uses small *provisional* display
fallbacks (clearly reported in the UI) for latest-message selection, counts,
ordering and insight text. They never assign or change a risk tier.
"""

from __future__ import annotations

import importlib
import os
import traceback
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ui.contract import (
    ALLOWED_INPUT_COLUMNS,
    ContractError,
    validate_raw_inputs,
    validate_scored,
)

EVENTS_API = {
    "latest": "select_latest_messages",      # (scored) -> one row per (source, event_id)
    "summary_24h": "summarize_upcoming_24h",  # (scored) -> dict of counts
    "queue": "build_action_queue",            # (latest) -> sorted DataFrame
    "insights": "build_insights",             # (scored) -> list[str]
    "replay": "select_replay",                # (scored, horizon_hours) -> one row per event
}

SUMMARY_KEYS = ["total_encounters", "distance_dangerous", "critical", "elevated"]

# Display reference values from the frozen rules in ROADMAP.md section 3. They
# draw chart guide lines and label the "below 1 km" card; tiers always come from
# the scored data. If src/risk.py exposes constants with these names, use them.
DISPLAY_THRESHOLDS = {
    "LOG10_PC_CRITICAL": -4.0,
    "LOG10_PC_ELEVATED": -6.0,
    "MISS_DISTANCE_CRITICAL_M": 1000.0,
    "MISS_DISTANCE_ELEVATED_M": 5000.0,
}

TIER_ORDER = {"critical": 0, "needs_review": 1, "elevated": 2, "safe": 3}

DEFAULT_PATHS = {
    "test_raw": os.path.join("data", "test_data.csv"),
    "train_raw": os.path.join("data", "train_data.csv"),
    "scored_export": os.path.join("outputs", "scored_test.csv"),
}


class PipelineUnavailable(RuntimeError):
    """A teammate module or input file the chosen data source needs is missing."""

    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.details = details


@dataclass
class EventsBackend:
    funcs: dict[str, Callable] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    import_error: str = ""

    @property
    def provisional(self) -> bool:
        return bool(self.missing)


def _import_optional(module: str):
    try:
        return importlib.import_module(module), ""
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == module or module.startswith(exc.name + ".")):
            return None, f"{module.replace('.', '/')}.py not found yet"
        return None, traceback.format_exc()
    except Exception:  # broken teammate module: surface the traceback
        return None, traceback.format_exc()


def load_events_backend() -> EventsBackend:
    module, error = _import_optional("src.events")
    backend = EventsBackend(import_error=error)
    for key, name in EVENTS_API.items():
        fn = getattr(module, name, None) if module else None
        if callable(fn):
            backend.funcs[key] = fn
        else:
            backend.missing.append(name)
    return backend


def display_thresholds() -> dict[str, float]:
    values = dict(DISPLAY_THRESHOLDS)
    module, _ = _import_optional("src.risk")
    for name in values:
        if module is not None and isinstance(getattr(module, name, None), (int, float)):
            values[name] = float(getattr(module, name))
    return values


def teammate_status() -> dict[str, str]:
    """Readable status for each teammate entry point, for the sidebar."""
    status = {}
    for module, func in [("src.model", "predict_log10_pc"), ("src.risk", "score_rows")]:
        mod, err = _import_optional(module)
        if mod is None:
            status[f"{module}.{func}"] = "missing" if "not found" in err else "error"
        else:
            status[f"{module}.{func}"] = "ready" if callable(getattr(mod, func, None)) else "missing"
    events = load_events_backend()
    for name in EVENTS_API.values():
        status[f"src.events.{name}"] = "missing" if name in events.missing else "ready"
    return status


# ---------------------------------------------------------------- loading data

def load_scored_csv(path_or_buffer, *, allow_synthetic: bool = False):
    if isinstance(path_or_buffer, str) and not os.path.exists(path_or_buffer):
        raise PipelineUnavailable(
            f"Scored export not found at '{path_or_buffer}'.",
            "Generate it with the risk pipeline, or upload a scored CSV instead.",
        )
    try:
        frame = pd.read_csv(path_or_buffer)
    except Exception as exc:
        raise ContractError(f"Could not read the CSV file: {exc}") from exc
    return validate_scored(frame, allow_synthetic=allow_synthetic)


def run_live_pipeline(source: str, raw_path: str):
    """Read raw telemetry, predict with src.model, score with src.risk."""
    if not os.path.exists(raw_path):
        raise PipelineUnavailable(
            f"Input file not found: '{raw_path}'.",
            "The datasets are shared outside Git. Ask the lead for the CSV and place it in data/.",
        )
    model, model_err = _import_optional("src.model")
    predict = getattr(model, "predict_log10_pc", None) if model else None
    if not callable(predict):
        raise PipelineUnavailable(
            "Predictions unavailable: src/model.py predict_log10_pc() is not ready (Teammate 1).",
            model_err,
        )
    risk, risk_err = _import_optional("src.risk")
    score_rows = getattr(risk, "score_rows", None) if risk else None
    if not callable(score_rows):
        raise PipelineUnavailable(
            "Risk scoring unavailable: src/risk.py score_rows() is not ready (Teammate 2).",
            risk_err,
        )

    header = pd.read_csv(raw_path, nrows=0)
    validate_raw_inputs(header, raw_path)
    # Read only allowlisted columns: `risk` and everything else never reaches the model.
    frame = pd.read_csv(raw_path, usecols=ALLOWED_INPUT_COLUMNS)[ALLOWED_INPUT_COLUMNS]
    frame.insert(0, "row_id", np.arange(len(frame), dtype="int64"))
    frame.insert(0, "source", source)

    try:
        preds = predict(frame.drop(columns=["source", "row_id"]))
    except Exception:
        raise PipelineUnavailable("src/model.py predict_log10_pc() raised an error.", traceback.format_exc())
    preds = np.asarray(preds, dtype="float64").reshape(-1)
    if len(preds) != len(frame):
        raise PipelineUnavailable(
            f"predict_log10_pc() returned {len(preds)} values for {len(frame)} rows.",
            "The contract requires one estimate per input row, in input order.",
        )
    try:
        scored = score_rows(frame, preds)
    except Exception:
        raise PipelineUnavailable("src/risk.py score_rows() raised an error.", traceback.format_exc())
    return validate_scored(pd.DataFrame(scored))


# --------------------------------------------- event logic (teammate or fallback)

def _fallback_latest(scored: pd.DataFrame) -> pd.DataFrame:
    pre_tca = scored[scored["time_to_tca_days"] >= 0]
    ordered = pre_tca.sort_values(["time_to_tca_days", "row_id"], ascending=[True, False])
    return ordered.drop_duplicates(["source", "event_id"], keep="first").reset_index(drop=True)


def _fallback_summary(latest: pd.DataFrame, miss_limit_m: float) -> dict[str, int]:
    window = latest[(latest["tca_hours"] >= 0) & (latest["tca_hours"] <= 24)]
    return {
        "total_encounters": len(window),
        "distance_dangerous": int((window["miss_distance_m"] < miss_limit_m).sum()),
        "critical": int((window["risk_tier"] == "critical").sum()),
        "elevated": int((window["risk_tier"] == "elevated").sum()),
    }


def _fallback_queue(latest: pd.DataFrame) -> pd.DataFrame:
    keyed = latest.assign(
        _tier=latest["risk_tier"].map(TIER_ORDER),
        _pc=latest["predicted_log10_pc"].fillna(-np.inf),
    )
    keyed = keyed.sort_values(["_tier", "tca_hours", "_pc"], ascending=[True, True, False])
    return keyed.drop(columns=["_tier", "_pc"]).reset_index(drop=True)


def _fallback_insights(scored: pd.DataFrame, latest: pd.DataFrame) -> list[str]:
    if latest.empty:
        return ["No pre-TCA encounters match the current filters."]
    notes = []
    with_pc = latest.dropna(subset=["predicted_log10_pc"])
    if not with_pc.empty:
        top = with_pc.loc[with_pc["predicted_log10_pc"].idxmax()]
        notes.append(
            f"Highest estimated risk: event {top.event_id} ({top.c_object_type}), "
            f"estimated Pc {top.predicted_pc:.1e}, {top.tca_hours:.1f} h to TCA."
        )
    closest = latest.loc[latest["miss_distance_m"].idxmin()]
    notes.append(
        f"Closest predicted approach: event {closest.event_id}, "
        f"{closest.miss_distance_km:.2f} km miss distance."
    )
    counts = latest["c_object_type"].value_counts()
    notes.append("Encounters by object type: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".")

    ordered = scored.sort_values(["source", "event_id", "time_to_tca_days", "row_id"],
                                 ascending=[True, True, False, True])
    delta = ordered.groupby(["source", "event_id"])["predicted_log10_pc"].diff()
    ordered = ordered.assign(_delta=delta)
    last = ordered.groupby(["source", "event_id"]).tail(1).dropna(subset=["_delta"])
    keys = set(zip(latest["source"], latest["event_id"]))
    last = last[[k in keys for k in zip(last["source"], last["event_id"])]]
    if not last.empty:
        riser = last.loc[last["_delta"].idxmax()]
        if riser._delta > 0:
            notes.append(
                f"Largest rise in estimated log risk: event {riser.event_id}, "
                f"+{riser._delta:.2f} since its previous message (a rising trend is not a tier change)."
            )
    return notes


def event_views(scored: pd.DataFrame, backend: EventsBackend, thresholds: dict[str, float],
                replay_hours: float | None = None):
    """Return (latest-per-event, action queue, 24 h summary, insights)."""
    if replay_hours is not None:
        if "replay" not in backend.funcs:
            raise PipelineUnavailable(
                "Training replay needs src/events.py select_replay() (Teammate 2).",
                backend.import_error,
            )
        latest = pd.DataFrame(backend.funcs["replay"](scored, replay_hours))
    elif "latest" in backend.funcs:
        latest = pd.DataFrame(backend.funcs["latest"](scored))
    else:
        latest = _fallback_latest(scored)

    queue = pd.DataFrame(backend.funcs["queue"](latest)) if "queue" in backend.funcs else _fallback_queue(latest)

    if "summary_24h" in backend.funcs:
        # In replay mode, pass the replay selection (one row per event) instead.
        summary = dict(backend.funcs["summary_24h"](latest if replay_hours is not None else scored))
    else:
        summary = _fallback_summary(latest, thresholds["MISS_DISTANCE_CRITICAL_M"])
    summary = {k: int(summary.get(k, 0)) for k in SUMMARY_KEYS}

    if "insights" in backend.funcs:
        insights = [str(s) for s in backend.funcs["insights"](scored)]
    else:
        insights = _fallback_insights(scored, latest)
    return latest, queue, summary, insights
