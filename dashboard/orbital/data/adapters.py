"""Data sources for the dashboard: synthetic demos and real model outputs.

* Real outputs are read from ``outputs/event_predictions.csv`` and
  ``outputs/cdm_history.csv`` at the repository root (or ORBITAL_OUTPUTS_DIR).
* If the model team uses different column names, add them to COLUMN_ALIASES;
  that is the only mapping point.
* A missing or invalid real dataset raises DataUnavailable. The dashboard never
  substitutes synthetic values for missing real predictions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import contract
from .synthetic_encounters import SPECS, make_synthetic

REPO_ROOT = Path(__file__).resolve().parents[3]

# Teammate column name -> v2 contract name. Extend when the model output is final.
COLUMN_ALIASES = {
    "events": {
        "final_risk_tier": "predicted_final_tier",
        "predicted_tier": "predicted_final_tier",
        "predicted_risk": "predicted_final_log10_pc",
        "predicted_final_risk": "predicted_final_log10_pc",
        "time_to_tca": "cutoff_time_to_tca_days",
        "observed_risk": "observed_log10_pc",
        "risk": "observed_log10_pc",
        "miss_distance": "observed_miss_distance_m",
        "true_final_tier": "actual_final_tier",
    },
    "history": {
        "time_to_tca": "time_to_tca_days",
        "miss_distance": "miss_distance_m",
        "relative_speed": "relative_speed_mps",
        "relative_position_r": "relative_position_r_m",
        "relative_position_t": "relative_position_t_m",
        "relative_position_n": "relative_position_n_m",
        "risk": "observed_log10_pc",
    },
}

REAL_KEY = "real"
SOURCE_LABELS = {**{k: s.label for k, s in SPECS.items()}, REAL_KEY: "Real model outputs (outputs/)"}


class DataUnavailable(RuntimeError):
    def __init__(self, title: str, detail: str = ""):
        super().__init__(title)
        self.title = title
        self.detail = detail


@dataclass
class Dataset:
    key: str
    label: str
    synthetic: bool
    events: pd.DataFrame
    history: pd.DataFrame
    model_version: str
    loaded_at: str
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def outputs_dir() -> Path:
    return Path(os.environ.get("ORBITAL_OUTPUTS_DIR", REPO_ROOT / "outputs"))


def _apply_aliases(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    renames = {old: new for old, new in COLUMN_ALIASES[table].items()
               if old in frame.columns and new not in frame.columns}
    return frame.rename(columns=renames)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _finish(key, label, synthetic, events, history, files) -> Dataset:
    events, ev_report = contract.validate_events(events)
    history, hist_report = contract.validate_history(history)
    warnings = ev_report.warnings + hist_report.warnings + contract.check_links(events, history)
    versions = sorted(events["model_version"].unique())
    if len(versions) > 1:
        warnings.append("Several model versions in one file: " + ", ".join(versions))
    return Dataset(key=key, label=label, synthetic=synthetic, events=events, history=history,
                   model_version=", ".join(versions), loaded_at=_now(), files=files, warnings=warnings)


def load_real(directory: Path | None = None) -> Dataset:
    folder = Path(directory) if directory else outputs_dir()
    events_path, history_path = folder / "event_predictions.csv", folder / "cdm_history.csv"
    if not events_path.exists():
        hint = ""
        if (folder / "test_predictions.csv").exists():
            hint = (" Found outputs/test_predictions.csv, but that file has one prediction per CDM "
                    "(row level). The updated brief needs one predicted FINAL tier per event.")
        raise DataUnavailable(
            "Real predictions unavailable: outputs/event_predictions.csv not found." + hint,
            "Ask the model team to export event-level predictions in the v2 contract "
            "(see dashboard/README.md), then press Reload.",
        )
    if not history_path.exists():
        raise DataUnavailable(
            "CDM history unavailable: outputs/cdm_history.csv not found.",
            "Trend charts and R/T/N plots need one row per CDM.",
        )
    try:
        events = _apply_aliases(pd.read_csv(events_path), "events")
        history = _apply_aliases(pd.read_csv(history_path), "history")
    except Exception as exc:  # unreadable CSV
        raise DataUnavailable(f"Could not read the model output files: {exc}") from exc
    try:
        return _finish(REAL_KEY, SOURCE_LABELS[REAL_KEY], False, events, history,
                       [str(events_path), str(history_path)])
    except contract.ContractError as exc:
        raise DataUnavailable(f"Model outputs do not match the v2 contract. {exc}") from exc


def load_synthetic(key: str) -> Dataset:
    events, history = make_synthetic(key)
    return _finish(key, SPECS[key].label, True, events, history, [])


def load(key: str) -> Dataset:
    if key == REAL_KEY:
        return load_real()
    if key in SPECS:
        return load_synthetic(key)
    raise DataUnavailable(f"Unknown data source '{key}'.")
