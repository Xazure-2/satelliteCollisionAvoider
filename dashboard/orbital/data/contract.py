"""Proposed data contract v2 (event-level final-risk prediction) and validation.

STATUS: PROPOSED — not yet approved by the whole team. Column names live only
here and in ``adapters.COLUMN_ALIASES`` so they can change without touching UI.

Two tables:

* events  — one row per (source, event_id): observed risk at the last eligible
  CDM (time_to_tca >= 2 days) and the model's predicted FINAL risk.
* history — one row per CDM, used for trend charts and R/T/N geometry.

Units (organiser clarification, 31 Oct 2026): time_to_tca in days, distances
and relative positions in metres, relative speed in m/s, Pc = 10 ** risk, and
risk = -30 is the "approximately zero" floor.

This module validates shape and integrity only. It never assigns a risk tier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIERS = ["critical", "elevated", "safe", "needs_review"]
SPLITS = {"train", "validation", "holdout", "test"}
HELD_OUT_SPLITS = {"validation", "holdout"}
LOG10_PC_FLOOR = -30.0
MIN_INPUT_TCA_DAYS = 2.0

EVENT_REQUIRED = [
    "source",
    "event_id",
    "split",
    "cutoff_time_to_tca_days",
    "predicted_final_tier",
    "model_version",
]
EVENT_OPTIONAL = [
    "c_object_type",
    "cutoff_row_id",
    "n_cdms_used",
    # Current observed risk at the last eligible CDM.
    "observed_log10_pc",
    "observed_miss_distance_m",
    "observed_tier",
    # Predicted FINAL risk.
    "predicted_final_log10_pc",
    "predicted_final_pc",
    "predicted_final_at_floor",
    "predicted_final_miss_distance_m",
    "final_miss_distance_basis",
    "predicted_final_log10_pc_lower",
    "predicted_final_log10_pc_upper",
    "p_safe",
    "p_elevated",
    "p_critical",
    "alert_reasons",
    "quality_flags",
    "recommended_action",
    # Held-out evaluation only; must be empty for test rows.
    "actual_final_log10_pc",
    "actual_final_miss_distance_m",
    "actual_final_tier",
]
EVENT_NUMERIC = [
    "event_id",
    "cutoff_time_to_tca_days",
    "cutoff_row_id",
    "n_cdms_used",
    "observed_log10_pc",
    "observed_miss_distance_m",
    "predicted_final_log10_pc",
    "predicted_final_pc",
    "predicted_final_miss_distance_m",
    "predicted_final_log10_pc_lower",
    "predicted_final_log10_pc_upper",
    "p_safe",
    "p_elevated",
    "p_critical",
    "actual_final_log10_pc",
    "actual_final_miss_distance_m",
]
EVENT_LISTS = ["alert_reasons", "quality_flags"]
ACTUAL_COLUMNS = ["actual_final_log10_pc", "actual_final_miss_distance_m", "actual_final_tier"]
MISS_DISTANCE_BASES = {"predicted", "observed_at_cutoff"}

HISTORY_REQUIRED = ["source", "event_id", "row_id", "time_to_tca_days", "miss_distance_m"]
HISTORY_OPTIONAL = [
    "c_object_type",
    "relative_speed_mps",
    "relative_position_r_m",
    "relative_position_t_m",
    "relative_position_n_m",
    "observed_log10_pc",
    "used_for_prediction",
    "quality_flags",
]
HISTORY_NUMERIC = [
    "event_id",
    "row_id",
    "time_to_tca_days",
    "miss_distance_m",
    "relative_speed_mps",
    "relative_position_r_m",
    "relative_position_t_m",
    "relative_position_n_m",
    "observed_log10_pc",
]


class ContractError(ValueError):
    """The data cannot be displayed because it breaks the contract."""


@dataclass
class Validation:
    warnings: list[str] = field(default_factory=list)


def _parse_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"could not parse {text!r} as a JSON list") from exc
    if not isinstance(parsed, list):
        raise ContractError(f"expected a JSON list, got {text!r}")
    return [str(v) for v in parsed]


def _require(frame: pd.DataFrame, required: list[str], table: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ContractError(f"{table}: missing required column(s): {', '.join(missing)}.")


def _numeric(frame: pd.DataFrame, columns: list[str], table: str) -> None:
    for col in columns:
        if col not in frame.columns:
            continue
        converted = pd.to_numeric(frame[col], errors="coerce")
        bad = converted.isna() & frame[col].notna() & (frame[col].astype(str).str.strip() != "")
        if bad.any():
            raise ContractError(f"{table}: column '{col}' has {int(bad.sum())} non-numeric value(s).")
        frame[col] = converted.astype(float)


def _tiers(series: pd.Series, column: str, *, allow_missing: bool) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    values = values.mask(values.isin(["", "nan", "none"]))
    unknown = set(values.dropna().unique()) - set(TIERS)
    if unknown:
        raise ContractError(f"events: column '{column}' has unknown tier(s): {', '.join(sorted(unknown))}.")
    if not allow_missing and values.isna().any():
        raise ContractError(f"events: column '{column}' is empty for {int(values.isna().sum())} event(s).")
    return values.astype(object).where(values.notna(), None)


def validate_events(frame: pd.DataFrame) -> tuple[pd.DataFrame, Validation]:
    report = Validation()
    if frame is None or frame.empty:
        raise ContractError("events: the table has no rows.")
    _require(frame, EVENT_REQUIRED, "events")
    df = frame.copy()
    for col in EVENT_OPTIONAL:
        if col not in df.columns:
            df[col] = np.nan
    _numeric(df, EVENT_NUMERIC, "events")
    if df["event_id"].isna().any():
        raise ContractError("events: 'event_id' is missing for some rows.")
    df["event_id"] = df["event_id"].astype("int64")
    df["source"] = df["source"].astype(str)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    bad_split = set(df["split"].unique()) - SPLITS
    if bad_split:
        raise ContractError(f"events: unknown split value(s): {', '.join(sorted(bad_split))}.")
    if df.duplicated(["source", "event_id"]).any():
        raise ContractError("events: duplicate (source, event_id) pairs; expected one row per encounter.")

    early = df["cutoff_time_to_tca_days"] < MIN_INPUT_TCA_DAYS
    if early.any():
        raise ContractError(
            f"events: {int(early.sum())} prediction(s) use a cutoff CDM with time_to_tca < 2 days, "
            "which the 31 Oct rules forbid as model input."
        )

    df["predicted_final_tier"] = _tiers(df["predicted_final_tier"], "predicted_final_tier", allow_missing=False)
    df["observed_tier"] = _tiers(df["observed_tier"], "observed_tier", allow_missing=True)
    df["actual_final_tier"] = _tiers(df["actual_final_tier"], "actual_final_tier", allow_missing=True)

    leaked = (df["split"] == "test") & (df["actual_final_tier"].notna() | df["actual_final_log10_pc"].notna())
    if leaked.any():
        raise ContractError(
            f"events: {int(leaked.sum())} test row(s) carry actual final labels. The test set has no "
            "final labels; check the export for leakage."
        )

    basis = df["final_miss_distance_basis"].astype("string").str.strip().str.lower()
    unknown_basis = set(basis.dropna().unique()) - MISS_DISTANCE_BASES
    if unknown_basis:
        raise ContractError("events: unknown final_miss_distance_basis: " + ", ".join(sorted(unknown_basis)))
    df["final_miss_distance_basis"] = basis.astype(object).where(basis.notna(), None)

    floor = df["predicted_final_log10_pc"] <= LOG10_PC_FLOOR + 1e-9
    at_floor = df["predicted_final_at_floor"]
    df["predicted_final_at_floor"] = floor | at_floor.astype(str).str.lower().isin(["true", "1"])
    missing_pred = df["predicted_final_log10_pc"].isna() & (df["predicted_final_tier"] != "needs_review")
    if missing_pred.any():
        report.warnings.append(
            f"{int(missing_pred.sum())} event(s) have a predicted tier but no predicted Pc value."
        )
    if df["observed_tier"].isna().all():
        report.warnings.append("No observed tier supplied: 'current observed' risk shows as unavailable.")
    if df[["predicted_final_log10_pc_lower", "predicted_final_log10_pc_upper", "p_critical"]].isna().all().all():
        report.warnings.append("No uncertainty fields supplied: uncertainty shows as 'not provided'.")

    for col in EVENT_LISTS:
        try:
            df[col] = df[col].map(_parse_list)
        except ContractError as exc:
            raise ContractError(f"events: column '{col}': {exc}") from exc
    for col in ["c_object_type", "recommended_action"]:
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    df["c_object_type"] = df["c_object_type"].fillna("UNKNOWN").astype(str)
    df["model_version"] = df["model_version"].astype(str)
    return df.reset_index(drop=True), report


def validate_history(frame: pd.DataFrame) -> tuple[pd.DataFrame, Validation]:
    report = Validation()
    if frame is None or frame.empty:
        raise ContractError("history: the table has no rows.")
    _require(frame, HISTORY_REQUIRED, "history")
    df = frame.copy()
    for col in HISTORY_OPTIONAL:
        if col not in df.columns:
            df[col] = np.nan
    _numeric(df, HISTORY_NUMERIC, "history")
    for col in ["event_id", "row_id"]:
        if df[col].isna().any():
            raise ContractError(f"history: '{col}' is missing for some rows.")
        df[col] = df[col].astype("int64")
    df["source"] = df["source"].astype(str)
    if (df["miss_distance_m"] < 0).any():
        raise ContractError("history: negative miss_distance_m values (metres must be >= 0).")
    if df.duplicated(["source", "row_id"]).any():
        raise ContractError("history: duplicate (source, row_id) pairs.")
    used = df["used_for_prediction"].astype(str).str.lower().isin(["true", "1"])
    if (used & (df["time_to_tca_days"] < MIN_INPUT_TCA_DAYS)).any():
        raise ContractError("history: CDMs under 2 days are marked used_for_prediction, which the rules forbid.")
    df["used_for_prediction"] = used
    try:
        df["quality_flags"] = df["quality_flags"].map(_parse_list)
    except ContractError as exc:
        raise ContractError(f"history: column 'quality_flags': {exc}") from exc
    if df["observed_log10_pc"].isna().all():
        report.warnings.append("History has no observed_log10_pc values: the risk evolution chart will be empty.")
    if df[["relative_position_r_m", "relative_position_t_m", "relative_position_n_m"]].isna().all().all():
        report.warnings.append("History has no relative positions: the R/T/N plot will be empty.")
    return df.reset_index(drop=True), report


def check_links(events: pd.DataFrame, history: pd.DataFrame) -> list[str]:
    keys = set(zip(history["source"], history["event_id"]))
    missing = [k for k in zip(events["source"], events["event_id"]) if k not in keys]
    if missing:
        return [f"{len(missing)} encounter(s) have no CDM history; their trend charts will be empty."]
    return []
