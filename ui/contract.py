"""Scored-row data contract from ROADMAP.md section 5, and validation helpers.

This module only checks that incoming data has the agreed shape. It never
assigns risk tiers or applies thresholds; that is the job of ``src/risk.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Column order exactly as listed in ROADMAP.md section 5.
SCORED_COLUMNS = [
    "source",
    "row_id",
    "event_id",
    "predicted_log10_pc",
    "predicted_pc",
    "risk_tier",
    "miss_distance_m",
    "miss_distance_km",
    "time_to_tca_days",
    "tca_hours",
    "c_object_type",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "alert_reasons",
    "quality_flags",
    "recommended_action",
    "model_version",
]

NUMERIC_COLUMNS = [
    "row_id",
    "event_id",
    "predicted_log10_pc",
    "predicted_pc",
    "miss_distance_m",
    "miss_distance_km",
    "time_to_tca_days",
    "tca_hours",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
]

LIST_COLUMNS = ["alert_reasons", "quality_flags"]

CONTRACT_SOURCES = {"train_replay", "test"}
# Dashboard-only value, never produced by the real pipeline.
SYNTHETIC_SOURCE = "synthetic"

RISK_TIERS = ["critical", "needs_review", "elevated", "safe"]

# Raw telemetry columns allowed into the model (ROADMAP.md section 5).
ALLOWED_INPUT_COLUMNS = [
    "event_id",
    "time_to_tca",
    "miss_distance",
    "relative_speed",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "c_object_type",
    "c_sigma_r",
    "t_sigma_r",
    "max_risk_estimate",
    "F10",
    "AP",
]

# Columns of the planned submission file, which is NOT a dashboard export.
SUBMISSION_COLUMNS = {"row_id", "event_id", "predicted_risk"}


class ContractError(ValueError):
    """Raised when data cannot be displayed because it breaks the contract."""


@dataclass
class ValidationReport:
    warnings: list[str] = field(default_factory=list)


def _parse_list(value) -> list[str]:
    """Accept a real list, a JSON string list, or an empty value."""
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


def validate_scored(frame: pd.DataFrame, *, allow_synthetic: bool = False) -> tuple[pd.DataFrame, ValidationReport]:
    """Check a scored table against the contract and return a clean copy.

    Raises ContractError with a readable message for problems that make the
    data impossible to display. Softer problems are returned as warnings.
    """
    report = ValidationReport()
    if frame is None or frame.empty:
        raise ContractError("The scored table is empty: it contains no rows.")

    columns = set(frame.columns)
    missing = [c for c in SCORED_COLUMNS if c not in columns]
    if missing:
        hint = ""
        if SUBMISSION_COLUMNS <= columns:
            hint = (
                " This looks like the scoring submission file "
                "(row_id, event_id, predicted_risk). The dashboard needs the richer "
                "scored export produced by src/risk.py score_rows()."
            )
        raise ContractError("Missing required column(s): " + ", ".join(missing) + "." + hint)

    df = frame[SCORED_COLUMNS + [c for c in frame.columns if c not in SCORED_COLUMNS]].copy()

    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = converted.isna() & df[col].notna()
        if bad.any():
            raise ContractError(f"Column '{col}' has {int(bad.sum())} non-numeric value(s).")
        df[col] = converted

    for col in ["row_id", "event_id"]:
        if df[col].isna().any():
            raise ContractError(f"Column '{col}' has missing values; it is a required key.")
        df[col] = df[col].astype("int64")

    sources = set(df["source"].astype(str).unique())
    allowed_sources = CONTRACT_SOURCES | ({SYNTHETIC_SOURCE} if allow_synthetic else set())
    unknown_sources = sources - allowed_sources
    if unknown_sources:
        raise ContractError(
            "Unknown 'source' value(s): " + ", ".join(sorted(unknown_sources))
            + ". Expected 'train_replay' or 'test'."
        )
    df["source"] = df["source"].astype(str)

    df["risk_tier"] = df["risk_tier"].astype(str).str.strip().str.lower()
    unknown_tiers = set(df["risk_tier"].unique()) - set(RISK_TIERS)
    if unknown_tiers:
        raise ContractError(
            "Unknown 'risk_tier' value(s): " + ", ".join(sorted(unknown_tiers))
            + ". Expected safe, elevated, critical or needs_review."
        )

    if df.duplicated(["source", "row_id"]).any():
        raise ContractError("Duplicate (source, row_id) pairs found; each source row must appear once.")

    for col in LIST_COLUMNS:
        try:
            df[col] = df[col].map(_parse_list)
        except ContractError as exc:
            raise ContractError(f"Column '{col}': {exc}") from exc

    df["c_object_type"] = df["c_object_type"].fillna("UNKNOWN").astype(str)
    df["recommended_action"] = df["recommended_action"].fillna("").astype(str)
    df["model_version"] = df["model_version"].fillna("unknown").astype(str)

    no_estimate = df["predicted_log10_pc"].isna() | ~np.isfinite(df["predicted_log10_pc"].fillna(0))
    unexpected = no_estimate & (df["risk_tier"] != "needs_review")
    if unexpected.any():
        report.warnings.append(
            f"{int(unexpected.sum())} row(s) have no usable Pc estimate but are not marked "
            "needs_review. Their tier came from the risk pipeline; please check src/risk.py."
        )
    if no_estimate.any():
        report.warnings.append(
            f"Predictions unavailable for {int(no_estimate.sum())} row(s); they have no Pc value "
            "in the table and are left out of Pc charts."
        )

    if df["miss_distance_m"].isna().any():
        report.warnings.append(
            f"{int(df['miss_distance_m'].isna().sum())} row(s) have no miss distance."
        )

    versions = sorted(df["model_version"].unique())
    if len(versions) > 1:
        report.warnings.append("Multiple model versions in one table: " + ", ".join(versions))

    return df.reset_index(drop=True), report


def validate_raw_inputs(frame: pd.DataFrame, path: str) -> None:
    """Check a raw telemetry CSV has the columns the model is allowed to use."""
    missing = [c for c in ALLOWED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ContractError(f"{path} is missing input column(s): " + ", ".join(missing) + ".")


def to_export_csv(frame: pd.DataFrame) -> bytes:
    """Serialize with list fields as JSON strings, as the contract requires."""
    out = frame.copy()
    for col in LIST_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(lambda v: json.dumps(list(v)))
    ordered = [c for c in SCORED_COLUMNS if c in out.columns]
    return out[ordered].to_csv(index=False).encode("utf-8")
