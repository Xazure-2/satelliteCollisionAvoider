"""Deterministic collision-risk rules and the shared scoring contract.

This module turns a model's predicted ``log10`` collision probability and the
conjunction miss distance into an auditable risk tier. It also produces the
row-level scored table (ROADMAP.md section 5) that the dashboard, event
summaries and exports consume.

The probability is a learned **estimated Pc** (a model estimate of the
supplied risk label), not a physically computed collision probability.

Design principles
-----------------
* **Deterministic.** The same inputs always produce the same outputs, with the
  same reason and flag ordering.
* **Fail-safe, never fail-silent.** A missing, non-finite, out-of-range or
  wrongly typed input never yields ``safe``. If one valid input proves the
  event is critical, the result stays ``critical`` and carries a quality
  warning. In every other case where safety cannot be proven, the result is
  ``needs_review``.
* **Label-blind.** Ground-truth columns such as ``risk`` are never read.
* **SI units internally.** Distances and relative positions are in metres and
  ``time_to_tca`` is in days. Kilometres and hours are derived exactly once,
  for display only.

Thresholds (inclusive or exclusive exactly as specified)
--------------------------------------------------------
* critical : ``log10_pc >= -4``  OR  ``miss_distance_m < 1000``
* elevated : ``log10_pc >= -6``  OR  ``miss_distance_m <= 5000``  (when not critical)
* safe     : neither condition holds AND both inputs are valid

Boundary values: ``-4`` is critical, ``-6`` is elevated, and both ``1000 m``
and ``5000 m`` are elevated. No floating-point tolerance is applied; the log
value is compared directly so probability round-off cannot move a boundary.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Final, Iterable, NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "PC_CRITICAL_LOG10",
    "PC_ELEVATED_LOG10",
    "DIST_CRITICAL_M",
    "DIST_ELEVATED_M",
    "LOG10_PC_CRITICAL",
    "LOG10_PC_ELEVATED",
    "MISS_DISTANCE_CRITICAL_M",
    "MISS_DISTANCE_ELEVATED_M",
    "LOG10_PC_CLIP_MIN",
    "LOG10_PC_CLIP_MAX",
    "EXTREME_SIGMA_M",
    "TIER_CRITICAL",
    "TIER_ELEVATED",
    "TIER_SAFE",
    "TIER_NEEDS_REVIEW",
    "RISK_TIERS",
    "VALID_SOURCES",
    "REASON_PC_CRITICAL",
    "REASON_PC_ELEVATED",
    "REASON_DIST_CRITICAL",
    "REASON_DIST_ELEVATED",
    "FLAG_CRITICAL_PARTIAL",
    "FLAG_SAFETY_UNDETERMINED",
    "FLAG_PREDICTION_CLIPPED",
    "FLAG_EVENT_ID_MISSING",
    "FLAG_TIME_TO_TCA_MISSING",
    "FLAG_TIME_TO_TCA_NEGATIVE",
    "FLAG_OBJECT_TYPE_MISSING",
    "ACTION_CRITICAL",
    "ACTION_ELEVATED",
    "ACTION_SAFE",
    "ACTION_NEEDS_REVIEW",
    "ACTION_NEEDS_REVIEW_ELEVATED",
    "PC_FIELD",
    "DIST_FIELD",
    "UNKNOWN_OBJECT_TYPE",
    "DEFAULT_MODEL_VERSION",
    "OUTPUT_COLUMNS",
    "EXTRA_OUTPUT_COLUMNS",
    "SCORED_COLUMNS",
    "CONTRACT_COLUMNS",
    "LIST_COLUMNS",
    "RiskAssessment",
    "classify_risk",
    "describe_alerts",
    "score_rows",
]

# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
PC_CRITICAL_LOG10: Final[float] = -4.0  # log10_pc >= -4    -> critical
PC_ELEVATED_LOG10: Final[float] = -6.0  # log10_pc >= -6    -> elevated
DIST_CRITICAL_M: Final[float] = 1000.0  # distance <  1000 m -> critical
DIST_ELEVATED_M: Final[float] = 5000.0  # distance <= 5000 m -> elevated

# Names the dashboard (ui/backend.py display_thresholds) looks up.
LOG10_PC_CRITICAL: Final[float] = PC_CRITICAL_LOG10
LOG10_PC_ELEVATED: Final[float] = PC_ELEVATED_LOG10
MISS_DISTANCE_CRITICAL_M: Final[float] = DIST_CRITICAL_M
MISS_DISTANCE_ELEVATED_M: Final[float] = DIST_ELEVATED_M

# Documented display clipping (ROADMAP.md section 4, Teammate 1 step 5). The
# lower bound reflects the observed label floor, not a physical limit.
LOG10_PC_CLIP_MIN: Final[float] = -30.0
LOG10_PC_CLIP_MAX: Final[float] = 0.0

# Radial sigmas above this are treated as potential sentinel values (matches
# the model's ``c_sigma_extreme`` / ``t_sigma_extreme`` features).
EXTREME_SIGMA_M: Final[float] = 1e6

# --------------------------------------------------------------------------- #
# Tiers
# --------------------------------------------------------------------------- #
TIER_CRITICAL: Final[str] = "critical"
TIER_ELEVATED: Final[str] = "elevated"
TIER_SAFE: Final[str] = "safe"
TIER_NEEDS_REVIEW: Final[str] = "needs_review"
RISK_TIERS: Final[tuple[str, ...]] = (
    TIER_CRITICAL,
    TIER_ELEVATED,
    TIER_NEEDS_REVIEW,
    TIER_SAFE,
)

VALID_SOURCES: Final[tuple[str, ...]] = ("train_replay", "test")

# --------------------------------------------------------------------------- #
# Alert reasons (stable, machine-readable codes)
# --------------------------------------------------------------------------- #
REASON_PC_CRITICAL: Final[str] = "log10_pc>=-4"
REASON_PC_ELEVATED: Final[str] = "log10_pc>=-6"
REASON_DIST_CRITICAL: Final[str] = "miss_distance<1000m"
REASON_DIST_ELEVATED: Final[str] = "miss_distance<=5000m"

# --------------------------------------------------------------------------- #
# Quality flags
# --------------------------------------------------------------------------- #
PC_FIELD: Final[str] = "predicted_log10_pc"
DIST_FIELD: Final[str] = "miss_distance_m"

# Per-input flags are generated as f"{field}_{suffix}".
_SUFFIX_MISSING: Final[str] = "missing"
_SUFFIX_INVALID_TYPE: Final[str] = "invalid_type"
_SUFFIX_NON_FINITE: Final[str] = "non_finite"
_SUFFIX_OUT_OF_RANGE: Final[str] = "out_of_range"  # log10_pc > 0, i.e. Pc > 1
_SUFFIX_NEGATIVE: Final[str] = "negative"
_SUFFIX_EXTREME: Final[str] = "extreme"

FLAG_CRITICAL_PARTIAL: Final[str] = "critical_with_incomplete_inputs"
FLAG_SAFETY_UNDETERMINED: Final[str] = "safety_undetermined"
FLAG_PREDICTION_CLIPPED: Final[str] = "prediction_clipped"
FLAG_EVENT_ID_MISSING: Final[str] = "event_id_missing"
FLAG_TIME_TO_TCA_MISSING: Final[str] = "time_to_tca_days_missing"
FLAG_TIME_TO_TCA_NEGATIVE: Final[str] = "time_to_tca_days_negative"
FLAG_OBJECT_TYPE_MISSING: Final[str] = "c_object_type_missing"

# --------------------------------------------------------------------------- #
# Recommended actions. A critical flag requests maneuver planning; it never
# prescribes or executes a burn.
# --------------------------------------------------------------------------- #
ACTION_CRITICAL: Final[str] = (
    "Initiate maneuver planning: escalate to flight dynamics and request refined "
    "orbit determination. This requests planning; it does not prescribe or "
    "execute a burn."
)
ACTION_ELEVATED: Final[str] = (
    "Review: request additional tracking or updated CDMs and re-assess before TCA."
)
ACTION_SAFE: Final[str] = (
    "Monitor: no action required beyond standard CDM processing."
)
ACTION_NEEDS_REVIEW: Final[str] = (
    "Manual review: inputs are missing or invalid, so safety cannot be "
    "determined automatically. Do not treat as safe."
)
ACTION_NEEDS_REVIEW_ELEVATED: Final[str] = (
    "Manual review: available inputs already meet elevated thresholds and "
    "missing or invalid inputs prevent ruling out critical. Treat as at least "
    "elevated."
)

# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
UNKNOWN_OBJECT_TYPE: Final[str] = "UNKNOWN"
DEFAULT_MODEL_VERSION: Final[str] = "unknown"

# Required columns, in the order listed in ROADMAP.md section 5.
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
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
)
# Additive diagnostics, appended after the required columns.
EXTRA_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "predicted_log10_pc_raw",
    "alert_details",
)
SCORED_COLUMNS: Final[tuple[str, ...]] = OUTPUT_COLUMNS + EXTRA_OUTPUT_COLUMNS
CONTRACT_COLUMNS: Final[list[str]] = list(OUTPUT_COLUMNS)  # backward-compatible name
# Serialized as JSON strings in CSV.
LIST_COLUMNS: Final[tuple[str, ...]] = ("alert_reasons", "quality_flags", "alert_details")

# Accepted input column names. Exactly one alias of each group may be present.
_MISS_DISTANCE_ALIASES: Final[tuple[str, ...]] = ("miss_distance_m", "miss_distance")
_TIME_TO_TCA_ALIASES: Final[tuple[str, ...]] = ("time_to_tca_days", "time_to_tca")
_ROW_ID_COLUMN: Final[str] = "row_id"
_EVENT_ID_COLUMN: Final[str] = "event_id"
_SOURCE_COLUMN: Final[str] = "source"
_MODEL_VERSION_COLUMN: Final[str] = "model_version"
_OBJECT_TYPE_COLUMN: Final[str] = "c_object_type"
_RELATIVE_POSITION_COLUMNS: Final[tuple[str, ...]] = (
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
)
# Optional model inputs checked for data quality only when present:
# (column, nonnegative, sigma-like).
_OPTIONAL_INPUT_CHECKS: Final[tuple[tuple[str, bool, bool], ...]] = (
    ("relative_speed", True, False),
    ("c_sigma_r", True, True),
    ("t_sigma_r", True, True),
    ("max_risk_estimate", False, False),
    ("F10", True, False),
    ("AP", True, False),
)


class RiskAssessment(NamedTuple):
    """Result of :func:`classify_risk`.

    Attributes:
        risk_tier: One of ``critical``, ``elevated``, ``safe`` or ``needs_review``.
        alert_reasons: Every satisfied trigger, in deterministic order
            (probability first, then distance).
        recommended_action: Operator-facing action text for the tier.
        quality_flags: Data-quality warnings, in deterministic order.
    """

    risk_tier: str
    alert_reasons: tuple[str, ...]
    recommended_action: str
    quality_flags: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Scalar parsing
# --------------------------------------------------------------------------- #
def _parse_number(value: Any, field: str) -> tuple[float | None, str | None]:
    """Strictly parse a scalar into a finite float.

    Args:
        value: Candidate scalar.
        field: Field name used to build the quality flag.

    Returns:
        A tuple ``(number, flag)``. Exactly one element is ``None``.
        ``number`` is a finite float when parsing succeeds. Otherwise ``flag``
        holds a quality flag describing why parsing failed.

    Notes:
        Strings and booleans are rejected on purpose. A numeric-looking string
        signals an upstream parsing bug and must not be trusted silently.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None, f"{field}_{_SUFFIX_MISSING}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return None, f"{field}_{_SUFFIX_INVALID_TYPE}"
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None, f"{field}_{_SUFFIX_NON_FINITE}"
    if math.isnan(number):
        return None, f"{field}_{_SUFFIX_MISSING}"
    if math.isinf(number):
        return None, f"{field}_{_SUFFIX_NON_FINITE}"
    return number, None


def _validated_pc(value: Any) -> tuple[float | None, str | None]:
    """Parse ``log10(Pc)`` and enforce ``log10(Pc) <= 0``, i.e. ``Pc <= 1``."""
    number, flag = _parse_number(value, PC_FIELD)
    if number is not None and number > 0.0:
        return None, f"{PC_FIELD}_{_SUFFIX_OUT_OF_RANGE}"
    return number, flag


def _validated_distance(value: Any) -> tuple[float | None, str | None]:
    """Parse the miss distance in metres and enforce ``distance >= 0``."""
    number, flag = _parse_number(value, DIST_FIELD)
    if number is not None and number < 0.0:
        return None, f"{DIST_FIELD}_{_SUFFIX_NEGATIVE}"
    return number, flag


def _finite_or_nan(value: Any, field: str) -> float:
    """Return a finite float for display, or NaN if ``value`` is not finite numeric."""
    number, _ = _parse_number(value, field)
    return float("nan") if number is None else number


# --------------------------------------------------------------------------- #
# Core rule
# --------------------------------------------------------------------------- #
def classify_risk(predicted_log10_pc: Any, miss_distance_m: Any) -> RiskAssessment:
    """Classify one conjunction into a deterministic risk tier.

    Args:
        predicted_log10_pc: Predicted ``log10`` collision probability. A valid
            value is a finite real number ``<= 0``.
        miss_distance_m: Miss distance in metres. A valid value is a finite
            real number ``>= 0``.

    Returns:
        A :class:`RiskAssessment`.

    Rules:
        * ``critical`` if ``log10_pc >= -4`` or ``miss_distance_m < 1000``,
          evaluated on whichever inputs are valid. If one input is invalid,
          the result is still critical and the flag
          ``critical_with_incomplete_inputs`` is added.
        * Otherwise, if either input is invalid: ``needs_review`` with the flag
          ``safety_undetermined``. Elevated triggers from the valid input are
          kept in ``alert_reasons``.
        * Otherwise ``elevated`` if ``log10_pc >= -6`` or
          ``miss_distance_m <= 5000``.
        * Otherwise ``safe``.

    Alert reasons:
        For each input, the strictest satisfied threshold is reported. A
        critical probability implies the elevated probability rule, so it is
        not duplicated. Triggers from different inputs are always all kept.
        For example, a critical probability combined with an elevated distance
        yields both reasons.
    """
    quality_flags: list[str] = []

    pc, pc_flag = _validated_pc(predicted_log10_pc)
    if pc_flag is not None:
        quality_flags.append(pc_flag)

    distance, dist_flag = _validated_distance(miss_distance_m)
    if dist_flag is not None:
        quality_flags.append(dist_flag)

    alert_reasons: list[str] = []
    is_critical = False

    if pc is not None:
        if pc >= PC_CRITICAL_LOG10:
            alert_reasons.append(REASON_PC_CRITICAL)
            is_critical = True
        elif pc >= PC_ELEVATED_LOG10:
            alert_reasons.append(REASON_PC_ELEVATED)

    if distance is not None:
        if distance < DIST_CRITICAL_M:
            alert_reasons.append(REASON_DIST_CRITICAL)
            is_critical = True
        elif distance <= DIST_ELEVATED_M:
            alert_reasons.append(REASON_DIST_ELEVATED)

    inputs_complete = pc is not None and distance is not None

    if is_critical:
        tier, action = TIER_CRITICAL, ACTION_CRITICAL
        if not inputs_complete:
            quality_flags.append(FLAG_CRITICAL_PARTIAL)
    elif not inputs_complete:
        tier = TIER_NEEDS_REVIEW
        action = ACTION_NEEDS_REVIEW_ELEVATED if alert_reasons else ACTION_NEEDS_REVIEW
        quality_flags.append(FLAG_SAFETY_UNDETERMINED)
    elif alert_reasons:
        tier, action = TIER_ELEVATED, ACTION_ELEVATED
    else:
        tier, action = TIER_SAFE, ACTION_SAFE

    return RiskAssessment(
        risk_tier=tier,
        alert_reasons=tuple(alert_reasons),
        recommended_action=action,
        quality_flags=tuple(quality_flags),
    )


# --------------------------------------------------------------------------- #
# Readable explanations
# --------------------------------------------------------------------------- #
def _decimals_against(value: float, thresholds: tuple[float, ...], scale: float = 1.0) -> int:
    """Smallest number of decimals (2 to 9) at which ``value / scale`` never
    looks equal to any threshold unless the value really is equal (e.g.
    999.6 m prints ``0.9996`` km, not ``1.00`` km).
    """
    scaled = value / scale
    for decimals in range(2, 10):
        rounded = float(f"{scaled:.{decimals}f}")
        if all((rounded == t / scale) == (value == t) for t in thresholds):
            return decimals
    return 9


def _fmt_against(value: float, thresholds: tuple[float, ...], scale: float = 1.0) -> str:
    return f"{value / scale:.{_decimals_against(value, thresholds, scale)}f}"


def _fmt_estimated_pc(log10_pc: float) -> str:
    """``'3.2e-04 (log10 -3.50)'``, with extra digits near a threshold."""
    decimals = _decimals_against(log10_pc, (PC_CRITICAL_LOG10, PC_ELEVATED_LOG10))
    return f"{10.0 ** log10_pc:.{decimals - 1}e} (log10 {log10_pc:.{decimals}f})"


def describe_alerts(predicted_log10_pc: Any, miss_distance_m: Any) -> list[str]:
    """Return one readable sentence per alert reason from :func:`classify_risk`.

    The sentences are in the same order as ``alert_reasons``, for example
    ``"Miss distance 0.72 km is below 1 km."``. Probabilities are always
    labelled as *estimated* Pc. Invalid inputs produce no sentence; they are
    reported through quality flags instead.
    """
    reasons = classify_risk(predicted_log10_pc, miss_distance_m).alert_reasons
    pc, _ = _validated_pc(predicted_log10_pc)
    distance, _ = _validated_distance(miss_distance_m)
    distance_thresholds = (DIST_CRITICAL_M, DIST_ELEVATED_M)
    details: list[str] = []
    for reason in reasons:
        if reason in (REASON_PC_CRITICAL, REASON_PC_ELEVATED):
            assert pc is not None
            label = "critical threshold 1e-4" if reason == REASON_PC_CRITICAL else "elevated threshold 1e-6"
            details.append(f"Estimated Pc {_fmt_estimated_pc(pc)} is at or above the {label}.")
        elif reason == REASON_DIST_CRITICAL:
            assert distance is not None
            details.append(
                f"Miss distance {_fmt_against(distance, distance_thresholds, 1000.0)} km is below 1 km."
            )
        elif reason == REASON_DIST_ELEVATED:
            assert distance is not None
            details.append(
                f"Miss distance {_fmt_against(distance, distance_thresholds, 1000.0)} km "
                "is at or below 5 km."
            )
    return details


# --------------------------------------------------------------------------- #
# Frame helpers
# --------------------------------------------------------------------------- #
def _resolve_column(
    frame: pd.DataFrame, aliases: tuple[str, ...], *, required: bool
) -> str | None:
    """Return the single alias present in ``frame``.

    Raises:
        ValueError: If more than one alias is present (ambiguous input), or if
            none is present and ``required`` is true.
    """
    present = [alias for alias in aliases if alias in frame.columns]
    if len(present) > 1:
        raise ValueError(
            f"Ambiguous input: columns {present} are aliases of the same field; "
            "keep exactly one."
        )
    if not present:
        if required:
            raise ValueError(f"Missing required column; expected one of {list(aliases)}")
        return None
    return present[0]


def _as_prediction_array(predictions: Any, expected_length: int) -> np.ndarray:
    """Convert predictions to a 1-D object array, aligned by position.

    pandas objects are read positionally (``to_numpy``), never by index label.
    This prevents silent misalignment after shuffling or filtering. Arrays of
    shape ``(n, 1)`` are accepted and flattened.

    Raises:
        TypeError: If ``predictions`` is not array-like.
        ValueError: If the shape or length is incompatible.
    """
    if predictions is None or isinstance(predictions, (str, bytes)):
        raise TypeError("predicted_log10_pc must be array-like, not a scalar or string")

    if isinstance(predictions, pd.DataFrame):
        if predictions.shape[1] != 1:
            raise ValueError("predicted_log10_pc DataFrame must have exactly one column")
        values = predictions.iloc[:, 0].to_numpy(dtype=object)
    elif isinstance(predictions, (pd.Series, pd.Index)):
        values = predictions.to_numpy(dtype=object)
    elif isinstance(predictions, np.ndarray):
        array = predictions
        if array.ndim == 2 and array.shape[1] == 1:
            array = array.reshape(-1)
        if array.ndim != 1:
            raise ValueError(
                f"predicted_log10_pc must be 1-D or (n, 1); got shape {predictions.shape}"
            )
        values = array.astype(object)
    elif isinstance(predictions, Iterable):
        items = list(predictions)
        values = np.empty(len(items), dtype=object)
        for index, item in enumerate(items):
            values[index] = item  # nested items stay as scalars -> invalid_type
    else:
        raise TypeError(
            f"predicted_log10_pc must be array-like, got {type(predictions).__name__}"
        )

    if len(values) != expected_length:
        raise ValueError(
            f"predicted_log10_pc has length {len(values)} but frame has "
            f"{expected_length} rows"
        )
    return values


def _resolve_row_ids(frame: pd.DataFrame) -> np.ndarray:
    """Return the original CSV row positions.

    If the loader kept a ``row_id`` column (needed when the frame has been
    filtered or reordered), it is validated and used. Otherwise the 0-based
    position within ``frame`` is used. The DataFrame index and ``event_id``
    are never used.

    Raises:
        ValueError: If a ``row_id`` column exists but is not unique,
            non-negative and integer-valued.
    """
    n_rows = len(frame)
    if _ROW_ID_COLUMN not in frame.columns:
        return np.arange(n_rows, dtype=np.int64)

    ids = pd.to_numeric(frame[_ROW_ID_COLUMN], errors="coerce")
    if ids.isna().any():
        raise ValueError("row_id column contains missing or non-numeric values")
    values = ids.to_numpy(dtype=float)
    if n_rows and (
        not np.all(np.isfinite(values))
        or not np.all(values == np.floor(values))
        or np.any(values < 0)
    ):
        raise ValueError("row_id values must be non-negative integers")
    if pd.Index(values).has_duplicates:
        raise ValueError("row_id values must be unique")
    return values.astype(np.int64)


def _resolve_sources(frame: pd.DataFrame, source: str | None) -> np.ndarray:
    """Return one validated source label per row.

    A ``source`` column in ``frame`` (as the dashboard supplies) is used when
    present; an explicit ``source`` argument must then agree with it. Every
    value must be one of :data:`VALID_SOURCES`.
    """
    if source is not None:
        if not isinstance(source, str) or source.strip() not in VALID_SOURCES:
            raise ValueError(f"source must be one of {list(VALID_SOURCES)}, got {source!r}")
        source = source.strip()

    if _SOURCE_COLUMN in frame.columns:
        column = frame[_SOURCE_COLUMN]
        if column.isna().any():
            raise ValueError("source column contains missing values")
        values = column.astype(str).str.strip()
        unknown = sorted(set(values) - set(VALID_SOURCES))
        if unknown:
            raise ValueError(
                f"Unknown source value(s) {unknown}; expected one of {list(VALID_SOURCES)}"
            )
        if source is not None and (values != source).any():
            raise ValueError(f"source argument {source!r} disagrees with the source column")
        return values.to_numpy(dtype=object)

    if source is None:
        if len(frame) == 0:
            return np.empty(0, dtype=object)
        raise ValueError(
            "source is required: pass source='train_replay' or 'test', or include a source column"
        )
    return np.full(len(frame), source, dtype=object)


def _resolve_model_versions(frame: pd.DataFrame, model_version: str | None) -> np.ndarray:
    """Return one model version per row (argument, then column, then ``unknown``)."""
    if model_version is not None:
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        return np.full(len(frame), model_version.strip(), dtype=object)
    if _MODEL_VERSION_COLUMN in frame.columns:
        values = frame[_MODEL_VERSION_COLUMN].astype("string").str.strip()
        values = values.mask(values.eq(""), pd.NA).fillna(DEFAULT_MODEL_VERSION)
        return values.astype(str).to_numpy(dtype=object)
    return np.full(len(frame), DEFAULT_MODEL_VERSION, dtype=object)


def _numeric_column(frame: pd.DataFrame, column: str | None) -> np.ndarray:
    """Return a float array for ``column``. Non-numeric values become NaN.

    If ``column`` is None, an all-NaN array is returned.
    """
    if column is None:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _dedupe(items: Iterable[str]) -> list[str]:
    """Remove duplicates while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _input_quality_flags(
    frame: pd.DataFrame, time_days: np.ndarray, object_missing: np.ndarray
) -> list[list[str]]:
    """Row-level data-quality flags for keys and optional telemetry inputs.

    These flags never change the tier; the tier depends only on estimated Pc
    and miss distance.
    """
    n_rows = len(frame)
    flags: list[list[str]] = [[] for _ in range(n_rows)]

    def add(mask: np.ndarray, flag: str) -> None:
        for index in np.flatnonzero(mask):
            flags[index].append(flag)

    add(frame[_EVENT_ID_COLUMN].isna().to_numpy(), FLAG_EVENT_ID_MISSING)
    add(~np.isfinite(time_days), FLAG_TIME_TO_TCA_MISSING)
    with np.errstate(invalid="ignore"):
        add(time_days < 0, FLAG_TIME_TO_TCA_NEGATIVE)
    add(object_missing, FLAG_OBJECT_TYPE_MISSING)

    for column in _RELATIVE_POSITION_COLUMNS:
        if column in frame.columns:
            add(~np.isfinite(_numeric_column(frame, column)), f"{column}_{_SUFFIX_MISSING}")

    for column, nonnegative, sigma_like in _OPTIONAL_INPUT_CHECKS:
        if column not in frame.columns:
            continue
        values = _numeric_column(frame, column)
        add(~np.isfinite(values), f"{column}_{_SUFFIX_MISSING}")
        with np.errstate(invalid="ignore"):
            if nonnegative:
                add(values < 0, f"{column}_{_SUFFIX_NEGATIVE}")
            if sigma_like:
                add(values > EXTREME_SIGMA_M, f"{column}_{_SUFFIX_EXTREME}")
    return flags


def _empty_scored_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in SCORED_COLUMNS})


# --------------------------------------------------------------------------- #
# Public scoring entry point
# --------------------------------------------------------------------------- #
def score_rows(
    frame: pd.DataFrame,
    predicted_log10_pc: Any,
    *,
    source: str | None = None,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Score every row of ``frame`` and return the shared output contract.

    Args:
        frame: Input telemetry rows. Required columns: ``event_id``, one of
            ``miss_distance`` / ``miss_distance_m`` (metres) and one of
            ``time_to_tca`` / ``time_to_tca_days`` (days). Optional:
            ``source``, ``row_id``, ``c_object_type``, relative positions
            (metres), ``model_version`` and the optional model inputs, which
            are checked for data quality only. ``risk`` and other label
            columns are never read.
        predicted_log10_pc: 1-D array-like with one raw model estimate per
            row, aligned by position (never by index label).
        source: ``"train_replay"`` or ``"test"``. Optional when ``frame`` has
            a ``source`` column; if both are given they must agree.
        model_version: Artifact identifier. Defaults to a ``model_version``
            column, else ``"unknown"``.

    Returns:
        DataFrame with :data:`OUTPUT_COLUMNS` followed by
        :data:`EXTRA_OUTPUT_COLUMNS`, one row per input row, in input order
        and with the input index. List columns hold Python lists.

        Finite predictions are clipped to ``[-30, 0]`` (flag
        ``prediction_clipped``) before classification, so the stored
        ``predicted_log10_pc`` is exactly the value that set the tier. The
        unclipped value is kept in ``predicted_log10_pc_raw``. A missing or
        non-finite prediction gives ``needs_review`` unless the miss distance
        alone proves critical.

    Raises:
        TypeError: If ``frame`` is not a DataFrame or predictions are not array-like.
        ValueError: On missing required columns, ambiguous aliases, invalid
            source, invalid ``row_id`` values or a prediction length mismatch.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    n_rows = len(frame)
    predictions = _as_prediction_array(predicted_log10_pc, n_rows)
    sources = _resolve_sources(frame, source)
    versions = _resolve_model_versions(frame, model_version)
    if n_rows == 0:
        return _empty_scored_frame()

    if _EVENT_ID_COLUMN not in frame.columns:
        raise ValueError("frame must contain an event_id column")
    distance_column = _resolve_column(frame, _MISS_DISTANCE_ALIASES, required=True)
    time_column = _resolve_column(frame, _TIME_TO_TCA_ALIASES, required=True)

    row_ids = _resolve_row_ids(frame)
    distances = frame[distance_column].to_numpy(dtype=object)
    time_days = _numeric_column(frame, time_column)

    if _OBJECT_TYPE_COLUMN in frame.columns:
        object_types = frame[_OBJECT_TYPE_COLUMN].astype("string").str.strip()
        object_missing = (object_types.isna() | object_types.eq("")).to_numpy(dtype=bool)
        object_types = object_types.mask(object_missing, UNKNOWN_OBJECT_TYPE)
        object_values = object_types.astype(str).to_numpy(dtype=object)
    else:
        object_missing = np.ones(n_rows, dtype=bool)
        object_values = np.full(n_rows, UNKNOWN_OBJECT_TYPE, dtype=object)

    data_flags = _input_quality_flags(frame, time_days, object_missing)
    positions = {
        column: _numeric_column(frame, column if column in frame.columns else None)
        for column in _RELATIVE_POSITION_COLUMNS
    }

    raw_values = np.full(n_rows, np.nan)
    used_values = np.full(n_rows, np.nan)
    tiers: list[str] = []
    reasons: list[list[str]] = []
    details: list[list[str]] = []
    actions: list[str] = []
    flags: list[list[str]] = []
    distances_m = np.full(n_rows, np.nan)

    for i in range(n_rows):
        raw, _ = _parse_number(predictions[i], PC_FIELD)
        row_flags: list[str] = []
        if raw is None:
            pc_input: Any = predictions[i]  # classify_risk reports why it is unusable
        else:
            raw_values[i] = raw
            pc_input = min(max(raw, LOG10_PC_CLIP_MIN), LOG10_PC_CLIP_MAX)
            used_values[i] = pc_input
            if pc_input != raw:
                row_flags.append(FLAG_PREDICTION_CLIPPED)

        distance = distances[i]
        distances_m[i] = _finite_or_nan(distance, DIST_FIELD)
        assessment = classify_risk(pc_input, distance)

        tiers.append(assessment.risk_tier)
        reasons.append(list(assessment.alert_reasons))
        details.append(describe_alerts(pc_input, distance))
        actions.append(assessment.recommended_action)
        flags.append(_dedupe([*assessment.quality_flags, *row_flags, *data_flags[i]]))

    with np.errstate(over="ignore", invalid="ignore"):
        predicted_pc = np.where(np.isfinite(used_values), 10.0 ** used_values, np.nan)
        time_hours = time_days * 24.0
        distances_km = distances_m / 1000.0

    out = pd.DataFrame(
        {
            "source": sources,
            "row_id": row_ids,
            "event_id": frame[_EVENT_ID_COLUMN].to_numpy(),
            "predicted_log10_pc": used_values,
            "predicted_pc": predicted_pc,
            "risk_tier": tiers,
            "miss_distance_m": distances_m,
            "miss_distance_km": distances_km,
            "time_to_tca_days": time_days,
            "tca_hours": time_hours,
            "c_object_type": object_values,
            "relative_position_r": positions["relative_position_r"],
            "relative_position_t": positions["relative_position_t"],
            "relative_position_n": positions["relative_position_n"],
            "alert_reasons": reasons,
            "quality_flags": flags,
            "recommended_action": actions,
            "model_version": versions,
            "predicted_log10_pc_raw": raw_values,
            "alert_details": details,
        },
        columns=list(SCORED_COLUMNS),
    )
    out.index = frame.index
    return out
