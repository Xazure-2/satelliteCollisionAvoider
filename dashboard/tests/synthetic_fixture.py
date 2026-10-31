"""SYNTHETIC encounter data in the proposed v2 contract, for demos and tests.

Everything here is invented. Rows carry source="synthetic" and a
model_version starting with "SYNTHETIC-", so they can never pass for real
predictions.

Tiers are labelled by ``_label_tier`` below: a stand-in for Teammate 2's risk
module, used ONLY to label generated data so that every synthetic value agrees
with the organiser's thresholds. The dashboard UI never calls it; it displays
whatever tier the data provides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contract import LOG10_PC_FLOOR

OBJECT_TYPES = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN"]
OBJECT_WEIGHTS = [0.55, 0.2, 0.15, 0.1]
ACTIONS = {
    "critical": "Human review: start manoeuvre planning assessment",
    "elevated": "Analyst review",
    "safe": "Monitor",
    "needs_review": "Data-quality review",
}


@dataclass(frozen=True)
class SyntheticSpec:
    key: str
    label: str
    seed: int
    n_test: int
    n_holdout: int
    critical_share: float      # share of events whose predicted final tier is critical
    elevated_share: float
    escalation_share: float    # share of non-safe predictions that escalate from a safer observed tier
    with_uncertainty: bool
    miss_basis: str            # "predicted" or "observed_at_cutoff"
    n_missing_predictions: int
    model_version: str


SPECS = {
    "synthetic_a": SyntheticSpec(
        key="synthetic_a", label="Synthetic demo A · baseline", seed=11, n_test=220, n_holdout=180,
        critical_share=0.07, elevated_share=0.22, escalation_share=0.55, with_uncertainty=True,
        miss_basis="predicted", n_missing_predictions=3, model_version="SYNTHETIC-A-v2",
    ),
    "synthetic_b": SyntheticSpec(
        key="synthetic_b", label="Synthetic demo B · high alert, sparse fields", seed=29, n_test=90,
        n_holdout=70, critical_share=0.2, elevated_share=0.3, escalation_share=0.7, with_uncertainty=False,
        miss_basis="observed_at_cutoff", n_missing_predictions=6, model_version="SYNTHETIC-B-v2",
    ),
}


def _label_tier(log_pc: float, miss_m: float) -> str:
    """Synthetic-data labeller following the 31 Oct thresholds (see module docstring)."""
    if not np.isfinite(log_pc) or not np.isfinite(miss_m) or miss_m < 0:
        return "needs_review"
    if log_pc >= -4 or miss_m < 1000:
        return "critical"
    if log_pc >= -6 or miss_m <= 5000:
        return "elevated"
    return "safe"


def _sample(tier: str, rng: np.random.Generator) -> tuple[float, float]:
    """Return (log10 Pc, miss distance m) that land in ``tier``."""
    if tier == "critical":
        if rng.random() < 0.6:
            return rng.uniform(-4.0, -2.3), rng.uniform(1500, 12000)
        return rng.uniform(-9, -4.5), rng.uniform(150, 950)
    if tier == "elevated":
        if rng.random() < 0.6:
            return rng.uniform(-5.9, -4.2), rng.uniform(5500, 25000)
        return rng.uniform(-12, -6.5), rng.uniform(1100, 4900)
    if rng.random() < 0.25:
        return LOG10_PC_FLOOR, rng.uniform(8000, 60000)
    return rng.uniform(-14, -6.6), rng.uniform(6000, 60000)


def _safer(tier: str, rng: np.random.Generator) -> str:
    if tier == "critical":
        return "safe" if rng.random() < 0.45 else "elevated"
    if tier == "elevated":
        return "safe"
    return "safe"


def _actual_from_prediction(pred: str, rng: np.random.Generator) -> str:
    """Make held-out truth mostly agree with predictions, with realistic misses."""
    r = rng.random()
    if pred == "critical":
        return "critical" if r < 0.72 else ("elevated" if r < 0.92 else "safe")
    if pred == "elevated":
        return "elevated" if r < 0.7 else ("critical" if r < 0.8 else "safe")
    if pred == "safe":
        return "safe" if r < 0.94 else ("critical" if r < 0.965 else "elevated")
    return ["safe", "elevated", "critical"][int(rng.integers(0, 3))]


def _path(start: float, end: float, n: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    base = np.linspace(start, end, n)
    noise = rng.normal(0, scale, n)
    noise[-1] = 0.0
    return base + noise


def make_synthetic(key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = SPECS[key]
    rng = np.random.default_rng(spec.seed)
    events, history = [], []
    row_id = 0
    total = spec.n_test + spec.n_holdout
    missing_ids = set(rng.choice(spec.n_test, size=spec.n_missing_predictions, replace=False).tolist())

    for event_id in range(total):
        split = "test" if event_id < spec.n_test else "holdout"
        obj = str(rng.choice(OBJECT_TYPES, p=OBJECT_WEIGHTS))
        r = rng.random()
        target = "critical" if r < spec.critical_share else (
            "elevated" if r < spec.critical_share + spec.elevated_share else "safe")
        escalates = target != "safe" and rng.random() < spec.escalation_share
        obs_target = _safer(target, rng) if escalates else target
        if target == "safe" and rng.random() < 0.12:
            obs_target = "elevated"  # de-escalation case

        pred_log, pred_miss = _sample(target, rng)
        obs_log, obs_miss = _sample(obs_target, rng)
        if spec.miss_basis == "observed_at_cutoff":
            pred_miss = obs_miss
        observed_tier = _label_tier(obs_log, obs_miss)
        predicted_tier = _label_tier(pred_log, pred_miss)

        cutoff = float(np.clip(rng.choice([rng.uniform(2.0, 3.0), rng.uniform(3.0, 5.0), rng.uniform(5.0, 6.9)],
                                          p=[0.35, 0.4, 0.25]), 2.0, 6.95))
        n_pre = int(rng.integers(3, 9))
        gaps = rng.uniform(0.2, 0.6, n_pre - 1)
        pre_times = cutoff + np.concatenate([[0.0], np.cumsum(gaps)])[::-1]
        pre_times = np.minimum(pre_times, 7.0)

        speed = float(rng.uniform(800, 15000))
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        start_log = obs_log if obs_log <= LOG10_PC_FLOOR else obs_log - rng.uniform(-0.6, 1.2)
        pre_logs = np.clip(_path(start_log, obs_log, n_pre, rng, 0.25), LOG10_PC_FLOOR, -0.5)
        if obs_log <= LOG10_PC_FLOOR:
            pre_logs[:] = LOG10_PC_FLOOR
        pre_miss = np.clip(_path(obs_miss * rng.uniform(0.85, 1.25), obs_miss, n_pre, rng, obs_miss * 0.03), 50, None)

        times, logs, misses, used = list(pre_times), list(pre_logs), list(pre_miss), [True] * n_pre
        actual_tier = actual_log = actual_miss = None
        if split == "holdout":
            actual_tier = _actual_from_prediction(predicted_tier, rng)
            actual_log, actual_miss = _sample(actual_tier, rng)
            actual_tier = _label_tier(actual_log, actual_miss)
            n_post = int(rng.integers(2, 6))
            final_t = float(rng.uniform(0.05, 0.9))
            post_times = np.linspace(cutoff, final_t, n_post + 1)[1:]
            post_logs = np.clip(_path(obs_log, actual_log, n_post + 1, rng, 0.2)[1:], LOG10_PC_FLOOR, -0.5)
            post_miss = np.clip(_path(obs_miss, actual_miss, n_post + 1, rng, 40)[1:], 50, None)
            post_logs[-1], post_miss[-1] = actual_log, actual_miss
            times += list(post_times)
            logs += list(post_logs)
            misses += list(post_miss)
            used += [False] * n_post

        cutoff_row = row_id + n_pre - 1
        for t, lg, m, u in zip(times, logs, misses, used):
            jitter = direction + rng.normal(0, 0.03, 3)
            rr, tt, nn = jitter / np.linalg.norm(jitter) * m
            flags = []
            if rng.random() < 0.03:
                flags.append("c_sigma_r_extreme")
            history.append({
                "source": "synthetic", "event_id": event_id, "row_id": row_id,
                "time_to_tca_days": float(t), "miss_distance_m": float(m),
                "relative_speed_mps": speed + float(rng.normal(0, 2)),
                "relative_position_r_m": float(rr), "relative_position_t_m": float(tt),
                "relative_position_n_m": float(nn), "observed_log10_pc": float(lg),
                "used_for_prediction": bool(u), "c_object_type": obj, "quality_flags": flags,
            })
            row_id += 1

        pred_value = pred_log
        reasons = []
        if predicted_tier in ("critical", "elevated"):
            if pred_log >= -4:
                reasons.append(f"Predicted final Pc {10 ** pred_log:.1e} ≥ 1e-4")
            elif pred_log >= -6:
                reasons.append(f"Predicted final Pc {10 ** pred_log:.1e} ≥ 1e-6")
            if pred_miss < 1000:
                reasons.append(f"Final miss distance {pred_miss:,.0f} m < 1,000 m")
            elif pred_miss <= 5000:
                reasons.append(f"Final miss distance {pred_miss:,.0f} m within 1,000–5,000 m")
        flags = []
        if event_id in missing_ids:
            predicted_tier, pred_value, reasons = "needs_review", np.nan, ["Model output unavailable"]
            flags.append("missing_model_output")
        if pred_value is not None and np.isfinite(pred_value) and pred_value <= LOG10_PC_FLOOR:
            flags.append("prediction_at_floor")

        width = rng.uniform(0.4, 1.6)
        p_crit = {"critical": rng.uniform(0.55, 0.95), "elevated": rng.uniform(0.1, 0.4),
                  "safe": rng.uniform(0.0, 0.08)}.get(predicted_tier, np.nan)
        p_elev = np.nan if not np.isfinite(p_crit) else rng.uniform(0.05, 1 - p_crit)
        uncertain = spec.with_uncertainty and np.isfinite(pred_value) and pred_value > LOG10_PC_FLOOR
        events.append({
            "source": "synthetic", "event_id": event_id, "split": split, "c_object_type": obj,
            "cutoff_row_id": cutoff_row, "cutoff_time_to_tca_days": cutoff, "n_cdms_used": n_pre,
            "observed_log10_pc": obs_log, "observed_miss_distance_m": obs_miss, "observed_tier": observed_tier,
            "predicted_final_log10_pc": pred_value,
            "predicted_final_pc": 10 ** pred_value if np.isfinite(pred_value) else np.nan,
            "predicted_final_at_floor": bool(np.isfinite(pred_value) and pred_value <= LOG10_PC_FLOOR),
            "predicted_final_miss_distance_m": pred_miss, "final_miss_distance_basis": spec.miss_basis,
            "predicted_final_tier": predicted_tier,
            "predicted_final_log10_pc_lower": pred_value - width if uncertain else np.nan,
            "predicted_final_log10_pc_upper": min(pred_value + width, 0) if uncertain else np.nan,
            "p_critical": p_crit if spec.with_uncertainty else np.nan,
            "p_elevated": p_elev if spec.with_uncertainty else np.nan,
            "p_safe": (1 - p_crit - p_elev) if spec.with_uncertainty and np.isfinite(p_crit) else np.nan,
            "alert_reasons": reasons, "quality_flags": flags,
            "recommended_action": ACTIONS[predicted_tier], "model_version": spec.model_version,
            "actual_final_log10_pc": actual_log, "actual_final_miss_distance_m": actual_miss,
            "actual_final_tier": actual_tier,
        })
    return pd.DataFrame(events), pd.DataFrame(history)
