"""Clearly labelled SYNTHETIC demonstration data in the scored-row contract.

Used only until the real model (src/model.py) and risk pipeline (src/risk.py)
are ready. Every row has source="synthetic" and model_version
"synthetic-demo-v0", so it can never be mistaken for real predictions.

Risk tiers here are authored per scenario, not computed: each scenario samples
Pc and miss distance from inside a hand-picked band for its intended tier.
The real tier logic lives in src/risk.py and is not reproduced here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_MODEL_VERSION = "synthetic-demo-v0"

OBJECT_TYPES = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN", "OTHER"]
OBJECT_WEIGHTS = [0.5, 0.2, 0.15, 0.1, 0.05]

# (log10 Pc band, miss distance band in metres, authored tier, reason kinds)
BANDS = {
    "safe": ((-12.0, -6.6), (8_000, 60_000), "safe", []),
    "elevated_pc": ((-5.8, -4.3), (6_500, 25_000), "elevated", ["pc_elevated"]),
    "elevated_distance": ((-9.0, -6.6), (1_300, 4_700), "elevated", ["distance_elevated"]),
    "critical_pc": ((-3.8, -2.6), (5_500, 15_000), "critical", ["pc_critical"]),
    "critical_distance": ((-9.0, -6.6), (250, 900), "critical", ["distance_critical"]),
    "critical_both": ((-3.6, -2.4), (150, 850), "critical", ["pc_critical", "distance_critical"]),
}

ACTIONS = {
    "safe": "Monitor",
    "elevated": "Review",
    "critical": "Initiate maneuver planning",
    "needs_review": "Review data quality",
}

# Scenario name -> list of band phases from oldest to newest message.
SCENARIOS = {
    "safe": ["safe"],
    "elevated_pc": ["elevated_pc"],
    "elevated_distance": ["elevated_distance"],
    "critical_pc": ["critical_pc"],
    "critical_distance": ["critical_distance"],
    "critical_both": ["critical_both"],
    "escalation": ["safe", "elevated_pc", "critical_pc"],
    "rising_but_elevated": ["elevated_pc"],
    "de_escalation": ["elevated_pc", "safe"],
    "needs_review": ["safe"],
}

SCENARIO_MIX = (
    ["safe"] * 18
    + ["elevated_pc"] * 5
    + ["elevated_distance"] * 4
    + ["critical_pc"] * 2
    + ["critical_distance"] * 2
    + ["critical_both"] * 1
    + ["escalation"] * 2
    + ["rising_but_elevated"] * 1
    + ["de_escalation"] * 2
    + ["needs_review"] * 1
)


def _reason_text(kind: str, log_pc: float, miss_m: float) -> str:
    pc = 10 ** log_pc
    km = miss_m / 1000
    return {
        "pc_critical": f"Estimated Pc {pc:.1e} is at or above 1e-4.",
        "pc_elevated": f"Estimated Pc {pc:.1e} is at or above 1e-6.",
        "distance_critical": f"Miss distance {km:.2f} km is below 1 km.",
        "distance_elevated": f"Miss distance {km:.2f} km is at or below 5 km.",
    }[kind]


def make_synthetic_scored(seed: int = 7) -> pd.DataFrame:
    """Build a deterministic synthetic scored table (one row per message)."""
    rng = np.random.default_rng(seed)
    rows = []
    row_id = 0
    for event_id, scenario in enumerate(SCENARIO_MIX):
        n_msgs = int(rng.integers(4, 10))
        # About 40% of events have a latest message within 24 h of TCA.
        latest_tca = rng.uniform(0.05, 1.0) if rng.random() < 0.4 else rng.uniform(1.05, 3.5)
        gaps = rng.uniform(0.25, 0.7, size=n_msgs - 1)
        times = latest_tca + np.concatenate([[0.0], np.cumsum(gaps)])[::-1]
        times = np.minimum(times, 6.95)
        obj = rng.choice(OBJECT_TYPES, p=OBJECT_WEIGHTS)
        phases = SCENARIOS[scenario]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        # Positions within each band drift smoothly so histories look plausible.
        u_pc0, u_d0 = rng.uniform(0.2, 0.8, size=2)
        pc_slope, d_slope = rng.normal(0, 0.25, size=2)
        if scenario == "rising_but_elevated":
            u_pc0, pc_slope = 0.08, 0.85

        for i, t in enumerate(times):
            phase = phases[min(int(i * len(phases) / n_msgs), len(phases) - 1)]
            (pc_lo, pc_hi), (d_lo, d_hi), tier, kinds = BANDS[phase]
            frac = i / max(n_msgs - 1, 1)
            u_pc = np.clip(u_pc0 + pc_slope * frac + rng.normal(0, 0.03), 0.02, 0.98)
            u_d = np.clip(u_d0 + d_slope * frac + rng.normal(0, 0.03), 0.02, 0.98)
            log_pc = pc_lo + (pc_hi - pc_lo) * u_pc
            miss = d_lo + (d_hi - d_lo) * u_d
            drift = direction + rng.normal(scale=0.05, size=3)
            r, tt, n = drift / np.linalg.norm(drift) * miss
            reasons = [_reason_text(k, log_pc, miss) for k in kinds]
            flags: list[str] = []
            pred = log_pc
            if scenario == "needs_review" and i == n_msgs - 1:
                tier, pred, reasons = "needs_review", np.nan, ["Model estimate unavailable."]
                flags = ["missing_model_output"]
            if scenario == "elevated_distance" and event_id % 2 == 0 and i == n_msgs - 1:
                flags.append("extreme_c_sigma_r")
            if scenario == "safe" and event_id % 7 == 3:
                flags.append("missing_F10_AP_imputed")
            rows.append(
                {
                    "source": "synthetic",
                    "row_id": row_id,
                    "event_id": event_id,
                    "predicted_log10_pc": pred,
                    "predicted_pc": 10 ** pred if np.isfinite(pred) else np.nan,
                    "risk_tier": tier,
                    "miss_distance_m": miss,
                    "miss_distance_km": miss / 1000,
                    "time_to_tca_days": t,
                    "tca_hours": 24 * t,
                    "c_object_type": obj,
                    "relative_position_r": r,
                    "relative_position_t": tt,
                    "relative_position_n": n,
                    "alert_reasons": reasons,
                    "quality_flags": flags,
                    "recommended_action": ACTIONS[tier],
                    "model_version": SYNTHETIC_MODEL_VERSION,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)
