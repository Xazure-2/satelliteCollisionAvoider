"""Unit tests for src/risk.py: frozen thresholds, fail-safe rules and score_rows contract."""

import math
import unittest

import numpy as np
import pandas as pd

from src import risk
from src.risk import (
    ACTION_CRITICAL,
    ACTION_ELEVATED,
    ACTION_NEEDS_REVIEW,
    ACTION_NEEDS_REVIEW_ELEVATED,
    ACTION_SAFE,
    EXTRA_OUTPUT_COLUMNS,
    FLAG_CRITICAL_PARTIAL,
    FLAG_PREDICTION_CLIPPED,
    FLAG_SAFETY_UNDETERMINED,
    OUTPUT_COLUMNS,
    REASON_DIST_CRITICAL,
    REASON_DIST_ELEVATED,
    REASON_PC_CRITICAL,
    REASON_PC_ELEVATED,
    classify_risk,
    describe_alerts,
    score_rows,
)

# Column order exactly as listed in ROADMAP.md section 5.
ROADMAP_COLUMNS = [
    "source", "row_id", "event_id", "predicted_log10_pc", "predicted_pc", "risk_tier",
    "miss_distance_m", "miss_distance_km", "time_to_tca_days", "tca_hours", "c_object_type",
    "relative_position_r", "relative_position_t", "relative_position_n", "alert_reasons",
    "quality_flags", "recommended_action", "model_version",
]


def telemetry(n=4, **overrides):
    """Small raw telemetry frame in the allowed-input shape."""
    frame = pd.DataFrame({
        "event_id": np.arange(n) // 2,
        "time_to_tca": np.linspace(2.0, 0.5, n),
        "miss_distance": np.linspace(800.0, 12000.0, n),
        "relative_speed": np.full(n, 14000.0),
        "relative_position_r": np.arange(n, dtype=float) * 10,
        "relative_position_t": np.arange(n, dtype=float) * 100,
        "relative_position_n": np.arange(n, dtype=float) * -5,
        "c_object_type": ["DEBRIS", "PAYLOAD"] * (n // 2) + ["DEBRIS"] * (n % 2),
        "c_sigma_r": np.full(n, 250.0),
        "t_sigma_r": np.full(n, 5.0),
        "max_risk_estimate": np.full(n, -7.0),
        "F10": np.full(n, 90.0),
        "AP": np.full(n, 10.0),
    })
    for key, value in overrides.items():
        frame[key] = value
    return frame


class ThresholdConstantTests(unittest.TestCase):
    def test_thresholds_are_frozen(self):
        self.assertEqual(risk.PC_CRITICAL_LOG10, -4.0)
        self.assertEqual(risk.PC_ELEVATED_LOG10, -6.0)
        self.assertEqual(risk.DIST_CRITICAL_M, 1000.0)
        self.assertEqual(risk.DIST_ELEVATED_M, 5000.0)

    def test_dashboard_threshold_names(self):
        # ui/backend.py display_thresholds() reads these names from src.risk.
        self.assertEqual(risk.LOG10_PC_CRITICAL, -4.0)
        self.assertEqual(risk.LOG10_PC_ELEVATED, -6.0)
        self.assertEqual(risk.MISS_DISTANCE_CRITICAL_M, 1000.0)
        self.assertEqual(risk.MISS_DISTANCE_ELEVATED_M, 5000.0)


class ClassifyBoundaryTests(unittest.TestCase):
    def assertTier(self, log_pc, distance, tier, reasons):
        result = classify_risk(log_pc, distance)
        self.assertEqual(result.risk_tier, tier, (log_pc, distance))
        self.assertEqual(list(result.alert_reasons), reasons, (log_pc, distance))
        self.assertEqual(result.quality_flags, ())

    def test_pc_boundaries(self):
        self.assertTier(-4.0, 6000.0, "critical", [REASON_PC_CRITICAL])
        self.assertTier(np.nextafter(-4.0, -np.inf), 6000.0, "elevated", [REASON_PC_ELEVATED])
        self.assertTier(-6.0, 6000.0, "elevated", [REASON_PC_ELEVATED])
        self.assertTier(np.nextafter(-6.0, -np.inf), 6000.0, "safe", [])
        self.assertTier(0.0, 6000.0, "critical", [REASON_PC_CRITICAL])
        self.assertTier(-30.0, 6000.0, "safe", [])

    def test_distance_boundaries(self):
        self.assertTier(-7.0, np.nextafter(1000.0, 0.0), "critical", [REASON_DIST_CRITICAL])
        self.assertTier(-7.0, 1000.0, "elevated", [REASON_DIST_ELEVATED])
        self.assertTier(-7.0, 5000.0, "elevated", [REASON_DIST_ELEVATED])
        self.assertTier(-7.0, np.nextafter(5000.0, np.inf), "safe", [])
        self.assertTier(-7.0, 0.0, "critical", [REASON_DIST_CRITICAL])

    def test_conflicting_triggers_keep_both_reasons(self):
        self.assertTier(-3.0, 3000.0, "critical", [REASON_PC_CRITICAL, REASON_DIST_ELEVATED])
        self.assertTier(-5.0, 500.0, "critical", [REASON_PC_ELEVATED, REASON_DIST_CRITICAL])
        self.assertTier(-3.0, 500.0, "critical", [REASON_PC_CRITICAL, REASON_DIST_CRITICAL])
        self.assertTier(-5.0, 3000.0, "elevated", [REASON_PC_ELEVATED, REASON_DIST_ELEVATED])

    def test_numpy_and_integer_scalars(self):
        self.assertEqual(classify_risk(np.float32(-4.0), np.int64(999)).risk_tier, "critical")
        self.assertEqual(classify_risk(-7, 6000).risk_tier, "safe")

    def test_actions_by_tier(self):
        self.assertEqual(classify_risk(-3, 6000).recommended_action, ACTION_CRITICAL)
        self.assertEqual(classify_risk(-5, 6000).recommended_action, ACTION_ELEVATED)
        self.assertEqual(classify_risk(-7, 6000).recommended_action, ACTION_SAFE)
        self.assertIn("maneuver planning", ACTION_CRITICAL)
        self.assertIn("does not prescribe or execute a burn", ACTION_CRITICAL)
        self.assertTrue(ACTION_SAFE.startswith("Monitor"))
        self.assertTrue(ACTION_ELEVATED.startswith("Review"))

    def test_deterministic(self):
        self.assertEqual(classify_risk(-5.5, 4000.0), classify_risk(-5.5, 4000.0))


class ClassifyInvalidInputTests(unittest.TestCase):
    INVALID_PC = [None, float("nan"), float("inf"), float("-inf"), pd.NA, "-5", True, [1], 0.5]
    INVALID_DIST = [None, float("nan"), float("inf"), pd.NA, "700", False, -1.0]

    def test_invalid_inputs_never_safe(self):
        # Every pair with at least one invalid input; the other side is either
        # invalid or a value that would otherwise be safe.
        pcs = self.INVALID_PC + [-8.0]
        dists = self.INVALID_DIST + [20000.0]
        for i, pc in enumerate(pcs):
            for j, dist in enumerate(dists):
                if i == len(pcs) - 1 and j == len(dists) - 1:
                    continue  # the only fully valid pair
                with self.subTest(pc=pc, dist=dist):
                    self.assertNotEqual(classify_risk(pc, dist).risk_tier, "safe")

    def test_missing_pc_is_needs_review(self):
        result = classify_risk(float("nan"), 20000.0)
        self.assertEqual(result.risk_tier, "needs_review")
        self.assertEqual(result.recommended_action, ACTION_NEEDS_REVIEW)
        self.assertEqual(result.quality_flags, ("predicted_log10_pc_missing", FLAG_SAFETY_UNDETERMINED))

    def test_missing_distance_is_needs_review(self):
        result = classify_risk(-8.0, None)
        self.assertEqual(result.risk_tier, "needs_review")
        self.assertIn("miss_distance_m_missing", result.quality_flags)

    def test_valid_critical_input_survives_missing_other(self):
        result = classify_risk(None, 500.0)
        self.assertEqual(result.risk_tier, "critical")
        self.assertEqual(result.alert_reasons, (REASON_DIST_CRITICAL,))
        self.assertIn(FLAG_CRITICAL_PARTIAL, result.quality_flags)
        result = classify_risk(-3.0, float("nan"))
        self.assertEqual(result.risk_tier, "critical")
        self.assertIn(FLAG_CRITICAL_PARTIAL, result.quality_flags)

    def test_elevated_evidence_with_missing_input_stays_review(self):
        result = classify_risk(float("nan"), 3000.0)
        self.assertEqual(result.risk_tier, "needs_review")
        self.assertEqual(result.alert_reasons, (REASON_DIST_ELEVATED,))
        self.assertEqual(result.recommended_action, ACTION_NEEDS_REVIEW_ELEVATED)

    def test_specific_flags(self):
        self.assertIn("predicted_log10_pc_non_finite", classify_risk(float("inf"), 9000).quality_flags)
        self.assertIn("predicted_log10_pc_invalid_type", classify_risk("-5", 9000).quality_flags)
        self.assertIn("predicted_log10_pc_invalid_type", classify_risk(True, 9000).quality_flags)
        self.assertIn("predicted_log10_pc_out_of_range", classify_risk(0.5, 9000).quality_flags)
        self.assertIn("miss_distance_m_negative", classify_risk(-7, -1).quality_flags)
        self.assertIn("miss_distance_m_invalid_type", classify_risk(-7, "700").quality_flags)


class DescribeAlertsTests(unittest.TestCase):
    def test_roadmap_example(self):
        self.assertEqual(describe_alerts(-8.0, 720.0), ["Miss distance 0.72 km is below 1 km."])

    def test_sentence_per_reason_in_order(self):
        details = describe_alerts(-3.5, 4200.0)
        self.assertEqual(len(details), 2)
        self.assertTrue(details[0].startswith("Estimated Pc 3.2e-04 (log10 -3.50)"))
        self.assertIn("critical threshold 1e-4", details[0])
        self.assertEqual(details[1], "Miss distance 4.20 km is at or below 5 km.")

    def test_near_boundary_text_is_not_misleading(self):
        self.assertEqual(describe_alerts(-8.0, 999.6), ["Miss distance 0.9996 km is below 1 km."])
        self.assertEqual(describe_alerts(-8.0, 1000.0), ["Miss distance 1.00 km is at or below 5 km."])
        text = describe_alerts(-4.00001, 9000.0)[0]
        self.assertIn("elevated threshold", text)
        self.assertIn("(log10 -4.00001)", text)
        self.assertNotIn("1.0e-04", text)
        self.assertEqual(describe_alerts(-8.0, 1000.0004),
                         ["Miss distance 1.0000004 km is at or below 5 km."])

    def test_invalid_inputs_give_no_sentence(self):
        self.assertEqual(describe_alerts(None, float("nan")), [])
        self.assertEqual(describe_alerts(-8.0, 20000.0), [])


class ScoreRowsContractTests(unittest.TestCase):
    def setUp(self):
        self.frame = telemetry(4)
        self.preds = np.array([-7.0, -5.0, -3.0, -8.0])

    def test_columns_match_roadmap_contract(self):
        out = score_rows(self.frame, self.preds, source="test", model_version="m1")
        self.assertEqual(list(OUTPUT_COLUMNS), ROADMAP_COLUMNS)
        self.assertEqual(list(out.columns), ROADMAP_COLUMNS + list(EXTRA_OUTPUT_COLUMNS))
        self.assertEqual(risk.CONTRACT_COLUMNS, ROADMAP_COLUMNS)

    def test_order_index_and_keys_preserved(self):
        frame = self.frame.iloc[[3, 1, 0, 2]].copy()
        frame.index = [40, 10, 0, 20]
        out = score_rows(frame, self.preds, source="test")
        self.assertEqual(list(out.index), [40, 10, 0, 20])
        self.assertEqual(out["row_id"].tolist(), [0, 1, 2, 3])  # positional, not index labels
        self.assertEqual(out["event_id"].tolist(), frame["event_id"].tolist())
        self.assertEqual(out["predicted_log10_pc"].tolist(), self.preds.tolist())

    def test_row_id_column_used_when_present(self):
        frame = self.frame.assign(row_id=[7, 3, 9, 1])
        out = score_rows(frame, self.preds, source="test")
        self.assertEqual(out["row_id"].tolist(), [7, 3, 9, 1])
        with self.assertRaises(ValueError):
            score_rows(self.frame.assign(row_id=[1, 1, 2, 3]), self.preds, source="test")
        with self.assertRaises(ValueError):
            score_rows(self.frame.assign(row_id=[0, 1, 2.5, 3]), self.preds, source="test")

    def test_predictions_aligned_by_position(self):
        series = pd.Series(self.preds, index=[3, 2, 1, 0])
        out = score_rows(self.frame, series, source="test")
        self.assertEqual(out["predicted_log10_pc"].tolist(), self.preds.tolist())
        out2 = score_rows(self.frame, self.preds.reshape(-1, 1), source="test")
        pd.testing.assert_frame_equal(out, out2)

    def test_units_converted_exactly_once(self):
        out = score_rows(self.frame, self.preds, source="test")
        np.testing.assert_array_equal(out["miss_distance_m"], self.frame["miss_distance"])
        np.testing.assert_array_equal(out["miss_distance_km"], self.frame["miss_distance"] / 1000.0)
        np.testing.assert_array_equal(out["time_to_tca_days"], self.frame["time_to_tca"])
        np.testing.assert_array_equal(out["tca_hours"], self.frame["time_to_tca"] * 24.0)
        for axis in "rtn":
            col = f"relative_position_{axis}"
            np.testing.assert_array_equal(out[col], self.frame[col])
        np.testing.assert_array_equal(out["predicted_pc"], 10.0 ** self.preds)

    def test_contract_input_aliases(self):
        renamed = self.frame.rename(columns={"miss_distance": "miss_distance_m",
                                             "time_to_tca": "time_to_tca_days"})
        a = score_rows(self.frame, self.preds, source="test")
        b = score_rows(renamed, self.preds, source="test")
        pd.testing.assert_frame_equal(a, b)
        both = self.frame.assign(miss_distance_m=self.frame["miss_distance"])
        with self.assertRaises(ValueError):
            score_rows(both, self.preds, source="test")

    def test_tiers_match_classify_risk(self):
        out = score_rows(self.frame, self.preds, source="test")
        for i, row in out.iterrows():
            expected = classify_risk(self.preds[i], self.frame.loc[i, "miss_distance"])
            self.assertEqual(row["risk_tier"], expected.risk_tier)
            self.assertEqual(row["alert_reasons"], list(expected.alert_reasons))
            self.assertEqual(row["recommended_action"], expected.recommended_action)
            self.assertEqual(row["alert_details"],
                             describe_alerts(self.preds[i], self.frame.loc[i, "miss_distance"]))
        self.assertEqual(out["risk_tier"].tolist(), ["critical", "elevated", "critical", "safe"])

    def test_list_fields_are_lists(self):
        out = score_rows(self.frame, self.preds, source="test")
        for col in ("alert_reasons", "quality_flags", "alert_details"):
            self.assertTrue(all(isinstance(v, list) for v in out[col]), col)


class ScoreRowsSourceAndVersionTests(unittest.TestCase):
    def setUp(self):
        self.frame = telemetry(2)
        self.preds = [-7.0, -7.0]

    def test_dashboard_call_style(self):
        # ui/backend.py: frame has source + row_id columns, called with two positional args.
        frame = self.frame.copy()
        frame.insert(0, "row_id", np.arange(len(frame), dtype="int64"))
        frame.insert(0, "source", "train_replay")
        out = score_rows(frame, np.asarray(self.preds))
        self.assertEqual(out["source"].tolist(), ["train_replay"] * 2)
        self.assertEqual(out["model_version"].tolist(), ["unknown"] * 2)

    def test_source_argument_and_validation(self):
        self.assertEqual(score_rows(self.frame, self.preds, source="test")["source"].tolist(), ["test"] * 2)
        for bad in ["train", "", None]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                score_rows(self.frame, self.preds, source=bad)
        with self.assertRaises(ValueError):
            score_rows(self.frame.assign(source=["test", "other"]), self.preds)
        with self.assertRaises(ValueError):
            score_rows(self.frame.assign(source="test"), self.preds, source="train_replay")

    def test_mixed_sources_in_one_frame(self):
        out = score_rows(self.frame.assign(source=["test", "train_replay"]), self.preds)
        self.assertEqual(out["source"].tolist(), ["test", "train_replay"])

    def test_model_version(self):
        out = score_rows(self.frame, self.preds, source="test", model_version="abc-M3")
        self.assertEqual(out["model_version"].tolist(), ["abc-M3"] * 2)
        out = score_rows(self.frame.assign(model_version=["v1", None]), self.preds, source="test")
        self.assertEqual(out["model_version"].tolist(), ["v1", "unknown"])
        with self.assertRaises(ValueError):
            score_rows(self.frame, self.preds, source="test", model_version=" ")


class ScoreRowsPredictionTests(unittest.TestCase):
    def test_clipping_is_flagged_and_drives_tier(self):
        frame = telemetry(4, miss_distance=[20000.0] * 4)
        out = score_rows(frame, [0.5, -35.0, -30.0, 0.0], source="test")
        self.assertEqual(out["predicted_log10_pc"].tolist(), [0.0, -30.0, -30.0, 0.0])
        self.assertEqual(out["predicted_log10_pc_raw"].tolist(), [0.5, -35.0, -30.0, 0.0])
        self.assertEqual(out["risk_tier"].tolist(), ["critical", "safe", "safe", "critical"])
        clipped = [FLAG_PREDICTION_CLIPPED in f for f in out["quality_flags"]]
        self.assertEqual(clipped, [True, True, False, False])
        np.testing.assert_array_equal(out["predicted_pc"], [1.0, 1e-30, 1e-30, 1.0])

    def test_unusable_predictions_never_safe(self):
        frame = telemetry(4, miss_distance=[20000.0, 20000.0, 500.0, 3000.0])
        out = score_rows(frame, [np.nan, np.inf, None, np.nan], source="test")
        self.assertEqual(out["risk_tier"].tolist(),
                         ["needs_review", "needs_review", "critical", "needs_review"])
        self.assertTrue(out["predicted_log10_pc"].isna().all())
        self.assertTrue(out["predicted_pc"].isna().all())
        self.assertIn("predicted_log10_pc_non_finite", out.loc[1, "quality_flags"])
        self.assertIn(FLAG_CRITICAL_PARTIAL, out.loc[2, "quality_flags"])

    def test_bad_prediction_containers(self):
        frame = telemetry(2)
        with self.assertRaises(ValueError):
            score_rows(frame, [-7.0], source="test")
        with self.assertRaises(TypeError):
            score_rows(frame, -7.0, source="test")
        with self.assertRaises(TypeError):
            score_rows(frame, "-7", source="test")
        with self.assertRaises(ValueError):
            score_rows(frame, np.zeros((2, 2)), source="test")


class ScoreRowsDataQualityTests(unittest.TestCase):
    def test_missing_and_invalid_distance(self):
        frame = telemetry(3, miss_distance=[np.nan, -5.0, 20000.0])
        out = score_rows(frame, [-8.0, -8.0, -8.0], source="test")
        self.assertEqual(out["risk_tier"].tolist(), ["needs_review", "needs_review", "safe"])
        self.assertIn("miss_distance_m_missing", out.loc[0, "quality_flags"])
        self.assertIn("miss_distance_m_negative", out.loc[1, "quality_flags"])

    def test_input_flags(self):
        frame = telemetry(
            4,
            time_to_tca=[-0.1, np.nan, 1.0, 1.0],
            c_object_type=[None, "", "DEBRIS", "PAYLOAD"],
            c_sigma_r=[63781363.0, np.nan, -1.0, 10.0],
            F10=[np.nan, 90.0, 90.0, 90.0],
            event_id=[1, None, 2, 2],
        )
        out = score_rows(frame, [-8.0] * 4, source="test")
        flags = out["quality_flags"].tolist()
        self.assertIn("time_to_tca_days_negative", flags[0])
        self.assertIn("c_object_type_missing", flags[0])
        self.assertIn("c_sigma_r_extreme", flags[0])
        self.assertIn("F10_missing", flags[0])
        self.assertIn("time_to_tca_days_missing", flags[1])
        self.assertIn("c_object_type_missing", flags[1])
        self.assertIn("c_sigma_r_missing", flags[1])
        self.assertIn("event_id_missing", flags[1])
        self.assertIn("c_sigma_r_negative", flags[2])
        self.assertEqual(flags[3], [])
        self.assertEqual(out["c_object_type"].tolist(), ["UNKNOWN", "UNKNOWN", "DEBRIS", "PAYLOAD"])
        # Data-quality flags on optional inputs never change the tier.
        for i in range(4):
            self.assertEqual(out.loc[i, "risk_tier"],
                             classify_risk(-8.0, frame.loc[i, "miss_distance"]).risk_tier)

    def test_negative_tca_keeps_tier(self):
        frame = telemetry(2, time_to_tca=[-0.05, 0.5], miss_distance=[500.0, 500.0])
        out = score_rows(frame, [-8.0, -8.0], source="train_replay")
        self.assertEqual(out["risk_tier"].tolist(), ["critical", "critical"])
        self.assertEqual(out["tca_hours"].tolist(), [-0.05 * 24, 12.0])

    def test_required_columns(self):
        frame = telemetry(2)
        for col in ("event_id", "miss_distance", "time_to_tca"):
            with self.subTest(col=col), self.assertRaises(ValueError):
                score_rows(frame.drop(columns=col), [-7.0, -7.0], source="test")
        with self.assertRaises(TypeError):
            score_rows(frame.to_dict(), [-7.0, -7.0], source="test")

    def test_minimal_frame_without_optional_columns(self):
        frame = pd.DataFrame({"event_id": [5], "time_to_tca": [1.0], "miss_distance": [700.0]})
        out = score_rows(frame, [-8.0], source="test")
        self.assertEqual(out.loc[0, "risk_tier"], "critical")
        self.assertTrue(math.isnan(out.loc[0, "relative_position_r"]))
        self.assertEqual(out.loc[0, "quality_flags"], ["c_object_type_missing"])

    def test_empty_frame(self):
        out = score_rows(telemetry(2).iloc[0:0], [], source="test")
        self.assertEqual(len(out), 0)
        self.assertEqual(list(out.columns), ROADMAP_COLUMNS + list(EXTRA_OUTPUT_COLUMNS))


class LabelBlindnessTests(unittest.TestCase):
    def test_risk_column_never_affects_output(self):
        frame = telemetry(4)
        preds = [-7.0, -5.0, -3.0, -8.0]
        base = score_rows(frame, preds, source="test", model_version="m")
        for risk_values in ([0.0] * 4, [-30.0] * 4, [np.nan] * 4):
            with self.subTest(risk=risk_values):
                out = score_rows(frame.assign(risk=risk_values), preds, source="test", model_version="m")
                pd.testing.assert_frame_equal(base, out)
                self.assertNotIn("risk", out.columns)


if __name__ == "__main__":
    unittest.main()
