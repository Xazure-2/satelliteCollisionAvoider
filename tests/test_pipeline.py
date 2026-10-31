"""Event logic and integration tests: model -> score_rows -> events -> dashboard contract.

The dashboard checks need Teammate 3's ``ui`` package. It is imported from the
repository when merged, or from ``DASHBOARD_UI_PATH`` (a directory containing
``ui/``) otherwise; those tests are skipped if neither is available.
The model checks use Teammate 1's ``src.model`` with a tiny pipeline fitted on
synthetic data, so no trained artifact or real dataset is needed.
"""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import events
from src.risk import OUTPUT_COLUMNS, SCORED_COLUMNS, score_rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _optional_import(name):
    try:
        return importlib.import_module(name)
    except Exception:  # missing module or missing dependency
        return None


def _load_dashboard():
    path = os.environ.get("DASHBOARD_UI_PATH")
    if path and path not in sys.path:
        sys.path.append(path)
    return _optional_import("ui.contract"), _optional_import("ui.backend")


UI_CONTRACT, UI_BACKEND = _load_dashboard()
MODEL = _optional_import("src.model")


def synthetic_scored():
    """Synthetic fixture, hand-counted for the 24-hour window (not real telemetry).

    Latest pre-TCA message per (source, event_id), hours = 24 * time_to_tca:

    event          rows      latest  hours  log10Pc  miss m  tier          in window
    test/1         0,1       1       12     -3       8000    critical(Pc)  yes
    test/2         2         2       24     -8       800     critical(d)   yes (24 h edge)
    test/3         3         3       0      -5       9000    elevated(Pc)  yes (0 h edge)
    test/4         4         4       24.0024 -2      9000    critical      no
    test/5         5         -       -      -        -       post-TCA only no (excluded)
    test/6         6,7       7       6      -8       3000    elevated(d)   yes (tie: higher row_id)
    train_replay/1 8         8       18     -9       20000   safe          yes (ID collides with test/1)
    test/7         9         9       12     nan      20000   needs_review  yes
    """
    frame = pd.DataFrame(
        [
            # source, event_id, time_to_tca, miss, pred, object
            ("test", 1, 3.0, 9000.0, -7.0, "DEBRIS"),
            ("test", 1, 0.5, 8000.0, -3.0, "DEBRIS"),
            ("test", 2, 1.0, 800.0, -8.0, "PAYLOAD"),
            ("test", 3, 0.0, 9000.0, -5.0, "DEBRIS"),
            ("test", 4, 1.0001, 9000.0, -2.0, "ROCKET BODY"),
            ("test", 5, -0.1, 500.0, -2.0, "DEBRIS"),
            ("test", 6, 0.25, 9000.0, -3.0, "UNKNOWN"),
            ("test", 6, 0.25, 3000.0, -8.0, "UNKNOWN"),
            ("train_replay", 1, 0.75, 20000.0, -9.0, "PAYLOAD"),
            ("test", 7, 0.5, 20000.0, np.nan, "DEBRIS"),
        ],
        columns=["source", "event_id", "time_to_tca", "miss_distance", "pred", "c_object_type"],
    )
    frame.insert(0, "row_id", np.arange(len(frame)))
    preds = frame.pop("pred").to_numpy()
    return score_rows(frame, preds, model_version="fixture")


EXPECTED_24H = {
    "total_encounters": 6,
    "distance_dangerous": 1,
    "critical": 2,
    "elevated": 2,
    "needs_review": 1,
    "safe": 1,
    "pc_critical": 1,
    "distance_only_critical": 1,
    "pc_elevated": 1,
    "distance_only_elevated": 1,
}


def keys(frame):
    return list(zip(frame["source"], frame["event_id"]))


class LatestMessageTests(unittest.TestCase):
    def setUp(self):
        self.scored = synthetic_scored()

    def test_one_row_per_event_with_expected_choice(self):
        latest = events.select_latest_messages(self.scored)
        self.assertEqual(list(latest.columns), list(self.scored.columns))
        self.assertEqual(len(latest), len(set(keys(latest))))
        chosen = dict(zip(keys(latest), latest["row_id"]))
        self.assertEqual(chosen, {("test", 1): 1, ("test", 2): 2, ("test", 3): 3, ("test", 4): 4,
                                  ("test", 6): 7, ("train_replay", 1): 8, ("test", 7): 9})

    def test_negative_tca_never_selected(self):
        latest = events.select_latest_messages(self.scored)
        self.assertNotIn(("test", 5), keys(latest))
        self.assertTrue((latest["time_to_tca_days"] >= 0).all())

    def test_tie_breaker_highest_row_id_regardless_of_order(self):
        shuffled = self.scored.iloc[::-1].reset_index(drop=True)
        latest = events.select_latest_messages(shuffled)
        self.assertEqual(dict(zip(keys(latest), latest["row_id"]))[("test", 6)], 7)

    def test_source_id_collision_kept_separate(self):
        latest = events.select_latest_messages(self.scored)
        self.assertIn(("test", 1), keys(latest))
        self.assertIn(("train_replay", 1), keys(latest))

    def test_matches_teammate_model_latest_indices(self):
        if MODEL is None:
            self.skipTest("src.model not importable")
        frame = self.scored.rename(columns={"time_to_tca_days": "time_to_tca"})
        positions = sorted(MODEL.latest_indices(frame))
        mine = sorted(events.select_latest_messages(self.scored)["row_id"])
        self.assertEqual(sorted(frame["row_id"].iloc[positions]), mine)

    def test_replay_cutoff_never_selects_newer_message(self):
        frame = pd.DataFrame({"event_id": [9, 9, 9], "time_to_tca": [3.0, 1.2, 0.4],
                              "miss_distance": [9000.0] * 3})
        scored = score_rows(frame, [-8.0] * 3, source="train_replay")
        pick = lambda c: events.select_latest_messages(scored, c)["time_to_tca_days"].tolist()
        self.assertEqual(pick(None), [0.4])
        self.assertEqual(pick(1.0), [1.2])
        self.assertEqual(pick(1.2), [1.2])  # inclusive
        self.assertEqual(pick(2.0), [3.0])
        self.assertEqual(pick(5.0), [])
        for bad in (-1, float("nan"), float("inf"), "x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                events.select_latest_messages(scored, bad)

    def test_missing_event_ids_are_not_merged(self):
        frame = pd.DataFrame({"event_id": [None, None], "time_to_tca": [0.5, 0.4],
                              "miss_distance": [9000.0, 9000.0]})
        scored = score_rows(frame, [-8.0, -8.0], source="test")
        self.assertEqual(len(events.select_latest_messages(scored)), 2)

    def test_requires_event_key_columns(self):
        with self.assertRaises(ValueError):
            events.select_latest_messages(self.scored.drop(columns="source"))
        with self.assertRaises(TypeError):
            events.select_latest_messages([1, 2])

    def test_empty(self):
        self.assertEqual(len(events.select_latest_messages(self.scored.iloc[0:0])), 0)


class Summary24hTests(unittest.TestCase):
    def setUp(self):
        self.scored = synthetic_scored()

    def test_hand_counted_fixture(self):
        summary = events.summarize_upcoming_24h(self.scored)
        self.assertEqual({k: summary[k] for k in events.SUMMARY_COUNT_KEYS}, EXPECTED_24H)
        self.assertEqual(summary["time_basis"], "original")
        self.assertEqual(summary["data_mode"], "latest_message")
        self.assertEqual(summary["window_hours"], 24.0)

    def test_duplicate_messages_counted_once(self):
        dup = self.scored.copy()
        dup["row_id"] = dup["row_id"] + 100
        doubled = pd.concat([self.scored, dup], ignore_index=True)
        summary = events.summarize_upcoming_24h(doubled)
        self.assertEqual({k: summary[k] for k in events.SUMMARY_COUNT_KEYS}, EXPECTED_24H)

    def test_tier_counts_partition_total(self):
        s = events.summarize_upcoming_24h(self.scored)
        self.assertEqual(s["critical"] + s["elevated"] + s["needs_review"] + s["safe"],
                         s["total_encounters"])
        self.assertEqual(s["pc_critical"] + s["distance_only_critical"], s["critical"])
        self.assertEqual(s["pc_elevated"] + s["distance_only_elevated"], s["elevated"])

    def test_test_data_style_view_is_empty(self):
        # Every test-file message is >= 2 days before TCA, so the honest count is zero.
        frame = pd.DataFrame({"event_id": [0, 0, 1], "time_to_tca": [2.0002, 2.5, 6.99],
                              "miss_distance": [500.0, 600.0, 700.0]})
        scored = score_rows(frame, [-2.0] * 3, source="test")
        summary = events.summarize_upcoming_24h(scored)
        self.assertTrue(all(summary[k] == 0 for k in events.SUMMARY_COUNT_KEYS))

    def test_empty_and_bad_basis(self):
        summary = events.summarize_upcoming_24h(self.scored.iloc[0:0])
        self.assertTrue(all(summary[k] == 0 for k in events.SUMMARY_COUNT_KEYS))
        with self.assertRaises(ValueError):
            events.summarize_upcoming_24h(self.scored, time_basis="replay")
        with self.assertRaises(ValueError):
            events.summarize_upcoming_24h(self.scored, time_basis="wall_clock")

    def test_alias(self):
        self.assertEqual(events.summarize_24h(self.scored), events.summarize_upcoming_24h(self.scored))


class ReplayTests(unittest.TestCase):
    def setUp(self):
        frame = pd.DataFrame({
            "event_id": [1, 1, 1, 2, 2, 3],
            "time_to_tca": [3.0, 2.1, 0.2, 1.5, 0.9, 0.5],
            "miss_distance": [9000.0, 800.0, 700.0, 9000.0, 9000.0, 9000.0],
            "c_object_type": "DEBRIS",
        })
        self.scored = score_rows(frame, [-8.0, -8.0, -3.0, -5.0, -8.0, -8.0], source="train_replay")

    def test_selection_and_labels(self):
        replay = events.select_replay(self.scored, 48)
        self.assertEqual(list(replay.columns), list(self.scored.columns) + events.REPLAY_COLUMNS)
        self.assertEqual(replay["event_id"].tolist(), [1])  # only event 1 has a message >= 2 days
        row = replay.iloc[0]
        self.assertEqual(row["time_to_tca_days"], 2.1)          # original, unchanged
        self.assertAlmostEqual(row["tca_hours"], 2.1 * 24)      # original, unchanged
        self.assertEqual(row["replay_tca_hours"], 48.0)
        self.assertEqual(row["replay_horizon_hours"], 48.0)
        self.assertAlmostEqual(row["message_staleness_hours"], 2.1 * 24 - 48)
        self.assertEqual(row["data_mode"], "historical_replay")
        self.assertIn("Historical replay", row["replay_label"])
        self.assertIn("not a real-time", row["replay_label"])
        self.assertEqual(row["risk_tier"], "critical")  # 800 m at the replay point

    def test_never_uses_messages_after_replay_point(self):
        replay = events.select_replay(self.scored, 12)
        chosen = dict(zip(replay["event_id"], replay["time_to_tca_days"]))
        self.assertEqual(chosen, {1: 2.1, 2: 0.9, 3: 0.5})
        self.assertTrue((replay["tca_hours"] >= 12).all())

    def test_replay_summary_uses_replay_tca_hours(self):
        replay = events.select_replay(self.scored, 12)
        replay_summary = events.summarize_upcoming_24h(replay)
        self.assertEqual(replay_summary["time_basis"], "replay")
        self.assertEqual(replay_summary["data_mode"], "historical_replay")
        self.assertEqual(replay_summary["total_encounters"], 3)
        self.assertEqual(replay_summary["critical"], 1)
        self.assertEqual(replay_summary["distance_dangerous"], 1)
        # The same rows judged on their original message age: only events 2 and 3 are <= 24 h.
        original = events.summarize_upcoming_24h(replay, time_basis="original")
        self.assertEqual(original["total_encounters"], 2)
        self.assertEqual(events.summarize_upcoming_24h(replay, time_basis="replay"), replay_summary)

    def test_horizon_beyond_24h_counts_nothing(self):
        summary = events.summarize_upcoming_24h(events.select_replay(self.scored, 36))
        self.assertEqual(summary["total_encounters"], 0)

    def test_no_fabricated_timestamps(self):
        replay = events.select_replay(self.scored, 12)
        for col in replay.columns:
            self.assertFalse(any(w in col.lower() for w in ("timestamp", "epoch", "date", "datetime")), col)

    def test_invalid_horizon_and_alias(self):
        for bad in (-1, float("nan"), None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                events.select_replay(self.scored, bad)
        pd.testing.assert_frame_equal(events.select_training_replay(self.scored, 12),
                                      events.select_replay(self.scored, 12))


class ActionQueueTests(unittest.TestCase):
    def setUp(self):
        self.latest = events.select_latest_messages(synthetic_scored())

    def test_order_ranks_and_reasons(self):
        queue = events.build_action_queue(self.latest)
        self.assertEqual(list(queue.columns), list(self.latest.columns) + events.QUEUE_COLUMNS)
        self.assertEqual(queue["priority_rank"].tolist(), list(range(1, len(self.latest) + 1)))
        # critical (shortest TCA first), needs_review, elevated (shortest first), safe
        self.assertEqual(keys(queue), [("test", 1), ("test", 2), ("test", 4), ("test", 7),
                                       ("test", 3), ("test", 6), ("train_replay", 1)])
        reasons = dict(zip(keys(queue), queue["reason_text"]))
        self.assertIn("Miss distance 0.80 km is below 1 km.", reasons[("test", 2)])
        self.assertIn("Estimated Pc", reasons[("test", 1)])
        self.assertIn("Needs manual review", reasons[("test", 7)])
        self.assertIn("predicted_log10_pc_missing", reasons[("test", 7)])
        self.assertIn("No trigger", reasons[("train_replay", 1)])

    def test_tie_breakers_tca_then_estimated_pc(self):
        frame = pd.DataFrame({"event_id": [1, 2, 3], "time_to_tca": [0.5, 0.5, 0.2],
                              "miss_distance": [9000.0] * 3})
        scored = score_rows(frame, [-3.5, -2.0, -3.9], source="test")
        queue = events.build_action_queue(scored)
        self.assertEqual(queue["event_id"].tolist(), [3, 2, 1])

    def test_rows_never_dropped_and_alias(self):
        scored = synthetic_scored()
        self.assertEqual(len(events.build_action_queue(scored)), len(scored))
        pd.testing.assert_frame_equal(events.prioritize_events(self.latest),
                                      events.build_action_queue(self.latest))
        self.assertEqual(len(events.build_action_queue(self.latest.iloc[0:0])), 0)


class InsightTests(unittest.TestCase):
    def setUp(self):
        self.scored = synthetic_scored()

    def test_content_and_determinism(self):
        lines = events.build_insights(self.scored)
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(s, str) for s in lines))
        self.assertEqual(lines, events.build_insights(self.scored.iloc[::-1].reset_index(drop=True)))
        text = "\n".join(lines)
        self.assertIn("model-estimated Pc", text)
        self.assertIn("Highest estimated risk: event test/4", text)
        self.assertIn("Closest predicted encounter: event test/2", text)
        self.assertIn("miss distance 0.800 km", text)
        self.assertIn("Events by object type: DEBRIS 3, PAYLOAD 2, ROCKET BODY 1, UNKNOWN 1.", text)
        self.assertIn("Rising estimate: event test/1, log10 Pc -7.00 -> -3.00 (+4.00)", text)
        self.assertIn("1 event(s) have no eligible pre-TCA message", text)
        self.assertEqual(events.generate_insights(self.scored), text)

    def test_rising_trend_without_tier_change_is_explained(self):
        frame = pd.DataFrame({"event_id": [1, 1], "time_to_tca": [1.0, 0.5],
                              "miss_distance": [9000.0, 9000.0]})
        scored = score_rows(frame, [-9.0, -7.0], source="test")
        text = "\n".join(events.build_insights(scored))
        self.assertIn("tier unchanged (safe)", text)
        self.assertIn("not by itself a tier change", text)

    def test_replay_insights_are_labelled(self):
        lines = events.build_insights(self.scored, horizon_hours=12)
        self.assertTrue(lines[0].startswith("Historical replay"))

    def test_empty(self):
        self.assertIn("No scored encounters available.", events.build_insights(self.scored.iloc[0:0]))


class RiskChangeTests(unittest.TestCase):
    def test_consecutive_deltas_in_chronological_order(self):
        frame = pd.DataFrame({"row_id": [0, 1, 2, 3, 4], "event_id": [1, 1, 1, 1, 2],
                              "time_to_tca": [0.5, 2.0, 1.0, 1.0, 0.3],
                              "miss_distance": [9000.0] * 5})
        scored = score_rows(frame, [-4.0, -8.0, -6.5, -6.0, -5.0], source="test")
        changes = events.compute_risk_changes(scored)
        # Event 1 chronological: row1 (2.0) -> row2 (1.0) -> row3 (1.0, higher row_id) -> row0 (0.5)
        self.assertEqual(changes["row_id"].tolist(), [2, 3, 0])
        self.assertEqual(changes["previous_row_id"].tolist(), [1, 2, 3])
        np.testing.assert_allclose(changes["delta_log10_pc"], [1.5, 0.5, 2.0])
        self.assertEqual(changes["trend"].tolist(), ["rising"] * 3)
        self.assertEqual(changes["tier_changed"].tolist(), [False, True, True])
        cut = events.compute_risk_changes(scored, replay_cutoff_days=1.0)
        self.assertEqual(cut["row_id"].tolist(), [2, 3])


class ExportTests(unittest.TestCase):
    def test_round_trip(self):
        scored = synthetic_scored()
        with tempfile.TemporaryDirectory() as tmp:
            path = events.export_scored_csv(scored, Path(tmp) / "out" / "scored.csv")
            raw = pd.read_csv(path)
            back = events.read_scored_csv(path)
        self.assertEqual(list(raw.columns), list(SCORED_COLUMNS))
        self.assertEqual(raw.loc[1, "alert_reasons"], '["log10_pc>=-4"]')
        self.assertEqual(raw.loc[0, "quality_flags"], "[]")
        for col in ("alert_reasons", "quality_flags", "alert_details"):
            self.assertEqual(back[col].tolist(), scored[col].tolist())
        self.assertEqual(back["risk_tier"].tolist(), scored["risk_tier"].tolist())
        np.testing.assert_array_equal(back["predicted_log10_pc"], scored["predicted_log10_pc"])


@unittest.skipIf(UI_CONTRACT is None or UI_BACKEND is None,
                 "dashboard ui package not available (set DASHBOARD_UI_PATH)")
class DashboardContractTests(unittest.TestCase):
    def test_scored_rows_pass_dashboard_validation(self):
        scored = synthetic_scored()
        self.assertEqual(list(OUTPUT_COLUMNS), UI_CONTRACT.SCORED_COLUMNS)
        df, report = UI_CONTRACT.validate_scored(scored)
        self.assertEqual(df["risk_tier"].tolist(), scored["risk_tier"].tolist())
        # Rows without a usable estimate must already be needs_review (no tier mismatch warning).
        self.assertFalse(any("not marked needs_review" in w for w in report.warnings), report.warnings)

    def test_dashboard_export_round_trip_keeps_tiers(self):
        scored = synthetic_scored()
        df, _ = UI_CONTRACT.validate_scored(scored)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dash.csv"
            path.write_bytes(UI_CONTRACT.to_export_csv(df))
            reloaded, _ = UI_BACKEND.load_scored_csv(str(path))
            mine = Path(tmp) / "mine.csv"
            events.export_scored_csv(scored, mine)
            reloaded_mine, _ = UI_BACKEND.load_scored_csv(str(mine))
        self.assertEqual(reloaded["risk_tier"].tolist(), scored["risk_tier"].tolist())
        self.assertEqual(reloaded_mine["risk_tier"].tolist(), scored["risk_tier"].tolist())
        self.assertEqual(reloaded_mine["alert_reasons"].tolist(), scored["alert_reasons"].tolist())

    def test_backend_uses_real_events_api(self):
        backend = UI_BACKEND.load_events_backend()
        self.assertEqual(backend.missing, [], backend.import_error)
        self.assertFalse(backend.provisional)
        self.assertEqual(UI_BACKEND.display_thresholds(), {
            "LOG10_PC_CRITICAL": -4.0, "LOG10_PC_ELEVATED": -6.0,
            "MISS_DISTANCE_CRITICAL_M": 1000.0, "MISS_DISTANCE_ELEVATED_M": 5000.0})

    def test_event_views_match_events_module(self):
        scored, _ = UI_CONTRACT.validate_scored(synthetic_scored())
        backend = UI_BACKEND.load_events_backend()
        thresholds = UI_BACKEND.display_thresholds()
        latest, queue, summary, insights = UI_BACKEND.event_views(scored, backend, thresholds)
        self.assertEqual(summary, {k: EXPECTED_24H[k] for k in UI_BACKEND.SUMMARY_KEYS})
        self.assertEqual(len(latest), 7)
        self.assertEqual(queue["priority_rank"].tolist(), list(range(1, 8)))
        self.assertEqual(insights, events.build_insights(scored))

    def test_event_views_replay_mode(self):
        scored, _ = UI_CONTRACT.validate_scored(synthetic_scored())
        backend = UI_BACKEND.load_events_backend()
        latest, _, summary, _ = UI_BACKEND.event_views(
            scored, backend, UI_BACKEND.display_thresholds(), replay_hours=12)
        self.assertTrue((latest["replay_tca_hours"] == 12).all())
        self.assertTrue((latest["data_mode"] == "historical_replay").all())
        # Replay basis: every selected event is 12 h from the aligned TCA.
        self.assertEqual(summary["total_encounters"], len(latest))

    def test_live_pipeline_scoring_call(self):
        # Mirrors ui/backend.run_live_pipeline: allowlisted columns + source/row_id, positional call.
        raw = pd.read_csv(FIXTURES / "telemetry.csv")
        frame = raw[UI_CONTRACT.ALLOWED_INPUT_COLUMNS].copy()
        frame.insert(0, "row_id", np.arange(len(frame), dtype="int64"))
        frame.insert(0, "source", "test")
        preds = np.linspace(-9.0, -3.0, len(frame))
        df, _ = UI_CONTRACT.validate_scored(pd.DataFrame(score_rows(frame, preds)))
        self.assertEqual(len(df), len(raw))


@unittest.skipIf(MODEL is None, "src.model (Teammate 1) not importable")
class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from threadpoolctl import threadpool_limits

        rng = np.random.default_rng(42)
        n = 80
        train = pd.DataFrame({c: rng.uniform(1, 10, n) for c in MODEL.FEATURES if c != "c_object_type"})
        train["miss_distance"] *= 1000.0
        train["c_object_type"] = np.tile(["DEBRIS", "PAYLOAD"], n // 2)
        # Synthetic training target only; never a real label and never a test input.
        target = -10.0 + train["miss_distance"] / 1500.0
        cls.pipeline = MODEL.make_pipeline(
            parameters=dict(MODEL.PARAMETERS, max_iter=5, min_samples_leaf=2))
        with threadpool_limits(limits=1):
            cls.pipeline.fit(train, target)
        cls.raw = pd.read_csv(FIXTURES / "telemetry.csv")

    def predict(self, frame):
        return MODEL.predict_log10_pc(frame, self.pipeline)

    def test_end_to_end_scoring(self):
        preds = self.predict(self.raw)
        out = score_rows(self.raw, preds, source="test", model_version="fixture-model")
        self.assertEqual(len(out), len(self.raw))
        self.assertEqual(out["row_id"].tolist(), list(range(len(self.raw))))
        self.assertTrue(np.isfinite(out["predicted_log10_pc"]).all())
        self.assertTrue(out["risk_tier"].isin(["safe", "elevated", "critical"]).all())
        latest = events.select_latest_messages(out)
        self.assertEqual(len(latest), self.raw["event_id"].nunique())

    def test_changing_or_dropping_risk_does_not_change_predictions(self):
        base_pred = self.predict(self.raw)
        base = score_rows(self.raw, base_pred, source="test", model_version="m")
        rng = np.random.default_rng(0)
        variants = {
            "risk=0": self.raw.assign(risk=0.0),
            "risk=random": self.raw.assign(risk=rng.uniform(-30, 0, len(self.raw))),
            "risk=-30": self.raw.assign(risk=-30.0),
        }
        for name, frame in variants.items():
            with self.subTest(name):
                pred = self.predict(frame)
                np.testing.assert_array_equal(pred, base_pred)
                pd.testing.assert_frame_equal(score_rows(frame, pred, source="test", model_version="m"), base)
                dropped = frame.drop(columns="risk")
                np.testing.assert_array_equal(self.predict(dropped), base_pred)

    def test_tiers_agree_with_model_rule_and_prediction_frame(self):
        grid_pc = np.array([-35, -30, -7, -6.000001, -6, -5, -4.000001, -4, -1, 0, 0.5, np.nan])
        grid_d = np.array([0, 999.999, 1000, 4999.9, 5000, 5000.1, 20000, np.nan, -1])
        pc, dist = [a.ravel() for a in np.meshgrid(grid_pc, grid_d)]
        frame = pd.DataFrame({"event_id": np.arange(len(pc)), "time_to_tca": 1.0, "miss_distance": dist})
        mine = score_rows(frame, pc, source="test")
        theirs = MODEL.risk_tier(np.clip(pc, -30, 0), dist)
        np.testing.assert_array_equal(mine["risk_tier"].to_numpy(), theirs)

        bundle = {"pipeline": self.pipeline, "metadata": {"model_version": "fixture-model"}}
        exported = MODEL.prediction_frame(self.raw, bundle, source="test")
        scored = score_rows(self.raw, self.predict(self.raw), source="test", model_version="fixture-model")
        self.assertEqual(exported["risk_tier"].tolist(), scored["risk_tier"].tolist())
        np.testing.assert_array_equal(exported["predicted_log10_pc"], scored["predicted_log10_pc"])
        np.testing.assert_array_equal(exported["predicted_pc"], scored["predicted_pc"])


if __name__ == "__main__":
    unittest.main()
