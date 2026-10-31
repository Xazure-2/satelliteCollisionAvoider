"""Data-layer tests. Run from dashboard/:  python -m unittest discover -s tests -v"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from orbital.data import adapters, analytics, contract  # noqa: E402
from orbital.data.synthetic_encounters import _label_tier, make_synthetic  # noqa: E402
from orbital.data.synthetic_globe import make_population  # noqa: E402


class SyntheticDatasets(unittest.TestCase):
    def test_both_datasets_validate_and_differ(self):
        a, b = adapters.load("synthetic_a"), adapters.load("synthetic_b")
        self.assertTrue(a.synthetic and b.synthetic)
        self.assertNotEqual(len(a.events), len(b.events))
        self.assertNotEqual(analytics.tier_counts(a.events), analytics.tier_counts(b.events))
        self.assertTrue(a.model_version.startswith("SYNTHETIC"))

    def test_synthetic_tiers_follow_thresholds(self):
        for key in ("synthetic_a", "synthetic_b"):
            ev, hist = make_synthetic(key)
            for r in ev.to_dict("records"):
                self.assertEqual(r["observed_tier"], _label_tier(r["observed_log10_pc"], r["observed_miss_distance_m"]))
                if r["predicted_final_tier"] != "needs_review":
                    self.assertEqual(r["predicted_final_tier"],
                                     _label_tier(r["predicted_final_log10_pc"], r["predicted_final_miss_distance_m"]))

    def test_threshold_boundaries(self):
        self.assertEqual(_label_tier(-4.0, 9000), "critical")
        self.assertEqual(_label_tier(-6.0, 9000), "elevated")
        self.assertEqual(_label_tier(-10, 999.9), "critical")
        self.assertEqual(_label_tier(-10, 1000), "elevated")
        self.assertEqual(_label_tier(-10, 5000), "elevated")
        self.assertEqual(_label_tier(-10, 5000.1), "safe")

    def test_cutoff_and_history_rules(self):
        for key in ("synthetic_a", "synthetic_b"):
            ev, hist = make_synthetic(key)
            self.assertTrue((ev["cutoff_time_to_tca_days"] >= 2).all())
            used = hist[hist["used_for_prediction"]]
            self.assertTrue((used["time_to_tca_days"] >= 2).all())
            self.assertTrue(ev.loc[ev["split"] == "test", "actual_final_tier"].isna().all())
            test_keys = set(ev.loc[ev["split"] == "test", "event_id"])
            self.assertTrue((hist[hist["event_id"].isin(test_keys)]["time_to_tca_days"] >= 2).all())

    def test_observed_values_match_cutoff_cdm(self):
        ev, hist = make_synthetic("synthetic_a")
        by_row = hist.set_index("row_id")
        for r in ev.head(40).to_dict("records"):
            cdm = by_row.loc[r["cutoff_row_id"]]
            self.assertAlmostEqual(cdm["miss_distance_m"], r["observed_miss_distance_m"], places=6)
            self.assertAlmostEqual(cdm["time_to_tca_days"], r["cutoff_time_to_tca_days"], places=6)


class ContractValidation(unittest.TestCase):
    def setUp(self):
        self.events, self.history = make_synthetic("synthetic_b")

    def test_missing_required_column(self):
        with self.assertRaisesRegex(contract.ContractError, "predicted_final_tier"):
            contract.validate_events(self.events.drop(columns=["predicted_final_tier"]))

    def test_early_cutoff_rejected(self):
        bad = self.events.copy()
        bad.loc[0, "cutoff_time_to_tca_days"] = 1.5
        with self.assertRaisesRegex(contract.ContractError, "< 2 days"):
            contract.validate_events(bad)

    def test_test_label_leak_rejected(self):
        bad = self.events.copy()
        bad.loc[bad["split"] == "test", "actual_final_tier"] = "safe"
        with self.assertRaisesRegex(contract.ContractError, "leakage"):
            contract.validate_events(bad)

    def test_unknown_tier_rejected(self):
        bad = self.events.copy()
        bad.loc[0, "predicted_final_tier"] = "orange"
        with self.assertRaisesRegex(contract.ContractError, "orange"):
            contract.validate_events(bad)

    def test_sparse_fields_warn_not_fail(self):
        sparse = self.events.drop(columns=["p_critical", "predicted_final_log10_pc_lower",
                                           "predicted_final_log10_pc_upper", "observed_tier"])
        df, report = contract.validate_events(sparse)
        self.assertTrue(any("uncertainty" in w for w in report.warnings))
        self.assertTrue(any("observed tier" in w for w in report.warnings))

    def test_history_used_under_two_days_rejected(self):
        bad = self.history.copy()
        bad.loc[0, "time_to_tca_days"] = 1.0
        bad.loc[0, "used_for_prediction"] = True
        with self.assertRaisesRegex(contract.ContractError, "under 2 days"):
            contract.validate_history(bad)


class RealAdapter(unittest.TestCase):
    def test_missing_outputs_is_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(adapters.DataUnavailable, "event_predictions.csv not found"):
                adapters.load_real(Path(d))

    def test_row_level_file_gets_specific_hint(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "test_predictions.csv").write_text("row_id,event_id,predicted_risk\n0,0,-5\n")
            with self.assertRaises(adapters.DataUnavailable) as ctx:
                adapters.load_real(Path(d))
            self.assertIn("row level", ctx.exception.title)

    def test_real_files_with_aliases_load(self):
        events, history = make_synthetic("synthetic_b")
        events = events.assign(source="test", model_version="m1").rename(
            columns={"predicted_final_tier": "final_risk_tier"})
        history = history.assign(source="test").rename(columns={"miss_distance_m": "miss_distance"})
        for frame in (events, history):
            for col in ("alert_reasons", "quality_flags"):
                if col in frame:
                    frame[col] = frame[col].map(json.dumps)
        with tempfile.TemporaryDirectory() as d:
            events.to_csv(Path(d) / "event_predictions.csv", index=False)
            history.to_csv(Path(d) / "cdm_history.csv", index=False)
            ds = adapters.load_real(Path(d))
        self.assertFalse(ds.synthetic)
        self.assertEqual(len(ds.events), len(events))

    def test_invalid_real_file_is_not_replaced_by_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "event_predictions.csv").write_text("event_id\n1\n")
            (Path(d) / "cdm_history.csv").write_text("event_id\n1\n")
            with self.assertRaisesRegex(adapters.DataUnavailable, "v2 contract"):
                adapters.load_real(Path(d))


class Analytics(unittest.TestCase):
    def setUp(self):
        self.ds = adapters.load("synthetic_a")

    def test_formatting(self):
        self.assertEqual(analytics.fmt_pc(-30.0), "≈ 0")
        self.assertEqual(analytics.fmt_pc(-4.0), "1.0e-04")
        self.assertEqual(analytics.fmt_pc(float("nan")), "—")
        self.assertEqual(analytics.fmt_countdown(2.5), "T−2 d 12 h")
        self.assertEqual(analytics.fmt_m(1234.4), "1,234 m")

    def test_filters_change_counts(self):
        ev = self.ds.events
        all_counts = analytics.tier_counts(analytics.apply_filters(ev, analytics.Filters()))
        crit = analytics.apply_filters(ev, analytics.Filters(tiers=("critical",)))
        self.assertEqual(len(crit), all_counts["critical"])
        narrow = analytics.apply_filters(ev, analytics.Filters(tca_min=2.0, tca_max=3.0))
        self.assertTrue((narrow["cutoff_time_to_tca_days"] < 3.0 + 1e-9).all())
        one = analytics.apply_filters(ev, analytics.Filters(search="17"))
        self.assertTrue(all("17" in str(e) for e in one["event_id"]))

    def test_windows_partition_scope(self):
        test = analytics.scope_frame(self.ds.events, "test")
        windows = analytics.warning_windows(test)
        self.assertEqual(sum(w["total"] for w in windows), len(test))

    def test_priority_queue_order(self):
        test = analytics.scope_frame(self.ds.events, "test")
        q = analytics.sort_queue(test, "priority", False)
        ranks = q["predicted_final_tier"].map(analytics.QUEUE_ORDER).tolist()
        self.assertEqual(ranks, sorted(ranks))

    def test_validation_metrics_consistent(self):
        m = analytics.validation_metrics(self.ds.events)
        self.assertEqual(sum(map(sum, m["matrix"])), m["n"])
        self.assertEqual(m["tp"] + m["fn"], m["critical_actual"])
        self.assertEqual(len(m["missed"]), m["fn"])
        self.assertTrue(0 <= m["accuracy"] <= 1)

    def test_transitions(self):
        self.assertEqual(analytics.transition("safe", "critical"), "escalation")
        self.assertEqual(analytics.transition("critical", "safe"), "de-escalation")
        self.assertEqual(analytics.transition(None, "safe"), "unknown")


class DecisionSupport(unittest.TestCase):
    def ev(self, **kw):
        base = {"predicted_final_tier": "safe", "observed_tier": "safe", "predicted_final_log10_pc": -8.0,
                "predicted_final_miss_distance_m": 9000.0, "final_miss_distance_basis": "predicted",
                "predicted_final_at_floor": False, "quality_flags": []}
        base.update(kw)
        return base

    def test_pc_trigger_explained(self):
        t = analytics.threshold_checks(self.ev(predicted_final_tier="critical", predicted_final_log10_pc=-3.5))
        self.assertEqual([r["triggered"] for r in t["rows"]], ["yes", "no"])
        self.assertIn("CRITICAL because predicted final Pc", t["summary"])
        self.assertTrue(t["consistent"])

    def test_distance_boundaries(self):
        crit = analytics.threshold_checks(self.ev(predicted_final_tier="critical", predicted_final_miss_distance_m=999))
        self.assertEqual(crit["rows"][1]["level"], "critical")
        at_1000 = analytics.threshold_checks(self.ev(predicted_final_tier="elevated", predicted_final_miss_distance_m=1000))
        self.assertEqual(at_1000["rows"][1]["level"], "elevated")
        at_5000 = analytics.threshold_checks(self.ev(predicted_final_tier="elevated", predicted_final_miss_distance_m=5000))
        self.assertEqual(at_5000["rows"][1]["level"], "elevated")

    def test_inconsistent_tier_is_reported_not_changed(self):
        t = analytics.threshold_checks(self.ev(predicted_final_tier="safe", predicted_final_log10_pc=-3.0))
        self.assertFalse(t["consistent"])

    def test_missing_values_not_invented(self):
        t = analytics.threshold_checks(self.ev(predicted_final_log10_pc=float("nan")))
        self.assertEqual(t["rows"][0]["value"], "not provided")
        r = analytics.reliability(self.ev())
        self.assertEqual(r["level"], "not assessed")
        self.assertEqual(r["probability"], "—")

    def test_reliability_from_probability_and_interval(self):
        r = analytics.reliability(self.ev(predicted_final_tier="critical", p_critical=0.9,
                                          predicted_final_log10_pc_lower=-4.5, predicted_final_log10_pc_upper=-3.0))
        self.assertEqual(r["crosses"], ["1e-4"])
        self.assertEqual(r["level"], "moderate")  # high probability, but the range spans a threshold

    def test_next_steps(self):
        rel = {"level": "high", "crosses": []}
        self.assertEqual(analytics.next_step(self.ev(predicted_final_tier="critical"), rel)["key"], "refer")
        self.assertEqual(analytics.next_step(self.ev(predicted_final_tier="needs_review"), rel)["key"], "request")
        self.assertEqual(analytics.next_step(self.ev(predicted_final_tier="elevated"), rel)["key"], "request")
        self.assertEqual(analytics.next_step(self.ev(), rel)["key"], "monitor")
        self.assertEqual(analytics.next_step(self.ev(quality_flags=["x"]), rel)["key"], "request")
        text = " ".join(analytics.next_step(self.ev(predicted_final_tier="critical"), rel)["reasons"]).lower()
        self.assertNotIn("perform", text)
        self.assertIn("does not", text)

    def test_every_synthetic_event_is_explainable(self):
        for key in ("synthetic_a", "synthetic_b"):
            ds = adapters.load(key)
            for ev in ds.events.to_dict("records"):
                t = analytics.threshold_checks(ev)
                self.assertTrue(t["consistent"], (key, ev["event_id"]))
                analytics.next_step(ev, analytics.reliability(ev))


class GlobePopulation(unittest.TestCase):
    def test_population_is_synthetic_and_valid(self):
        objs = make_population(3000)
        self.assertGreaterEqual(len(objs), 3000)
        self.assertTrue(all(o["id"].startswith("SIM-") for o in objs))
        self.assertTrue(all(-90 <= o["lat"] <= 90 and -180 <= o["lng"] <= 180 for o in objs))
        pairs = [o for o in objs if o["partner"]]
        self.assertTrue(pairs and all(o["proximity"] in ("close", "near") for o in pairs))
        self.assertFalse(any("event_id" in o or "pc" in o for o in objs))
        self.assertTrue(all(math.isfinite(o["alt_km"]) for o in objs))

    def test_deterministic(self):
        self.assertEqual(make_population(500)[:5], make_population(500)[:5])


if __name__ == "__main__":
    np.seterr(all="ignore")
    unittest.main()
