import sys
import unittest
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from src.dashboard_export import export_tables
from src.final_tier import event_features, final_labels, make_classifier, LATEST_ONLY
from src.row_model import CappedLogPc
from tests.test_model import fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dashboard'))
from orbital.data import contract  # noqa: E402


class DashboardExportTests(unittest.TestCase):
    def setUp(self):
        x = fixture(); x['time_to_tca'] = np.tile([5., 3., 1., 0.2], 20); x['max_risk_estimate'] = -3.
        with threadpool_limits(limits=1):
            row = CappedLogPc().fit(x, x.risk)
            tier = make_classifier(LATEST_ONLY).set_params(estimator__max_iter=3, estimator__min_samples_leaf=2) \
                .fit(event_features(x), np.tile(['Safe', 'Elevated', 'Critical', 'Safe'], 5))
        self.x = x
        self.row = {'pipeline': row, 'metadata': {'model_version': 'row', 'floor': -6.001}}
        self.tier = {'pipeline': tier, 'metadata': {'thresholds': [.5, .5], 'model_version': 'tier'}}

    def test_test_export_passes_dashboard_contract_without_labels(self):
        events, history = export_tables(self.x.drop(columns='risk'), self.row, self.tier, 'test', 'test')
        ev, _ = contract.validate_events(events)
        hist, _ = contract.validate_history(history)
        self.assertEqual(len(ev), 20)
        self.assertTrue((ev.cutoff_time_to_tca_days >= 2).all())
        self.assertTrue(ev.actual_final_tier.isna().all())
        self.assertTrue(hist.observed_log10_pc.isna().all())          # test risk never read
        self.assertEqual(int(hist.used_for_prediction.sum()), 40)      # the two >= 2-day CDMs per event
        self.assertEqual(contract.check_links(ev, hist), [])

    def test_holdout_export_carries_outcomes_and_rejects_leaks(self):
        events, _ = export_tables(self.x.drop(columns='risk'), self.row, self.tier, 'train', 'holdout',
                                  observed_risk=self.x.risk, labels=final_labels(self.x))
        ev, _ = contract.validate_events(events)
        self.assertTrue(ev.actual_final_tier.notna().all())
        cut = self.x[self.x.time_to_tca.eq(3.)].set_index('event_id').risk
        np.testing.assert_allclose(ev.set_index('event_id').observed_log10_pc.sort_index(), cut.sort_index())
        with self.assertRaises(ValueError):
            export_tables(self.x, self.row, self.tier, 'test', 'test')    # risk column present
        with self.assertRaises(ValueError):
            export_tables(self.x.drop(columns='risk'), self.row, self.tier, 'test', 'test', labels=final_labels(self.x))


if __name__ == '__main__':
    unittest.main()
