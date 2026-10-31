import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.events import snapshot, event_table, window_counts, action_queue, risk_trends, build_tables
from src.risk import score_rows
from src.row_model import CappedLogPc
from src.final_tier import event_features, make_classifier, LATEST_ONLY
from tests.test_model import fixture


def scored_fixture():
    # Synthetic events (labelled synthetic): relative times chosen to test window edges at 0 and 1 day.
    frame = pd.DataFrame({
        'event_id': [1, 1, 2, 2, 3, 4, 5, 5],
        'row_id': [0, 1, 2, 3, 4, 5, 6, 7],
        'time_to_tca': [3.0, 1.0, 2.5, 0.0, 1.0001, -0.05, 0.5, 0.5],
        'miss_distance': [6000., 800., 9000., 4000., 300., 200., 7000., 6500.],
        'c_object_type': ['DEBRIS'] * 8})
    for a in 'rtn':
        frame[f'relative_position_{a}'] = 1.0
    return score_rows(frame, [-7., -7., -7., -7., -7., -3., -7., -5.])


class PipelineTests(unittest.TestCase):
    def test_snapshot_latest_pre_tca_message_with_ties_and_cutoff(self):
        s = scored_fixture()
        latest = snapshot(s).set_index('event_id')
        self.assertEqual(latest.loc[1, 'row_id'], 1)
        self.assertEqual(latest.loc[2, 'row_id'], 3)          # time_to_tca == 0 is pre-TCA
        self.assertNotIn(4, latest.index)                      # only an after-TCA message
        self.assertEqual(latest.loc[5, 'row_id'], 7)           # tie on time: later row wins
        replay = snapshot(s, cutoff_days=2.0).set_index('event_id')
        self.assertEqual(sorted(replay.index), [1, 2])         # nothing newer than the cutoff is selected
        self.assertAlmostEqual(replay.loc[1, 'message_age_days'], 1.0)

    def test_window_counts_hand_counted(self):
        events = event_table(scored_fixture())
        c = window_counts(events, 1.0)
        # in window: event 1 (1.0 d, 0.8 km -> critical), 2 (0.0 d, 4 km -> elevated), 5 (0.5 d, 6.5 km, Pc 1e-5 -> elevated)
        # event 3 at 1.0001 d is outside; event 4 has only an after-TCA message.
        self.assertEqual((c['encounters'], c['critical'], c['elevated'], c['below_1km_now']), (3, 1, 2, 1))
        self.assertEqual(c['critical_pc_driven'], 0)

    def test_queue_order_and_unique_events(self):
        q = action_queue(event_table(scored_fixture()))
        self.assertTrue(q[['source', 'event_id']].drop_duplicates().shape[0] == len(q))
        self.assertEqual(q.priority_tier.tolist()[:2], ['critical', 'critical'])
        self.assertEqual(q.event_id.tolist()[:2], [1, 3])     # same tier: sooner TCA first

    def test_forecast_raises_only_while_current(self):
        s = scored_fixture()
        forecasts = pd.DataFrame({'source': 'test', 'event_id': [1, 2], 'predicted_tier': ['Critical', 'Critical'],
                                  'persistence_tier': ['Elevated', 'Safe'], 'predicted_final_miss_distance_m': [700., 900.]})
        now = event_table(s, forecasts).set_index('event_id')
        self.assertFalse(now.loc[2, 'forecast_current'])       # newer messages supersede the forecast
        self.assertEqual(now.loc[2, 'priority_tier'], 'elevated')
        replay = event_table(s, forecasts, cutoff_days=2.0).set_index('event_id')
        self.assertTrue(replay.loc[2, 'forecast_current'])
        self.assertEqual(replay.loc[2, 'priority_tier'], 'critical')
        self.assertTrue(replay.loc[2, 'reasons'][0].startswith('Model forecasts a critical final tier'))

    def test_trends(self):
        _, rise = risk_trends(scored_fixture())
        self.assertAlmostEqual(rise.set_index('event_id').loc[5, 'largest_log_pc_rise'], 2.0)

    def test_end_to_end_ignores_risk_and_is_deterministic(self):
        x = fixture(); x['time_to_tca'] = np.tile([5., 3., 1., 0.2], 20); x['max_risk_estimate'] = -3.
        with threadpool_limits(limits=1):
            row = CappedLogPc().fit(x, x.risk)
            f = event_features(x)
            tier = make_classifier(LATEST_ONLY).set_params(estimator__max_iter=3, estimator__min_samples_leaf=2) \
                .fit(f, np.tile(['Safe', 'Elevated', 'Critical', 'Safe'], 5))
        row_bundle = {'pipeline': row, 'metadata': {'model_version': 'row-test', 'floor': -6.001}}
        tier_bundle = {'pipeline': tier, 'metadata': {'thresholds': [.5, .5], 'model_version': 'tier-test'}}
        a = build_tables(x, row_bundle, tier_bundle)
        b = build_tables(x.assign(risk=99.).sample(frac=1, random_state=1), row_bundle, tier_bundle)
        key = ['event_id']
        pd.testing.assert_frame_equal(a['events'].sort_values(key).reset_index(drop=True)[['event_id', 'priority_tier', 'predicted_log10_pc']],
                                      b['events'].sort_values(key).reset_index(drop=True)[['event_id', 'priority_tier', 'predicted_log10_pc']])
        self.assertEqual(a['counts'], b['counts'])
        self.assertEqual(len(a['rows']), len(x))
        self.assertTrue(all(isinstance(s, str) for s in a['insights']))


if __name__ == '__main__':
    unittest.main()
