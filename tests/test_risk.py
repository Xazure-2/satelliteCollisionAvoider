import unittest
import numpy as np
import pandas as pd
from src.risk import assess, most_severe, score_rows, format_pc, PRIORITY
from src.model import risk_tier


class RiskRuleTests(unittest.TestCase):
    def test_threshold_boundaries(self):
        # (log_pc, distance) -> tier, including exact boundaries from the brief
        cases = [(-4, 9000, 'critical'), (-4.0001, 9000, 'elevated'), (-6, 9000, 'elevated'), (-6.0001, 9000, 'safe'),
                 (-30, 999.99, 'critical'), (-30, 1000, 'elevated'), (-30, 5000, 'elevated'), (-30, 5000.01, 'safe')]
        p, d, expected = zip(*cases)
        self.assertEqual(assess(p, d).risk_tier.tolist(), list(expected))

    def test_conflicting_triggers_keep_both_reasons(self):
        r = assess([-3.5], [500]).iloc[0]
        self.assertEqual(r.risk_tier, 'critical')
        self.assertEqual((r.pc_trigger, r.distance_trigger), ('critical', 'critical'))
        self.assertEqual(len(r.alert_reasons), 2)
        self.assertIn('Miss distance 0.50 km is below 1 km', r.alert_reasons)

    def test_invalid_inputs_never_default_to_safe(self):
        r = assess([np.nan, -10, 0.5, -10, np.nan], [9000, np.nan, 9000, -5, 500])
        self.assertEqual(r.risk_tier.tolist(), ['needs_review', 'needs_review', 'needs_review', 'needs_review', 'critical'])
        self.assertTrue(any('unavailable' in s for s in r.alert_reasons[4]))  # critical kept, quality warning too

    def test_matches_model_risk_tier(self):
        rng = np.random.default_rng(0)
        p = np.concatenate([rng.uniform(-9, 0.5, 400), [np.nan, -4, -6]])
        d = np.concatenate([rng.uniform(-100, 8000, 400), [np.nan, 1000, 5000]])
        self.assertEqual(assess(p, d).risk_tier.tolist(), risk_tier(p, d).tolist())

    def test_most_severe_and_priority(self):
        self.assertEqual(most_severe(['safe', 'elevated', 'critical', 'safe'], ['critical', 'safe', 'elevated', None]).tolist(),
                         ['critical', 'elevated', 'critical', 'safe'])
        self.assertLess(PRIORITY['needs_review'], PRIORITY['elevated'])

    def test_score_rows_contract_units_and_order(self):
        frame = pd.DataFrame({'event_id': [3, 1, 3], 'row_id': [10, 11, 12], 'time_to_tca': [2.5, 1.0, -0.1],
                              'miss_distance': [800., 12000., 3000.], 'relative_position_r': [1., 2., 3.],
                              'relative_position_t': [4., 5., 6.], 'relative_position_n': [7., 8., 9.],
                              'c_object_type': ['DEBRIS', None, 'PAYLOAD'], 'risk': [0., 0., 0.]})
        out = score_rows(frame, [-7., -6.001, -3.])
        self.assertEqual(out.row_id.tolist(), [10, 11, 12])
        np.testing.assert_allclose(out.tca_hours, [60., 24., -2.4])
        self.assertEqual(out.miss_distance_km.tolist(), [0.8, 12., 3.])
        self.assertEqual(out.risk_tier.tolist(), ['critical', 'safe', 'critical'])
        self.assertEqual(out.c_object_type[1], 'UNKNOWN')
        self.assertIn('after_tca_historical', out.quality_flags[2])
        self.assertNotIn('risk', out)
        self.assertEqual(format_pc(-6.001), '< 1e-6')


if __name__ == '__main__':
    unittest.main()
