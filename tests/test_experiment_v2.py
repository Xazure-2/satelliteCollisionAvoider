import unittest
import numpy as np
import pandas as pd
from src.experiment_v2 import PhysicsFeatures, training_target, to_log_pc, best_threshold
from tests.test_model import fixture


class ExperimentV2Tests(unittest.TestCase):
    def test_both_sigma_placeholders_become_missing_with_flag(self):
        x = fixture(); x.loc[0, 'c_sigma_r'] = 63781363.; x.loc[1, 't_sigma_r'] = 63781361.; x.loc[2, 'c_sigma_r'] = 2e6
        t = PhysicsFeatures().fit_transform(x)
        self.assertTrue(np.isnan(t.loc[0, 'c_sigma_r'])); self.assertEqual(t.loc[0, 'c_sigma_placeholder'], 1)
        self.assertTrue(np.isnan(t.loc[1, 't_sigma_r'])); self.assertEqual(t.loc[1, 't_sigma_placeholder'], 1)
        self.assertEqual(t.loc[2, 'c_sigma_r'], 2e6); self.assertEqual(t.loc[2, 'c_sigma_placeholder'], 0)
        self.assertNotIn('risk', t); self.assertNotIn('event_id', t)

    def test_space_weather_fills_only_from_earlier_messages(self):
        x = fixture().iloc[:4].copy()
        x['time_to_tca'] = [4., 3., 2., 1.]; x['F10'] = [np.nan, 100., np.nan, 120.]
        t = PhysicsFeatures().fit_transform(x)
        self.assertTrue(np.isnan(t.F10.iloc[0]))  # no earlier message: stays missing
        self.assertEqual(t.F10.iloc[2], 100.)

    def test_targets_and_upper_bound(self):
        frame = pd.DataFrame({'risk': [-30., -5., -3.], 'max_risk_estimate': [-10., -4., -2.]})
        floor = dict(target='floor', floor=-6.001); gap = dict(target='gap', floor=-8.)
        np.testing.assert_allclose(training_target(frame, floor), [-6.001, -5., -3.])
        np.testing.assert_allclose(training_target(frame, gap), [0., 1., 1.])
        np.testing.assert_allclose(to_log_pc(frame, np.array([-7., -1., -2.5]), floor), [-6.001, -4., -2.5])
        np.testing.assert_allclose(to_log_pc(frame, np.array([-1., 1., 0.5]), gap), [-8., -5., -2.5])

    def test_threshold_prefers_higher_cut_on_ties(self):
        self.assertEqual(best_threshold(np.array([True, False]), np.array([-3.5, -6.5])), -3.5)


if __name__ == '__main__':
    unittest.main()
