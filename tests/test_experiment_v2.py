import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model import PARAMETERS, PhysicsFeatures, make_physics_pipeline, pc_tier, risk_tier, prediction_frame
from src.experiment_v2 import training_target, to_log_pc, best_threshold
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

    def test_space_weather_fill_handles_duplicate_index(self):
        x = fixture().iloc[:4].copy(); x.index = [7, 7, 8, 8]
        x['time_to_tca'] = [4., 3., 2., 1.]; x['F10'] = [100., np.nan, np.nan, 120.]
        self.assertEqual(PhysicsFeatures().fit_transform(x).F10.tolist(), [100., 100., 100., 120.])

    def test_positive_log_pc_needs_review(self):
        self.assertEqual(pc_tier([0.5, 0., -5.]).tolist(), ['needs_review', 'critical', 'elevated'])
        self.assertEqual(risk_tier([0.5, 0.5], [6000, 999]).tolist(), ['needs_review', 'critical'])

    def test_prediction_frame_scores_invalid_rows_individually(self):
        x = fixture()
        with threadpool_limits(limits=1):
            model = make_physics_pipeline(parameters=dict(PARAMETERS, max_iter=3, min_samples_leaf=2)).fit(x, x.risk)
        bundle = {'pipeline': model, 'metadata': {'model_version': 'test'}}
        bad = x.copy()
        bad.loc[0, 'miss_distance'] = -1           # unusable distance: needs_review
        bad.loc[1, 'time_to_tca'] = np.nan          # unusable input, but distance proves critical
        bad.loc[1, 'miss_distance'] = 500.
        bad.loc[2, 'c_sigma_r'] = 63781361.
        out = prediction_frame(bad, bundle)
        self.assertEqual(len(out), len(x))
        self.assertTrue(np.isnan(out.predicted_log10_pc_raw[:2]).all())
        self.assertTrue(np.isfinite(out.predicted_log10_pc_raw[2:]).all())
        self.assertEqual(out.risk_tier[:2].tolist(), ['needs_review', 'critical'])
        self.assertIn('miss_distance_missing_or_invalid', out.quality_flags[0])
        self.assertIn('c_sigma_r_placeholder', out.quality_flags[2])
        valid = prediction_frame(x.iloc[3:], bundle)
        np.testing.assert_array_equal(valid.predicted_log10_pc_raw, out.predicted_log10_pc_raw[3:])

    def test_threshold_prefers_higher_cut_on_ties(self):
        self.assertEqual(best_threshold(np.array([True, False]), np.array([-3.5, -6.5])), -3.5)


if __name__ == '__main__':
    unittest.main()
