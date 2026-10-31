import unittest
import numpy as np
import pandas as pd
from src.tier_probability import metrics, calibration_features, predict_probabilities, make_calibrator
from src.final_tier import TIER_NAMES


class TierProbabilityTests(unittest.TestCase):
    def test_metrics_align_columns_with_labels(self):
        truth = np.array(['Safe', 'Elevated', 'Critical'])
        perfect = pd.DataFrame(np.eye(3) * 0.998 + 0.001, columns=TIER_NAMES)
        swapped = perfect[['Critical', 'Elevated', 'Safe']].set_axis(TIER_NAMES, axis=1)
        good, bad = metrics(truth, perfect), metrics(truth, swapped)
        self.assertLess(good['log_loss'], 0.01)
        self.assertGreater(bad['log_loss'], 1.0)
        self.assertLess(good['brier'], 0.01)

    def test_calibration_features_use_floor_and_forecast(self):
        f = pd.DataFrame({'latest_miss_distance': [500., 3000., 9000.], 'latest_max_risk_estimate': [-5., -6., -9.],
                          'latest_time_to_tca': [2.1, 2.5, 3.0]})
        X = calibration_features(f, -np.log10([800., 4000., 20000.]))
        np.testing.assert_allclose(X.forecast_log_distance, np.log10([800., 4000., 20000.]))
        self.assertEqual(X.floor_critical.tolist(), [1., 0., 0.])
        self.assertEqual(X.floor_elevated.tolist(), [0., 1., 0.])

    def test_calibrator_refuses_other_model_version(self):
        tier = {'metadata': {'model_version': 'a', 'spec': {'kind': 'change_regressor'}}, 'pipeline': None}
        prob = {'metadata': {'tier_model_version': 'b'}, 'calibrator': make_calibrator()}
        with self.assertRaises(ValueError):
            predict_probabilities(pd.DataFrame(), tier, prob)


if __name__ == '__main__':
    unittest.main()
