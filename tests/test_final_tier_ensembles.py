import unittest
import numpy as np
import pandas as pd
from src.final_tier_ensembles import sample_configs, calibrate, stack, SEARCH_SPACE


def dev_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    tier = np.tile(['Safe', 'Elevated', 'Critical'], n // 3)
    distance = np.where(tier == 'Critical', 500., np.where(tier == 'Elevated', 3000., 20000.)) * rng.uniform(.8, 1.2, n)
    return pd.DataFrame({'final_tier': tier, 'final_time_to_tca': 0.5, 'latest_miss_distance': distance,
                         'latest_max_risk_estimate': -10.})


class EnsembleTests(unittest.TestCase):
    def test_configs_are_unique_include_defaults_and_stay_in_space(self):
        configs = sample_configs(8)
        self.assertEqual(configs[0], {})
        self.assertEqual(len({tuple(sorted(c.items())) for c in configs}), 8)
        for c in configs[1:]:
            self.assertTrue(all(c[k] in SEARCH_SPACE[k] for k in c))

    def test_calibration_and_stack_are_out_of_fold_probabilities(self):
        dev = dev_frame()
        fold = (np.arange(len(dev)) // 3) % 3  # every fold holds all tiers
        score = -np.log10(dev.latest_miss_distance)
        raw = pd.DataFrame({'critical': score, 'elevated': score})
        prob = calibrate(dev, raw, fold)
        self.assertTrue(((prob >= 0) & (prob <= 1)).all().all())
        self.assertGreater(prob.critical[dev.final_tier == 'Critical'].mean(), prob.critical[dev.final_tier == 'Safe'].mean())
        stacked = stack(dev, {'R': raw}, fold)
        self.assertTrue((stacked.elevated >= stacked.critical - 1e-12).all())


if __name__ == '__main__':
    unittest.main()
