import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from src.final_pc import FLOOR, fit, predict, metrics, spec, report
from src.final_tier import event_features
from tests.test_model import fixture


def events():
    x = fixture(); x['time_to_tca'] = np.tile([5., 3., 1., 0.2], 20)
    f = event_features(x)
    return x, f.assign(final_risk=np.linspace(-30, -3, len(f)))


class FinalPcTests(unittest.TestCase):
    def test_forecast_is_floored_and_never_positive(self):
        x, f = events()
        s = dict(spec(event_features(x)), params={'max_iter': 5, 'min_samples_leaf': 2})
        with threadpool_limits(limits=1):
            model = fit(f, s['columns'], s)
        p = predict(model, f)
        self.assertTrue((p >= FLOOR).all() and (p <= 0).all())
        self.assertEqual(len(p), len(f))

    def test_metrics_and_report(self):
        truth = np.array([-30., -7., -5., -3.5])
        pred = np.array([FLOOR, FLOOR, -5.5, -4.5])
        m = metrics(truth, pred)
        self.assertEqual(m['high_risk_events'], 2)
        self.assertEqual((m['elevated_or_worse']['tp'], m['elevated_or_worse']['fn']), (2, 0))
        self.assertEqual((m['critical']['tp'], m['critical']['fn']), (0, 1))
        text = report({'cv': {'model': m, 'floor_everywhere': m, 'latest_max_risk_estimate': m}})
        self.assertIn('does not change the tier', text)


if __name__ == '__main__':
    unittest.main()
