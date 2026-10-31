import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from src.model import PARAMETERS, prediction_frame, predict_log10_pc
from src.row_model import CappedLogPc, FLOOR
from tests.test_model import fixture


class RowModelTests(unittest.TestCase):
    def fitted(self, x):
        model = CappedLogPc()
        with threadpool_limits(limits=1):
            model.fit(x, x.risk)
        return model

    def test_predictions_are_floored_and_capped_at_max_risk(self):
        x = fixture(); x['risk'] = np.linspace(-30, -2, len(x)); x['max_risk_estimate'] = -3.
        x.loc[0, 'max_risk_estimate'] = -7.   # cap below the floor still floors
        p = predict_log10_pc(x, self.fitted(x))
        self.assertTrue((p >= FLOOR).all())
        self.assertTrue((p <= np.maximum(x.max_risk_estimate, FLOOR) + 1e-12).all())
        self.assertEqual(p[0], FLOOR)

    def test_export_and_serialization(self):
        x = fixture(); x['max_risk_estimate'] = -3.
        model = self.fitted(x)
        bundle = {'pipeline': model, 'metadata': {'model_version': 'test', 'floor': FLOOR}}
        out = prediction_frame(x.drop(columns='risk'), bundle)
        self.assertEqual(len(out), len(x))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.joblib'; joblib.dump(bundle, path)
            np.testing.assert_array_equal(out.predicted_log10_pc_raw, prediction_frame(x, joblib.load(path)).predicted_log10_pc_raw)


if __name__ == '__main__':
    unittest.main()
