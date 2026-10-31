"""Production per-message Pc estimator for the dashboard (row-level experiment v2 winner).

Physics features, target floored at -6.001 and predictions capped at `max_risk_estimate` (the maximum
Pc over covariance scaling), selected by grouped CV in reports/experiment_v2.md. Values at the floor
mean "below 1e-6" (Safe by Pc). Fitted on train+validation events only; the holdout is never used.
The final-tier event model (src/final_tier.py) is separate and drives the scored tier predictions.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from threadpoolctl import threadpool_limits

from src.model import ROOT, FEATURES, make_physics_pipeline

ARTIFACT = ROOT / 'artifacts/row_pc_model.joblib'
FLOOR = -6.001
MODEL_VERSION = 'row-pc-v2-F6'


class CappedLogPc(RegressorMixin, BaseEstimator):
    """Physics pipeline whose log10(Pc) output is floored and capped at max_risk_estimate."""

    def __init__(self, floor=FLOOR):
        self.floor = floor

    def fit(self, X, y):
        self.pipeline_ = make_physics_pipeline()
        self.pipeline_.fit(X, np.maximum(np.asarray(y, dtype=float), self.floor))
        return self

    def predict(self, X):
        raw = self.pipeline_.predict(X)
        upper = np.asarray(X['max_risk_estimate'], dtype=float)
        return np.maximum(np.minimum(raw, upper), self.floor)

    @property
    def named_steps(self):
        # prediction_frame / predict_log10_pc look up the feature and category steps here.
        return self.pipeline_.named_steps


def train(root=ROOT, path=None):
    from src.experiment import load_partitioned_data
    data, splits, manifest = load_partitioned_data(Path(root))
    pool = data.loc[data.partition.isin(['train', 'validation'])]
    model = CappedLogPc()
    with threadpool_limits(limits=4):
        model.fit(pool, pool.risk)
    bundle = {'pipeline': model, 'metadata': dict(
        model_version=MODEL_VERSION, floor=FLOOR, cap='max_risk_estimate', feature_order=FEATURES,
        training_rows=int(len(pool)), training_events=int(pool.event_id.nunique()),
        split_sha256=manifest['split_sha256'], selection='reports/experiment_v2.md (F6_floor_-6.001_physics)')}
    joblib.dump(bundle, path or ARTIFACT)
    return bundle


def load_row_model(path=None):
    # Only load trusted, locally produced joblib files.
    return joblib.load(path or ARTIFACT)
