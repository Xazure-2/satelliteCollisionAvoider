"""Row-level log10(Pc) estimation; IDs and labels never enter the feature matrix."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['time_to_tca', 'miss_distance', 'relative_speed', 'relative_position_r',
            'relative_position_t', 'relative_position_n', 'c_object_type', 'c_sigma_r',
            't_sigma_r', 'max_risk_estimate', 'F10', 'AP']
NUMERIC = [c for c in FEATURES if c != 'c_object_type']
OPTIONAL = {'c_object_type', 'c_sigma_r', 't_sigma_r', 'F10', 'AP'}
NONNEGATIVE = {'time_to_tca', 'miss_distance', 'relative_speed', 'c_sigma_r', 't_sigma_r', 'F10', 'AP'}
PARAMETERS = dict(loss='squared_error', learning_rate=0.05, max_iter=300,
                  max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                  early_stopping=False, random_state=42)
TIERS = ['safe', 'elevated', 'critical', 'needs_review']
# Placeholder "no covariance" sigma: 10 x Earth's equatorial radius (6,378,136.3 m).
# The files contain both 63,781,363 and 63,781,361, so match with a small tolerance.
SIGMA_PLACEHOLDER_M = 10 * 6378136.3
SIGMA_PLACEHOLDER_TOLERANCE_M = 10.0


def is_sigma_placeholder(values):
    return np.isclose(np.asarray(values, dtype=float), SIGMA_PLACEHOLDER_M, rtol=0, atol=SIGMA_PLACEHOLDER_TOLERANCE_M)


def numeric_names(use_max_risk=True, derived=False):
    names = [c for c in NUMERIC if use_max_risk or c != 'max_risk_estimate']
    names += [f'{c}_missing' for c in names]
    names += ['c_sigma_extreme', 't_sigma_extreme']
    if derived:
        names += ['abs_radial_position', 'combined_radial_sigma', 'radial_sigma_ratio']
    return names


def invalid_required(frame, use_max_risk=True):
    """Per-row reasons for unusable required telemetry; such rows are scored needs_review, never safe."""
    reasons = [[] for _ in range(len(frame))]
    for col in [c for c in FEATURES if c not in OPTIONAL and (use_max_risk or c != 'max_risk_estimate')]:
        if col not in frame:
            continue
        values = pd.to_numeric(frame[col], errors='coerce').astype(float)
        bad = ~np.isfinite(values)
        if col in NONNEGATIVE:
            bad |= values.lt(0)
        for i in np.flatnonzero(bad.to_numpy()):
            reasons[i].append(f'{col}_missing_or_invalid')
    return reasons


def clean_features(frame, use_max_risk=True, strict=True):
    """Coerce permitted telemetry. strict=True raises on invalid required values; otherwise they become NaN."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('Pass a pandas DataFrame with named telemetry columns.')
    columns = [c for c in FEATURES if use_max_risk or c != 'max_risk_estimate']
    missing = set(columns) - set(frame.columns) - OPTIONAL
    if missing:
        raise ValueError(f'Missing required telemetry columns: {sorted(missing)}')
    out = frame.reindex(columns=columns).copy()
    for col in [c for c in columns if c != 'c_object_type']:
        values = pd.to_numeric(out[col], errors='coerce').astype(float)
        bad = ~np.isfinite(values)
        if col in NONNEGATIVE:
            bad |= values.lt(0)
        if strict and col not in OPTIONAL and bad.any():
            raise ValueError(f'{col}: {int(bad.sum())} missing/invalid values; review these rows.')
        out[col] = values.mask(bad)
    out['c_object_type'] = out.c_object_type.fillna('__MISSING__').astype(str)
    return out


class TelemetryFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, derived=False, use_max_risk=True):
        self.derived = derived
        self.use_max_risk = use_max_risk

    def fit(self, X, y=None):
        clean_features(X, self.use_max_risk)
        self.feature_names_in_ = np.asarray([c for c in FEATURES if self.use_max_risk or c != 'max_risk_estimate'])
        return self

    def transform(self, X):
        out = clean_features(X, self.use_max_risk)
        for col in [c for c in out.columns if c != 'c_object_type']:
            out[f'{col}_missing'] = out[col].isna().astype(float)
        out['c_sigma_extreme'] = out.c_sigma_r.gt(1e6).astype(float)
        out['t_sigma_extreme'] = out.t_sigma_r.gt(1e6).astype(float)
        if self.derived:
            out['abs_radial_position'] = out.relative_position_r.abs()
            out['combined_radial_sigma'] = np.hypot(out.c_sigma_r, out.t_sigma_r)
            denominator = out.combined_radial_sigma.where(out.combined_radial_sigma.gt(0))
            out['radial_sigma_ratio'] = out.abs_radial_position / denominator
            out['radial_sigma_ratio'] = out.radial_sigma_ratio.replace([np.inf, -np.inf], np.nan)
        return out


def make_pipeline(imputation='median', derived=False, use_max_risk=True, dummy=None, parameters=None):
    if imputation not in {'median', 'native'}:
        raise ValueError('imputation must be median or native')
    numeric = SimpleImputer(strategy='median', keep_empty_features=True) if imputation == 'median' else 'passthrough'
    preprocessing = ColumnTransformer([
        ('numeric', numeric, numeric_names(use_max_risk, derived)),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['c_object_type']),
    ], remainder='drop', sparse_threshold=0)
    estimator = DummyRegressor(strategy=dummy) if dummy else HistGradientBoostingRegressor(**(parameters or PARAMETERS))
    return Pipeline([('features', TelemetryFeatures(derived, use_max_risk)),
                     ('preprocess', preprocessing), ('regressor', estimator)])


class PhysicsFeatures(TransformerMixin, BaseEstimator):
    """Experiment-v2 features: allowed telemetry plus geometry/uncertainty terms; IDs and labels never become predictors.

    Placeholder sigmas become missing with flags. event_id, when present, is used only to
    forward-fill daily space weather from the same event's earlier messages (larger time_to_tca).
    """

    def __init__(self, physics=True):
        self.physics = physics

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        out = clean_features(X)
        if 'event_id' in X:
            ids = X.event_id.to_numpy()
            order = np.lexsort((-X.time_to_tca.to_numpy(dtype=float), ids))
            filled = pd.DataFrame(out[['F10', 'AP']].to_numpy()[order]).groupby(ids[order]).ffill().to_numpy()
            weather = np.empty_like(filled)
            weather[order] = filled
            out[['F10', 'AP']] = weather
        for side in ['c', 't']:
            col = f'{side}_sigma_r'
            placeholder = is_sigma_placeholder(out[col])
            out[f'{side}_sigma_placeholder'] = placeholder.astype(float)
            out[col] = out[col].mask(placeholder)
        if self.physics:
            d = out.miss_distance
            sigma = np.hypot(out.c_sigma_r, out.t_sigma_r)
            sigma = sigma.where(sigma.gt(0))
            out['combined_radial_sigma'] = sigma
            out['miss_over_sigma'] = d / sigma
            out['radial_over_sigma'] = out.relative_position_r.abs() / sigma
            out['radial_fraction'] = out.relative_position_r.abs() / d.where(d.gt(0))
            # Alfano-style maximum Pc ~ R^2 / (e d^2); this backs out an implied log10(R^2) object-size proxy.
            out['log10_hbr2_proxy'] = out.max_risk_estimate + np.log10(np.e * d.where(d.gt(0)) ** 2)
        return out


def physics_numeric_columns(physics=True):
    base = ['time_to_tca', 'miss_distance', 'relative_speed', 'relative_position_r', 'relative_position_t',
            'relative_position_n', 'c_sigma_r', 't_sigma_r', 'max_risk_estimate', 'F10', 'AP',
            'c_sigma_placeholder', 't_sigma_placeholder']
    return base + (['combined_radial_sigma', 'miss_over_sigma', 'radial_over_sigma', 'radial_fraction',
                    'log10_hbr2_proxy'] if physics else [])


def make_physics_pipeline(physics=True, parameters=None):
    preprocessing = ColumnTransformer([
        ('numeric', 'passthrough', physics_numeric_columns(physics)),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['c_object_type']),
    ], remainder='drop', sparse_threshold=0)
    return Pipeline([('features', PhysicsFeatures(physics)), ('preprocess', preprocessing),
                     ('regressor', HistGradientBoostingRegressor(**(parameters or PARAMETERS)))])


def pc_tier(log_pc):
    values = np.asarray(log_pc, dtype=float)
    return np.select([~np.isfinite(values) | (values > 0), values >= -4, values >= -6],
                     ['needs_review', 'critical', 'elevated'], default='safe')


def risk_tier(log_pc, distance):
    p, d = np.asarray(log_pc, dtype=float), np.asarray(distance, dtype=float)
    vp, vd = np.isfinite(p) & (p <= 0), np.isfinite(d) & (d >= 0)
    return np.select([(vp & (p >= -4)) | (vd & (d < 1000)),
                      vp & vd & ((p >= -6) | (d <= 5000)), vp & vd],
                     ['critical', 'elevated', 'safe'], default='needs_review')


def latest_indices(frame):
    selected = frame.loc[np.isfinite(frame.time_to_tca) & frame.time_to_tca.ge(0)].copy()
    selected['_position'] = np.flatnonzero(np.isfinite(frame.time_to_tca) & frame.time_to_tca.ge(0))
    keys = ['source', 'event_id'] if 'source' in selected else ['event_id']
    return (selected.sort_values(keys + ['time_to_tca', 'row_id'],
                                 ascending=[True] * len(keys) + [True, False])
            .drop_duplicates(keys)['_position'].to_numpy())


def load_bundle(path=None):
    # Only load trusted, locally produced joblib files.
    return joblib.load(path or ROOT / 'artifacts/selected_model.joblib')


def predict_log10_pc(frame, model=None):
    """Return raw estimates in input order. Optional model avoids reloading for repeated calls."""
    pipeline = (load_bundle() if model is None else model)
    if isinstance(pipeline, dict):
        pipeline = pipeline['pipeline']
    if frame.empty:
        clean_features(frame, getattr(pipeline.named_steps['features'], 'use_max_risk', True))
        return np.empty(0, dtype=float)
    result = np.asarray(pipeline.predict(frame), dtype=float)
    if result.shape != (len(frame),) or not np.isfinite(result).all():
        raise ValueError('Model did not produce one finite estimate per input row.')
    return result


def prediction_frame(frame, bundle, source='test'):
    """Export table; rows with invalid required telemetry get NaN estimates and needs_review (or critical by distance)."""
    if 'event_id' not in frame or frame.event_id.isna().any():
        raise ValueError('Exports require a nonmissing event_id for every row.')
    invalid = invalid_required(frame)
    valid = np.array([not r for r in invalid], dtype=bool)
    raw = np.full(len(frame), np.nan)
    if valid.any():
        raw[valid] = predict_log10_pc(frame.loc[valid], bundle)
    clipped = np.clip(raw, -30, 0)
    prepared = clean_features(frame, strict=False)
    flags = [list(r) for r in invalid]
    for col in [c for c in NUMERIC if c in OPTIONAL]:
        for i in np.flatnonzero(prepared[col].isna()):
            flags[i].append(f'{col}_missing_or_invalid')
    for col in ['c_sigma_r', 't_sigma_r']:
        placeholder = is_sigma_placeholder(prepared[col])
        for i in np.flatnonzero(placeholder):
            flags[i].append(f'{col}_placeholder')
        for i in np.flatnonzero(prepared[col].gt(1e6) & ~placeholder):
            flags[i].append(f'{col}_extreme')
    categories = bundle['pipeline'].named_steps['preprocess'].named_transformers_['category'].categories_[0]
    for i in np.flatnonzero(~prepared.c_object_type.isin(categories)):
        flags[i].append('unseen_object_type')
    for i in np.flatnonzero(prepared.c_object_type.eq('__MISSING__')):
        flags[i].append('missing_object_type')
    was_clipped = np.isfinite(raw) & (raw != clipped)
    for i in np.flatnonzero(was_clipped):
        flags[i].append('prediction_clipped')
    import json
    distance = prepared.miss_distance.to_numpy()
    return pd.DataFrame({
        'source': source, 'row_id': frame['row_id'].to_numpy() if 'row_id' in frame else np.arange(len(frame)),
        'event_id': frame.event_id.to_numpy(), 'predicted_log10_pc_raw': raw,
        'predicted_log10_pc': clipped, 'predicted_pc': 10.0 ** clipped,
        'prediction_clipped': was_clipped,
        'pc_tier': pc_tier(clipped), 'risk_tier': risk_tier(clipped, distance),
        'miss_distance_m': distance, 'miss_distance_km': distance / 1000,
        'time_to_tca_days': prepared.time_to_tca.to_numpy(), 'tca_hours': prepared.time_to_tca.to_numpy() * 24,
        'c_object_type': prepared.c_object_type.to_numpy(),
        'quality_flags': [json.dumps(f) for f in flags], 'model_version': bundle['metadata']['model_version'],
    })
