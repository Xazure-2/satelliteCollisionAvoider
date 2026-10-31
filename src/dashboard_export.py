"""Export model results in the Reflex dashboard's v2 contract (dashboard/orbital/data/contract.py).

Two tables, written to outputs/:

* event_predictions.csv: one row per (source, event_id), from the final-tier model at the cutoff CDM
  (the latest message with time_to_tca >= 2 days).
* cdm_history.csv: one row per CDM of those events, for trend charts and R/T/N geometry.

Mapping decisions:
- predicted_final_tier and predicted_final_miss_distance_m come from the final-tier model (basis
  "predicted"); p_safe/p_elevated/p_critical come from the calibrated probability layer
  (src/tier_probability.py). The model does not forecast final Pc, so predicted_final_log10_pc is empty.
- observed_tier is the tier implied by the observed miss distance at the cutoff CDM, the floor the
  model can only raise. Test `risk` is never read, so observed Pc is empty for test events.
- Held-out training events (source "train", split "holdout") are out of sample for both models and
  carry observed risk history and actual final outcomes for evaluation display only.
"""
import json

import numpy as np
import pandas as pd

from src.final_tier import HORIZON_DAYS, predict_events, tier_from
from src.model import prediction_frame
from src.risk import ACTIONS

TIER_NAME = {'Safe': 'safe', 'Elevated': 'elevated', 'Critical': 'critical'}


def cutoff_rows(messages):
    """Latest message with time_to_tca >= 2 days per event; ties broken by the later row."""
    early = messages.loc[messages.time_to_tca.ge(HORIZON_DAYS)]
    return (early.sort_values(['event_id', 'time_to_tca', 'row_id'], ascending=[True, True, False])
            .drop_duplicates('event_id').set_index('event_id'))


def reasons_for(cutoff_m, forecast_m, observed, predicted, pc_text, cuts_m):
    r = [f'Miss distance at the cutoff CDM is {cutoff_m / 1000:.2f} km ({observed} by distance)']
    if np.isfinite(forecast_m) and cuts_m:
        r.append(f'Model forecasts a final miss distance of {forecast_m / 1000:.2f} km ({predicted} final tier); '
                 f'alert thresholds on the forecast, tuned for recall: critical <= {cuts_m[0] / 1000:.2f} km, '
                 f'elevated <= {cuts_m[1] / 1000:.2f} km')
    if predicted != observed:
        r.insert(0, f'Forecast raises the tier from {observed} (observed at cutoff) to {predicted}')
    r.append(f'Estimated Pc at the cutoff CDM (row model): {pc_text}')
    return r


def export_tables(messages, row_bundle, tier_bundle, source, split, observed_risk=None, labels=None, prob_bundle=None):
    """Build (events, history) for one source. `messages` holds event_id, row_id and permitted telemetry only.

    observed_risk: optional Series of log10(Pc) aligned to messages (held-out training only).
    labels: optional final_labels() frame indexed by event_id (held-out training only).
    prob_bundle: optional tier-probability calibrator (src/tier_probability.py) for p_safe/p_elevated/p_critical.
    """
    if 'risk' in messages:
        raise ValueError('Pass permitted telemetry only; observed risk goes in observed_risk for held-out data.')
    if split == 'test' and (observed_risk is not None or labels is not None):
        raise ValueError('Test events carry no observed risk or final labels.')
    usable = messages.event_id.isin(messages.loc[messages.time_to_tca.ge(HORIZON_DAYS), 'event_id'])
    messages = messages.loc[usable]
    if observed_risk is not None:
        observed_risk = observed_risk.loc[usable]
    rows = prediction_frame(messages, row_bundle, source=source)
    rows.index = messages.index
    forecasts = predict_events(messages.loc[messages.time_to_tca.ge(HORIZON_DAYS)], tier_bundle).set_index('event_id')
    cut = cutoff_rows(messages)
    est = rows.set_index('row_id').reindex(cut.row_id)
    observed = tier_from(np.full(len(cut), -30.0), cut.miss_distance.to_numpy())  # distance only
    observed = np.array([TIER_NAME[t] for t in observed])
    predicted = forecasts.predicted_tier.map(TIER_NAME).reindex(cut.index).to_numpy()
    distance_model = 'predicted_final_miss_distance_m' in forecasts
    forecast_m = (forecasts.predicted_final_miss_distance_m.reindex(cut.index).to_numpy() if distance_model
                  else np.full(len(cut), np.nan))
    pc_text = [('< 1e-6' if p <= -6.001 + 1e-9 else f'{10 ** p:.1e}') if np.isfinite(p) else 'unavailable'
               for p in est.predicted_log10_pc.to_numpy()]
    flags = [json.loads(f) for f in est.quality_flags]
    # Change-regressor thresholds are -log10(forecast final distance in metres).
    cuts_m = [10 ** -t for t in tier_bundle['metadata']['thresholds']] if distance_model else None
    events = pd.DataFrame({
        'source': source, 'event_id': cut.index.to_numpy(), 'split': split,
        'cutoff_time_to_tca_days': cut.time_to_tca.to_numpy(), 'cutoff_row_id': cut.row_id.to_numpy(),
        'n_cdms_used': forecasts.message_count.reindex(cut.index).to_numpy(),
        'predicted_final_tier': predicted, 'model_version': tier_bundle['metadata']['model_version'],
        'c_object_type': cut.c_object_type.fillna('UNKNOWN').astype(str).to_numpy(),
        'observed_miss_distance_m': cut.miss_distance.to_numpy(), 'observed_tier': observed,
        'predicted_final_miss_distance_m': forecast_m,
        'final_miss_distance_basis': 'predicted' if distance_model else None,
        'alert_reasons': [json.dumps(reasons_for(c, f, o, p, t, cuts_m)) for c, f, o, p, t in
                          zip(cut.miss_distance.to_numpy(), forecast_m, observed, predicted, pc_text)],
        'quality_flags': [json.dumps(f) for f in flags],
        'recommended_action': [ACTIONS[t] for t in predicted],
    })
    for col in ['p_safe', 'p_elevated', 'p_critical']:
        if col in forecasts:
            events[col] = forecasts[col].reindex(cut.index).to_numpy()
    if prob_bundle is not None:
        from src.final_tier import event_features
        from src.tier_probability import predict_probabilities
        proba = predict_probabilities(event_features(messages), tier_bundle, prob_bundle).reindex(cut.index)
        events['p_safe'], events['p_elevated'], events['p_critical'] = (
            proba.Safe.to_numpy(), proba.Elevated.to_numpy(), proba.Critical.to_numpy())
    if observed_risk is not None:
        by_row = pd.Series(observed_risk.to_numpy(), index=messages.row_id.to_numpy())
        events['observed_log10_pc'] = by_row.reindex(cut.row_id).to_numpy()
    if labels is not None:
        lab = labels.reindex(cut.index)
        events['actual_final_log10_pc'] = lab.final_risk.to_numpy()
        events['actual_final_miss_distance_m'] = lab.final_miss_distance.to_numpy()
        events['actual_final_tier'] = lab.final_tier.str.lower().to_numpy()
    history = pd.DataFrame({
        'source': source, 'event_id': messages.event_id.to_numpy(), 'row_id': messages.row_id.to_numpy(),
        'time_to_tca_days': messages.time_to_tca.to_numpy(), 'miss_distance_m': messages.miss_distance.to_numpy(),
        'relative_speed_mps': messages.relative_speed.to_numpy(),
        'relative_position_r_m': messages.relative_position_r.to_numpy(),
        'relative_position_t_m': messages.relative_position_t.to_numpy(),
        'relative_position_n_m': messages.relative_position_n.to_numpy(),
        'observed_log10_pc': observed_risk.to_numpy() if observed_risk is not None else np.nan,
        'used_for_prediction': messages.time_to_tca.ge(HORIZON_DAYS).to_numpy(),
        'c_object_type': messages.c_object_type.fillna('UNKNOWN').astype(str).to_numpy(),
        'quality_flags': rows.quality_flags.to_numpy(),
    })
    return events, history
