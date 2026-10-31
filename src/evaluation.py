"""Evaluation on rows and events, with event-cluster bootstrap intervals."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix
from src.model import latest_indices, pc_tier, risk_tier, TIERS


def detection(truth, scores, threshold=-4):
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pred = scores >= threshold
    tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
    fn, tn = int((truth & ~pred).sum()), int((~truth & ~pred).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, positives=tp+fn, predicted_positives=tp+fp,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f2=5*tp/(5*tp+4*fn+fp) if tp+fn else None,
                average_precision=float(average_precision_score(truth, scores)) if tp+fn else None)


def metrics(frame, prediction):
    y, p = frame.risk.to_numpy(), np.asarray(prediction)
    if not len(y):
        return dict(rows=0, events=0, mae=None, rmse=None, pc_critical=detection([], []))
    error = p-y
    return dict(rows=len(y), events=int(frame.event_id.nunique()),
                mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                pc_critical=detection(y >= -4, p))


def evaluate(frame, prediction):
    p = np.asarray(prediction)
    result = metrics(frame, p)
    idx = latest_indices(frame)
    last, lastp = frame.iloc[idx], p[idx]
    result['latest_pc'] = metrics(last, lastp)
    result['latest_pc']['false_alerts_per_1000_events'] = 1000 * result['latest_pc']['pc_critical']['fp']/len(last) if len(last) else None
    true_tier = risk_tier(frame.risk, frame.miss_distance)
    pred_tier = risk_tier(np.clip(p, -30, 0), frame.miss_distance)
    # Binary scores are used only to count combined-rule detections; no probability interpretation.
    combined = detection(true_tier[idx] == 'critical', (pred_tier[idx] == 'critical').astype(float), threshold=.5)
    combined.pop('average_precision')
    combined['false_alerts_per_1000_events'] = 1000*combined['fp']/len(idx) if len(idx) else None
    result['latest_combined'] = combined
    result['tier_labels'] = TIERS
    result['pc_confusion_matrix'] = confusion_matrix(pc_tier(frame.risk), pc_tier(p), labels=TIERS).tolist()
    result['combined_confusion_matrix'] = confusion_matrix(true_tier, pred_tier, labels=TIERS).tolist()
    result['clipped_predictions'] = int(((p < -30) | (p > 0)).sum())
    return result


def slice_metrics(frame, prediction):
    masks = {'all': np.ones(len(frame), bool), 'floor_-30': frame.risk.eq(-30),
             'non_floor': frame.risk.ne(-30), 'risk_ge_-6': frame.risk.ge(-6),
             'risk_ge_-4': frame.risk.ge(-4), 'near_-6_+-0.5': frame.risk.sub(-6).abs().le(.5),
             'near_-4_+-0.5': frame.risk.sub(-4).abs().le(.5), 'test_horizon_ge_2_days': frame.time_to_tca.ge(2)}
    for label, mask in [('0_to_24h', frame.time_to_tca.between(0, 1)),
                        ('24_to_48h', frame.time_to_tca.gt(1) & frame.time_to_tca.le(2)),
                        ('48_to_96h', frame.time_to_tca.gt(2) & frame.time_to_tca.le(4)),
                        ('above_96h', frame.time_to_tca.gt(4))]:
        masks[f'time/{label}'] = mask
    for category in sorted(frame.c_object_type.fillna('__MISSING__').unique()):
        masks[f'object/{category}'] = frame.c_object_type.fillna('__MISSING__').eq(category)
    missing = frame[['c_sigma_r', 't_sigma_r', 'F10', 'AP', 'c_object_type']].isna().any(axis=1)
    extreme = frame.c_sigma_r.gt(1e6) | frame.t_sigma_r.gt(1e6)
    for name, flag in [('missing_optional', missing), ('extreme_sigma', extreme)]:
        masks[f'{name}/yes'], masks[f'{name}/no'] = flag, ~flag
    return {name: metrics(frame.loc[np.asarray(mask)], np.asarray(prediction)[np.asarray(mask)]) for name, mask in masks.items()}


def bootstrap_events(frame, prediction, repetitions=200, seed=42):
    """Resample whole events, retaining every message in each sampled event.

    Intervals describe metric sampling variability conditional on this fitted model;
    they are not physical Pc intervals or model-training uncertainty estimates.
    """
    y, p = frame.risk.to_numpy(), np.asarray(prediction)
    truth, pred = y >= -4, p >= -4
    stats = pd.DataFrame({'event_id': frame.event_id.to_numpy(), 'n': 1,
        'absolute': np.abs(y-p), 'squared': (y-p)**2,
        'tp': (truth & pred).astype(int), 'fp': (~truth & pred).astype(int), 'fn': (truth & ~pred).astype(int)})
    grouped = stats.groupby('event_id', sort=True).sum()
    idx = latest_indices(frame)
    latest = stats.iloc[idx].set_index('event_id').reindex(grouped.index)
    array = np.column_stack([grouped[['n','absolute','squared','tp','fp','fn']].to_numpy(),
                             latest[['tp','fp','fn']].to_numpy()])
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ['mae','rmse','pc_recall','pc_f2','latest_pc_recall','latest_pc_f2','latest_false_alerts_per_1000_events']}
    for _ in range(repetitions):
        totals = array[rng.integers(0,len(array),len(array))].sum(axis=0)
        n, absolute, squared, tp, fp, fn, ltp, lfp, lfn = totals
        values = [absolute/n, np.sqrt(squared/n), tp/(tp+fn) if tp+fn else np.nan,
                  5*tp/(5*tp+4*fn+fp) if tp+fn else np.nan,
                  ltp/(ltp+lfn) if ltp+lfn else np.nan,
                  5*ltp/(5*ltp+4*lfn+lfp) if ltp+lfn else np.nan, 1000*lfp/len(array)]
        for name, value in zip(samples, values):
            samples[name].append(value)
    result = {}
    for name, values in samples.items():
        valid = np.asarray(values)[np.isfinite(values)]
        result[name] = {'low_95': float(np.quantile(valid,.025)) if len(valid) else None,
                        'high_95': float(np.quantile(valid,.975)) if len(valid) else None,
                        'valid_replicates': len(valid), 'undefined_replicates': repetitions-len(valid)}
    return {'repetitions': repetitions, 'seed': seed, 'resampling_unit': 'whole event', 'intervals': result}


def select_candidate(results, names):
    best_rmse = min(results[n]['validation']['rmse'] for n in names)
    eligible = [n for n in names if results[n]['validation']['rmse'] <= 1.01*best_rmse]
    complexity = {'M1':0, 'M2':0, 'M3':1}
    def order(name):
        m = results[name]['validation']; f2 = m['pc_critical']['f2']
        return (-(f2 if f2 is not None else -1), m['pc_critical']['fp'], m['mae'], complexity.get(name,0), name)
    return min(eligible, key=order)
