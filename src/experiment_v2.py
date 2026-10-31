"""Second experiment: physics-informed features, floored/gap targets, grouped CV, validated alert threshold.

Selection uses grouped cross-validation on train+validation events only. The holdout is not
loaded here; any later holdout use is a disclosed second look (the M3 ledger stays untouched).
Headline metrics use rows with time_to_tca >= 2 days, matching every supplied test row.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from src.experiment import load_partitioned_data, write_json
from src.model import ROOT, PARAMETERS, SIGMA_PLACEHOLDER_M, make_physics_pipeline

CRITICAL, ELEVATED = -4.0, -6.0
HEADLINE_HORIZON_DAYS = 2.0
FOLDS = 5


# target: 'raw' log10(Pc); 'floor' = max(risk, floor); 'gap' = max_risk_estimate - max(risk, floor) >= 0.
CANDIDATES = {
    'R0_raw_m3_features': dict(physics=False, target='raw', floor=None),
    'R1_raw_physics': dict(physics=True, target='raw', floor=None),
    'F8_floor_-8_physics': dict(physics=True, target='floor', floor=-8.0),
    'F6_floor_-6.001_physics': dict(physics=True, target='floor', floor=-6.001),
    'G8_gap_floor_-8_physics': dict(physics=True, target='gap', floor=-8.0),
}


def training_target(frame, config):
    y = frame.risk.to_numpy(dtype=float)
    if config['floor'] is not None:
        y = np.maximum(y, config['floor'])
    if config['target'] == 'gap':
        y = np.maximum(frame.max_risk_estimate.to_numpy(dtype=float) - y, 0)
    return y


def to_log_pc(frame, raw, config):
    """Map model output to log10(Pc), never above the max-Pc upper bound or below the training floor."""
    upper = frame.max_risk_estimate.to_numpy(dtype=float)
    p = upper - np.maximum(raw, 0) if config['target'] == 'gap' else np.minimum(raw, upper)
    return np.maximum(p, config['floor']) if config['floor'] is not None else p


def detection(truth, score, threshold):
    truth, alert = np.asarray(truth, bool), np.asarray(score) >= threshold
    tp, fp, fn = int((truth & alert).sum()), int((~truth & alert).sum()), int((truth & ~alert).sum())
    return dict(tp=tp, fp=fp, fn=fn, positives=tp + fn,
                precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
                f2=5 * tp / (5 * tp + 4 * fn + fp) if tp + fn else None)


def score(frame, log_pc, alert_score=None, alert_threshold=CRITICAL):
    """Kelvins-style high-risk MSE (risk >= -6, predictions clipped at -6.001) plus tier detection."""
    y, p = frame.risk.to_numpy(), np.asarray(log_pc, dtype=float)
    high = y >= ELEVATED
    mse_hr = float(np.mean((np.maximum(p[high], -6.001) - y[high]) ** 2)) if high.any() else None
    crit = y >= CRITICAL
    a = p if alert_score is None else alert_score
    return dict(rows=len(y), events=int(frame.event_id.nunique()), mse_high_risk=mse_hr,
                critical=detection(crit, a, alert_threshold),
                critical_ap=float(average_precision_score(crit, p)) if crit.any() else None,
                elevated_or_worse=detection(high, p, ELEVATED))


def best_threshold(truth, log_pc, grid=np.arange(-7.0, -2.99, 0.05)):
    """Alert threshold on predicted log10(Pc) maximizing F2 for true Pc >= 1e-4; ties -> higher threshold."""
    best = max(grid, key=lambda t: ((detection(truth, log_pc, t)['f2'] or 0), t))
    return float(round(best, 2))


def run(root=ROOT, folds=FOLDS):
    root = Path(root)
    data, _, _ = load_partitioned_data(root)
    pool = data.loc[data.partition.isin(['train', 'validation'])].reset_index(drop=True)  # holdout excluded
    event_has_critical = pool.groupby('event_id').risk.transform('max').ge(CRITICAL).astype(int)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_of = np.empty(len(pool), int)
    for k, (_, idx) in enumerate(splitter.split(pool, event_has_critical, pool.event_id)):
        fold_of[idx] = k
    out_dir = root / 'artifacts/experiment_v2'
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = pool.time_to_tca.ge(HEADLINE_HORIZON_DAYS).to_numpy()
    oof = pd.DataFrame({'row_id': pool.row_id, 'event_id': pool.event_id, 'fold': fold_of, 'risk': pool.risk,
                        'time_to_tca': pool.time_to_tca, 'max_risk_estimate': pool.max_risk_estimate})
    oof['B2_max_risk_estimate'] = pool.max_risk_estimate.to_numpy()
    for name, config in CANDIDATES.items():
        start = time.time()
        prediction = np.empty(len(pool))
        for k in range(folds):
            fit, held = fold_of != k, fold_of == k
            pipeline = make_physics_pipeline(config['physics'])
            with threadpool_limits(limits=4):
                pipeline.fit(pool.loc[fit], training_target(pool.loc[fit], config))
                prediction[held] = to_log_pc(pool.loc[held], pipeline.predict(pool.loc[held]), config)
        oof[name] = prediction
        print(f'{name}: {time.time() - start:.0f}s', flush=True)
    oof.to_csv(out_dir / 'oof_predictions.csv', index=False)

    results = {}
    crit = pool.risk.ge(CRITICAL).to_numpy()
    for name in ['B2_max_risk_estimate', *CANDIDATES]:
        p = oof[name].to_numpy()
        # Validated alert threshold: chosen on the other folds' out-of-fold rows at the test horizon, applied to this fold.
        alert = np.zeros(len(p), bool)
        thresholds = []
        for k in range(folds):
            other, this = (fold_of != k) & horizon, fold_of == k
            t = best_threshold(crit[other], p[other])
            thresholds.append(t)
            alert[this] = p[this] >= t
        results[name] = dict(
            headline_ge_2_days=score(pool.loc[horizon], p[horizon]),
            headline_ge_2_days_tuned_alert=score(pool.loc[horizon], p[horizon], alert[horizon].astype(float), .5),
            all_rows=score(pool, p), fold_thresholds=thresholds,
            final_alert_threshold=best_threshold(crit[horizon], p[horizon]))
    selection = select(results)
    summary = dict(protocol=dict(folds=folds, grouping='event_id', stratified_on='event has risk >= -4',
                                 pool='train+validation partitions; holdout not loaded',
                                 headline=f'rows with time_to_tca >= {HEADLINE_HORIZON_DAYS} days',
                                 sigma_placeholder_m=SIGMA_PLACEHOLDER_M, parameters=PARAMETERS),
                   candidates=CANDIDATES, results=results, selection=selection)
    write_json(out_dir / 'results.json', summary)
    (root / 'reports/experiment_v2.md').write_text(report(summary))
    return summary


def select(results):
    """Highest out-of-fold F2 with the validated alert threshold at the test horizon; ties -> lower high-risk MSE."""
    learned = [n for n in results if n != 'B2_max_risk_estimate']
    key = lambda n: (results[n]['headline_ge_2_days_tuned_alert']['critical']['f2'] or 0,
                     -(results[n]['headline_ge_2_days']['mse_high_risk'] or np.inf))
    best = max(learned, key=key)
    baseline = results['B2_max_risk_estimate']['headline_ge_2_days_tuned_alert']['critical']['f2'] or 0
    return dict(selected=best, beats_max_risk_baseline=bool(key(best)[0] > baseline),
                rule='max out-of-fold F2 (validated alert threshold, time_to_tca >= 2 d); ties by lower high-risk MSE')


def fmt(v):
    return 'n/a' if v is None else f'{v:.3f}' if isinstance(v, float) else str(v)


def report(summary):
    rows = []
    for name, r in summary['results'].items():
        h, t = r['headline_ge_2_days'], r['headline_ge_2_days_tuned_alert']
        rows.append([name, fmt(h['mse_high_risk']), fmt(h['critical']['recall']), fmt(h['critical']['f2']),
                     fmt(t['critical']['recall']), fmt(t['critical']['precision']), fmt(t['critical']['f2']),
                     f"{t['critical']['tp']}/{t['critical']['fp']}/{t['critical']['fn']}",
                     fmt(h['critical_ap']), fmt(h['elevated_or_worse']['recall']), fmt(r['final_alert_threshold'])])
    header = ['candidate', 'MSE risk>=-6', 'recall@-4', 'F2@-4', 'recall@tuned', 'precision@tuned', 'F2@tuned',
              'TP/FP/FN@tuned', 'AP', 'recall@-6', 'alert threshold']
    table = '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + ' --- |' * len(header)] +
                      ['| ' + ' | '.join(r) + ' |' for r in rows])
    s, p = summary['selection'], summary['protocol']
    h = summary['results'][next(iter(summary['results']))]['headline_ge_2_days']
    return f"""# Experiment v2: physics features, floored targets, grouped CV

Selected **{s['selected']}**. Beats the `max_risk_estimate` baseline on the selection metric: **{s['beats_max_risk_baseline']}**.

Selection rule: {s['rule']}.

## Protocol

- {p['folds']}-fold grouped CV by `event_id`, stratified on whether an event has any `risk >= -4`; pool: {p['pool']}.
- Headline: {p['headline']} ({h['rows']} rows, {h['events']} events), matching every supplied test row.
- `@-4`: alert when predicted log10(Pc) >= -4. `@tuned`: alert threshold chosen on the other folds' out-of-fold predictions (F2 for true Pc >= 1e-4), applied to the held-out fold. The final threshold is refit on all out-of-fold rows; it is an alerting cut, not a change to the reported Pc.
- MSE uses only rows with true `risk >= -6`, with predictions clipped at -6.001 (Kelvins challenge convention).
- Every learned prediction is capped at `max_risk_estimate` (the maximum Pc over covariance scaling) and floored at the training floor.
- Placeholder sigmas ({p['sigma_placeholder_m']:.1f} m = 10 Earth radii, ±10 m) are set to missing with a flag. F10/AP are forward-filled from earlier messages of the same event.
- The M3 holdout ledger is untouched; this experiment never loads holdout rows.

## Out-of-fold results, time_to_tca >= 2 days

{table}

B2 is the one-line rule "Pc = max_risk_estimate". R0 uses the original M3 feature set with the new placeholder handling, cap and evaluation.

## Limitations

- Labels are calculated log10(Pc) per message, not observed collisions; no physical calibration is claimed.
- The derived features use radial sigmas only; full covariance and object size are not in the permitted columns.
- Fold thresholds vary; see `artifacts/experiment_v2/results.json`. A holdout check of the selected candidate would be a disclosed second look.
"""


if __name__ == '__main__':
    run()
