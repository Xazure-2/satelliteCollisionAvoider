"""Final-tier follow-up: ensembles, stacking, nested hyperparameter search, row-level Pc features.

Every candidate uses the same 5 event folds as src/final_tier.py, development events only (the
holdout is never loaded), fold-validated thresholds, and the persistence floor (hybrid). Results are
compared with the current selection (HR1) and persistence using paired event bootstraps.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.experiment import write_json
from src.final_tier import (DISTANCE_GRID, HORIZON_DAYS, ROOT, assign_folds, candidates, evaluate_scores, fit_spec,
                            load, model_scores, oof_model_scores, paired_difference, persistence, table, fmt, matrix)
from src.model import make_physics_pipeline

SEARCH_SPACE = dict(learning_rate=[0.02, 0.05, 0.1], max_iter=[150, 300, 600], max_leaf_nodes=[7, 15, 31, 63],
                    min_samples_leaf=[10, 20, 50, 100], l2_regularization=[0.0, 1.0, 5.0], max_features=[0.5, 0.8, 1.0])
SEARCH_ITERATIONS = 12
INNER_FOLDS = 3
ROW_PC_FLOOR = -8.0
ROW_PC_COLUMNS = ['rowpc_latest', 'rowpc_max', 'rowpc_change']


def sample_configs(n=SEARCH_ITERATIONS, seed=42):
    rng = np.random.default_rng(seed)
    configs = [{}]  # the current defaults compete too
    while len(configs) < n:
        c = {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}
        c = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in c.items()}
        if c not in configs:
            configs.append(c)
    return configs


def with_params(spec, params):
    return dict(spec, params=params)


def nested_search(spec, dev, fold, grids):
    """Outer folds as usual; in each, pick hyperparameters by inner 3-fold CV (hybrid forecast-event F2)."""
    configs = sample_configs()
    scores = pd.DataFrame(index=dev.index, columns=['critical', 'elevated'], dtype=float)
    chosen = []
    for k in np.unique(fold):
        outer = dev.loc[fold != k]
        inner_fold = assign_folds(outer, INNER_FOLDS)
        best = None
        for params in configs:
            s = oof_model_scores(with_params(spec, params), outer, inner_fold)
            f2 = evaluate_scores(outer, s, inner_fold, grids, hybrid=True)[0]['forecast']['critical']['f2']
            if best is None or f2 > best[0]:
                best = (f2, params)
        chosen.append(dict(fold=int(k), inner_f2=best[0], params=best[1]))
        model = fit_spec(with_params(spec, best[1]), outer)
        scores.loc[fold == k] = model_scores(model, spec, dev.loc[fold == k]).to_numpy()
        print(f'  nested fold {k}: inner F2 {best[0]:.3f} with {best[1] or "defaults"}', flush=True)
    return scores, chosen


def calibrate(dev, scores, fold):
    """Map regressor scores to probabilities with per-fold logistic fits on the other folds' forecast events."""
    forecast = (dev.final_time_to_tca < HORIZON_DAYS).to_numpy()
    out = pd.DataFrame(index=dev.index, columns=['critical', 'elevated'], dtype=float)
    for k in np.unique(fold):
        fit = (fold != k) & forecast
        for col, positive in [('critical', dev.final_tier == 'Critical'), ('elevated', dev.final_tier != 'Safe')]:
            lr = LogisticRegression().fit(scores.loc[fit, [col]].to_numpy(), positive[fit])
            out.loc[fold == k, col] = lr.predict_proba(scores.loc[fold == k, [col]].to_numpy())[:, 1]
    return out


def stack(dev, base_scores, fold):
    """Logistic meta-learner on out-of-fold base scores + persistence + latest distance (Wolpert-style stacking)."""
    floor = persistence(dev, False)
    X = pd.concat([s.add_prefix(f'{n}_') for n, s in base_scores.items()], axis=1)
    X['floor_critical'] = (floor == 'Critical').astype(float)
    X['floor_elevated'] = (floor == 'Elevated').astype(float)
    X['log_latest_distance'] = np.log10(np.clip(dev.latest_miss_distance, 1, None))
    forecast = (dev.final_time_to_tca < HORIZON_DAYS).to_numpy()
    out = pd.DataFrame(index=dev.index, columns=['critical', 'elevated'], dtype=float)
    for k in np.unique(fold):
        fit = (fold != k) & forecast
        meta = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        meta.fit(X.loc[fit], dev.final_tier[fit])
        proba = pd.DataFrame(meta.predict_proba(X.loc[fold == k]), columns=meta.classes_)
        out.loc[fold == k, 'critical'] = proba['Critical'].to_numpy()
        out.loc[fold == k, 'elevated'] = (proba['Critical'] + proba['Elevated']).to_numpy()
    return out


def row_pc_features(messages, dev, fold):
    """Out-of-fold row-level log10(Pc) estimates on >= 2-day messages, aggregated per event.

    The row model is experiment v2's physics pipeline with the target floored at -8 and capped at
    max_risk_estimate, trained on >= 2-day messages of the other folds' events.
    """
    early = messages.loc[messages.event_id.isin(dev.index) & messages.time_to_tca.ge(HORIZON_DAYS)].copy()
    event_fold = pd.Series(fold, index=dev.index)
    row_fold = event_fold.reindex(early.event_id).to_numpy()
    prediction = np.empty(len(early))
    for k in np.unique(fold):
        model = make_physics_pipeline()
        with threadpool_limits(limits=4):
            model.fit(early.loc[row_fold != k], np.maximum(early.risk[row_fold != k], ROW_PC_FLOOR))
            raw = model.predict(early.loc[row_fold == k])
        prediction[row_fold == k] = np.maximum(np.minimum(raw, early.max_risk_estimate[row_fold == k]), ROW_PC_FLOOR)
    early['rowpc'] = prediction
    early = early.sort_values(['event_id', 'time_to_tca'], ascending=[True, False])
    g = early.groupby('event_id').rowpc
    return pd.DataFrame({'rowpc_latest': g.last(), 'rowpc_max': g.max(),
                         'rowpc_change': g.last() - g.first()}).reindex(dev.index)


def run(root=ROOT):
    root = Path(root)
    started = time.time()
    data, _, features, messages = load(root)
    dev = data.loc[data.partition.isin(['train', 'validation'])].copy()
    fold = assign_folds(dev)
    specs = candidates(features)
    log = lambda m: print(f'[{time.time() - started:5.0f}s] {m}', flush=True)

    base = {}
    for name in ['C3_history_balanced', 'F3_forecast_events_only', 'M3_median_imputed_indicators', 'R1_distance_change_regressor']:
        base[name] = oof_model_scores(specs[name], dev, fold)
        log(f'base {name}')
    r1_prob = calibrate(dev, base['R1_distance_change_regressor'], fold)

    scores, kinds = {}, {}
    scores['HR1 current selection'] = base['R1_distance_change_regressor']; kinds['HR1 current selection'] = 'distance'
    scores['HF3 forecast-only classifier'] = base['F3_forecast_events_only']
    scores['E1 mean(F3, R1 calibrated)'] = (base['F3_forecast_events_only'] + r1_prob) / 2
    scores['E2 mean(C3, F3, M3, R1 calibrated)'] = (base['C3_history_balanced'] + base['F3_forecast_events_only']
                                                   + base['M3_median_imputed_indicators'] + r1_prob) / 4
    scores['S1 logistic stack'] = stack(dev, {'C3': base['C3_history_balanced'], 'F3': base['F3_forecast_events_only'],
                                              'M3': base['M3_median_imputed_indicators'], 'R1': base['R1_distance_change_regressor']}, fold)
    log('ensembles and stack')

    chosen = {}
    scores['T1 R1 nested hyperparameter search'], chosen['R1'] = nested_search(specs['R1_distance_change_regressor'], dev, fold, DISTANCE_GRID)
    kinds['T1 R1 nested hyperparameter search'] = 'distance'
    log('nested search R1')
    scores['T2 F3 nested hyperparameter search'], chosen['F3'] = nested_search(specs['F3_forecast_events_only'], dev, fold, None)
    log('nested search F3')

    rowpc = row_pc_features(messages, dev, fold)
    dev_pc = dev.join(rowpc)
    log(f'row-level Pc features (missing for {int(rowpc.isna().any(axis=1).sum())} events)')
    for label, name, kind in [('P1 R1 + row-level Pc', 'R1_distance_change_regressor', 'distance'),
                              ('P2 F3 + row-level Pc', 'F3_forecast_events_only', None)]:
        spec = dict(specs[name], columns=specs[name]['columns'] + ROW_PC_COLUMNS)
        scores[label] = oof_model_scores(spec, dev_pc, fold)
        kinds[label] = kind
    log('row-level Pc candidates')

    results, predictions = {}, {}
    for name, s in scores.items():
        grids = DISTANCE_GRID if kinds.get(name) == 'distance' else None
        results[name], predictions[name] = evaluate_scores(dev, s, fold, grids, hybrid=True)
    forecast = (dev.final_time_to_tca < HORIZON_DAYS).to_numpy()
    truth = dev.final_tier[forecast]
    persist = persistence(dev, False)
    for name in results:
        results[name]['vs_current'] = paired_difference(truth, predictions[name][forecast], predictions['HR1 current selection'][forecast])
        results[name]['vs_persistence'] = paired_difference(truth, predictions[name][forecast], persist[forecast])
    results['P0 persistence'] = dict(forecast=evaluate_persistence(dev))
    out_dir = root / 'artifacts/final_tier'
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / 'ensemble_results.json', dict(results=results, chosen_hyperparameters=chosen,
                                                       search_space=SEARCH_SPACE, iterations=SEARCH_ITERATIONS))
    pd.DataFrame(predictions, index=dev.index).assign(fold=fold, forecast=forecast).to_csv(out_dir / 'ensemble_cv_predictions.csv')
    (root / 'reports/final_tier_ensembles.md').write_text(report(results, chosen, len(dev), int(forecast.sum())))
    log('done')
    return results


def evaluate_persistence(dev):
    from src.final_tier import tier_metrics
    forecast = (dev.final_time_to_tca < HORIZON_DAYS).to_numpy()
    return tier_metrics(dev.final_tier[forecast], persistence(dev, False)[forecast])


def report(results, chosen, n_dev, n_forecast):
    header = ['candidate', 'Critical recall', 'Critical precision', 'Critical F2', 'TP/FP/FN', 'macro-F1',
              'ΔF2 vs HR1 [95%]', 'Δmisses vs HR1 [95%]', 'ΔF2 vs persistence [95%]']
    ci = lambda d, k, n=3: f"{d[k]['low_95']:+.{n}f} to {d[k]['high_95']:+.{n}f}"
    rows = []
    for name, r in results.items():
        m, c = r['forecast'], r['forecast']['critical']
        extra = ([ci(r['vs_current'], 'critical_f2'), ci(r['vs_current'], 'critical_misses', 0), ci(r['vs_persistence'], 'critical_f2')]
                 if 'vs_current' in r else ['—', '—', '—'])
        rows.append([name, fmt(c['recall']), fmt(c['precision']), fmt(c['f2']), f"{c['tp']}/{c['fp']}/{c['fn']}",
                     fmt(m['macro_f1']), *extra])
    learned = {n: r for n, r in results.items() if 'vs_current' in r}
    best = max(learned, key=lambda n: (learned[n]['forecast']['critical']['f2'], learned[n]['forecast']['macro_f1']))
    chosen_text = '\n'.join(f"- {model} fold {c['fold']}: inner F2 {c['inner_f2']:.3f}, {c['params'] or 'defaults'}"
                            for model, cs in chosen.items() for c in cs)
    return f"""# Final-tier follow-up: ensembles, hyperparameter search, row-level Pc

Generated by `python -m src.final_tier_ensembles`. Development events only ({n_dev} events, {n_forecast} forecast events); the holdout is not loaded. Same 5 event folds, fold-validated thresholds and persistence floor as [final_tier_report.md](final_tier_report.md). See [literature_review.md](literature_review.md) for the motivation behind each idea.

Highest forecast-event Critical F2 here: **{best}** ({fmt(learned[best]['forecast']['critical']['f2'])}). Paired bootstrap intervals compare each candidate with the current selection (HR1) and with persistence on the same resampled events.

## Results (forecast events, hybrid with persistence floor)

{table(header, rows)}

- **E1/E2:** probability averages. The distance-change regressor's score is mapped to probabilities by per-fold logistic fits on the other folds.
- **S1:** logistic meta-learner on out-of-fold base scores (C3, F3, M3, R1), the persistence tier and latest distance, trained per fold on the other folds' forecast events.
- **T1/T2:** nested random search ({SEARCH_ITERATIONS} configurations including the defaults, inner {INNER_FOLDS}-fold CV inside each outer fold), so the reported score is not tuned on the events it is scored on.
- **P1/P2:** add out-of-fold row-level log10(Pc) estimates (experiment v2 physics model, target floored at {ROW_PC_FLOOR}, capped at `max_risk_estimate`) aggregated per event: latest, max, change.

## Hyperparameters chosen in the nested search

{chosen_text}

Best-config choices that vary across folds indicate a flat optimum: the defaults are about as good as anything in the search space.

Comparing more variants on the same folds raises the risk that the best CV score is luck; intervals that straddle zero against HR1 mean "no detectable difference".

Best candidate confusion matrix (forecast events):

{matrix(learned[best]['forecast']['confusion_matrix'])}
"""


if __name__ == '__main__':
    run()
