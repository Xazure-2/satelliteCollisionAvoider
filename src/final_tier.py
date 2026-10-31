"""Event-level task: predict each event's final tier from messages with time_to_tca >= 2 days.

Organizer definition (data/challenge-1-event-labels.xlsx): the final message is the event's smallest
time_to_tca; its risk and miss distance give the tier (Critical, then Elevated, then Safe). Only the
permitted telemetry from messages at least 2 days before closest approach enters the features;
`risk` is never an input. Model selection uses grouped CV on the train+validation partitions of the
frozen event split; the holdout partition (15% of events) is evaluated once, guarded by a ledger.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from src.experiment import sha256, write_json
from src.model import FEATURES, PARAMETERS, ROOT, PhysicsFeatures, physics_numeric_columns

TIER_NAMES = ['Safe', 'Elevated', 'Critical']
HORIZON_DAYS = 2.0
FOLDS = 5
CLASSIFIER = dict({k: v for k, v in PARAMETERS.items() if k != 'loss'}, loss='log_loss')
TREND_COLUMNS = ['miss_distance', 'max_risk_estimate', 'relative_position_r', 'relative_position_t',
                 'relative_position_n', 'c_sigma_r', 't_sigma_r']


def tier_from(log_pc, distance_m):
    """Organizer rules; log_pc = -30 is the 'too small to compute' placeholder (Pc ~ 0)."""
    r, d = np.asarray(log_pc, float), np.asarray(distance_m, float)
    return np.select([(r >= -4) | (d < 1000), (r >= -6) | (d <= 5000)], ['Critical', 'Elevated'], 'Safe')


def final_labels(messages):
    """One row per event from its last message (smallest time_to_tca, which may be after TCA)."""
    last = messages.loc[messages.groupby('event_id').time_to_tca.idxmin()]
    usable = messages.loc[messages.time_to_tca.ge(HORIZON_DAYS)].event_id.unique()
    return pd.DataFrame({'event_id': last.event_id.to_numpy(), 'final_time_to_tca': last.time_to_tca.to_numpy(),
                         'final_risk': last.risk.to_numpy(), 'final_miss_distance': last.miss_distance.to_numpy(),
                         'final_tier': tier_from(last.risk, last.miss_distance),
                         'usable': last.event_id.isin(usable).to_numpy()}).set_index('event_id')


def check_against_sheet(labels, path):
    sheet = pd.read_excel(path, sheet_name='train_event_labels', header=2).set_index('event_id')
    if len(sheet) != len(labels) or not sheet.index.sort_values().equals(labels.index.sort_values()):
        raise ValueError('Organizer label sheet covers different events than train_data.csv.')
    if not (sheet.final_tier.reindex(labels.index) == labels.final_tier).all():
        raise ValueError('Organizer final_tier differs from tiers recomputed from train_data.csv.')
    if not ((sheet.usable_for_training.reindex(labels.index) == 'yes') == labels.usable).all():
        raise ValueError('Organizer usable_for_training differs from the 2-day eligibility rule.')


def event_features(messages):
    """Aggregate permitted telemetry of messages with time_to_tca >= 2 days into one row per event."""
    early = messages.loc[messages.time_to_tca.ge(HORIZON_DAYS)].copy()
    if early.empty:
        raise ValueError('No messages at least 2 days before closest approach.')
    early = early.sort_values(['event_id', 'time_to_tca'], ascending=[True, False], kind='mergesort')
    rows = PhysicsFeatures().fit_transform(early)
    rows['event_id'] = early.event_id.to_numpy()
    numeric = physics_numeric_columns()
    group = rows.groupby('event_id', sort=True)
    latest = group[numeric + ['c_object_type']].last()
    out = latest.add_prefix('latest_')
    first = group[numeric].first()
    out['message_count'] = group.size()
    out['observed_span_days'] = first.time_to_tca - latest.time_to_tca
    for col in ['miss_distance', 'max_risk_estimate', 'miss_over_sigma']:
        out[f'min_{col}'] = group[col].min()
        out[f'max_{col}'] = group[col].max()
    out['std_miss_distance'] = group.miss_distance.std()
    t = rows.time_to_tca
    for col in TREND_COLUMNS:
        # Least-squares slope per day toward TCA (positive = value grows as TCA approaches).
        y = rows[col]
        ok = y.notna()
        g = pd.DataFrame({'e': rows.event_id[ok], 'x': -t[ok], 'y': y[ok]})
        g['xy'], g['xx'] = g.x * g.y, g.x * g.x
        m = g.groupby('e')[['x', 'y', 'xy', 'xx']].mean()
        var = m.xx - m.x ** 2
        out[f'slope_{col}'] = ((m.xy - m.x * m.y) / var.where(var > 1e-9)).reindex(out.index)
        out[f'change_{col}'] = latest[col] - first[col]
    # Linear extrapolation of miss distance to TCA; a heuristic feature, not a propagation.
    out['extrapolated_miss_distance'] = (latest.miss_distance + out.slope_miss_distance * latest.time_to_tca).clip(lower=0)
    out['sigma_shrink_c'] = latest.c_sigma_r / first.c_sigma_r.where(first.c_sigma_r > 0)
    out['sigma_shrink_t'] = latest.t_sigma_r / first.t_sigma_r.where(first.t_sigma_r > 0)
    out['latest_distance_tier'] = pd.Categorical(tier_from(np.full(len(latest), -30.), latest.miss_distance),
                                                 categories=TIER_NAMES).codes.astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


LATEST_ONLY = [f'latest_{c}' for c in physics_numeric_columns()]


def make_classifier(columns, class_weight=None):
    preprocessing = ColumnTransformer([
        ('numeric', 'passthrough', columns),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['latest_c_object_type']),
    ], remainder='drop', sparse_threshold=0)
    return Pipeline([('preprocess', preprocessing),
                     ('classifier', HistGradientBoostingClassifier(**CLASSIFIER, class_weight=class_weight))])


LABEL_COLUMNS = {'final_tier', 'final_risk', 'final_miss_distance', 'final_time_to_tca', 'partition'}


def candidates(feature_frame):
    """feature_frame must come straight from event_features; label columns are rejected."""
    if LABEL_COLUMNS & set(feature_frame.columns):
        raise ValueError('Pass event_features output only; label columns must not become features.')
    all_numeric = [c for c in feature_frame.columns if c != 'latest_c_object_type']
    return {
        'C1_latest_message': dict(columns=LATEST_ONLY, class_weight=None),
        'C2_latest_plus_history': dict(columns=all_numeric, class_weight=None),
        'C3_history_balanced': dict(columns=all_numeric, class_weight='balanced'),
    }


def persistence(features, use_max_risk):
    """Baselines: apply the tier rules to the latest >= 2-day message (Pc proxy: max_risk_estimate, or ignored)."""
    pc = features.latest_max_risk_estimate if use_max_risk else np.full(len(features), -30.)
    return tier_from(pc, features.latest_miss_distance)


def ordered_probabilities(model, X):
    proba = pd.DataFrame(model.predict_proba(X), columns=model.classes_, index=X.index)
    return proba.reindex(columns=TIER_NAMES, fill_value=0.0)


def decide(proba, critical_threshold, elevated_threshold):
    """Critical if P(Critical) >= t_c; else Elevated if P(Elevated or Critical) >= t_e; else Safe."""
    pc, pe = proba.Critical.to_numpy(), (proba.Critical + proba.Elevated).to_numpy()
    return np.select([pc >= critical_threshold, pe >= elevated_threshold], ['Critical', 'Elevated'], 'Safe')


def f2(truth, prediction):
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    tp, fp, fn = (truth & prediction).sum(), (~truth & prediction).sum(), (truth & ~prediction).sum()
    return 5 * tp / (5 * tp + 4 * fn + fp) if tp + fn else 0.0


def tune_thresholds(truth, proba, grid=np.round(np.arange(0.05, 0.951, 0.025), 3)):
    """Pick t_c maximizing Critical F2, then t_e maximizing (Elevated or worse) F2; ties -> higher threshold."""
    truth = np.asarray(truth)
    tc = max(grid, key=lambda t: (f2(truth == 'Critical', proba.Critical >= t), t))
    worse = truth != 'Safe'
    te = max(grid, key=lambda t: (f2(worse, decide(proba, tc, t) != 'Safe'), t))
    return float(tc), float(te)


def tier_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    crit_t, crit_p = truth == 'Critical', prediction == 'Critical'
    tp = int((crit_t & crit_p).sum())
    return dict(events=len(truth),
                critical=dict(tp=tp, fp=int((~crit_t & crit_p).sum()), fn=int((crit_t & ~crit_p).sum()),
                              positives=int(crit_t.sum()),
                              recall=tp / crit_t.sum() if crit_t.any() else None,
                              precision=tp / crit_p.sum() if crit_p.any() else None,
                              f2=f2(crit_t, crit_p)),
                elevated_or_worse_recall=float(((prediction != 'Safe') & (truth != 'Safe')).sum() / max((truth != 'Safe').sum(), 1)),
                macro_f1=float(f1_score(truth, prediction, labels=TIER_NAMES, average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(truth, prediction)),
                accuracy=float((truth == prediction).mean()),
                confusion_matrix=confusion_matrix(truth, prediction, labels=TIER_NAMES).tolist())


def bootstrap(truth, prediction, repetitions=1000, seed=42):
    """Event bootstrap (one row per event) for Critical recall/precision/F2 and macro-F1."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    rng, stats = np.random.default_rng(seed), {k: [] for k in ['critical_recall', 'critical_precision', 'critical_f2', 'macro_f1']}
    for _ in range(repetitions):
        i = rng.integers(0, len(truth), len(truth))
        m = tier_metrics(truth[i], prediction[i])
        for k, v in [('critical_recall', m['critical']['recall']), ('critical_precision', m['critical']['precision']),
                     ('critical_f2', m['critical']['f2']), ('macro_f1', m['macro_f1'])]:
            stats[k].append(np.nan if v is None else v)
    return {k: dict(low_95=float(np.nanquantile(v, .025)), high_95=float(np.nanquantile(v, .975))) for k, v in stats.items()}


def load(root=ROOT):
    root = Path(root)
    messages = pd.read_csv(root / 'data/train_data.csv', usecols=['event_id', *FEATURES, 'risk'])
    labels = final_labels(messages)
    sheet = root / 'data/challenge-1-event-labels.xlsx'
    if sheet.exists():
        check_against_sheet(labels, sheet)
    splits = pd.read_csv(root / 'artifacts/event_splits.csv').set_index('event_id').partition
    features = event_features(messages)
    data = features.join(labels[['final_tier', 'final_risk', 'final_miss_distance', 'final_time_to_tca']], how='inner')
    data['partition'] = splits.reindex(data.index)
    if data.partition.isna().any():
        raise ValueError('Usable events missing from the frozen event split.')
    return data, labels, features


def cross_validate(dev, specs, folds=FOLDS):
    """Out-of-fold probabilities; thresholds for each fold are tuned on the other folds' out-of-fold rows."""
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold = np.empty(len(dev), int)
    for k, (_, idx) in enumerate(kfold.split(dev, dev.final_tier)):
        fold[idx] = k
    results, oof = {}, {}
    for name, spec in specs.items():
        proba = pd.DataFrame(index=dev.index, columns=TIER_NAMES, dtype=float)
        for k in range(folds):
            model = make_classifier(spec['columns'], spec['class_weight'])
            with threadpool_limits(limits=4):
                model.fit(dev.loc[fold != k], dev.final_tier[fold != k])
                proba.loc[fold == k] = ordered_probabilities(model, dev.loc[fold == k]).to_numpy()
        prediction = np.empty(len(dev), object)
        thresholds = []
        for k in range(folds):
            tc, te = tune_thresholds(dev.final_tier[fold != k], proba.loc[fold != k])
            thresholds.append([tc, te])
            prediction[fold == k] = decide(proba.loc[fold == k], tc, te)
        results[name] = dict(cv=tier_metrics(dev.final_tier, prediction), fold_thresholds=thresholds,
                             thresholds=tune_thresholds(dev.final_tier, proba))
        oof[name] = proba
    for name, use in [('P0_persist_distance', False), ('P1_persist_distance_maxrisk', True)]:
        results[name] = dict(cv=tier_metrics(dev.final_tier, persistence(dev, use)))
    return results, oof


def select(results):
    """Highest CV Critical F2 among learned candidates; ties -> macro-F1."""
    learned = [n for n in results if n.startswith('C')]
    return max(learned, key=lambda n: (results[n]['cv']['critical']['f2'], results[n]['cv']['macro_f1']))


def run(root=ROOT, finalize=False):
    root = Path(root)
    out_dir = root / 'artifacts/final_tier'
    out_dir.mkdir(parents=True, exist_ok=True)
    data, labels, features = load(root)
    dev = data.loc[data.partition.isin(['train', 'validation'])]
    specs = candidates(features)
    results, oof = cross_validate(dev, specs)
    selected = select(results)
    spec = specs[selected]
    tc, te = results[selected]['thresholds']
    model = make_classifier(spec['columns'], spec['class_weight'])
    with threadpool_limits(limits=4):
        model.fit(dev, dev.final_tier)
    identity = dict(selected=selected, columns=spec['columns'], class_weight=spec['class_weight'],
                    thresholds=[tc, te], parameters=CLASSIFIER, dev_events=int(len(dev)),
                    train_sha256=sha256(root / 'data/train_data.csv'),
                    split_sha256=sha256(root / 'artifacts/event_splits.csv'),
                    sources={p: sha256(root / p) for p in ['src/final_tier.py', 'src/model.py']})
    model_version = 'final-tier-' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    bundle = dict(pipeline=model, metadata=dict(identity, model_version=model_version, tiers=TIER_NAMES,
                                                horizon_days=HORIZON_DAYS))
    import joblib
    joblib.dump(bundle, out_dir / 'final_tier_model.joblib')
    write_json(out_dir / 'cv_results.json', dict(results=results, selected=selected, identity=identity))

    holdout_result = None
    ledger = out_dir / 'holdout_ledger.json'
    if ledger.exists():
        saved = json.loads(ledger.read_text())
        if saved['model_version'] != model_version:
            raise RuntimeError('The final-tier holdout was already opened for a different model. '
                               'Do not retune on it; report the saved result and arrange a new protocol.')
        holdout_result = saved['result']
    elif finalize:
        holdout = data.loc[data.partition.eq('holdout')]
        prediction = decide(ordered_probabilities(model, holdout), tc, te)
        holdout_result = dict(selected=tier_metrics(holdout.final_tier, prediction),
                              selected_bootstrap=bootstrap(holdout.final_tier, prediction),
                              persistence_distance=tier_metrics(holdout.final_tier, persistence(holdout, False)),
                              persistence_bootstrap=bootstrap(holdout.final_tier, persistence(holdout, False)))
        pd.DataFrame({'event_id': holdout.index, 'final_tier': holdout.final_tier, 'predicted_tier': prediction}) \
            .to_csv(out_dir / 'holdout_predictions.csv', index=False)
        with ledger.open('x') as f:
            json.dump(dict(model_version=model_version, result=holdout_result), f, indent=2)
    test = export_test(root, bundle)
    (root / 'reports/final_tier_report.md').write_text(
        report(results, selected, identity, model_version, holdout_result, data, labels, test))
    return results, selected, holdout_result


def predict_events(messages, bundle):
    """One row per event: tier probabilities and decision from its >= 2-day messages."""
    features = event_features(messages)
    tc, te = bundle['metadata']['thresholds']
    proba = ordered_probabilities(bundle['pipeline'], features)
    out = pd.DataFrame({'event_id': features.index, 'predicted_tier': decide(proba, tc, te),
                        'p_safe': proba.Safe.to_numpy(), 'p_elevated': proba.Elevated.to_numpy(),
                        'p_critical': proba.Critical.to_numpy(),
                        'latest_time_to_tca_days': features.latest_time_to_tca.to_numpy(),
                        'latest_miss_distance_m': features.latest_miss_distance.to_numpy(),
                        'persistence_tier': persistence(features, False),
                        'message_count': features.message_count.to_numpy(),
                        'model_version': bundle['metadata']['model_version']})
    return out.reset_index(drop=True)


def export_test(root, bundle):
    test = pd.read_csv(Path(root) / 'data/test_data.csv', usecols=['event_id', *FEATURES])
    if test.time_to_tca.lt(HORIZON_DAYS).any():
        raise ValueError('Test messages closer than 2 days to TCA; the event model expects only >= 2-day inputs.')
    predictions = predict_events(test, bundle)
    if len(predictions) != test.event_id.nunique():
        raise ValueError('Expected one prediction per test event.')
    outputs = Path(root) / 'outputs'
    outputs.mkdir(exist_ok=True)
    predictions.insert(0, 'source', 'test')
    predictions.to_csv(outputs / 'test_event_predictions.csv', index=False)
    return predictions


def fmt(v):
    return 'n/a' if v is None else f'{v:.3f}' if isinstance(v, float) else str(v)


def metric_row(name, m):
    c = m['critical']
    return [name, fmt(c['recall']), fmt(c['precision']), fmt(c['f2']), f"{c['tp']}/{c['fp']}/{c['fn']}",
            fmt(m['elevated_or_worse_recall']), fmt(m['macro_f1']), fmt(m['balanced_accuracy']), fmt(m['accuracy'])]


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + ' --- |' * len(header)] +
                     ['| ' + ' | '.join(r) + ' |' for r in rows])


def matrix(cm):
    return '```text\n' + 'truth \\ predicted  ' + '  '.join(f'{t:>8}' for t in TIER_NAMES) + '\n' + '\n'.join(
        f'{t:<18} ' + '  '.join(f'{v:>8}' for v in row) for t, row in zip(TIER_NAMES, cm)) + '\n```'


def report(results, selected, identity, model_version, holdout, data, labels, test):
    header = ['model', 'Critical recall', 'Critical precision', 'Critical F2', 'TP/FP/FN', 'Elevated+ recall',
              'macro-F1', 'balanced acc.', 'accuracy']
    order = ['P0_persist_distance', 'P1_persist_distance_maxrisk', *[n for n in results if n.startswith('C')]]
    cv = table(header, [metric_row(n, results[n]['cv']) for n in order])
    dev = data.loc[data.partition.isin(['train', 'validation'])]
    counts = data.groupby('partition').final_tier.value_counts().unstack().reindex(columns=TIER_NAMES)
    tc, te = identity['thresholds']
    sel = results[selected]['cv']
    if holdout:
        h, hb, p, pb = holdout['selected'], holdout['selected_bootstrap'], holdout['persistence_distance'], holdout['persistence_bootstrap']
        ci = lambda b, k: f"{b[k]['low_95']:.3f}–{b[k]['high_95']:.3f}"
        holdout_text = f"""## Holdout result (evaluated once)

{h['events']} events, {h['critical']['positives']} Critical. 95% event-bootstrap intervals in brackets.

| model | Critical recall | Critical precision | Critical F2 | macro-F1 |
| --- | --- | --- | --- | --- |
| {selected} | {fmt(h['critical']['recall'])} [{ci(hb, 'critical_recall')}] | {fmt(h['critical']['precision'])} [{ci(hb, 'critical_precision')}] | {fmt(h['critical']['f2'])} [{ci(hb, 'critical_f2')}] | {fmt(h['macro_f1'])} [{ci(hb, 'macro_f1')}] |
| P0 persistence | {fmt(p['critical']['recall'])} [{ci(pb, 'critical_recall')}] | {fmt(p['critical']['precision'])} [{ci(pb, 'critical_precision')}] | {fmt(p['critical']['f2'])} [{ci(pb, 'critical_f2')}] | {fmt(p['macro_f1'])} [{ci(pb, 'macro_f1')}] |

Selected model confusion matrix:

{matrix(h['confusion_matrix'])}

Persistence confusion matrix:

{matrix(p['confusion_matrix'])}

The holdout events were never used for EDA, feature design or selection in this task. Their row-level labels were scored once for the earlier M3 row-level model; that result did not inform this experiment."""
    else:
        holdout_text = '## Holdout result\n\nNot opened yet. Run `python scripts/train_final_tier.py --finalize` once the selection is frozen.'
    tiers = test.predicted_tier.value_counts().reindex(TIER_NAMES, fill_value=0)
    return f"""# Final-tier model report

**Task (organizer clarification):** predict each event's final tier, from its last message before closest approach, using only messages with `time_to_tca >= 2` days. Tiers follow the brief: Critical if risk >= -4 or miss distance < 1,000 m; Elevated if not Critical and (risk >= -6 or miss distance <= 5,000 m); otherwise Safe. `risk = -30` is a placeholder for Pc ~ 0.

**Selected: {selected}**, model version `{model_version}`. CV Critical recall {fmt(sel['critical']['recall'])}, precision {fmt(sel['critical']['precision'])}, F2 {fmt(sel['critical']['f2'])}.

## Data

- Labels recomputed from `train_data.csv` and checked against `challenge-1-event-labels.xlsx` (identical tiers and eligibility).
- {int(labels.usable.sum())} of {len(labels)} training events have at least one message 2+ days out; the others cannot be scored and are excluded.
- Critical events are almost all distance-driven: {int(((labels.final_miss_distance < 1000) & labels.usable).sum())} of {int(((labels.final_tier == 'Critical') & labels.usable).sum())} usable Critical events have a final miss distance below 1 km; {int(((labels.final_risk >= -4) & labels.usable).sum())} have Pc >= 1e-4.
- Frozen event split (seed 42) reused: train+validation for development, holdout for one final evaluation.

{table(['partition', *TIER_NAMES], [[i, *map(str, r)] for i, r in zip(counts.index, counts.to_numpy())])}

## Features (one row per event; `risk` never used)

Latest >= 2-day message: the 12 permitted columns plus physics terms (combined radial sigma with placeholder sigmas set to missing, miss distance / sigma, radial fraction, object-size proxy from `max_risk_estimate`). History over all >= 2-day messages: message count, observed span, min/max of miss distance, `max_risk_estimate` and miss/sigma, per-day slopes and net changes of distance, risk bound, relative position and sigmas, sigma shrinkage, and a linear extrapolation of miss distance to TCA (a heuristic, not orbit propagation).

## Cross-validation on development events ({len(dev)} events, 5 stratified folds)

Decision rule: Critical if P(Critical) >= t_c; else Elevated if P(Elevated or Critical) >= t_e; else Safe. Each fold's thresholds are tuned on the other folds' out-of-fold predictions (Critical F2, then Elevated-or-worse F2). Final thresholds: t_c = {tc}, t_e = {te}. Persistence baselines apply the tier rules to the latest >= 2-day message (P0 ignores Pc; P1 uses `max_risk_estimate`, an upper bound, as the Pc proxy).

{cv}

Selected-model CV confusion matrix:

{matrix(sel['confusion_matrix'])}

{holdout_text}

## Test demo predictions

`outputs/test_event_predictions.csv`: one row per test event ({len(test)} events), with tier probabilities, the decision, the persistence tier and latest geometry. Test events have no final outcomes, so these are not scored. Predicted tiers: {', '.join(f'{k} {v}' for k, v in tiers.items())}. Test event IDs restart from 0 and are unrelated to training IDs.

## Limitations

- Critical is mostly a miss-distance outcome; the model forecasts how the screening distance evolves over the last two days, which the permitted columns describe only partially (no along-track/cross-track sigmas, no orbits).
- F2 favours recall: false alarms are cheaper than missed Critical events. Thresholds can be moved for a different operating point; report the new confusion matrix if so.
- Probabilities are classifier scores tuned for ranking and thresholds, not calibrated collision probabilities.
"""


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true', help='Evaluate the holdout once (guarded by a ledger).')
    run(finalize=parser.parse_args().finalize)
