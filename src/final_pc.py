"""Forecast of each event's FINAL collision probability (log10 Pc), alongside the frozen final-tier model.

The final-tier model decides tiers from a miss-distance forecast and does not forecast Pc. This module
adds a separate event-level regressor for the final log10(Pc), so the dashboard and exports can show a
predicted final Pc. It never changes the tier or the tier probabilities.

- Inputs: the same event features as the final-tier model (messages >= 2 days out; risk never an input).
- Target: final log10(Pc) floored at -8; 41% of labels are the -30 "too small to compute" placeholder.
  A floor at -6.001 (the Kelvins convention) piled predictions just above 1e-6 and flagged ~2,300
  development events at 3% precision; at -8 the forecast says "below 1e-6" for almost every event, as
  the data supports (only 1.1% of forecast events end at Pc >= 1e-6). A floored forecast means
  "below 1e-8".
- Evaluation: same 5 event folds, scored on development forecast events (final message inside 2 days)
  with the Kelvins high-risk MSE and Pc-threshold detection; one ledger-guarded holdout check.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.final_tier import HORIZON_DAYS, ROOT, assign_folds, candidates, load, make_model

FLOOR = -8.0
ARTIFACT = ROOT / 'artifacts/final_tier/final_pc.joblib'
LEDGER = ROOT / 'artifacts/final_tier/final_pc_holdout.json'
MODEL_VERSION = 'final-pc-v1'


def spec(features):
    """Regressor spec reusing the final-tier history features (event_features output only)."""
    base = candidates(features)['C3_history_balanced']
    return dict(base, kind='change_regressor', class_weight=None)


def fit(train, columns, spec_):
    model = make_model(spec_, columns)
    model.fit(train, np.maximum(train.final_risk.to_numpy(dtype=float), FLOOR))
    return model


def predict(model, features):
    return np.clip(model.predict(features), FLOOR, 0.0)


def detection(truth, pred, threshold):
    t, p = truth >= threshold, pred >= threshold
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    return dict(positives=int(t.sum()), tp=tp, fp=fp, fn=fn,
                recall=tp / (tp + fn) if tp + fn else None, precision=tp / (tp + fp) if tp + fp else None)


def metrics(truth, pred):
    truth, pred = np.asarray(truth, dtype=float), np.asarray(pred, dtype=float)
    high = truth >= -6
    return dict(events=int(len(truth)), high_risk_events=int(high.sum()),
                mse_high_risk=float(np.mean((np.maximum(pred[high], FLOOR) - truth[high]) ** 2)) if high.any() else None,
                mae_floored=float(np.mean(np.abs(pred - np.maximum(truth, FLOOR)))),
                elevated_or_worse=detection(truth, pred, -6.0), critical=detection(truth, pred, -4.0))


def baselines(frame):
    """Floor everywhere ("below 1e-8"), and the latest message's max_risk_estimate (an upper bound)."""
    return {'floor_everywhere': np.full(len(frame), FLOOR),
            'latest_max_risk_estimate': np.clip(frame.latest_max_risk_estimate.to_numpy(dtype=float), FLOOR, 0.0)}


def run(root=ROOT, evaluate_holdout=False):
    root = Path(root)
    data, _, features, _ = load(root)
    dev = data.loc[data.partition.isin(['train', 'validation'])]
    fold = assign_folds(dev)
    forecast = (dev.final_time_to_tca < HORIZON_DAYS).to_numpy()
    s = spec(features)
    oof = np.empty(len(dev))
    for k in np.unique(fold):
        model = fit(dev.loc[fold != k], s['columns'], s)
        oof[fold == k] = predict(model, dev.loc[fold == k])
    truth = dev.final_risk.to_numpy()[forecast]
    cv = {'model': metrics(truth, oof[forecast])}
    cv.update({name: metrics(truth, b[forecast]) for name, b in baselines(dev).items()})
    model = fit(dev, s['columns'], s)
    joblib.dump(dict(pipeline=model, metadata=dict(model_version=MODEL_VERSION, floor=FLOOR, columns=s['columns'],
                                                    fitted_on='development events', target='final log10(Pc) floored at -8')),
                root / ARTIFACT.relative_to(ROOT))
    result = dict(cv=cv)
    ledger = root / LEDGER.relative_to(ROOT)
    if ledger.exists():
        result['holdout'] = json.loads(ledger.read_text())['result']
    elif evaluate_holdout:
        hold = data.loc[data.partition.eq('holdout')]
        hold = hold.loc[hold.final_time_to_tca < HORIZON_DAYS]
        t = hold.final_risk.to_numpy()
        result['holdout'] = {'model': metrics(t, predict(model, hold))}
        result['holdout'].update({name: metrics(t, b) for name, b in baselines(hold).items()})
        with ledger.open('x') as f:
            json.dump(dict(model_version=MODEL_VERSION, result=result['holdout']), f, indent=2)
    (root / 'reports/final_pc_report.md').write_text(report(result))
    return result


def load_final_pc_model(path=None):
    # Only load trusted, locally produced joblib files.
    return joblib.load(path or ARTIFACT)


def _fmt(v, digits=3):
    return 'n/a' if v is None else f'{v:.{digits}f}'


def report(result):
    names = {'model': 'Final-Pc forecaster', 'floor_everywhere': 'Always at the floor (below 1e-8)',
             'latest_max_risk_estimate': 'Latest `max_risk_estimate` (upper bound)'}

    def table(block):
        rows = ['| Forecast | MSE, final risk >= -6 | MAE (floored) | Elevated+ recall / precision | Critical recall / precision |',
                '| --- | --- | --- | --- | --- |']
        for k, m in block.items():
            e, c = m['elevated_or_worse'], m['critical']
            rows.append(f"| {names[k]} | {_fmt(m['mse_high_risk'])} | {_fmt(m['mae_floored'])} | "
                        f"{_fmt(e['recall'], 2)} / {_fmt(e['precision'], 2)} | {_fmt(c['recall'], 2)} / {_fmt(c['precision'], 2)} |")
        return '\n'.join(rows)
    m = result['cv']['model']
    text = f"""# Final-Pc forecast

Generated by `python scripts/train_final_pc.py`. An event-level regressor forecasts each event's **final** log10(Pc) from the same features as the final-tier model (messages at least 2 days before closest approach; `risk` is never an input). It supplies the dashboard's "Predicted final Pc". **It does not change the tier or the tier probabilities**, which come from the frozen final-tier model and its probability layer.

The target is floored at −8; 41% of labels are the −30 "too small to compute" placeholder. A forecast at the floor means "below 1e-8". We first tried the Kelvins convention (floor −6.001): its predictions piled up just above 1e-6, flagging about 2,300 development events as Pc ≥ 1e-6 at 3% precision. With the −8 floor the forecast says "below 1e-6" for almost every event, which is what the data supports: only about 1% of forecast events end with Pc ≥ 1e-6, and final Pc ≥ 1e-4 cannot be forecast from messages 2+ days out.

## Cross-validation (development forecast events: {m['events']}, of which {m['high_risk_events']} have final Pc >= 1e-6)

{table(result['cv'])}

MSE uses only events whose true final Pc is at least 1e-6 (forecasts as made, floor −8). MAE compares forecasts with labels floored at −8 over all forecast events, so always answering the floor scores well on it. Critical positives (final Pc >= 1e-4) are very rare: {m['critical']['positives']} events.
"""
    if 'holdout' in result:
        text += f"""
## Holdout (forecast events, evaluated once)

{table(result['holdout'])}

Recorded once in `artifacts/final_tier/final_pc_holdout.json`; nothing was tuned on it.
"""
    text += """
## How to read it

- **This is a weak forecast, by necessity.** It halves the high-risk error of always answering the floor, but catches only about 1 in 10 events that end at Pc ≥ 1e-6 (precision about 20%), and it cannot anticipate Pc ≥ 1e-4. Treat it as context.
- **Pc rarely decides the tier here:** 831 of 834 Critical training events were Critical because of miss distance. The final Pc forecast is context for operators; the tier comes from the miss-distance forecast.
- **It can disagree with the tier.** For example, an event can be Elevated by forecast miss distance while its Pc forecast is below 1e-6. The dashboard shows both; neither overrides the other.
- **It is a forecast of the calculated Pc in the final message,** not a physical probability of collision.
"""
    return text
