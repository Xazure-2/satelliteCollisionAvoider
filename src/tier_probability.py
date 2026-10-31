"""Tier probabilities for the frozen final-tier model: P(Safe), P(Elevated), P(Critical) per event.

The frozen model (src/final_tier.py, holdout ledger final-tier-d7496dd831de) decides the tier with a
recall-tuned rule; it outputs a forecast final miss distance but no probabilities. This module adds a
calibrator: a multinomial logistic regression on the model's out-of-fold forecast plus the persistence
tier and a few latest-message terms, fitted on development forecast events (final message inside 2 days,
like every test event). Tier decisions are unchanged; probabilities say how likely each final tier is.

Because the decision rule favours recall, a Critical flag can carry P(Critical) well below 0.5. Both are
reported: the flag for triage, the probability for confidence.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.final_tier import (HORIZON_DAYS, ROOT, TIER_NAMES, assign_folds, candidates, load, model_scores,
                            oof_model_scores, persistence)

ARTIFACT = ROOT / 'artifacts/final_tier/tier_probability.joblib'
SELECTED_SPEC = 'R1_distance_change_regressor'


def calibration_features(features, forecast_score):
    """forecast_score: -log10(forecast final miss distance), the frozen model's score."""
    floor = persistence(features, False)
    return pd.DataFrame({
        'forecast_log_distance': -np.asarray(forecast_score, dtype=float),
        'latest_log_distance': np.log10(np.clip(features.latest_miss_distance.to_numpy(dtype=float), 1, None)),
        'floor_critical': (floor == 'Critical').astype(float),
        'floor_elevated': (floor == 'Elevated').astype(float),
        'latest_max_risk_estimate': features.latest_max_risk_estimate.to_numpy(dtype=float),
        'latest_time_to_tca': features.latest_time_to_tca.to_numpy(dtype=float),
    }, index=features.index)


def make_calibrator():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def ordered(model, X):
    return pd.DataFrame(model.predict_proba(X), columns=model.classes_, index=X.index).reindex(columns=TIER_NAMES, fill_value=0.0)


def metrics(truth, proba):
    """Multiclass log loss and Brier score, plus Critical-vs-rest Brier and a reliability table."""
    truth = np.asarray(truth)
    onehot = np.column_stack([truth == t for t in TIER_NAMES]).astype(float)
    p = proba[TIER_NAMES].to_numpy()
    crit, pc = truth == 'Critical', proba.Critical.to_numpy()
    bins = np.array([0, .05, .1, .2, .4, .6, .8, 1.0001])
    which = np.digitize(pc, bins) - 1
    table = [dict(bin=f'{bins[b]:.2f}–{min(bins[b + 1], 1):.2f}', events=int((which == b).sum()),
                  mean_predicted=float(pc[which == b].mean()), observed_rate=float(crit[which == b].mean()))
             for b in range(len(bins) - 1) if (which == b).any()]
    ece = sum(r['events'] * abs(r['mean_predicted'] - r['observed_rate']) for r in table) / len(truth)
    alphabetical = sorted(TIER_NAMES)  # sklearn's log_loss expects columns in sorted label order
    return dict(events=int(len(truth)), log_loss=float(log_loss(truth, proba[alphabetical].to_numpy(), labels=alphabetical)),
                brier=float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
                critical_brier=float(np.mean((pc - crit) ** 2)), critical_ece=float(ece), reliability=table)


def baselines(dev, fold, forecast):
    """Out-of-fold class prior and P(final tier | persistence tier), on forecast events."""
    prior, conditional = [], []
    floor = pd.Series(persistence(dev, False), index=dev.index)
    for k in np.unique(fold):
        fit, this = (fold != k) & forecast, (fold == k) & forecast
        freq = dev.final_tier[fit].value_counts(normalize=True).reindex(TIER_NAMES, fill_value=0)
        prior.append(pd.DataFrame([freq.to_numpy()] * this.sum(), columns=TIER_NAMES, index=dev.index[this]))
        table = pd.crosstab(floor[fit], dev.final_tier[fit], normalize='index').reindex(columns=TIER_NAMES, fill_value=0)
        rows = table.reindex(floor[this]).fillna(freq).to_numpy()
        conditional.append(pd.DataFrame(rows, columns=TIER_NAMES, index=dev.index[this]))
    return pd.concat(prior).loc[dev.index[forecast]], pd.concat(conditional).loc[dev.index[forecast]]


def run(root=ROOT, evaluate_holdout=False):
    root = Path(root)
    data, _, features, _ = load(root)
    dev = data.loc[data.partition.isin(['train', 'validation'])]
    fold = assign_folds(dev)
    forecast = (dev.final_time_to_tca < HORIZON_DAYS).to_numpy()
    spec = candidates(features)[SELECTED_SPEC]
    scores = oof_model_scores(spec, dev, fold)                     # same folds and spec as the selection
    X = calibration_features(dev, scores.critical)
    oof = pd.DataFrame(index=dev.index[forecast], columns=TIER_NAMES, dtype=float)
    for k in np.unique(fold):
        fit, this = (fold != k) & forecast, (fold == k) & forecast
        model = make_calibrator().fit(X[fit], dev.final_tier[fit])
        oof.loc[dev.index[this]] = ordered(model, X[this]).to_numpy()
    truth = dev.final_tier[forecast]
    prior, conditional = baselines(dev, fold, forecast)
    cv = dict(calibrator=metrics(truth, oof), class_prior=metrics(truth, prior),
              persistence_conditional=metrics(truth, conditional))
    calibrator = make_calibrator().fit(X[forecast], truth)
    tier_bundle = joblib.load(root / 'artifacts/final_tier/final_tier_model.joblib')
    bundle = dict(calibrator=calibrator, metadata=dict(
        tier_model_version=tier_bundle['metadata']['model_version'], features=list(X.columns),
        fitted_on='development forecast events, out-of-fold forecasts', events=int(forecast.sum())))
    joblib.dump(bundle, ARTIFACT)
    result = dict(cv=cv)
    ledger = root / 'artifacts/final_tier/probability_holdout.json'
    if ledger.exists():
        result['holdout'] = json.loads(ledger.read_text())['result']
    elif evaluate_holdout:
        hold = data.loc[data.partition.eq('holdout')]
        hold = hold.loc[hold.final_time_to_tca < HORIZON_DAYS]
        proba = predict_probabilities(hold, tier_bundle, bundle)
        floor = pd.Series(persistence(hold, False), index=hold.index)
        table = pd.crosstab(pd.Series(persistence(dev[forecast], False), index=dev.index[forecast]), truth,
                            normalize='index').reindex(columns=TIER_NAMES, fill_value=0)
        conditional_h = pd.DataFrame(table.reindex(floor).to_numpy(), columns=TIER_NAMES, index=hold.index)
        result['holdout'] = dict(calibrator=metrics(hold.final_tier, proba),
                                 persistence_conditional=metrics(hold.final_tier, conditional_h))
        with ledger.open('x') as f:
            json.dump(dict(tier_model_version=bundle['metadata']['tier_model_version'], result=result['holdout']), f, indent=2)
    (root / 'reports/tier_probability_report.md').write_text(report(result))
    return result


def predict_probabilities(features, tier_bundle, prob_bundle):
    """P(Safe/Elevated/Critical) for event feature rows (from event_features)."""
    if prob_bundle['metadata']['tier_model_version'] != tier_bundle['metadata']['model_version']:
        raise ValueError('Probability calibrator was fitted for a different final-tier model.')
    score = model_scores(tier_bundle['pipeline'], tier_bundle['metadata']['spec'], features).critical
    return ordered(prob_bundle['calibrator'], calibration_features(features, score))


def load_probability_model(path=None):
    # Only load trusted, locally produced joblib files.
    return joblib.load(path or ARTIFACT)


def fmt(v):
    return f'{v:.4f}'


def report(result):
    def rows(block):
        names = {'calibrator': 'Calibrated model probabilities', 'persistence_conditional': 'P(final tier given current distance tier)',
                 'class_prior': 'Class frequencies (no information)'}
        return '\n'.join(f"| {names[k]} | {fmt(m['log_loss'])} | {fmt(m['brier'])} | {fmt(m['critical_brier'])} | {fmt(m['critical_ece'])} |"
                         for k, m in block.items())
    header = '| Probabilities | Log loss | Brier (3-class) | Critical Brier | Critical ECE |\n| --- | --- | --- | --- | --- |'
    rel = '\n'.join(f"| {r['bin']} | {r['events']} | {r['mean_predicted']:.3f} | {r['observed_rate']:.3f} |"
                    for r in result['cv']['calibrator']['reliability'])
    text = f"""# Final-tier probabilities

Generated by `python scripts/train_tier_probability.py`. Each event gets P(Safe), P(Elevated) and P(Critical) for its **final** tier, alongside the frozen model's tier decision (unchanged). A multinomial logistic calibrator maps the frozen model's forecast final miss distance, the current distance tier, the latest distance, `max_risk_estimate` and time to TCA to probabilities. It is fitted on out-of-fold forecasts for development forecast events (final message inside 2 days, {result['cv']['calibrator']['events']} events); the cross-validated scores below are cross-fitted on the same 5 folds.

Lower is better for every column. Log loss and Brier score measure the full probability vector; ECE is the average gap between predicted P(Critical) and the observed Critical rate across bins (a constant forecast like the class frequencies scores near zero ECE while carrying no information, so read it with the Brier score).

## Cross-validation (development forecast events)

{header}
{rows(result['cv'])}

Reliability of P(Critical) (cross-validated):

| P(Critical) bin | Events | Mean predicted | Observed Critical rate |
| --- | --- | --- | --- |
{rel}
"""
    if 'holdout' in result:
        text += f"""
## Holdout (forecast events, evaluated once)

{header}
{rows(result['holdout'])}

The tier model's holdout ledger is unchanged; this is a separate one-time evaluation of the probability layer (`artifacts/final_tier/probability_holdout.json`). Nothing was tuned on it.
"""
    text += """
## How to read the probabilities

- **The tier flag and the probability answer different questions.** The flag is tuned for recall (F2), so an event can be flagged Critical with P(Critical) around 0.2–0.4: worth maneuver planning attention, but more likely than not to end below Critical.
- **Probabilities are calibrated against final tiers in historical data,** not physical collision probabilities. Pc remains the tier's input, not this output.
"""
    return text
