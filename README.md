# Satellite Collision Avoidance

Hackathon prototype for estimating collision risk from conjunction telemetry. See [ROADMAP.md](ROADMAP.md) for the team assignments and application plan.

## Start with the EDA notebook

Open [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) in Jupyter or VS Code. Select a Python environment with the dependencies below, then **Restart Kernel → Run All**. Launch from the repository root or its `notebooks/` directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eda.txt
```

Put the supplied files at `data/train_data.csv` and `data/test_data.csv`. Raw datasets, local environments, caches, and generated artifacts are ignored by Git. The notebook never modifies the raw CSV files.

The notebook inventories the full schema, then analyzes the 12 allowed predictors. It covers quality checks, missingness, uncertainty extremes, risk tiers, feature relationships, event histories, test feature shift, and possible encounter overlap. Test `risk` is never loaded. No model is trained.

Before detailed EDA, it reserves a reproducible 70/15/15 event split. Label relationships and example histories use only the 70% training partition. Previous repository inspection exposed aggregate full-training counts/ranges; that prior inspection is disclosed in the notebook.

Generated files:

- `artifacts/event_splits.csv`: eligible event IDs and their frozen partitions; reuse for modeling.
- `artifacts/eda_manifest.json`: file fingerprints, split settings, units, and environment versions.
- `reports/eda_summary.md`: computed findings and proposed modeling decisions.

Reruns verify the saved split and reuse it. If the inputs or split settings change, the notebook stops instead of silently replacing the split. Preserve and archive the old split and manifest before intentionally starting a new experiment. Do not delete them merely to get past a failed integrity check.

The notebook usually takes about 1–3 minutes on a laptop, with some extra time possible for first-run plotting caches. Charts contain a deterministic sample where needed; summary statistics use all eligible analysis rows. Inspect the final check cell and summary before proceeding to modeling.

## Interpretation

`risk = log10(Pc)`; `risk = -30` is a placeholder for a probability too small to compute (Pc ≈ 0). Tiers follow the hackathon brief: Critical if risk >= −4 or miss distance < 1,000 m; Elevated if not Critical and (risk >= −6 or miss distance <= 5,000 m); otherwise Safe.

Every test message is 2–7 days before closest approach, so a real next-24-hour view of the test data is empty; use `time_to_tca` windows on a training replay to demonstrate warnings. Relative positions describe encounter geometry at TCA, not satellite trajectories. Test event IDs restart at 0 and are never joined to training IDs. The project follows the brief's thresholds and does not claim operational certification.

## Models at a glance

| Purpose | Model | Train | Output | Report |
|---|---|---|---|---|
| **Scored task:** each event's final tier, with a probability, from messages >= 2 days out | Distance-change regressor + persistence floor; calibrated P(Safe/Elevated/Critical) | `python scripts/train_final_tier.py` then `python scripts/train_tier_probability.py` | `outputs/test_event_predictions.csv` (one row per test event) | [model_development.md](reports/model_development.md), [final_tier_report.md](reports/final_tier_report.md) |
| **Dashboard:** estimated Pc per message | Row-level v2: physics features, target floored at −6.001, capped at `max_risk_estimate` | `python scripts/train_row_model.py` then `python scripts/predict.py` | `outputs/test_predictions.csv` (one row per test message) | [experiment_v2.md](reports/experiment_v2.md) |
| Historical first pass | M3 row-level regressor (superseded) | `python scripts/train.py` (blocked: holdout already opened) | `artifacts/selected_model.joblib` | [model_report.md](reports/model_report.md) |

Start with [reports/model_development.md](reports/model_development.md): it tells the full iteration story and gives the headline holdout result (Critical recall 0.882 vs 0.789 for persistence on held-out forecast events).

```bash
python -m pip install -r requirements-model.txt
python -m unittest discover -s tests
python scripts/train_final_tier.py     # final-tier CV + selection; reuses the saved holdout result
python scripts/train_row_model.py      # refit the per-message Pc model on train+validation events
python scripts/predict.py              # per-message test predictions for the dashboard
```

Put `challenge-1-event-labels.xlsx` in `data/`; final-tier labels are recomputed from `train_data.csv` and checked against it. Holdout ledgers (`artifacts/holdout_evaluation.json` for M3, `artifacts/final_tier/holdout_ledger.json` for the final-tier model) block re-evaluating a different model on the same holdout. Do not delete them to reuse the holdout as validation; arrange a new protocol instead.

### Final-tier output (`outputs/test_event_predictions.csv`)

`source`, `event_id`, `predicted_tier`, `predicted_final_miss_distance_m`, `persistence_tier` (tier of the latest message), `latest_time_to_tca_days`, `latest_miss_distance_m`, `message_count`, `model_version`, and the calibrated probabilities `p_safe`, `p_elevated`, `p_critical` plus `p_predicted_tier`. Run `python scripts/train_tier_probability.py` after `train_final_tier.py` to add them; see [tier_probability_report.md](reports/tier_probability_report.md). The predictor only accepts messages at least 2 days before closest approach: **filter replayed training messages to `time_to_tca >= 2` before calling `predict_events`**, or it rejects the batch.

```python
from src.final_tier import predict_events
import joblib
bundle = joblib.load('artifacts/final_tier/final_tier_model.joblib')  # trusted local file only
events = predict_events(messages.loc[messages.time_to_tca >= 2], bundle)
```

### Per-message Pc output (`outputs/test_predictions.csv`)

Raw and clipped log10(Pc), `predicted_pc`, `pc_display` (`< 1e-6` at the floor), Pc-only and combined tiers, miss distance, hours to TCA, object type, quality flags and model version, one row per input message in original order.

```python
from src.row_model import load_row_model
from src.model import prediction_frame
table = prediction_frame(telemetry_frame, load_row_model())   # needs event_id; row_id optional
```

`prediction_frame` scores row by row: invalid required telemetry gets no estimate and a `needs_review` tier (or `critical` when a valid miss distance is below 1 km), so one bad message never blocks the batch or defaults to safe. Placeholder sigmas (10 Earth radii, 63,781,361–63,781,363 m) are flagged `*_placeholder`. `predict_log10_pc(frame, model)` is the strict variant and raises on invalid input; called without a model it still loads the historical M3 artifact.

**Caveat:** the row model is tuned for recall, as the brief's F2 metric favours. In cross-validation it catches 98% of messages that are truly Elevated-or-worse by Pc, but only about 29% of its Elevated-by-Pc flags are correct. Use the final-tier model to prioritize events, and per-message Pc for context. Neither output is a calibrated probability of an observed collision.

## Run the dashboard with real predictions

The Reflex dashboard in `dashboard/` reads `outputs/event_predictions.csv` and `outputs/cdm_history.csv` (its v2 contract). Generate them from the trained models, then start the app:

```bash
# 1. From the repository root: export predictions (about 5 s; checked with the dashboard's own loader)
python scripts/train_tier_probability.py   # only needed once, or after retraining (fits tier probabilities)
python scripts/export_dashboard_outputs.py

# 2. Start the dashboard (first run installs JavaScript dependencies, 1-2 minutes)
cd dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
reflex run --env prod --single-port
```

Open http://localhost:3000, choose **Real model outputs (outputs/)** in the top-bar data selector and press reload. If you retrain a model, rerun step 1 and reload.

What the export contains:

- **Test events** (`source=test`, 2,167): the final-tier prediction and forecast final miss distance at the cutoff CDM (latest message >= 2 days out), the observed-distance tier at that CDM, reasons, quality flags and full CDM history. Test `risk` is never read, so observed Pc is empty.
- **Held-out training events** (`source=train`, `split=holdout`, 1,800): the same predictions, out of sample for both models, plus observed risk history and actual final outcomes. These drive the Model Validation page, which reproduces the holdout result (108 of 118 Critical events caught, 41 false alarms).
- **Tier probabilities:** `p_safe`, `p_elevated`, `p_critical` from the calibrated probability layer, which drive the dashboard's reliability panel.
- **Not supplied:** predicted final Pc. The final-tier model forecasts tier and miss distance, not Pc, so that field shows "not provided" (one expected loader warning).
- **Alert thresholds:** the model flags Critical when the forecast final distance is <= 1.31 km (tuned for recall), and the latest message's distance tier is a floor. The encounter page's threshold check may report that a Critical tier is "not reproduced" from the 1 km rule; each event's reasons state the tuned thresholds and the floor.

## Risk rules and dashboard data

`src/risk.py` holds the brief's rules as pure functions: `assess(log_pc, miss_distance_m)` returns the tier, which trigger fired (Pc and/or distance), readable reasons such as "Miss distance 0.72 km is below 1 km", and a recommended action. Unusable inputs give `needs_review`, never safe, unless a valid input already proves Critical. `score_rows` builds the per-message table, converting units once (days → hours, metres → km).

`src/events.py` turns messages into event views:

- `snapshot(scored, cutoff_days=0)`: newest pre-TCA message per event. A positive cutoff replays what was known that many days before each closest approach.
- `event_table(...)`: one row per event with the current tier (latest message), the final-tier forecast, and the **priority tier**. While the latest message is 2+ days out, the forecast can raise priority, never lower it; once newer messages exist, they supersede the forecast (still shown, marked superseded).
- `window_counts(events, window_days=1)`: unique events with closest approach within the window, with Critical split by distance vs Pc and the number of forecast upgrades.
- `action_queue(events)`: Critical → needs review → Elevated → Safe, then soonest closest approach, then highest estimated Pc.
- `insights(events, scored)`: deterministic sentences (tier counts, window summary, upgrades, highest Pc, closest approach, critical events by object type, largest Pc rise).
- `build_tables(messages, row_bundle, tier_bundle, ...)`: runs everything; any `risk` column is ignored.

```bash
python scripts/build_dashboard_data.py                  # outputs/dashboard/{test,train_replay}_{rows,events}.csv + _summary.json
python scripts/build_dashboard_data.py --cutoff-days 2 --out outputs/dashboard_cutoff2   # replay "2 days before TCA"
```

- **`test`**: all test events. The 24-hour window is honestly empty (every test message is 2+ days out); the queue is driven by the final-tier forecasts.
- **`train_replay`**: holdout events only, so both models are out of sample, with observed outcomes in separate `observed_*` columns for comparison. At cutoff 0 the window shows events whose latest message is within a day of closest approach; at cutoff 2 it shows the forecasts as they stood 2 days out.
- List columns (`reasons`, `alert_reasons`, `quality_flags`) are JSON strings in the CSVs.

## Notebooks

`notebooks/02_modeling.ipynb` documents the historical M3 experiment (dummy baselines, median-imputed and native-missing HGB, derived features, ablation without `max_risk_estimate`). Its findings are superseded by the reports above; keep it for the audit trail.
