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

`risk = log10(Pc)`. A row with `risk >= -4` meets the challenge's critical Pc threshold. The application policy also marks distance below 1 km as critical. These two triggers are analyzed separately.

The supplied test messages are more than two days before closest approach; their next-24-hour view is therefore empty. Relative positions describe updated encounter geometry at TCA, not complete satellite trajectories. The project follows the hackathon's thresholds and does not claim operational certification.

## Final-tier task (organizer clarification)

The scored task is each event's **final tier** (from its last message before closest approach), predicted only from messages with `time_to_tca >= 2` days. Put `challenge-1-event-labels.xlsx` in `data/`; labels are recomputed from `train_data.csv` and checked against it.

```bash
python scripts/train_final_tier.py             # grouped CV, selection, test event predictions
python scripts/train_final_tier.py --finalize  # evaluate the holdout once (ledger-guarded)
```

Read [reports/model_development.md](reports/model_development.md) for the full iteration story and headline holdout result, and [reports/final_tier_report.md](reports/final_tier_report.md) for the generated tables. Outputs: `outputs/test_event_predictions.csv` (one row per test event, tier probabilities, decision, persistence tier). The row-level models below remain for per-message estimated Pc in the dashboard.

## Modeling experiment

Open [notebooks/02_modeling.ipynb](notebooks/02_modeling.ipynb) for the experiment, comparison plots, and selected-model diagnostics. The implementation fits two dummy baselines, median-imputed HGB, native-missing-value HGB, and a derived-feature variant, followed by an ablation without `max_risk_estimate`. It reuses the EDA event split and selects on validation only.

```bash
python -m pip install -r requirements-model.txt
python -m unittest discover -s tests -v
python scripts/train.py
python scripts/train.py --finalize
python scripts/predict.py
```

The first train command fits or reuses cached validation experiments. `--finalize` freezes the selection, checks inference compatibility, evaluates the holdout once, and exports all test rows. The notebook defaults to validation only; set its `FINALIZE` option deliberately when starting a new experiment. After finalization, reruns reuse the saved holdout metrics and do not predict the holdout again.

Read [reports/model_report.md](reports/model_report.md) for measured results, counts, slices, event-bootstrap intervals, and limitations. The selected preprocessing and estimator are stored together in `artifacts/selected_model.joblib`; only load trusted joblib files. Its metadata records the model version, units, feature order, training event IDs, and environment versions.

The holdout ledger protects the evaluated model: changing implementation, inputs, or experiment settings after finalization blocks a new automatic run. Preserve the evaluated experiment and arrange a new evaluation protocol before retuning. Do not delete the ledger to reuse the same holdout as validation.

Test exports:

- `outputs/test_predictions.csv`: raw and clipped log risk, estimated Pc, risk tiers, original row/event identifiers, quality flags, and model version.
- `outputs/test_submission.csv`: `row_id,event_id,predicted_risk`, with raw log10(Pc). Confirm the organizer's required column names before submitting.
- `tests/fixtures/telemetry.csv` and `expected_predictions.csv`: small reproducible, label-free inference handoff.

For app integration:

```python
from src.model import load_bundle, predict_log10_pc

model = load_bundle()
raw_log_risk = predict_log10_pc(telemetry_frame, model)
```

Calling `predict_log10_pc(frame)` alone loads the selected artifact automatically. Both forms preserve row order, ignore labels and extra columns, and accept unseen categories and missing optional inputs. `predict_log10_pc` raises a clear error on invalid required telemetry. `prediction_frame` (exports and dashboard) instead scores row by row: invalid rows get no estimate, a `needs_review` tier (or `critical` when a valid miss distance is below 1 km), and quality flags, so one bad message never blocks the batch or defaults to safe. Placeholder sigmas (10 Earth radii, 63,781,361–63,781,363 m) are flagged `*_placeholder`. This model estimates supplied risk labels; its Pc output is not a calibrated probability of observed collisions.
