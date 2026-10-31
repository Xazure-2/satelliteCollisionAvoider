# Experiment v2: physics features, floored targets, grouped CV

Selected **F6_floor_-6.001_physics**. Beats the `max_risk_estimate` baseline on the selection metric: **True**.

Selection rule: max out-of-fold F2 (validated alert threshold, time_to_tca >= 2 d); ties by lower high-risk MSE.

## Protocol

- 5-fold grouped CV by `event_id`, stratified on whether an event has any `risk >= -4`; pool: train+validation partitions; holdout not loaded.
- Headline: rows with time_to_tca >= 2.0 days (94858 rows, 10142 events), matching every supplied test row.
- `@-4`: alert when predicted log10(Pc) >= -4. `@tuned`: alert threshold chosen on the other folds' out-of-fold predictions (F2 for true Pc >= 1e-4), applied to the held-out fold. The final threshold is refit on all out-of-fold rows; it is an alerting cut, not a change to the reported Pc.
- MSE uses only rows with true `risk >= -6`, with predictions clipped at -6.001 (Kelvins challenge convention).
- Every learned prediction is capped at `max_risk_estimate` (the maximum Pc over covariance scaling) and floored at the training floor.
- Placeholder sigmas (63781363.0 m = 10 Earth radii, ±10 m) are set to missing with a flag. F10/AP are forward-filled from earlier messages of the same event.
- The M3 holdout ledger is untouched; this experiment never loads holdout rows.

## Out-of-fold results, time_to_tca >= 2 days

| candidate | MSE risk>=-6 | recall@-4 | F2@-4 | recall@tuned | precision@tuned | F2@tuned | TP/FP/FN@tuned | AP | recall@-6 | alert threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B2_max_risk_estimate | 0.295 | 1.000 | 0.698 | 0.943 | 0.350 | 0.705 | 517/959/31 | 0.521 | 1.000 | -3.950 |
| R0_raw_m3_features | 0.632 | 0.403 | 0.421 | 0.440 | 0.351 | 0.419 | 241/446/307 | 0.313 | 0.408 | -4.300 |
| R1_raw_physics | 0.497 | 0.451 | 0.473 | 0.637 | 0.362 | 0.553 | 349/616/199 | 0.448 | 0.450 | -4.500 |
| F8_floor_-8_physics | 0.317 | 0.473 | 0.509 | 0.812 | 0.490 | 0.718 | 445/463/103 | 0.695 | 0.567 | -4.650 |
| F6_floor_-6.001_physics | 0.164 | 0.507 | 0.544 | 0.867 | 0.483 | 0.748 | 475/508/73 | 0.747 | 0.984 | -4.550 |
| G8_gap_floor_-8_physics | 0.324 | 0.440 | 0.480 | 0.803 | 0.463 | 0.700 | 440/511/108 | 0.648 | 0.557 | -4.650 |

B2 is the one-line rule "Pc = max_risk_estimate". R0 uses the original M3 feature set with the new placeholder handling, cap and evaluation.

## Limitations

- Labels are calculated log10(Pc) per message, not observed collisions; no physical calibration is claimed.
- The derived features use radial sigmas only; full covariance and object size are not in the permitted columns.
- Fold thresholds vary; see `artifacts/experiment_v2/results.json`. A holdout check of the selected candidate would be a disclosed second look.
