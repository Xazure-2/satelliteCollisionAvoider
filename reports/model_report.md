# Row-level collision-risk model report

## Result and decision

Selected **M3**, model version `87c8704204c5-M3`, using validation only. M3 inherited preprocessing from **M2**. The selected model beats the mean-target dummy's validation RMSE. No held-out data were used to fit preprocessing or the estimator.

Among M1/M2/M3 within 1% of the lowest validation raw-log-risk RMSE, maximize Pc-critical F2 at -4; break ties by fewer false positives, lower MAE, simpler pipeline, then name.

The 1% tolerance is our practical balanced-selection default, not an organizer-supplied scoring formula. A0 is diagnostic only and never participates in selection. Scores use **raw** predicted log risk; displayed Pc uses `10 ** clip(raw, -30, 0)`. No oversampling, class weighting, target-floor removal, log transforms, PCA, or automatic row-based early stopping was used.

## Frozen data split and validation experiment

| partition | rows | events | Pc_critical_rows | Pc_critical_events |
| --- | --- | --- | --- | --- |
| train | 113195 | 9200 | 503 | 108 |
| validation | 24359 | 1971 | 107 | 23 |
| holdout | 24689 | 1972 | 105 | 23 |

| model | imputation | derived | RMSE | MAE | recall | F2 | AP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | median | False | 10.1568 | 9.5104 | 0.0000 | 0.0000 | 0.0044 | 0 | 107 |
| B1 | median | False | 10.0694 | 9.5570 | 0.0000 | 0.0000 | 0.0044 | 0 | 107 |
| M1 | median | False | 5.4106 | 3.7585 | 0.3645 | 0.3255 | 0.2315 | 132 | 68 |
| M2 | native | False | 5.3960 | 3.7525 | 0.3645 | 0.3328 | 0.2375 | 119 | 68 |
| M3 | native | True | 5.2728 | 3.5097 | 0.4860 | 0.4377 | 0.2998 | 114 | 55 |
| A0 | native | True | 5.7622 | 3.8701 | 0.0187 | 0.0229 | 0.0643 | 6 | 105 |

B0 = median dummy; B1 = mean dummy; M1 = median-imputed HGB; M2 = native-NaN HGB; M3 = selected preprocessing plus geometry features; A0 = selected architecture without `max_risk_estimate`.

All learned models use the same two extreme-sigma flags and numeric missingness indicators. Categories are one-hot encoded from training only; unknown categories are accepted. Derived features are absolute radial position, independent-error combined radial sigma, and their ratio (missing for invalid/zero denominator), calculated before imputation. Extreme sigmas remain measurements, not automatically discarded sentinels.

Parameters: `{"early_stopping": false, "l2_regularization": 1.0, "learning_rate": 0.05, "loss": "squared_error", "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20, "random_state": 42}`.

## Selected-model validation diagnostics

| level | rows/events | tp | fp | fn | tn | positives | predicted_positives | precision | recall | f2 | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rows | 24359 | 52 | 114 | 55 | 24138 | 107 | 166 | 0.3133 | 0.4860 | 0.4377 | 0.2998 |
| latest event | 1971 | 0 | 2 | 1 | 1968 | 1 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.3333 |

Latest-message Pc false alerts per 1,000 events: 1.0147

Latest-message combined-policy results:

| tp | fp | fn | tn | positives | predicted_positives | precision | recall | f2 | false_alerts_per_1000_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 144 | 2 | 0 | 1825 | 144 | 146 | 0.9863 | 1.0000 | 0.9972 | 1.0147 |

Confusion matrices: rows = truth, columns = prediction; order ['safe', 'elevated', 'critical', 'needs_review']. The fourth category is reserved for invalid inputs.

Pc only:

```text
[[22613   149    16     0]
 [  873   503    98     0]
 [   16    39    52     0]
 [    0     0     0     0]]
```

Combined policy:

```text
[[16988    34     3     0]
 [  398  4998    59     0]
 [    5    13  1861     0]
 [    0     0     0     0]]
```

### Slices (overlap; do not sum their row counts)

| slice | rows | events | MAE | RMSE | positives | recall | F2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 24359 | 1971 | 3.5097 | 5.2728 | 107 | 0.4860 | 0.4377 |
| floor_-30 | 9933 | 1290 | 3.1816 | 5.4656 | 0 | undefined | undefined |
| non_floor | 14426 | 1832 | 3.7355 | 5.1358 | 107 | 0.4860 | 0.4377 |
| risk_ge_-6 | 1581 | 311 | 2.1336 | 3.2831 | 107 | 0.4860 | 0.4498 |
| risk_ge_-4 | 107 | 23 | 1.4404 | 2.5302 | 107 | 0.4860 | 0.5417 |
| near_-6_+-0.5 | 1410 | 360 | 2.8571 | 3.9393 | 0 | undefined | undefined |
| near_-4_+-0.5 | 225 | 46 | 1.4283 | 2.3494 | 72 | 0.3333 | 0.3324 |
| test_horizon_ge_2_days | 16819 | 1805 | 3.7042 | 5.3799 | 92 | 0.5543 | 0.4784 |
| time/0_to_24h | 3851 | 1406 | 2.7265 | 4.7020 | 2 | 0.0000 | 0.0000 |
| time/24_to_48h | 3689 | 1377 | 3.4402 | 5.3424 | 13 | 0.0769 | 0.0943 |
| time/48_to_96h | 7058 | 1480 | 4.2001 | 5.9173 | 32 | 0.5000 | 0.4520 |
| time/above_96h | 9761 | 1575 | 3.3457 | 4.9552 | 60 | 0.5833 | 0.4916 |
| object/DEBRIS | 14105 | 1068 | 3.6824 | 5.4252 | 79 | 0.5570 | 0.5164 |
| object/PAYLOAD | 2386 | 218 | 1.6964 | 3.3846 | 7 | 0.1429 | 0.1667 |
| object/ROCKET BODY | 334 | 25 | 3.5268 | 5.5209 | 2 | 0.0000 | 0.0000 |
| object/TBA | 24 | 5 | 4.0559 | 6.0468 | 0 | undefined | undefined |
| object/UNKNOWN | 7510 | 655 | 3.7588 | 5.4575 | 19 | 0.3684 | 0.2692 |
| missing_optional/yes | 1041 | 73 | 3.6751 | 5.4834 | 23 | 0.6957 | 0.6504 |
| missing_optional/no | 23318 | 1902 | 3.5023 | 5.2632 | 84 | 0.4286 | 0.3822 |
| extreme_sigma/yes | 886 | 204 | 0.8361 | 1.1597 | 0 | undefined | undefined |
| extreme_sigma/no | 23473 | 1922 | 3.6106 | 5.3666 | 107 | 0.4860 | 0.4377 |

### Event-cluster bootstrap

| metric | low_95 | high_95 | valid_replicates | undefined_replicates |
| --- | --- | --- | --- | --- |
| mae | 3.3977 | 3.6403 | 200 | 0 |
| rmse | 5.1160 | 5.4573 | 200 | 0 |
| pc_recall | 0.3222 | 0.6492 | 200 | 0 |
| pc_f2 | 0.2810 | 0.5692 | 200 | 0 |
| latest_pc_recall | 0.0000 | 0.0000 | 128 | 72 |
| latest_pc_f2 | 0.0000 | 0.0000 | 128 | 72 |
| latest_false_alerts_per_1000_events | 0.0000 | 3.0441 | 200 | 0 |

200 resamples, seed 42, whole events sampled with replacement and all their rows retained. Percentile 95% intervals describe sampling variation conditional on this trained model. They do not cover model-training uncertainty, physical Pc uncertainty, or distribution shift. Undefined replicate counts are explicit. Average precision is computed from continuous raw log-risk scores; it is not trapezoidal PR area. Precision is undefined with no positive predictions; recall/F2/AP are undefined with no actual positives.

## Single holdout evaluation

The selection and model hash were frozen and compatibility checks passed before this evaluation. Reruns reuse the saved result; the holdout ledger blocks automatic evaluation of another model. These findings must not initiate a tuning cycle.

MAE: **3.4744**; RMSE: **5.1824**.

| level | rows/events | tp | fp | fn | tn | positives | predicted_positives | precision | recall | f2 | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rows | 24689 | 39 | 109 | 66 | 24475 | 105 | 148 | 0.2635 | 0.3714 | 0.3433 | 0.2235 |
| latest event | 1972 | 0 | 1 | 2 | 1969 | 2 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0657 |

Latest-message Pc false alerts per 1,000 events: 0.5071

Latest-message combined-policy results:

| tp | fp | fn | tn | positives | predicted_positives | precision | recall | f2 | false_alerts_per_1000_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 0 | 1 | 1843 | 129 | 128 | 1.0000 | 0.9922 | 0.9938 | 0.0000 |

Confusion matrices: rows = truth, columns = prediction; order ['safe', 'elevated', 'critical', 'needs_review']. The fourth category is reserved for invalid inputs.

Pc only:

```text
[[22984   101    11     0]
 [  986   404    98     0]
 [   40    26    39     0]
 [    0     0     0     0]]
```

Combined policy:

```text
[[17905    48     3     0]
 [  445  4613    67     0]
 [    2    22  1584     0]
 [    0     0     0     0]]
```

### Holdout slices

| slice | rows | events | MAE | RMSE | positives | recall | F2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 24689 | 1972 | 3.4744 | 5.1824 | 105 | 0.3714 | 0.3433 |
| floor_-30 | 10282 | 1339 | 3.0921 | 5.3185 | 0 | undefined | undefined |
| non_floor | 14407 | 1831 | 3.7472 | 5.0831 | 105 | 0.3714 | 0.3433 |
| risk_ge_-6 | 1593 | 289 | 2.6005 | 3.8704 | 105 | 0.3714 | 0.3501 |
| risk_ge_-4 | 105 | 23 | 2.8166 | 4.2689 | 105 | 0.3714 | 0.4248 |
| near_-6_+-0.5 | 1281 | 337 | 2.8547 | 3.9555 | 0 | undefined | undefined |
| near_-4_+-0.5 | 207 | 50 | 2.5377 | 3.9607 | 67 | 0.3582 | 0.3582 |
| test_horizon_ge_2_days | 17081 | 1800 | 3.7123 | 5.3357 | 86 | 0.4535 | 0.3971 |
| time/0_to_24h | 3848 | 1417 | 2.5779 | 4.4873 | 7 | 0.0000 | 0.0000 |
| time/24_to_48h | 3760 | 1402 | 3.3109 | 5.1397 | 12 | 0.0000 | 0.0000 |
| time/48_to_96h | 7163 | 1501 | 4.2081 | 5.8552 | 30 | 0.2333 | 0.2288 |
| time/above_96h | 9918 | 1602 | 3.3543 | 4.9265 | 56 | 0.5714 | 0.4734 |
| object/DEBRIS | 14032 | 1047 | 3.6582 | 5.3279 | 57 | 0.4912 | 0.4698 |
| object/PAYLOAD | 2528 | 244 | 1.5189 | 3.1298 | 23 | 0.0435 | 0.0538 |
| object/ROCKET BODY | 341 | 30 | 3.6825 | 5.6137 | 1 | 0.0000 | 0.0000 |
| object/TBA | 24 | 3 | 0.6670 | 0.8311 | 0 | undefined | undefined |
| object/UNKNOWN | 7764 | 650 | 3.7784 | 5.4338 | 24 | 0.4167 | 0.2890 |
| missing_optional/yes | 1180 | 86 | 3.0950 | 4.5991 | 2 | 0.0000 | 0.0000 |
| missing_optional/no | 23509 | 1895 | 3.4934 | 5.2100 | 103 | 0.3786 | 0.3578 |
| extreme_sigma/yes | 827 | 207 | 0.8499 | 1.2605 | 0 | undefined | undefined |
| extreme_sigma/no | 23862 | 1927 | 3.5654 | 5.2663 | 105 | 0.3714 | 0.3433 |

### Holdout event-cluster bootstrap

| metric | low_95 | high_95 | valid_replicates | undefined_replicates |
| --- | --- | --- | --- | --- |
| mae | 3.3645 | 3.5883 | 200 | 0 |
| rmse | 5.0241 | 5.3268 | 200 | 0 |
| pc_recall | 0.1968 | 0.5689 | 200 | 0 |
| pc_f2 | 0.1845 | 0.5082 | 200 | 0 |
| latest_pc_recall | 0.0000 | 0.0000 | 179 | 21 |
| latest_pc_f2 | 0.0000 | 0.0000 | 179 | 21 |
| latest_false_alerts_per_1000_events | 0.0000 | 1.5213 | 200 | 0 |

## Interpretation and limitations

- The target is each message's calculated log10(Pc), not an observed collision outcome or final-event forecast. Exponentiating a mean log-risk estimate does not generally yield mean physical Pc.
- The −30 floor remains in training. Consult non-floor and high-risk slices before drawing conclusions from aggregate errors.
- The latest-event positive sample is sparse; recall must be read with its actual denominator and bootstrap undefined counts. Combined-rule performance includes direct distance triggers and is not a measure of Pc estimation alone.
- `max_risk_estimate` is an explicitly permitted upstream risk-derived feature. Compare A0 with the selected model to see dependence on it; this diagnostic alone does not establish leakage.
- The EDA found no permitted-telemetry overlap candidates, but its rounded screen cannot prove independent physical encounters. IDs are source-local and never predictors.
- Test telemetry begins more than two days before TCA. Validation's `test_horizon_ge_2_days` slice is particularly relevant; no test labels were loaded or evaluated.
- Low critical recall would motivate a separately validated alert classifier in a later experiment, not silent threshold changes, inflated Pc values, or a hard low-risk gate. No auxiliary classifier was trained here.
- This model does not reconstruct complete covariance, physical collision probability, or maneuver commands; no calibration or operational-certification claim is made.

## Reproduction and artifacts

Run `python scripts/train.py` for cached validation experiments; add `--finalize` for the one-time holdout evaluation and test export. `python scripts/predict.py` regenerates exports with the frozen selected model. The train command reuses the EDA split and validates raw-file fingerprints. It never refits the selected model on validation/holdout.

Run directory: `artifacts/model_runs/87c8704204c5`. It contains candidate pipelines, validation predictions, detailed metrics, the frozen selection, and the manifest. Stable model: `artifacts/selected_model.joblib`; metadata: `artifacts/selected_model_metadata.json`. Handoff fixture: `tests/fixtures/telemetry.csv` and its expected predictions. Test exports preserve original row order and IDs; `test_submission.csv` contains `row_id,event_id,predicted_risk` where predicted risk is raw log10(Pc). Confirm the organizer's exact header before submission.

Checks: target and ID exclusion, column-order invariance, save/load equivalence, unknown category, missing optional values/columns, invalid required input rejection, early stopping disabled. Dependency versions: `{"joblib": "1.6.0", "numpy": "2.2.2", "pandas": "2.2.2", "scikit-learn": "1.9.1", "threadpoolctl": "3.7.0"}`. The manifest records data/source/split hashes and the model metadata includes feature order, units, seed, and training event IDs. Joblib files must come from a trusted source.

## Research informing this experiment

- [ESA challenge analysis](https://www.honda-ri.de/pubs/pdf/4688.pdf): strong baselines, tree models, and evaluation pitfalls; its final-event task differs from ours.
- [Supervised learning with missing values](https://arxiv.org/abs/1902.06931): motivates comparing simple imputation with direct missing-value handling.
- [Tabular-model benchmark](https://arxiv.org/abs/2207.08815): supports a boosted-tree starting point, without claiming universal superiority.
- [Satellite two-stage models](https://www.sciencedirect.com/science/article/pii/S246889672100094X): informs possible future alert classification, not a result claimed for our task.
- [NASA risk assessment](https://www.nasa.gov/cara/step-2-close-approach-risk-assessment/): physical Pc requires more than our permitted telemetry subset.
