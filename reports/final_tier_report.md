# Final-tier model report

**Task (organizer clarification):** predict each event's final tier, from its last message before closest approach, using only messages with `time_to_tca >= 2` days. Tiers follow the brief: Critical if risk >= -4 or miss distance < 1,000 m; Elevated if not Critical and (risk >= -6 or miss distance <= 5,000 m); otherwise Safe. `risk = -30` is a placeholder for Pc ~ 0.

**Selected: HR1_distance_change_regressor**, model version `final-tier-d7496dd831de`. On forecast events in CV: Critical recall 0.883 (persistence 0.813), precision 0.698 (0.796), F2 0.839 (0.810); missed Critical events 60 vs 96.

## Data

- Labels recomputed from `train_data.csv` and checked against `challenge-1-event-labels.xlsx` (identical tiers and eligibility).
- 11942 of 13154 training events have at least one message 2+ days out; the others cannot be scored and are excluded.
- Critical events are almost all distance-driven: 831 of 834 usable Critical events have a final miss distance below 1 km; 11 have Pc >= 1e-4.
- Frozen event split (seed 42) reused: train+validation for development, holdout for one final evaluation.
- **Forecast events** are those whose final message is inside 2 days of TCA. In the other events the final message *is* the latest >= 2-day message, so persistence is exact by construction. Test events' latest messages cluster just above 2 days (median 2.19 d vs 2.25 d in training, 95th percentile 2.93 d vs 5.76 d), consistent with events cropped at 2 days, so forecast events are the headline population.

| partition | Safe | Elevated | Critical |
| --- | --- | --- | --- |
| holdout | 1371 | 311 | 118 |
| train | 6247 | 1510 | 580 |
| validation | 1328 | 341 | 136 |

## Features (one row per event; `risk` never used)

Latest >= 2-day message: the 12 permitted columns plus physics terms (combined radial sigma with placeholder sigmas set to missing, miss distance / sigma, radial fraction, object-size proxy from `max_risk_estimate`). History over all >= 2-day messages: message count, observed span, min/max of miss distance, `max_risk_estimate` and miss/sigma, per-day slopes and net changes of distance, risk bound, relative position and sigmas, sigma shrinkage, and a linear extrapolation of miss distance to TCA (a heuristic, not orbit propagation).

## Cross-validation on development events (10142 events, 7558 forecast events, 5 stratified folds)

Decision rule: Critical if the Critical score >= t_c; else Elevated if the Elevated-or-worse score >= t_e; else Safe. Classifier scores are P(Critical) and P(Elevated or Critical); the change regressor's score is -log10(predicted final miss distance). Each fold's thresholds are tuned on the other folds' out-of-fold forecast events (Critical F2, then Elevated-or-worse F2). Final thresholds for the selected model: t_c = -3.116, t_e = -3.713.

Ideas compared (all on the same folds):

- **P0/P1 persistence:** tier rules on the latest >= 2-day message (P1 uses `max_risk_estimate`, an upper bound, as the Pc proxy).
- **C1/C3:** classifier on the latest message, or latest plus history with balanced class weights.
- **H-prefixed hybrids:** the persistence tier is a floor; the model can only raise it.
- **F3:** fit only on forecast events, matching the test population.
- **A3/AF3 prefix augmentation:** also fit on every >= 2-day prefix of each training event (weighted 1/prefixes per event), so the model sees the cut-off positions test events have.
- **M3:** median imputation with missingness indicators instead of native NaN handling (Josse et al. 2019); F10/AP are already forward-filled within events and placeholder sigmas set to missing.
- **R1:** regress the log change in miss distance from the latest >= 2-day message to the final message, then threshold the predicted final distance.

Selection: highest forecast-event Critical F2 among learned candidates, ties by macro-F1. Comparing about a dozen variants on the same folds makes the winner's CV score slightly optimistic; the holdout gives the unbiased number.

### Forecast events (headline)

| model | Critical recall | Critical precision | Critical F2 | TP/FP/FN | Elevated+ recall | macro-F1 | balanced acc. | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0_persist_distance | 0.813 | 0.796 | 0.810 | 418/107/96 | 0.910 | 0.867 | 0.868 | 0.931 |
| P1_persist_distance_maxrisk | 0.823 | 0.740 | 0.805 | 423/149/91 | 0.942 | 0.714 | 0.787 | 0.734 |
| C1_latest_message | 0.850 | 0.671 | 0.807 | 437/214/77 | 0.936 | 0.827 | 0.861 | 0.906 |
| C3_history_balanced | 0.854 | 0.722 | 0.824 | 439/169/75 | 0.939 | 0.846 | 0.875 | 0.914 |
| F3_forecast_events_only | 0.889 | 0.675 | 0.836 | 457/220/57 | 0.927 | 0.834 | 0.871 | 0.909 |
| A3_prefix_augmented | 0.874 | 0.672 | 0.824 | 449/219/65 | 0.939 | 0.824 | 0.865 | 0.899 |
| AF3_augmented_forecast_only | 0.897 | 0.629 | 0.826 | 461/272/53 | 0.930 | 0.819 | 0.863 | 0.903 |
| M3_median_imputed_indicators | 0.875 | 0.682 | 0.828 | 450/210/64 | 0.933 | 0.834 | 0.870 | 0.908 |
| R1_distance_change_regressor | 0.872 | 0.720 | 0.836 | 448/174/66 | 0.925 | 0.841 | 0.870 | 0.910 |
| HC3_history_balanced | 0.856 | 0.730 | 0.827 | 440/163/74 | 0.939 | 0.849 | 0.877 | 0.915 |
| HF3_forecast_events_only | 0.893 | 0.671 | 0.838 | 459/225/55 | 0.931 | 0.834 | 0.873 | 0.908 |
| HA3_prefix_augmented | 0.866 | 0.712 | 0.830 | 445/180/69 | 0.926 | 0.847 | 0.873 | 0.918 |
| HAF3_augmented_forecast_only | 0.875 | 0.656 | 0.821 | 450/236/64 | 0.929 | 0.830 | 0.866 | 0.911 |
| HR1_distance_change_regressor | 0.883 | 0.698 | 0.839 | 454/196/60 | 0.919 | 0.847 | 0.873 | 0.920 |

### All development events

| model | Critical recall | Critical precision | Critical F2 | TP/FP/FN | Elevated+ recall | macro-F1 | balanced acc. | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0_persist_distance | 0.866 | 0.853 | 0.863 | 620/107/96 | 0.887 | 0.886 | 0.881 | 0.937 |
| P1_persist_distance_maxrisk | 0.873 | 0.803 | 0.858 | 625/153/91 | 0.956 | 0.754 | 0.827 | 0.771 |
| C1_latest_message | 0.891 | 0.725 | 0.852 | 638/242/78 | 0.945 | 0.853 | 0.885 | 0.919 |
| C3_history_balanced | 0.894 | 0.775 | 0.867 | 640/186/76 | 0.948 | 0.870 | 0.897 | 0.926 |
| F3_forecast_events_only | 0.911 | 0.688 | 0.855 | 652/296/64 | 0.904 | 0.828 | 0.863 | 0.903 |
| A3_prefix_augmented | 0.895 | 0.691 | 0.845 | 641/287/75 | 0.929 | 0.827 | 0.867 | 0.901 |
| AF3_augmented_forecast_only | 0.904 | 0.639 | 0.834 | 647/366/69 | 0.893 | 0.806 | 0.846 | 0.893 |
| M3_median_imputed_indicators | 0.909 | 0.731 | 0.867 | 651/240/65 | 0.944 | 0.856 | 0.891 | 0.919 |
| R1_distance_change_regressor | 0.901 | 0.740 | 0.863 | 645/227/71 | 0.896 | 0.844 | 0.867 | 0.912 |
| HC3_history_balanced | 0.897 | 0.781 | 0.871 | 642/180/74 | 0.949 | 0.872 | 0.899 | 0.927 |
| HF3_forecast_events_only | 0.923 | 0.687 | 0.864 | 661/301/55 | 0.917 | 0.832 | 0.872 | 0.905 |
| HA3_prefix_augmented | 0.904 | 0.742 | 0.866 | 647/225/69 | 0.915 | 0.859 | 0.882 | 0.923 |
| HAF3_augmented_forecast_only | 0.911 | 0.679 | 0.853 | 652/308/64 | 0.903 | 0.832 | 0.865 | 0.909 |
| HR1_distance_change_regressor | 0.916 | 0.725 | 0.870 | 656/249/60 | 0.894 | 0.853 | 0.873 | 0.923 |

Paired event bootstrap, selected minus persistence on forecast events (1,000 resamples): Critical F2 [+0.010, +0.049] (share of resamples > 0: 1.00); Critical recall [+0.049, +0.093] (share of resamples > 0: 1.00); missed Critical events [-48, -25] (share of resamples > 0: 0.00).

Selected-model CV confusion matrix (forecast events):

```text
truth \ predicted      Safe  Elevated  Critical
Safe                   5385       205        18
Elevated                145      1113       178
Critical                 13        47       454
```

## Holdout result (evaluated once)

Forecast events (final message inside 2 days): 1334 events, 76 Critical. 95% event-bootstrap intervals in brackets. All holdout events: selected Critical F2 0.870, persistence 0.860.

| model | Critical recall | Critical precision | Critical F2 | macro-F1 |
| --- | --- | --- | --- | --- |
| HR1_distance_change_regressor | 0.882 [0.806–0.949] | 0.677 [0.580–0.768] | 0.831 [0.762–0.890] | 0.847 [0.815–0.879] |
| P0 persistence | 0.789 [0.700–0.883] | 0.811 [0.719–0.900] | 0.794 [0.716–0.870] | 0.870 [0.839–0.899] |

Selected model confusion matrix:

```text
truth \ predicted      Safe  Elevated  Critical
Safe                    979        27         4
Elevated                 29       191        28
Critical                  1         8        67
```

Persistence confusion matrix:

```text
truth \ predicted      Safe  Elevated  Critical
Safe                    986        21         3
Elevated                 32       205        11
Critical                  1        15        60
```

The holdout events were never used for EDA, feature design or selection in this task. Their row-level labels were scored once for the earlier M3 row-level model; that result did not inform this experiment.

## Test demo predictions

`outputs/test_event_predictions.csv`: one row per test event (2167 events), with the model scores (tier probabilities or predicted final distance), the decision, the persistence tier and latest geometry. Test events have no final outcomes, so these are not scored. Predicted tiers: Safe 1503, Elevated 436, Critical 228. Test event IDs restart from 0 and are unrelated to training IDs.

## Limitations

- Critical is mostly a miss-distance outcome; the model forecasts how the screening distance evolves over the last two days, which the permitted columns describe only partially (no along-track/cross-track sigmas, no orbits).
- F2 favours recall: false alarms are cheaper than missed Critical events. Thresholds can be moved for a different operating point; report the new confusion matrix if so.
- Probabilities are classifier scores tuned for ranking and thresholds, not calibrated collision probabilities.
