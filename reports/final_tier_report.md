# Final-tier model report

**Task (organizer clarification):** predict each event's final tier, from its last message before closest approach, using only messages with `time_to_tca >= 2` days. Tiers follow the brief: Critical if risk >= -4 or miss distance < 1,000 m; Elevated if not Critical and (risk >= -6 or miss distance <= 5,000 m); otherwise Safe. `risk = -30` is a placeholder for Pc ~ 0.

**Selected: C3_history_balanced**, model version `final-tier-5332b6694901`. CV Critical recall 0.894, precision 0.775, F2 0.867.

## Data

- Labels recomputed from `train_data.csv` and checked against `challenge-1-event-labels.xlsx` (identical tiers and eligibility).
- 11942 of 13154 training events have at least one message 2+ days out; the others cannot be scored and are excluded.
- Critical events are almost all distance-driven: 831 of 834 usable Critical events have a final miss distance below 1 km; 11 have Pc >= 1e-4.
- Frozen event split (seed 42) reused: train+validation for development, holdout for one final evaluation.

| partition | Safe | Elevated | Critical |
| --- | --- | --- | --- |
| holdout | 1371 | 311 | 118 |
| train | 6247 | 1510 | 580 |
| validation | 1328 | 341 | 136 |

## Features (one row per event; `risk` never used)

Latest >= 2-day message: the 12 permitted columns plus physics terms (combined radial sigma with placeholder sigmas set to missing, miss distance / sigma, radial fraction, object-size proxy from `max_risk_estimate`). History over all >= 2-day messages: message count, observed span, min/max of miss distance, `max_risk_estimate` and miss/sigma, per-day slopes and net changes of distance, risk bound, relative position and sigmas, sigma shrinkage, and a linear extrapolation of miss distance to TCA (a heuristic, not orbit propagation).

## Cross-validation on development events (10142 events, 5 stratified folds)

Decision rule: Critical if P(Critical) >= t_c; else Elevated if P(Elevated or Critical) >= t_e; else Safe. Each fold's thresholds are tuned on the other folds' out-of-fold predictions (Critical F2, then Elevated-or-worse F2). Final thresholds: t_c = 0.075, t_e = 0.1. Persistence baselines apply the tier rules to the latest >= 2-day message (P0 ignores Pc; P1 uses `max_risk_estimate`, an upper bound, as the Pc proxy).

| model | Critical recall | Critical precision | Critical F2 | TP/FP/FN | Elevated+ recall | macro-F1 | balanced acc. | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0_persist_distance | 0.866 | 0.853 | 0.863 | 620/107/96 | 0.887 | 0.886 | 0.881 | 0.937 |
| P1_persist_distance_maxrisk | 0.873 | 0.803 | 0.858 | 625/153/91 | 0.956 | 0.754 | 0.827 | 0.771 |
| C1_latest_message | 0.880 | 0.773 | 0.856 | 630/185/86 | 0.945 | 0.864 | 0.890 | 0.923 |
| C2_latest_plus_history | 0.888 | 0.770 | 0.862 | 636/190/80 | 0.940 | 0.865 | 0.891 | 0.924 |
| C3_history_balanced | 0.894 | 0.775 | 0.867 | 640/186/76 | 0.948 | 0.870 | 0.897 | 0.926 |

Selected-model CV confusion matrix:

```text
truth \ predicted      Safe  Elevated  Critical
Safe                   7175       367        33
Elevated                123      1575       153
Critical                 10        66       640
```

## Holdout result

Not opened yet. Run `python scripts/train_final_tier.py --finalize` once the selection is frozen.

## Test demo predictions

`outputs/test_event_predictions.csv`: one row per test event (2167 events), with tier probabilities, the decision, the persistence tier and latest geometry. Test events have no final outcomes, so these are not scored. Predicted tiers: Safe 1439, Elevated 515, Critical 213. Test event IDs restart from 0 and are unrelated to training IDs.

## Limitations

- Critical is mostly a miss-distance outcome; the model forecasts how the screening distance evolves over the last two days, which the permitted columns describe only partially (no along-track/cross-track sigmas, no orbits).
- F2 favours recall: false alarms are cheaper than missed Critical events. Thresholds can be moved for a different operating point; report the new confusion matrix if so.
- Probabilities are classifier scores tuned for ranking and thresholds, not calibrated collision probabilities.
