# Model development: how we iterated from a row-level regressor to a final-tier forecaster

This document records the full modelling process for Challenge #1: what we built first, what the reviews found, how the task changed, every experiment we ran, the results, and why the final model looks the way it does. Detailed generated tables are in the linked reports.

| Report | Contents |
|---|---|
| [eda_summary.md](eda_summary.md) | Data audit, frozen event split |
| [model_report.md](model_report.md) | First modelling pass (M3, row-level) |
| [experiment_v2.md](experiment_v2.md) | Row-level experiment v2 |
| [final_tier_report.md](final_tier_report.md) | Final-tier experiments, selection, **holdout result** |
| [final_tier_ensembles.md](final_tier_ensembles.md) | Ensembles, stacking, nested search, row-level Pc |
| [literature_review.md](literature_review.md) | Prior work and how it shaped the experiments |

## Headline result

**Task:** predict each conjunction event's final risk tier (Safe / Elevated / Critical) from messages received at least 2 days before closest approach.

**Model:** a gradient-boosted regressor forecasts how much the miss distance will change between the latest >= 2-day message and the final message. It flags Critical when the forecast final distance is about 1.3 km or less, and Elevated when it is about 5.2 km or less. The current tier from the latest message (persistence) acts as a floor the model can only raise.

**Holdout (evaluated once, 1,334 forecast events, 76 Critical):**

| | Critical recall | Critical precision | Critical F2 | Missed Critical |
|---|---|---|---|---|
| **Our model** | **0.882** [0.81–0.95] | 0.677 [0.58–0.77] | **0.831** [0.76–0.89] | **9** |
| Persistence baseline | 0.789 [0.70–0.88] | 0.811 [0.72–0.90] | 0.794 [0.72–0.87] | 16 |

- **Fewer missed collisions, with statistical support.** A paired bootstrap on the same holdout events gives 2 to 13 fewer missed Critical events and recall up 3 to 16 points; every resample favours the model.
- **F2 is higher, but the holdout is too small to call that significant:** the paired interval is −0.018 to +0.099, with 90% of resamples above zero. Cross-validation (514 Critical events) showed a significant F2 gain of +0.010 to +0.049.
- **The cost is false alarms:** 32 vs 14 non-Critical events flagged Critical (mostly Elevated events), the trade the F2 metric is designed to accept.
- **The model generalizes as expected:** holdout F2 0.831 vs 0.839 in cross-validation.

## 1. Starting point: the first modelling pass (M3)

The first pipeline treated the problem as **row-level regression**: predict each message's `risk` = log10(Pc) from the 12 permitted columns. It used a frozen 70/15/15 event split (seed 42), compared median/mean dummies with gradient-boosted trees (median imputation, native missing values, geometry features), and opened its holdout once. The selected model, M3, reached validation Pc-critical F2 0.44 (row level).

## 2. Review of M3: what was wrong

| Finding | Evidence | Consequence |
|---|---|---|
| A one-line rule beat the model | `max_risk_estimate` is the maximum Pc over covariance scaling (ESA data dictionary), so it bounds `risk` from above in 99.65% of rows. Flagging `max_risk_estimate >= -4`: recall 1.00, F2 0.57 vs M3's 0.49 / 0.44 | Add it as a baseline; cap predictions at it |
| The −30 floor distorted training | 41% of targets are the "too small to compute" placeholder; M3's MSE on rows with risk >= −6 was 10.8 vs 0.33 for raw `max_risk_estimate` | Floor the target; score MSE only on high-risk rows (Kelvins convention) |
| Placeholder sigmas treated as data | 63,781,363 m and 63,781,361 m = 10 × Earth's equatorial radius: a "no covariance" default | Set to missing with a flag; match both values with a tolerance |
| Circular or empty metrics | Combined-tier recall near 0.99 because miss distance is both input and label; latest-message metrics had 1–2 positives | Report Pc-only and horizon-matched metrics |
| Batch rejection | One invalid row rejected the whole prediction batch | Score rows individually with `needs_review` |

## 3. Row-level experiment v2

**Changes:** placeholder sigmas set to missing; physics features (miss distance / combined radial sigma, radial fraction, an object-size proxy backed out of `max_risk_estimate` via Alfano's maximum-Pc relation); F10/AP forward-filled within each event; targets floored at −8 or −6.001, or modelled as the gap below `max_risk_estimate`; predictions capped at `max_risk_estimate`. Evaluation used 5-fold grouped CV and scored only messages >= 2 days before TCA, matching the test set.

| Model | MSE (risk >= −6) | Critical F2 (fold-validated alert threshold) |
|---|---|---|
| `max_risk_estimate` rule | 0.295 | 0.705 |
| M3 features | 0.632 | 0.419 |
| Physics features, raw target | 0.497 | 0.553 |
| **Physics features, target floored at −6.001** | **0.164** | **0.748** |

**Lesson:** flooring the target did most of the work. The production code (`src/model.py`) gained the placeholder handling, per-row `needs_review` scoring and a `p > 0` rule, without changing the frozen M3 artifact's predictions.

## 4. The task changed: organizer clarification

The organizers clarified that the scored task is each event's **final tier**, from its last message before closest approach, predicted using only messages with `time_to_tca >= 2` days, with an event-level held-out split. They also supplied `challenge-1-event-labels.xlsx`.

- **Labels verified:** tiers, eligibility and message counts recomputed from `train_data.csv` match the sheet exactly.
- **Critical is a distance outcome:** 831 of 834 usable Critical events have a final miss distance below 1 km; only 11 have Pc >= 1e-4.
- **Persistence is strong:** applying the tier rules to the latest >= 2-day message gives Critical F2 0.86 over all development events.
- **That number is flattered.** 25% of development events have no messages inside 2 days, so their "final" message *is* the latest >= 2-day one and persistence is exact. Test events look cropped at 2 days (latest message: median 2.19 d, 95th percentile 2.93 d, vs 2.25 d / 5.76 d in training). **We therefore evaluate on forecast events** (final message inside 2 days: 7,558 development events, 514 Critical), where persistence scores F2 0.81.

**Protocol:** reuse the frozen split (85% development with 5 stratified folds, 15% holdout); one row per event; features only from >= 2-day messages; `risk` never an input. Decision rule: Critical if the Critical score passes threshold t_c, else Elevated if the Elevated-or-worse score passes t_e. Each fold's thresholds are tuned on the other folds' out-of-fold forecast events, maximizing Critical F2 and then Elevated-or-worse F2.

**Event features:** the latest >= 2-day message's permitted columns and physics terms, plus history over all >= 2-day messages: count, span, min/max of miss distance, `max_risk_estimate` and miss/sigma, per-day slopes and net changes, sigma shrinkage, and a linear extrapolation of miss distance to TCA.

## 5. Final-tier iteration 1: classifiers vs persistence

| Model (all development events) | Critical recall | Precision | F2 |
|---|---|---|---|
| Persistence | 0.866 | 0.853 | 0.863 |
| Classifier, latest message | 0.880 | 0.773 | 0.856 |
| Classifier, latest + history, balanced | 0.894 | 0.775 | 0.867 |

The learned classifier only tied persistence. This led to the forecast-event headline (section 4) and the next round of ideas.

## 6. Final-tier iteration 2: combining persistence with models

| Idea (forecast events) | Recall | Precision | F2 | Missed |
|---|---|---|---|---|
| Persistence | 0.813 | **0.796** | 0.810 | 96 |
| History classifier, balanced | 0.854 | 0.722 | 0.824 | 75 |
| Median imputation + missing indicators | 0.875 | 0.682 | 0.828 | 64 |
| Prefix augmentation (train on every >= 2-day prefix) | 0.874 | 0.672 | 0.824 | 65 |
| Train on forecast events only | 0.889 | 0.675 | 0.836 | 57 |
| Regress log change in miss distance | 0.872 | 0.720 | 0.836 | 66 |
| Forecast-only classifier + persistence floor | **0.893** | 0.671 | 0.838 | **55** |
| **Distance-change regressor + persistence floor (selected)** | 0.883 | 0.698 | **0.839** | 60 |

- **A persistence floor helps every model** a little: the model can only promote events, never demote them.
- **Forecasting the change in miss distance** works as well as classifying tiers directly, with better precision.
- **Imputation makes no real difference** for trees (Josse et al. 2019); the domain fixes (placeholders, space-weather fill) mattered more.
- **Augmentation did not help** on its own.
- **The gain over persistence is real in CV:** a paired bootstrap gives F2 +0.010 to +0.049 and 25 to 48 fewer missed Critical events.

## 7. Final-tier iteration 3: ensembles, tuning, row-level Pc

Motivated by the [literature review](literature_review.md), which found the Kelvins winner used a recall-promoting threshold cascade, the only top-team ensemble was a majority vote, and gains over persistence were small everywhere.

| Candidate | F2 | Missed | ΔF2 vs selected [95%] |
|---|---|---|---|
| **Selected (regressor + floor)** | **0.839** | 60 | — |
| Mean(forecast-only classifier, calibrated regressor) | 0.836 | 62 | −0.015 to +0.008 |
| Mean of four models | 0.837 | 59 | −0.015 to +0.012 |
| Logistic stack on out-of-fold scores | 0.830 | 67 | −0.022 to +0.003 |
| Regressor, nested random search (12 configs) | 0.834 | 63 | −0.013 to +0.004 |
| Classifier, nested random search | 0.835 | 61 | −0.021 to +0.013 |
| Regressor + row-level Pc features | 0.835 | 65 | −0.016 to +0.005 |
| Classifier + row-level Pc features | 0.831 | 60 | −0.025 to +0.007 |

- **None beat the selection;** every interval includes zero. All still beat persistence.
- **Ensembles have little to cancel:** the base models are trees on the same features and make correlated errors.
- **The hyperparameter optimum is flat:** chosen configurations differ across folds, and nested scores match the defaults.
- **Row-level Pc adds nothing:** Pc decides only 11 Critical events, and `max_risk_estimate` already carries that signal.

**Decision:** keep the distance-change regressor with the persistence floor and freeze it.

## 8. Holdout evaluation

Run once with `python scripts/train_final_tier.py --finalize`; the ledger (`artifacts/final_tier/holdout_ledger.json`) records model version `final-tier-d7496dd831de` and blocks any re-evaluation of a different model. Results are in the headline section above and [final_tier_report.md](final_tier_report.md).

| Holdout confusion matrix (forecast events) | Pred. Safe | Pred. Elevated | Pred. Critical |
|---|---|---|---|
| **Our model** — true Safe | 979 | 27 | 4 |
| true Elevated | 29 | 191 | 28 |
| true Critical | 1 | 8 | **67** |
| **Persistence** — true Safe | 986 | 21 | 3 |
| true Elevated | 32 | 205 | 11 |
| true Critical | 1 | 15 | 60 |

Across all 1,800 holdout events (including those where persistence is exact): Critical F2 0.870 vs 0.860, recall 0.915 vs 0.856.

**Disclosure:** the holdout events were never used for EDA, feature design or selection in this task. Their row-level labels were scored once for the earlier M3 model; that result did not inform any later decision.

## 9. Test predictions for the demo

`outputs/test_event_predictions.csv` holds one row per test event (2,167): 228 Critical, 436 Elevated, 1,503 Safe, with the predicted final miss distance, the persistence tier and latest geometry. The model raises 49 events from Elevated to Critical and 18 from Safe to Elevated, and never lowers a tier. Test events have no outcomes, so these are not scored. Test event IDs restart at 0 and are never joined to training IDs.

## 10. What we learned

1. **Beat the right baseline.** Persistence is strong, as in the Kelvins challenge; dummy baselines hide that.
2. **Evaluate on the population you will predict.** Removing events where persistence is trivially exact changed the picture.
3. **Domain fixes beat model complexity.** Floored targets, placeholder sigmas and the `max_risk_estimate` bound mattered more than ensembles or tuning.
4. **Promote, don't demote.** A persistence floor with recall-tuned thresholds is a principled version of the winning Kelvins approach.
5. **Report uncertainty honestly.** Paired bootstraps separate real gains (fewer missed Critical events) from noise (most model-vs-model differences).

## Limitations

- **Critical is mostly a miss-distance outcome.** The permitted columns lack along-track and cross-track sigmas and orbital state, which limits how well the last two days of distance evolution can be forecast.
- **The holdout is small** (76 Critical forecast events), so its intervals are wide.
- **Scores are not calibrated collision probabilities;** thresholds can move for a different operating point, with a new confusion matrix.
- **Tier thresholds follow the hackathon brief,** not certification against a specific NASA or ESA standard.

## Reproduce

```bash
python -m pip install -r requirements-model.txt
python -m unittest discover -s tests
python -m src.experiment_v2                    # row-level experiment v2
python scripts/train_final_tier.py             # final-tier CV, selection, test predictions
python -m src.final_tier_ensembles             # ensembles, nested search, row-level Pc (about 20 min)
python scripts/train_final_tier.py --finalize  # one-time holdout (already done; reruns reuse the saved result)
```
