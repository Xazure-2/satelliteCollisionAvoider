# Experiment log: from row-level Pc to final-tier forecasting

A narrative record of what we tried, what we found, and why the current model looks the way it does. Generated reports hold the full tables: [experiment_v2.md](experiment_v2.md) (row-level) and [final_tier_report.md](final_tier_report.md) (event-level). All numbers below are cross-validation results on development events only; **the holdout has not been opened for the final-tier task.**

## 1. Review of the first modelling pass (M3, row-level)

The original pipeline (see [model_report.md](model_report.md)) regressed each message's `risk` = log10(Pc). The review found:

- **A one-line rule beat the model.** `max_risk_estimate` is the maximum Pc over covariance scaling (ESA data dictionary), so it bounds `risk` from above in 99.65% of rows. On validation, flagging `max_risk_estimate >= -4` had Critical recall 1.00 and F2 0.57, vs M3's 0.49 and 0.44.
- **The −30 floor distorted training.** 41% of targets are the "too small to compute" placeholder. Squared-error regression pulled high-risk predictions down: M3's MSE on rows with risk >= −6 was 10.8, vs 0.33 for raw `max_risk_estimate`.
- **Placeholder sigmas were treated as measurements.** 63,781,363 m and 63,781,361 m are 10 × Earth's equatorial radius, a "no covariance" default, not a real uncertainty.
- **Some headline metrics were circular or uninformative.** Combined-tier recall near 0.99 was automatic, because miss distance is both an input and part of the label. Latest-message metrics had 1–2 positives.

## 2. Row-level experiment v2

5-fold grouped CV on train+validation events, scored on messages >= 2 days before TCA (like every test message):

| Model | MSE (risk >= −6) | Critical F2 (tuned alert) |
|---|---|---|
| `max_risk_estimate` rule | 0.295 | 0.705 |
| M3 features | 0.632 | 0.419 |
| Physics features + target floored at −6.001 | **0.164** | **0.748** |

Flooring the target did most of the work; physics features (miss distance / combined sigma, object-size proxy from `max_risk_estimate`) helped too. Placeholder sigmas are now set to missing with flags, and `prediction_frame` scores invalid rows individually (`needs_review`) instead of rejecting the batch.

## 3. The organizer clarification changed the task

The scored task is each event's **final tier** (from its last message before closest approach), predicted from messages >= 2 days out. Tiers: Critical if risk >= −4 or miss distance < 1,000 m; Elevated if not Critical and (risk >= −6 or miss distance <= 5,000 m); otherwise Safe.

What the data says about this task:

- **Critical is a distance outcome.** 831 of 834 usable Critical events have a final miss distance below 1 km; only 11 have Pc >= 1e-4.
- **Persistence is a strong baseline.** Applying the tier rules to the latest >= 2-day message gets Critical F2 0.86 across all development events.
- **That number is flattered.** 2,584 development events (25%) have no messages inside 2 days, so their "final" message *is* the latest >= 2-day one and persistence is exact. Test events look cropped at 2 days: their latest message has a median of 2.19 d (95th percentile 2.93 d), vs 2.25 d (5.76 d) in training. We therefore use **forecast events** (final message inside 2 days, 7,558 development events, 514 Critical) as the headline population. On those, persistence scores F2 0.81.

## 4. Final-tier experiments (forecast events, 5-fold CV)

| Idea | Critical recall | Critical precision | F2 | Missed Critical |
|---|---|---|---|---|
| Persistence (latest >= 2-day distance tier) | 0.813 | **0.796** | 0.810 | 96 |
| Classifier, latest message only | 0.850 | 0.671 | 0.807 | 77 |
| Classifier, latest + history, balanced | 0.854 | 0.722 | 0.824 | 75 |
| Median imputation + missing indicators | 0.875 | 0.682 | 0.828 | 64 |
| Prefix augmentation | 0.874 | 0.672 | 0.824 | 65 |
| Train on forecast events only | 0.889 | 0.675 | 0.836 | 57 |
| Regress log change in miss distance | 0.872 | 0.720 | 0.836 | 66 |
| Forecast-only + persistence floor | **0.893** | 0.671 | 0.838 | **55** |
| **Distance-change regressor + persistence floor (selected)** | 0.883 | 0.698 | **0.839** | 60 |

- **Selected model:** a gradient-boosted regressor predicts the log change in miss distance between the latest >= 2-day message and the final message. It flags Critical when the predicted final distance is <= about 1.3 km and Elevated when <= about 5.2 km. The persistence tier is a floor the model can only raise.
- **The gain over persistence is consistent.** A paired event bootstrap (1,000 resamples, same events for both) gives F2 +0.010 to +0.049 and 25 to 48 fewer missed Critical events; every resample favours the model. The cost is about 90 more false alarms, mostly Elevated events flagged Critical.
- **The top variants are statistically tied** (F2 0.836–0.839). Choosing among about a dozen variants on the same folds makes the winner's CV score slightly optimistic.
- **Imputation barely matters for trees,** consistent with Josse et al. (2019). The useful data fixes were the placeholder sigmas and filling F10/AP within events.
- **Prefix augmentation did not help** on its own; combined with forecast-only training it raised recall but cost too much precision.

## 5. Test demo predictions

`outputs/test_event_predictions.csv` holds one row per test event (2,167): 228 Critical, 436 Elevated, 1,503 Safe. The model never lowers a persistence tier; it raises 49 events from Elevated to Critical and 18 from Safe to Elevated. Test events have no outcomes, so these are not scored.

## 6. Ensembles, hyperparameter search and row-level Pc

Motivated by the [literature review](literature_review.md): the Kelvins winner used a recall-promoting threshold cascade, the only top-team ensemble was a majority vote, and the organizers' LightGBM gained about 1.3% F2 over persistence. Full table: [final_tier_ensembles.md](final_tier_ensembles.md). Same folds, persistence floor and fold-validated thresholds; paired bootstrap intervals against the current selection (HR1):

| Candidate | Critical F2 | Missed Critical | ΔF2 vs HR1 [95%] |
|---|---|---|---|
| **HR1 distance-change regressor + floor (current)** | **0.839** | 60 | — |
| E1 mean(forecast-only classifier, calibrated regressor) | 0.836 | 62 | −0.015 to +0.008 |
| E2 mean of four models | 0.837 | 59 | −0.015 to +0.012 |
| S1 logistic stack (out-of-fold base scores) | 0.830 | 67 | −0.022 to +0.003 |
| T1 regressor, nested random search (12 configs) | 0.834 | 63 | −0.013 to +0.004 |
| T2 classifier, nested random search | 0.835 | 61 | −0.021 to +0.013 |
| P1 regressor + row-level Pc features | 0.835 | 65 | −0.016 to +0.005 |
| P2 classifier + row-level Pc features | 0.831 | 60 | −0.025 to +0.007 |
| Persistence | 0.810 | 96 | — |

- **No idea beat HR1;** every interval against it includes zero. All still beat persistence.
- **Ensembles and stacking add nothing:** the base models make highly correlated errors (all are trees on the same features), so averaging has little to cancel.
- **Hyperparameter search finds a flat optimum:** chosen configurations differ across folds, and nested scores are no better than the defaults.
- **Row-level Pc features do not help:** only 11 usable Critical events are Pc-driven; the rest are miss-distance outcomes, and the row model's information is already in `max_risk_estimate`.
- **Decision:** keep HR1. The remaining error is about how miss distance evolves in the last two days, which the permitted columns describe only partly (no along-track or cross-track sigmas, no orbital state).

## 7. Open items

- Open the holdout once (`python scripts/train_final_tier.py --finalize`) for an unbiased estimate against persistence.
- Report and literature rewrite: Uriot et al. 2022 (Kelvins challenge), Foster & Estes / Alfano on Pc, NASA CA handbook, CCSDS 508.0 (CDM), ESA standards once editions are confirmed.

## Reproduce

```bash
python -m pip install -r requirements-model.txt
python -m unittest discover -s tests
python -m src.experiment_v2            # row-level experiment v2
python scripts/train_final_tier.py     # final-tier CV, selection and test event predictions
python -m src.final_tier_ensembles     # ensembles, nested search, row-level Pc (about 20 minutes)
```
