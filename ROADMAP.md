# Satellite Collision Avoidance — Hackathon Roadmap

Prepared 31 October 2026. Team: three implementation owners plus you as integration/demo lead. Schedule below assumes a 12-hour build; compress the timeboxes if needed without changing the integration order.

> **Status update (modelling complete).** The organizers clarified the task: predict each event's **final tier** from messages with `time_to_tca >= 2` days, and report metrics on an event-level held-out split of `train_data`. `test_data` has no outcomes and is for demo predictions. Teammate 1's work is done:
>
> - **Final-tier model** (scored task): `python scripts/train_final_tier.py` → `outputs/test_event_predictions.csv`, one row per test event. Holdout (evaluated once): Critical recall 0.882 vs 0.789 for persistence. See [reports/model_development.md](reports/model_development.md).
> - **Per-message Pc** (dashboard): `python scripts/train_row_model.py` then `python scripts/predict.py` → `outputs/test_predictions.csv`.
> - The row-level `test_submission.csv` below is obsolete; section 6 is resolved. Sections 3–5 and 7–9 still guide the risk-rules and dashboard work; the [README](README.md) has the current output columns and integration snippets.

## 1. What we will ship

A local dashboard that reads conjunction telemetry, estimates `log10(Pc)` with a trained model, converts it to collision probability, applies the challenge's risk rules, and prioritizes encounters for human maneuver planning. It will include an encounter geometry view, risk history, miss distance, a 24-hour alert summary, explanations, and downloadable test predictions.

**MVP stack:** Python, pandas, scikit-learn, joblib, Streamlit, Plotly. One process, direct function calls, CSV inputs and outputs. No separate API, database, account system, LLM, or orbital propagator is needed for the MVP. Pin versions after the first working install.

**Scientific scope:** this is a learned estimate of the supplied risk label. The selected columns do not contain the full covariance information, object sizes, and orbital state needed for an independent physical Pc calculation or reliable orbit reconstruction. Label displayed probabilities as **estimated Pc**. A critical flag requests maneuver planning; it does not prescribe or execute a burn.

## 2. What is actually in this repository

The following are local file inspection results, not model performance claims:

| Property | `data/train_data.csv` | `data/test_data.csv` |
|---|---:|---:|
| Rows / conjunction messages | 162,634 | 24,484 |
| Distinct event IDs | 13,154 | 2,167 |
| Columns | 103 | 103 |
| Messages per event | 1–23 | 1–17 |
| `time_to_tca` range, days | -0.1498 to 6.9938 | 2.0002 to 6.9932 |
| Missing `c_sigma_r` | 11 | 6 |
| Missing `F10` and missing `AP`, each | 6,822 | 1,329 |

Important findings and decisions:

- Both files contain `risk`. For this hackathon's row-prediction task, use **training `risk` only as the target**. Drop test `risk` before prediction and never use it to select a model, tune thresholds, or generate dashboard predictions.
- Event IDs are `0..13153` in training and `0..2166` in testing. All test ID values overlap training IDs, but there are **zero identical full rows across files after excluding `event_id`**. This suggests separate ID namespaces; it does not prove that encounters are unrelated. Use `(source, event_id)` as the display/grouping key, never join files by ID, and audit near-duplicate telemetry before claiming independent test encounters.
- Repeated rows describe updates to an encounter. Split validation by event, never randomly by row. Do not use `event_id` as a predictor.
- Training has only 716 rows with `risk >= -4` (about 0.44%). Under the combined Pc/distance rules it has 11,630 critical, 35,192 elevated, and 115,812 safe rows. Report Pc-only detection separately so distance triggers do not hide a weak Pc model.
- Training contains 391 messages with negative time to TCA. Exclude these from prospective model training/evaluation and active alert counts; retain them only in an explicitly historical view.
- No test message is within 24 hours of TCA. A truthful test-data 24-hour view is empty. Use a labeled training-data replay to demonstrate alerts; 9,445 training events have at least one message with `0 <= time_to_tca <= 1`.
- Radial sigma values reach about 63.8 million metres. Inspect these as potential invalid/sentinel uncertainty values; do not silently interpret them as ordinary measurements. Preserve flags and document any cleaning rule.

## 3. Freeze these rules and units first

Use the organizer's thresholds as the application policy. This does not establish certification against every NASA/ESA requirement.

| Priority | Tier | Rule using estimated Pc and distance |
|---|---|---|
| First | Critical — Maneuver Required | `Pc >= 1e-4 OR miss_distance_m < 1000` |
| Second | Elevated | `Pc >= 1e-6 OR miss_distance_m <= 5000`, after excluding critical |
| Otherwise | Safe | `Pc < 1e-6 AND miss_distance_m > 5000` |

At exactly 1,000 m or 5,000 m, distance alone is elevated. At exactly `1e-4`, Pc is critical; at exactly `1e-6`, Pc is elevated. Always retain both reasons if both triggers apply. Compare predicted log risk to `-4` and `-6` directly to avoid roundoff at probability boundaries.

The ESA data dictionary specifies days for `time_to_tca`, metres for distance and relative position, metres/second for speed, and metres for radial sigma. Convert only for display: `tca_hours = 24 * time_to_tca`, `miss_distance_km = miss_distance / 1000`. See the [ESA data dictionary](https://kelvins.esa.int/collision-avoidance-challenge/data/).

Validate finite values, nonnegative distances/sigmas/speeds, and required columns. Missing optional model inputs can be imputed with flags. If distance or model output is unusable, return `needs_review`; never default to safe. If a valid input already proves critical, retain critical even when the other input is missing, with a data-quality warning.

## 4. Three teammate assignments

### Teammate 1 — Data and ML

**Own:** `src/model.py`, `scripts/train.py`, `scripts/predict.py`, `reports/model_report.md`; generated artifacts under `artifacts/` and predictions under `outputs/`.

1. Build the explicit feature allowlist below. Keep `risk` as the training target and IDs as metadata.
2. Validate units, missingness, duplicates, extreme sigmas, target distribution, and possible cross-file encounter overlap. Save an auditable summary without training on test labels.
3. Freeze a reproducible 70/15/15 train/validation/holdout split of **training event IDs**, with seed 42. Balance rare critical-event presence where practical; record per-split counts. Fit every imputer, encoder, and transform only on the training partition.
4. Train a median-log-risk dummy baseline, then a modest `HistGradientBoostingRegressor` pipeline. Median-impute numeric columns with missingness indicators and encode object type with an unknown-safe encoder. Use log transforms for nonnegative skewed inputs only if validation improves. No large hyperparameter search.
5. Predict log risk; preserve raw output for diagnostics. Use `10 ** clip(predicted_log10_pc, -30, 0)` for display. The lower bound reflects the observed label floor, not proof that smaller probabilities are impossible. Flag clipped outputs.
6. Compare the learned model against the dummy baseline on regression and critical detection. Report MAE/RMSE in log space, Pc-critical precision/recall/F2 and PR-AUC, three-tier confusion matrix, and critical recall/false alarms per 1,000 events after latest-message aggregation. Include counts, including missed critical cases; report undefined metrics honestly when a slice has no positives.
7. Inspect errors around `-6` and `-4`, by object type, time-to-TCA band, and missing-data status. Use event-level bootstrap intervals if time permits. Do not claim physical probability calibration from regression accuracy alone.
8. Save the selected pipeline plus feature order, training split IDs, seed, units, model version, and dependency versions. Evaluate the holdout once after model selection; do not tune on it.
9. Generate one prediction for each of the 24,484 test rows, in original order, with original event IDs. Drop test `risk` and all non-allowlisted columns before inference.

**First handoff, hour 2:** `predict_log10_pc(frame)` returning one finite numeric estimate per valid input row; initially backed by the baseline. Provide a small input/output fixture with no label dependency.

**Done when:** a clean run trains and predicts reproducibly; unseen object types and missing optional inputs work; group leakage checks pass; the model report includes baselines, rare-class counts, limitations, and actual measured metrics. No promised accuracy target before seeing validation results.

### Teammate 2 — Risk rules, event summaries, and integration logic

> **Status: implemented** in `src/risk.py`, `src/events.py`, `scripts/build_dashboard_data.py`, `tests/test_risk.py` and `tests/test_pipeline.py`. Changes from the plan: events carry a *priority tier* combining the current assessment with the final-tier forecast (the forecast only applies while the latest message is 2+ days out), and windows use `time_to_tca` per the organizers. The dashboard can load `outputs/dashboard/*.csv` directly. See the README section "Risk rules and dashboard data".

**Own:** `src/risk.py`, `src/events.py`, `tests/test_risk.py`, `tests/test_pipeline.py`.

1. Implement the rules in section 3 as a pure function, independent of the model and UI. Include `risk_tier`, `alert_reasons`, `recommended_action`, and quality flags.
2. Implement `score_rows(frame, predicted_log10_pc)` using the shared output contract. Preserve input order and keys; convert units exactly once.
3. Implement latest-message selection per `(source, event_id)`: smaller nonnegative `time_to_tca` means a newer pre-TCA message. Use original row index as a deterministic tie-breaker. Do not select a newer row from beyond the chosen replay cutoff.
4. Implement unique-event counts for upcoming 24 hours (`0 <= tca_hours <= 24`): total encounters, distance-dangerous encounters (`miss_distance_m < 1000`), critical events, and elevated events. Keep distance-only and Pc-driven counters distinct; count each event once.
5. Sort the action queue by critical/elevated/safe, then shortest time to TCA, then highest estimated Pc. Include readable reasons such as “Miss distance 0.72 km is below 1 km.”
6. Build deterministic insight text: highest estimated risk, closest predicted encounter, counts by object type, and change in estimated log risk between consecutive messages for an event. A rising Pc trend is not automatically a tier change.
7. Implement replay selection as described in section 7 and export the scored table. Do not invent observation timestamps or real-world event dates.
8. Test threshold boundaries, conflicting triggers, missing/invalid inputs, negative TCA, duplicate messages, replay cutoff, source-ID collisions, and event deduplication. Include an integration check proving changing/dropping input `risk` does not change predictions.

**First handoff, hour 2:** working rule function, sample scored table, and exact column contract for the UI.

**Done when:** boundary checks pass, 24-hour counts agree with a small hand-counted fixture, and the same score produces the same tier in exports and the dashboard.

### Teammate 3 — Dashboard and visual story

**Own:** `app.py`, optional `ui/` helpers only if needed, and `reports/demo_script.md`.

1. Build a Streamlit layout from the shared sample output immediately; do not wait for model training.
2. Add data source selection (test predictions or labeled training replay), object-type/tier filters, and a time-to-TCA filter. Display the source and replay context prominently.
3. Show summary cards: upcoming encounters within 24 hours, encounters below 1 km, critical events, elevated events. Empty input must show a clear zero/empty state.
4. Add a sortable encounter table: event ID, estimated Pc in scientific notation, risk tier, miss distance in km, hours to TCA, object type, and trigger explanations. Pair colors with text labels.
5. Add an event detail panel: risk estimate history with `-6`/`-4` reference lines, miss-distance history, uncertainty/quality flags, and a local R/T/N relative-position view in km.
6. Label connected R/T/N points as **successive estimates of encounter geometry at TCA**. These are not measured flight paths. If an illustrative trajectory animation is added, mark it schematic; the allowed inputs cannot reconstruct real Earth-centred trajectories. Put actual orbit rendering after the MVP and only with additional authorized state data.
7. Show automatic insight text from Teammate 2 and add CSV download. Cache loaded data/model; keep training out of dashboard startup.
8. Prepare a 3-minute demo with one critical case, one safe case, one risk escalation, and the honest empty 24-hour test view followed by the replay demonstration.

**First handoff, hour 2:** runnable dashboard populated from the agreed fixture.

**Done when:** the dashboard runs from documented commands, filters and charts use the same event selection, the geometry caveat is visible, and the demo works locally without internet access.

### You — Integration lead and organizer liaison

Own `README.md`, dependency pins, this roadmap, merge order, acceptance checks, and the final pitch. Confirm the organizer's prediction target and required submission schema early (section 6). Distribute the data separately to teammates; the approximately 223 MB training CSV should not be added to ordinary Git history. Add `.gitignore` entries for `.DS_Store`, raw datasets, environments, caches, generated models, and bulk outputs when implementation starts. Keep small fixtures and reports tracked.

## 5. Shared interface: agree once, build in parallel

Allowed input columns:

```text
event_id, time_to_tca, miss_distance, relative_speed,
relative_position_r, relative_position_t, relative_position_n,
c_object_type, c_sigma_r, t_sigma_r, max_risk_estimate, F10, AP
```

`event_id` is a key only, leaving 12 model inputs. `risk` is never a predictor. Do not quietly add other columns from the 103-column files. `max_risk_estimate` is negative in these local files: preserve its supplied numeric representation as a feature; do not display it as raw Pc or substitute it for the target. Verify its encoding before building a baseline from it.

Internal scored-row contract (one row per source row):

| Field | Type / meaning |
|---|---|
| `source` | `train_replay` or `test` |
| `row_id` | Zero-based original CSV row index, excluding header |
| `event_id` | Original ID; unique only together with source at event level |
| `predicted_log10_pc` | Finite model estimate after documented display clipping |
| `predicted_pc` | `10 ** predicted_log10_pc` |
| `risk_tier` | `safe`, `elevated`, `critical`, or `needs_review` |
| `miss_distance_m`, `miss_distance_km` | Original distance and display conversion |
| `time_to_tca_days`, `tca_hours` | Original interval and display conversion |
| `c_object_type` | Original category or explicit unknown |
| `relative_position_r/t/n` | Three original fields, metres |
| `alert_reasons` | List of all satisfied triggers |
| `quality_flags` | List of missing, extreme, invalid, or clipped inputs/outputs |
| `recommended_action` | Monitor, review, or initiate maneuver planning |
| `model_version` | Reproducible artifact identifier |

For CSV, serialize list fields consistently as JSON strings. Keep the event-level tier predictions separate from the richer per-message dashboard export. ~~The proposed default row submission is `row_id,event_id,predicted_risk`.~~ *Superseded:* the deliverable is one tier prediction per event (`outputs/test_event_predictions.csv`); there is no row-level submission.

Suggested commands for the implementation team to make real:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-model.txt
python -m unittest discover -s tests
python scripts/train_final_tier.py     # final-tier model + outputs/test_event_predictions.csv
python scripts/train_row_model.py      # per-message Pc model
python scripts/predict.py              # outputs/test_predictions.csv
streamlit run app.py                   # planned (Teammate 3)
```

All but `streamlit run app.py` are implemented.

## 6. Resolve the scoring ambiguity without blocking the MVP

> **Resolved:** the organizers confirmed final-event scoring (final **tier**, not final risk), with messages >= 2 days out and an event-level held-out split of `train_data`. The plan below was implemented and extended. Notable changes: the model forecasts the change in miss distance (Critical is almost always a distance outcome), the latest-message tier serves as a floor, and evaluation focuses on events whose final message is inside 2 days. Details in [reports/model_development.md](reports/model_development.md). The original text is kept for the record.

**Original default:** follow this hackathon brief: estimate the supplied row-level `risk` from permitted telemetry and output all test rows.

**Organizer question for the lead:** “Are you scoring each test row's contemporaneous risk, or one final risk per event using messages available at least two days before TCA? Please share the submission template and scoring metric.”

This matters because the original [ESA challenge](https://kelvins.esa.int/collision-avoidance-challenge/challenge/) predicts the final event risk from earlier messages, and its [submission rules](https://kelvins.esa.int/collision-avoidance-challenge/submission-rules/) require `event_id,predicted_risk`, one row per event. That is a different target from reconstructing each message's risk.

If the organizers confirm final-event scoring, Teammate 1 changes the supervised examples before tuning: use the latest available feature row with `time_to_tca >= 2` for each training event, and the final pre-TCA training `risk` as its target. Prefer events with a label within one day of TCA; exclude ineligible events and report counts. Split by event before any aggregation or fitting. At test time use the latest permitted message per event and produce 2,167 predictions. Keep final-risk forecasts distinct from contemporaneous Pc in the UI. Earlier-message aggregation is an optional improvement after this baseline works; never use future messages or observed `risk` as input.

## 7. Honest 24-hour replay and visualization

The files provide relative time to TCA, not a shared wall-clock timestamp for every encounter. Therefore the MVP cannot claim that unrelated historical events all occur in the next real-world day.

- **Test view:** use each event's latest available message and its supplied TCA interval. The 24-hour count is zero for these files; show why rather than shifting times to fabricate alerts.
- **Training replay:** choose a synthetic horizon `h` hours before TCA. For each event, select its newest available nonnegative message satisfying `time_to_tca >= h / 24`. Its displayed remaining TCA is `h`; retain the message's original age and expose staleness. This intentionally aligns unrelated events to a common hypothetical TCA. Label all summary counts “historical replay, events aligned to TCA,” not a real fleet forecast.
- Use held-out training events for the replay demo where possible, keeping displayed predictions out of sample. Mark any true training labels used in evaluation charts separately from model estimates.
- Demonstrate the alert logic with small synthetic fixtures whose relative times differ across events; label them synthetic. This checks inclusion/exclusion at 0 and 24 hours without pretending to provide real observation epochs.
- Do not interpolate risk into an unobserved interval and present it as measured telemetry. Every chart should distinguish estimates, observed labels (where available), and schematic geometry.

## 8. Build sequence and integration checkpoints

| Time from start | Teammate 1 | Teammate 2 | Teammate 3 | Lead / integration gate |
|---|---|---|---|---|
| 0–30 min | Audit + split design | Freeze units/rules | Sketch layout + fixture | Confirm target/schema; assign file ownership |
| 30 min–2 h | Baseline predictor | Scoring + boundary tests | Dashboard on fixture | All three run against shared contract |
| 2–4 h | First learned model | Event selection + 24 h summaries | Table + event detail | Load real model and display one real event end to end |
| 4–6 h | Validation + error analysis | Replay + insight text + export | Geometry + history charts | Reconcile UI counts with exported data |
| 6–8 h | Select model; freeze holdout report | Integration checks | Usability + empty/error states | Reproduce test predictions and run demo |
| 8–10 h | Final artifacts + manifest | Fix integration defects | Polish demo story | Fresh-environment smoke test; freeze features |
| 10–12 h | Support final checks | Support final checks | Rehearse presentation | Submit exact schema, README, screenshots, and report |

Use separate `codex/model`, `codex/risk`, and `codex/dashboard` branches after the initial roadmap commit. Agree on interface changes in the team chat before changing shared column names. Merge baseline interfaces first, then the UI, then model improvements. Only the lead edits shared dependency pins and top-level README during parallel work.

## 9. Release checklist

- [ ] Only allowed feature columns reach inference; IDs and all `risk` columns are excluded.
- [ ] Validation events never appear in model fitting; all preprocessing is fitted on the training split.
- [ ] Row-level task or final-event task is explicitly documented, with the correct submission shape.
- [ ] Test export preserves expected IDs/order and contains finite values and no leaked labels.
- [ ] Model report includes rare-class recall, false positives, regression error, baseline comparison, and sample counts.
- [ ] Threshold equality, invalid inputs, and unit conversions pass focused tests.
- [ ] Alert counts use unique events and agree with the displayed table under the same filters.
- [ ] Empty 24-hour test view and replay caveat are visible and correct.
- [ ] Trajectory/geometry graphics distinguish available telemetry from illustration.
- [ ] README has setup, training, prediction, launch commands, data placement, and known limits.
- [ ] Clean-start demo works offline, with cached model artifact and a backup screenshot/video.

## 10. What to cut if time runs short

Keep the trained baseline, deterministic rules, event table, 24-hour logic, prediction export, and honest validation. Cut animation, SHAP, fancy styling, hyperparameter searches, and extra derived features first. If time remains, add validation-set permutation importance and better temporal features using only prior permitted telemetry. Do not spend hackathon time on autonomous burn optimization or a decorative Earth orbit that the data cannot support.

## References and standards scope

- The thresholds in section 3 are copied from the **provided hackathon brief** and are the acceptance policy for this application, including its distance overrides.
- [NASA/SP-20205011318 handbook, December 2020](https://www.nasa.gov/wp-content/uploads/2018/01/oce_51-nasa_spacecraft_conjunction_assessment_and_collision_avoidance_best_practices_handbook.pdf): consult section 6 and appendices M/N for risk assessment and Pc context. NASA's [risk-assessment overview](https://www.nasa.gov/cara/step-2-close-approach-risk-assessment/) explains why predicted trajectories, uncertainty, and combined object size matter to physical Pc. This prototype estimates labels rather than implementing that full physical calculation.
- [ESA dataset documentation](https://kelvins.esa.int/collision-avoidance-challenge/data/) supplies units and message/event semantics; [original scoring](https://kelvins.esa.int/collision-avoidance-challenge/scoring/) is background only until the hackathon organizers confirm their metric.
- The brief does not identify a specific ESA standard/version. Ask organizers for the intended reference before claiming standards compliance. Describe the demo as implementing the challenge's risk policy with NASA/ESA-informed methodology.
