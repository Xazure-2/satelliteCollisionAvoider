# Challenge #1 submission

`submission.csv`: one row per test event (2,167 events), the predicted **final** risk tier with its probability. Regenerate with `python scripts/make_submission.py` (or `./run_all.sh`).

| Column | Meaning |
|---|---|
| `event_id` | Test event ID (IDs restart in each file; never joined to training IDs) |
| `predicted_final_tier` | `Safe`, `Elevated` or `Critical`: the tier expected at the event's final message before closest approach |
| `probability` | Calibrated probability of the predicted tier |
| `p_safe`, `p_elevated`, `p_critical` | Calibrated probability of each final tier (sum to 1) |
| `requires_manoeuvre_planning` | `True` when the predicted tier is Critical |
| `predicted_final_miss_distance_m` | The model's forecast of the final miss distance, metres |
| `latest_time_to_tca_days` | Time to closest approach of the latest test message used (all test messages are 2+ days out) |
| `model_version` | Frozen final-tier model (`final-tier-d7496dd831de`) |

Current file: 228 Critical, 436 Elevated, 1,503 Safe.

## Tier definitions (hackathon brief, NASA/ESA-informed)

- **Critical (manoeuvre planning required):** Pc ≥ 1e-4 (risk ≥ −4) or miss distance < 1,000 m
- **Elevated:** not Critical, and 1e-6 ≤ Pc < 1e-4 or miss distance 1,000–5,000 m
- **Safe:** Pc < 1e-6 and miss distance > 5,000 m

## How the predictions are made

1. **Inputs:** only the permitted telemetry columns from each event's test messages (all 2+ days before closest approach). Test `risk` is never read.
2. **Tier:** a gradient-boosted regressor forecasts the change in miss distance up to the final message. An event is flagged Critical when the forecast final distance is ≤ 1.31 km, and Elevated when ≤ 5.16 km. These thresholds were tuned for recall (F2), so dangerous events are flagged early. The tier implied by the latest message's miss distance acts as a floor: the model can raise it, never lower it.
3. **Probability:** a calibrated multinomial logistic layer turns the forecast, current distance tier, latest distance, `max_risk_estimate` and time to TCA into P(Safe/Elevated/Critical).

Because the flag favours recall, an event can be flagged Critical with `probability` below 0.5. The flag says "plan for it"; the probability says how likely it is to end Critical.

## Validation (held-out training events, evaluated once)

- **Tier** (events whose final message is inside 2 days, 76 Critical): Critical recall 0.882 vs 0.789 for carrying the current tier forward; 9 vs 16 missed Critical events; F2 0.831 vs 0.794.
- **Probabilities:** log loss 0.212 vs 0.241 and Brier score 0.107 vs 0.114 for the current-distance-tier baseline; events given P(Critical) of 0.8–1.0 ended Critical 91% of the time.

Full details: [reports/model_development.md](../reports/model_development.md), [reports/tier_probability_report.md](../reports/tier_probability_report.md).
