# ORBITAL INTELLIGENCE — Reflex dashboard

AI-powered satellite collision early-warning interface for Airbus Challenge #1 (Teammate 3).
The original Streamlit dashboard (`../app.py`, `../ui/`) is untouched and remains the fallback.

> **Status:** real model outputs are available, including tier probabilities. Run `python scripts/export_dashboard_outputs.py` from the
> repository root to write `outputs/event_predictions.csv` and `outputs/cdm_history.csv` in the v2 contract, then
> choose **Real model outputs** in the top bar. Synthetic datasets remain available and clearly labelled.

## Run it

From the repository root, in GitHub Codespaces or locally (Python 3.11+, Node is downloaded by Reflex on first run):

```bash
cd dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
reflex run --env prod --single-port
```

Open the URL Reflex prints (port **3000**). In Codespaces, a pop-up offers **Open in Browser**; otherwise open the
**Ports** tab and click the globe icon next to 3000. Keep the port **Private** (the default) so only you can see it.
The first run takes a minute or two while Reflex installs its JavaScript dependencies.

**In Codespaces, always use `--single-port`.** It serves the page and the live connection from port 3000, which
is what the forwarded `https://…-3000.app.github.dev` address expects. Plain `reflex run` (hot reload on two
ports, 3000 and 8000) works locally, but the forwarded page cannot reach port 8000 without extra configuration.
After editing code, stop the app with `Ctrl+C` and start it again.

## Tests

```bash
cd dashboard
python -m unittest discover -s tests -v          # data layer: contract, adapters, synthetic data, analytics
```

Browser end-to-end checks (app must be running; needs Node and Playwright):

```bash
npm install playwright@1 && npx playwright install chromium
node tests/e2e_browser.mjs http://localhost:3000
```

They cover summary cards, filters, sorting, export, dataset switching, the missing-data error, encounter details,
chart units, validation maths, and globe rotate / zoom / hover / click / focus / reset / filters / 12,000 markers.

## Pages

| Page | What it shows |
|---|---|
| Mission Control | Clickable predicted-tier cards, early-warning windows, observed→predicted transitions, globe, priority queue |
| Global Tracking | Full-screen interactive globe (synthetic simulation) with filters and object panel |
| Collision Risk | Filters (tier, event ID, object type, time to TCA, population), sortable queue, CSV export, timeline |
| Encounter Analysis | Visual workspace. Left: **Encounter View** (each CDM drawn at its measured R/T/N position projected to 2-D; Earth backdrop, icons and approach arcs are illustrative and not to scale) or **Measured Geometry** (3-D R/T/N). Both support hover, click-to-inspect, zoom and reset. Right: observed → AI-predicted final tier, escalation, four measurement cards, model uncertainty, why this tier, recommended review step. Bottom: risk and miss-distance evolution, latest-6 CDM table, full telemetry (expandable) |
| Model Validation | Held-out confusion matrix, accuracy, critical recall/precision, missed criticals, limitations |
| Reports | Dataset provenance, data-quality warnings, contract summary, export |

## Architecture

```
dashboard/
├── rxconfig.py, requirements.txt
├── assets/                 orbital.css (theme), globe/ (Earth textures)
├── orbital/
│   ├── orbital.py          app + routes
│   ├── state.py            UI state ↔ data layer (the only place they meet)
│   ├── data/               NO UI CODE
│   │   ├── contract.py     v2 contract + validation (never assigns tiers)
│   │   ├── adapters.py     synthetic / real sources, COLUMN_ALIASES mapping
│   │   ├── analytics.py    filters, counts, windows, queue order, validation metrics, formatting
│   │   ├── synthetic_encounters.py   two synthetic encounter datasets (A, B)
│   │   └── synthetic_globe.py        separate synthetic orbital population for the globe
│   ├── components/         layout, widgets, Plotly charts, globe wrapper + orbit_globe.jsx
│   └── pages/              operations.py, analysis.py
└── tests/                  unit tests + browser e2e
```

* The UI never classifies risk. Tiers come from the data (the risk pipeline). The encounter page *explains* the
  supplied tier by comparing the predicted values with the published thresholds, and flags any mismatch rather
  than changing the tier (`analytics.threshold_checks`).
* "Reliability" is derived only from uncertainty the model supplies (tier probabilities and/or a Pc interval);
  without them it reads "not assessed". "Recommended next step" is triage guidance (continue monitoring, request
  additional data, refer for manoeuvre assessment); it never recommends or approves a manoeuvre.
* The encounter visualisation draws only measured relative positions (`orbital/data/encounter_view.py`). It never
  draws an orbit, an absolute position, a predicted final position, or a predicted miss distance the model did not
  supply
  (`analytics.reliability`, `analytics.next_step`).
* A missing or invalid real dataset shows an error. It is **never** silently replaced with synthetic data.
* The globe data is a separate synthetic simulation and is never linked to Airbus encounters.

## Connecting real model outputs

Put two CSV files in `outputs/` at the repository root (or set `ORBITAL_OUTPUTS_DIR`), then choose
**Real model outputs** in the top-bar selector and press reload.

**`outputs/event_predictions.csv`**: one row per `(source, event_id)`.

| Required | Meaning |
|---|---|
| `source` | e.g. `test`, `train` (kept separate; never join train and test by `event_id` alone) |
| `event_id` | original ID |
| `split` | `train`, `validation`, `holdout` or `test` |
| `cutoff_time_to_tca_days` | time_to_tca of the last CDM used (must be ≥ 2) |
| `predicted_final_tier` | `critical`, `elevated`, `safe` or `needs_review` (from the agreed risk logic) |
| `model_version` | reproducible model identifier |

Optional fields (shown when present, "not provided" otherwise):

- **Current observed risk** (at the cutoff CDM): `observed_log10_pc`, `observed_miss_distance_m`, `observed_tier`.
- **Predicted final risk:** `predicted_final_log10_pc`, `predicted_final_pc`, `predicted_final_at_floor`,
  `predicted_final_miss_distance_m`, and `final_miss_distance_basis` (`predicted` or `observed_at_cutoff`).
- **Uncertainty:** `predicted_final_log10_pc_lower` / `_upper` and `p_safe` / `p_elevated` / `p_critical`.
- **Explanation:** `alert_reasons` and `quality_flags` (JSON lists), `recommended_action`.
- **Context:** `c_object_type`, `cutoff_row_id`, `n_cdms_used`.
- **Held-out rows only:** `actual_final_log10_pc`, `actual_final_miss_distance_m`, `actual_final_tier`.
  These must be empty for test rows; the loader rejects test rows that carry labels.

**`outputs/cdm_history.csv`**: one row per CDM.

- **Required:** `source`, `event_id`, `row_id`, `time_to_tca_days`, `miss_distance_m`.
- **Optional:** `relative_speed_mps`, `relative_position_r_m`, `relative_position_t_m`, `relative_position_n_m`,
  `observed_log10_pc`, `used_for_prediction`, `c_object_type`, `quality_flags`.
- **Rule:** rows marked `used_for_prediction` must have `time_to_tca_days ≥ 2`.

If the model exports different column names, add them to `COLUMN_ALIASES` in `orbital/data/adapters.py`.
The UI needs no changes.

Units follow the 31 Oct 2026 organiser clarification:
- time_to_tca is in days;
- distances and relative positions are in metres;
- relative speed is in m/s;
- Pc = 10^risk, and risk = −30 is displayed as "≈ 0".

## Third-party assets

Earth textures (`assets/globe/`) are NASA Blue Marble / topography imagery (public domain) as distributed in the
[three-globe](https://github.com/vasturiano/three-globe) examples (MIT). The globe uses
[react-globe.gl](https://github.com/vasturiano/react-globe.gl) (MIT) and three.js (MIT), installed by Reflex.
