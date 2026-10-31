# Dashboard (`app.py` + `ui/`) — Streamlit fallback

> **Fallback.** The main dashboard is the Reflex app in `dashboard/`, fed by `python scripts/export_dashboard_outputs.py` (or `./run_all.sh`). This app predates the final models: its live pipeline uses the default `predict_log10_pc` artifact (the superseded M3 model), and its training replay expects a `select_replay()` that `src/events.py` does not provide. Demo script: [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

Owner: Teammate 3. Run from the repository root:

```bash
pip install streamlit plotly pandas numpy
streamlit run app.py
```

Exact versions will be pinned by the lead in `requirements.txt`. The dashboard was built against Streamlit 1.66, Plotly 6/7 and pandas 3. It needs Streamlit ≥ 1.46 for `width="stretch"`.

## Data sources (sidebar)

| Option | What it needs | Notes |
|---|---|---|
| Synthetic demo data | nothing | Labelled `source="synthetic"`, model `synthetic-demo-v0`. Tiers are authored per scenario. |
| Test predictions (live pipeline) | `data/test_data.csv`, `src.model.predict_log10_pc`, `src.risk.score_rows` | Reads only the 13 allowlisted columns, so `risk` never reaches the model. |
| Training replay (live pipeline) | as above with `data/train_data.csv`, plus `src.events.select_replay` | Horizon slider in hours. |
| Scored CSV export | a CSV in the ROADMAP section 5 contract | Default path `outputs/scored_test.csv`, or upload one. |

## What the dashboard expects from teammates

`ui/backend.py` is the only place that calls teammate code.

**Teammate 1, `src/model.py`**

- `predict_log10_pc(frame) -> array`: one value per row, in input order.
- `frame` holds the 13 allowed input columns from ROADMAP section 5.

**Teammate 2, `src/risk.py`**

- `score_rows(frame, predicted_log10_pc) -> DataFrame` with all contract columns.
- `frame` additionally has `source` and `row_id` columns, already filled in.
- Optional: float constants `LOG10_PC_CRITICAL`, `LOG10_PC_ELEVATED`, `MISS_DISTANCE_CRITICAL_M` and `MISS_DISTANCE_ELEVATED_M`. The dashboard uses them for chart guide lines only.

**Teammate 2, `src/events.py`** (proposed names; edit `EVENTS_API` in `ui/backend.py` if yours differ)

| Function | Input | Returns |
|---|---|---|
| `select_latest_messages(scored)` | scored rows | one row per `(source, event_id)` |
| `summarize_upcoming_24h(scored)` | scored rows | dict with `total_encounters`, `distance_dangerous`, `critical`, `elevated` |
| `build_action_queue(latest)` | latest rows | latest rows in action order |
| `build_insights(scored)` | scored rows | `list[str]` |
| `select_replay(scored, horizon_hours)` | scored rows, horizon | one row per event, with `tca_hours` set to the horizon |

Until these exist, the dashboard uses small provisional display fallbacks and says so on screen. The fallbacks never assign or change a risk tier: tiers always come from the scored data.

## Files

| File | Contents |
|---|---|
| `contract.py` | scored-row column list and validation, with readable error messages |
| `backend.py` | loading data, teammate adapters, provisional event fallbacks |
| `synthetic.py` | deterministic, clearly labelled synthetic scored rows |
| `charts.py` | Plotly figures |
| `theme.py` | tier colours/labels and card CSS |
