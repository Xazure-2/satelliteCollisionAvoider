# Three-minute demo script — Streamlit fallback dashboard

> **Fallback only.** The main dashboard is the Reflex app in `dashboard/` (demo script: [dashboard/DEMO_SCRIPT.md](../dashboard/DEMO_SCRIPT.md)). This Streamlit app predates the final models. Its "live pipeline" scores messages with the default `predict_log10_pc` artifact (the superseded M3 model), and its training replay expects a `select_replay()` function that `src/events.py` does not provide (`snapshot(scored, cutoff_days)` covers that need). Prefer its *Synthetic demo data* or *Test predictions* modes, or use the Reflex app.

**Goal:** show that the dashboard turns conjunction telemetry into a clear, honest, prioritised list of encounters that need a human decision.

**Before going on stage**

- Launch with `streamlit run app.py` and open the browser tab before the demo starts. The model must already be trained, because training never runs when the dashboard starts.
- Use a wide browser window (at least 1400 px) and zoom to 90% so the cards and table fit on screen.
- Decide which mode you are presenting:
  - **Real mode (preferred):** the model, risk pipeline and `data/*.csv` are all available. Use *Test predictions* and *Training replay*.
  - **Fallback mode:** use *Synthetic demo data*, and say aloud that it is synthetic. The orange banner says so too.
- Pick the case events in advance from the action queue (see "Choosing events" below) and write their IDs here: critical `____`, escalation `____`, safe `____`.
- Keep a backup screenshot or video in case the laptop or projector fails.

---

## 0:00–0:20 — The problem

> "Every day operators receive thousands of conjunction warnings, and most are harmless. The hard part is finding the few that need a maneuver decision *now*. Our dashboard estimates the collision probability for each encounter, applies the challenge's risk policy, and puts the encounters in the order an operator should look at them."

Point at the **Risk overview** cards: Critical, Elevated, Safe and Needs review, with each encounter counted once.

## 0:20–0:50 — The next 24 hours, told honestly

1. Select **Test predictions (live pipeline)**.
2. Point at the **Next 24 hours** cards, which show zeros, and the blue explanation underneath.

> "In the official test file, no message is within 24 hours of closest approach, so the honest answer is zero. We don't shift times to manufacture alerts."

3. Select **Training replay**, with the horizon at 12 h.

> "To show the alert logic, we replay held-out training events as each one looked 12 hours before its own closest approach. The blue banner tells you this is a historical replay, not a forecast."

## 0:50–1:40 — A critical case

1. Open the **Action queue** tab. It is already sorted by tier, then soonest TCA, then highest estimated Pc.
2. Click the top row, which is a 🔴 Critical encounter.
3. In the event detail, point out:
   - estimated Pc, miss distance and hours to TCA;
   - the red box showing *why* it is flagged ("Estimated Pc … is at or above 1e-4" and/or "Miss distance … is below 1 km");
   - the recommended action, **Initiate maneuver planning**.

> "Critical means a human starts maneuver planning. The system never commands a burn."

## 1:40–2:10 — A risk escalation

1. Choose the escalation event in the **Event detail** dropdown.
2. Point at the **Risk estimate history**: the line climbs across the dashed 1e-6 and 1e-4 lines, and the markers change colour from green to orange to red.

> "Each point is a new message. You can see the risk crossing the elevated and then the critical threshold as closest approach gets nearer."

3. Briefly show the **R/T/N geometry** plot.

> "These are successive estimates of where the other object will be at closest approach. They are not flight paths. Our data can't reconstruct real orbits, and we say so on screen."

## 2:10–2:30 — A safe case

Select a 🟢 Safe encounter (for example, filter *Risk tier* to Safe only). Show the high miss distance, a Pc far below 1e-6 and the action **Monitor**.

> "Most encounters look like this. Filtering them out is just as valuable as flagging the dangerous ones."

## 2:30–2:50 — The big picture

Open **Overview charts**:

- **Estimated Pc vs miss distance:** the dashed lines split the plot into the policy regions.
- **Encounters by hours to TCA:** the shaded band is the next 24 hours.

## 2:50–3:00 — Close

Open **Export & data** and point at the CSV download.

> "Every number here can be exported and audited. The model estimates risk, transparent rules decide the tier, and a human makes the call."

---

## Choosing events

- **Critical:** the row at priority 1 in the action queue.
- **Escalation:** in the event detail, look for a risk history whose markers go green → orange → red. In the synthetic data that is event **32** (also 33).
- **Safe:** filter Risk tier to Safe and take the top row. In the synthetic data that is event **4**.
- Synthetic data also includes **event 34** (risk rising while staying Elevated: a trend is not a tier change) and **event 37** (🟣 Needs review: the model estimate is unavailable).

Synthetic event IDs are stable because the generator uses a fixed seed. Real IDs depend on the model, so re-check them after the final training run.

## If something goes wrong

| What you see | Say / do |
|---|---|
| Red "Cannot show …" message | Click **Show synthetic demo data instead**, and say it is synthetic. |
| Old numbers after a teammate's update | Click **↻ Reload data** in the sidebar. |
| Replay error about `select_replay()` | Teammate 2's replay function isn't merged yet. Present *Test predictions* plus *Synthetic demo data*. |
| Charts slow to appear | Pre-open each tab once before the demo; data is cached after the first load. |
