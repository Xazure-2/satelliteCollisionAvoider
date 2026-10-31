# Three-minute demo — ORBITAL INTELLIGENCE

The official judging rubric wasn't shared with us. This script covers the criteria such challenges usually judge:
- the problem and its relevance;
- technical soundness and validation;
- the innovation, which here is early warning;
- usability for operators;
- data integrity and honesty.

Re-order it once the rubric is known.

## Before you go on stage

- Start the app (`cd dashboard && reflex run --env prod --single-port`) and open every page once, so the charts are cached.
- Use a 1920×1080 browser window at 100% zoom, and turn globe rotation **on** for the opening shot.
- Choose the data source:
  - **Real model outputs** if `outputs/event_predictions.csv` exists;
  - otherwise **Synthetic demo A**, and say so out loud.
- Write the event IDs you will use here. In synthetic demo A:
  - escalation: **166** (Elevated → Critical) or **39** (Safe → Critical);
  - missed critical on the validation page: **298**.
- Keep a backup screen recording.

---

### 0:00–0:25 · Problem (Mission Control)

> "Operators get thousands of conjunction messages. The organisers asked one question: using only messages at
> least two days before closest approach, what will the final risk tier be? We built Orbital Intelligence
> to answer that and to put the answer in front of an operator."

Point at the five cards and at the **SYNTHETIC DEMO / REAL MODEL DATA** badge in the top bar.

### 0:25–0:55 · The early-warning idea

Click the **Predicted critical** card. The queue filters immediately.

> "Each row shows the risk we observe now next to what the model predicts at the final message.
> Safe → Critical is the case that matters: the observed numbers look fine today, but the model expects
> the encounter to become critical."

Point at the **Escalations** count and the **2–3 d / 3–5 d / 5–7 d** windows.

### 0:55–1:45 · Investigate one encounter (Encounter Analysis)

Click the escalation row. Walk down the **right panel**, which answers the operator's five questions in order:

1. **How dangerous?** Observed ELEVATED → predicted CRITICAL, with the predicted final Pc.
2. **How soon?** The T− countdown, measured from the last eligible message. It is relative, not a live clock.
3. **Why?** "Critical because predicted final Pc … meets the critical threshold". The threshold rows show which
   criterion fired.
4. **How reliable?** Taken from the model's own probability and range. Point out when the range spans a threshold.
5. **What next?** *Refer for manoeuvre assessment*: "a specialist decides, the system does not."

Then rotate the **3D geometry** on the left. Say plainly: "relative to our satellite, not an orbit." Show the latest
eligible message, and say that the cyan sphere is the predicted miss *distance* (the model gives no direction).
Open the **Risk evolution** tab to show the 2-day cutoff and the prediction star.

### 1:45–2:20 · Credibility (Model Validation)

- **Confusion matrix:** held-out training events, split by event_id. The test set has no labels, so it is never used here.
- **Critical recall and precision:** give the counts as well as the percentages.
- **Missed critical:** open one (for example **298**) to show we report failures too.
- If synthetic: "These numbers demonstrate the page. Our real measured metrics are in the model report."

### 2:20–2:45 · Situational awareness (Global Tracking)

Drag to rotate, scroll to zoom, click a red-ring marker, then **Focus on object**, then **Global view**.

> "This globe is a labelled synthetic simulation. The challenge data has no absolute positions, so we
> don't place real encounters on Earth or invent trajectories."

### 2:45–3:00 · Close

> "Same interface, real outputs: the model team drops two CSVs in the agreed format and every card,
> chart and count updates. Early warning, honest data, a human in the loop."

---

## If something goes wrong

| Problem | What to do |
|---|---|
| "Real predictions unavailable" | Click **Show synthetic demo A instead** and say it's synthetic |
| Globe blank or slow | Turn rotation off, or reduce the population (Global Tracking). The rest of the app works without it |
| Reflex won't start | Present the Streamlit fallback: `streamlit run app.py` from the repository root |
