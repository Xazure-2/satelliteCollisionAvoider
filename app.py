"""Satellite Collision Avoidance — Streamlit dashboard (Teammate 3).

Run with:  streamlit run app.py

The dashboard only displays scored rows in the contract from ROADMAP.md
section 5. Predictions come from src/model.py, risk tiers from src/risk.py and
event summaries from src/events.py; see ui/backend.py for the connection points.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Collision Avoidance Dashboard", page_icon="🛰️", layout="wide")

from ui import backend, charts, theme  # noqa: E402
from ui.contract import ContractError, SCORED_COLUMNS, to_export_csv, validate_scored  # noqa: E402
from ui.synthetic import make_synthetic_scored  # noqa: E402

SOURCES = {
    "synthetic": "Synthetic demo data",
    "test": "Test predictions (live pipeline)",
    "train_replay": "Training replay (live pipeline)",
    "scored_file": "Scored CSV export",
}
TIER_CARD_ORDER = ["critical", "elevated", "safe", "needs_review"]

st.markdown(theme.CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ loaders

def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_data(show_spinner="Generating synthetic demo data…")
def load_synthetic():
    df, report = validate_scored(make_synthetic_scored(), allow_synthetic=True)
    return df, report.warnings


@st.cache_data(show_spinner="Running model and risk pipeline…")
def load_live(source: str, path: str, _mtime_key: float):
    df, report = backend.run_live_pipeline(source, path)
    return df, report.warnings


@st.cache_data(show_spinner="Loading scored export…")
def load_scored_path(path: str, _mtime_key: float):
    df, report = backend.load_scored_csv(path)
    return df, report.warnings


@st.cache_data(show_spinner="Loading uploaded file…")
def load_scored_upload(data: bytes, name: str):
    import io
    df, report = backend.load_scored_csv(io.BytesIO(data))
    return df, report.warnings


def use_synthetic():
    st.session_state["source"] = "synthetic"


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("🛰️ Data source")
    source = st.radio("Show", list(SOURCES), format_func=SOURCES.get, key="source",
                      label_visibility="collapsed")
    replay_hours = None
    raw_path = scored_path = None
    upload = None
    if source == "test":
        raw_path = st.text_input("Test telemetry CSV", backend.DEFAULT_PATHS["test_raw"])
    elif source == "train_replay":
        raw_path = st.text_input("Training telemetry CSV", backend.DEFAULT_PATHS["train_raw"])
        replay_hours = st.slider("Replay horizon: hours before TCA", 0, 72, 12,
                                 help="Each event shows its newest message at least this far before TCA.")
    elif source == "scored_file":
        scored_path = st.text_input("Scored export path", backend.DEFAULT_PATHS["scored_export"])
        upload = st.file_uploader("…or upload a scored CSV", type="csv")

    if st.button("↻ Reload data", width="stretch",
                 help="Clear the cache after a teammate updates their module or outputs."):
        st.cache_data.clear()
        st.rerun()

    with st.expander("Pipeline status"):
        for name, state in backend.teammate_status().items():
            icon = {"ready": "✅", "missing": "⏳", "error": "❌"}[state]
            st.markdown(f"{icon} `{name}` — {state}")


# ------------------------------------------------------------------ header

st.title("🛰️ Conjunction Risk Dashboard")
st.caption("Satellite collision avoidance · Estimated collision probability (Pc) from a learned model, the challenge risk policy, and a "
           "prioritised queue for human maneuver planning. A critical flag requests planning; it never "
           "commands a burn.")

try:
    if source == "synthetic":
        scored, warnings = load_synthetic()
        context = "synthetic"
    elif source in ("test", "train_replay"):
        scored, warnings = load_live(source, raw_path, _mtime(raw_path))
        context = source
    else:
        if upload is not None:
            scored, warnings = load_scored_upload(upload.getvalue(), upload.name)
            scored_path = upload.name
        else:
            scored, warnings = load_scored_path(scored_path, _mtime(scored_path))
        context = "scored_file"
except (backend.PipelineUnavailable, ContractError) as exc:
    st.error(f"**Cannot show \"{SOURCES[source]}\".** {exc}")
    details = getattr(exc, "details", "")
    if details:
        with st.expander("Details"):
            st.code(details)
    st.button("Show synthetic demo data instead", on_click=use_synthetic, type="primary")
    st.stop()

events_backend = backend.load_events_backend()
thresholds = backend.display_thresholds()
models = ", ".join(sorted(scored["model_version"].unique()))

if context == "synthetic":
    st.markdown(theme.banner("synthetic",
        "⚠️ <b>SYNTHETIC DEMONSTRATION DATA.</b> Generated by the dashboard to show the interface while the "
        "model and risk pipeline are being built. These are not real conjunctions or real predictions, and "
        "their tiers are authored by scenario, not computed by <code>src/risk.py</code>."),
        unsafe_allow_html=True)
elif replay_hours is not None:
    st.markdown(theme.banner("replay",
        f"⏪ <b>Historical replay, events aligned to TCA.</b> Each training event is shown as it looked "
        f"{replay_hours} h before its own closest approach. This is not a real fleet forecast. "
        f"Model: <code>{models}</code>"), unsafe_allow_html=True)
else:
    label = "test-set predictions" if context == "test" else f"scored export <code>{scored_path}</code>"
    st.markdown(theme.banner("real", f"✅ Showing {label}. Probabilities are <b>estimated Pc</b> from model "
                f"<code>{models}</code>."), unsafe_allow_html=True)

if events_backend.provisional:
    st.caption("ℹ️ `src/events.py` is not complete yet, so latest-message selection, counts, ordering and "
               "insight text use the dashboard's provisional display logic. Missing: "
               + ", ".join(f"`{m}`" for m in events_backend.missing))

try:
    latest_all, _, _, _ = backend.event_views(scored, events_backend, thresholds, replay_hours)
except backend.PipelineUnavailable as exc:
    st.error(f"**Cannot build event views.** {exc}")
    if exc.details:
        with st.expander("Details"):
            st.code(exc.details)
    st.button("Show synthetic demo data instead", on_click=use_synthetic, type="primary")
    st.stop()


# ------------------------------------------------------------------ filters

with st.sidebar:
    st.header("Filters")
    present_tiers = [t for t in charts.TIER_SEQUENCE if t in set(latest_all["risk_tier"])]
    tiers = st.multiselect("Risk tier", present_tiers, default=present_tiers, format_func=theme.TIER_LABELS.get)
    obj_options = sorted(latest_all["c_object_type"].unique())
    objects = st.multiselect("Object type", obj_options, default=obj_options)
    max_h = float(np.ceil(latest_all["tca_hours"].max())) if not latest_all.empty else 24.0
    max_h = max(max_h, 24.0)
    tca_range = st.slider("Hours to TCA", 0.0, max_h, (0.0, max_h), step=1.0)
    event_query = st.text_input("Event ID", placeholder="e.g. 12")

mask = (
    latest_all["risk_tier"].isin(tiers)
    & latest_all["c_object_type"].isin(objects)
    & latest_all["tca_hours"].between(tca_range[0], tca_range[1])
)
if event_query.strip():
    try:
        mask &= latest_all["event_id"] == int(event_query.strip())
    except ValueError:
        st.sidebar.error("Event ID must be a whole number.")
keys = set(zip(latest_all.loc[mask, "source"], latest_all.loc[mask, "event_id"]))
in_keys = pd.Series([k in keys for k in zip(scored["source"], scored["event_id"])], index=scored.index)
scored_f = scored[in_keys]

if keys:
    latest, queue, summary, insights = backend.event_views(scored_f, events_backend, thresholds, replay_hours)
else:
    latest = queue = latest_all.iloc[0:0]
    summary = {k: 0 for k in backend.SUMMARY_KEYS}
    insights = ["No encounters match the current filters."]


# ------------------------------------------------------------------ summary cards

st.subheader("Risk overview")
st.caption(f"{len(latest)} encounter(s) shown, counted once each by their latest message"
           + (" at the replay horizon" if replay_hours is not None else "") + ". Counts follow the filters.")
tier_counts = latest["risk_tier"].value_counts()
cols = st.columns(len(TIER_CARD_ORDER))
for col, tier in zip(cols, TIER_CARD_ORDER):
    n = int(tier_counts.get(tier, 0))
    share = f"{100 * n / len(latest):.0f}% of shown" if len(latest) else "no encounters"
    col.markdown(theme.card(theme.TIER_LABELS[tier], n, theme.TIER_COLORS[tier], share), unsafe_allow_html=True)

st.subheader("⏱️ Next 24 hours")
miss_km = thresholds["MISS_DISTANCE_CRITICAL_M"] / 1000
cols = st.columns(4)
cards_24h = [
    ("Upcoming encounters", summary["total_encounters"], "#4A6FA5", "0–24 h to TCA"),
    (f"Below {miss_km:g} km miss distance", summary["distance_dangerous"], theme.TIER_COLORS["critical"],
     "distance-driven, any Pc"),
    ("Critical events", summary["critical"], theme.TIER_COLORS["critical"], "maneuver planning requested"),
    ("Elevated events", summary["elevated"], theme.TIER_COLORS["elevated"], "review recommended"),
]
for col, (label, value, color, sub) in zip(cols, cards_24h):
    col.markdown(theme.card(label, value, color, sub), unsafe_allow_html=True)
if summary["total_encounters"] == 0:
    if context == "test":
        st.info("**No test encounter is within 24 hours of TCA.** Every message in the supplied test file is "
                "at least 2 days before closest approach, so an empty 24-hour view is the correct answer. "
                "Switch to *Training replay* to demonstrate the alert logic.")
    else:
        st.info("No encounters within 24 hours of TCA match the current filters.")

with st.container(border=True):
    st.markdown("**💡 Automatic insights**")
    for line in insights:
        st.markdown(f"- {line}")

if warnings:
    with st.expander(f"⚠️ Data quality warnings ({len(warnings)})"):
        for w in warnings:
            st.warning(w)


# ------------------------------------------------------------------ tabs

tab_queue, tab_charts, tab_export = st.tabs(["🚨 Action queue & event detail", "📊 Overview charts",
                                             "⬇️ Export & data"])


def _join(values) -> str:
    return "; ".join(values) if len(values) else "—"


with tab_queue:
    if queue.empty:
        st.info("No encounters match the current filters. Widen the tier, object or time filters in the sidebar.")
    else:
        sort_choice = st.radio(
            "Sort by", ["Action priority", "Highest estimated Pc", "Closest miss distance", "Soonest TCA"],
            horizontal=True, help="Action priority: tier, then soonest TCA, then highest Pc. You can also "
                                  "click any column header to sort.")
        view = queue.copy()
        view["priority"] = np.arange(1, len(view) + 1)
        if sort_choice == "Highest estimated Pc":
            view = view.sort_values("predicted_log10_pc", ascending=False, na_position="last")
        elif sort_choice == "Closest miss distance":
            view = view.sort_values("miss_distance_m")
        elif sort_choice == "Soonest TCA":
            view = view.sort_values("tca_hours")
        view = view.reset_index(drop=True)

        table = pd.DataFrame({
            "Priority": view["priority"],
            "Tier": view["risk_tier"].map(theme.TIER_LABELS),
            "Event ID": view["event_id"],
            "Estimated Pc": view["predicted_pc"],
            "Miss distance (km)": view["miss_distance_km"],
            "Hours to TCA": view["tca_hours"],
            "Object type": view["c_object_type"],
            "Why flagged": view["alert_reasons"].map(_join),
            "Recommended action": view["recommended_action"],
            "Quality flags": view["quality_flags"].map(_join),
        })
        if scored["source"].nunique() > 1:
            table.insert(2, "Source", view["source"])
        st.caption("Select a row to open its details below.")
        selection = st.dataframe(
            table, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(table)),
            on_select="rerun", selection_mode="single-row", key=f"queue_{source}_{sort_choice}",
            column_config={
                "Priority": st.column_config.NumberColumn(width="small"),
                "Estimated Pc": st.column_config.NumberColumn(format="%.2e", help="Estimated Pc = 10^predicted log10(Pc)"),
                "Miss distance (km)": st.column_config.NumberColumn(format="%.2f"),
                "Hours to TCA": st.column_config.NumberColumn(format="%.1f"),
                "Why flagged": st.column_config.TextColumn(width="large"),
            },
        )

        selected_rows = selection.selection.rows if selection else []
        options = list(range(len(view)))
        idx = st.selectbox(
            "Event detail", options, index=selected_rows[0] if selected_rows else 0,
            format_func=lambda i: f"#{view.at[i, 'priority']} · event {view.at[i, 'event_id']} · "
                                  f"{theme.TIER_LABELS[view.at[i, 'risk_tier']]}",
        )
        ev = view.iloc[idx]
        history = scored[(scored["source"] == ev.source) & (scored["event_id"] == ev.event_id)]
        history = history[history["time_to_tca_days"] >= 0]
        if replay_hours is not None:
            history = history[history["time_to_tca_days"] >= replay_hours / 24]

        with st.container(border=True):
            st.markdown(f"### Event {ev.event_id} — {theme.TIER_TITLES[ev.risk_tier]}")
            m = st.columns(5)
            m[0].metric("Estimated Pc", f"{ev.predicted_pc:.2e}" if np.isfinite(ev.predicted_pc) else "unavailable")
            m[1].metric("Miss distance", f"{ev.miss_distance_km:.2f} km" if np.isfinite(ev.miss_distance_km) else "—")
            m[2].metric("Time to TCA", f"{ev.tca_hours:.1f} h")
            m[3].metric("Object type", ev.c_object_type)
            m[4].metric("Messages", len(history))

            box = {"critical": st.error, "needs_review": st.warning, "elevated": st.warning,
                   "safe": st.success}[ev.risk_tier]
            reasons = ev.alert_reasons or ["No alert trigger satisfied."]
            box(f"**{ev.recommended_action or 'No action supplied'}** — " + " ".join(reasons))
            if ev.quality_flags:
                st.caption("Quality flags: " + ", ".join(f"`{f}`" for f in ev.quality_flags))
            if replay_hours is not None:
                st.caption("Replay: history is cut at the replay horizon; newer messages are hidden.")

            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.risk_history(history, thresholds), width="stretch")
            c2.plotly_chart(charts.miss_history(history, thresholds), width="stretch")
            st.plotly_chart(charts.rtn_geometry(history), width="stretch")
            st.caption("Each point is a successive estimate of where the secondary object will be relative to "
                       "the primary at TCA (radial / transverse / normal, km). These are not measured flight "
                       "paths, and the available data cannot reconstruct real orbits.")

with tab_charts:
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.tier_breakdown(latest), width="stretch")
    c2.plotly_chart(charts.tca_timeline(latest), width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.risk_map(latest, thresholds), width="stretch")
    c4.plotly_chart(charts.object_types(latest), width="stretch")
    st.caption("All charts use the latest message per encounter under the current filters. Dashed lines mark "
               "the policy thresholds for reference; tiers come from the risk pipeline.")

with tab_export:
    prefix = {"synthetic": "SYNTHETIC_", "train_replay": f"replay_{replay_hours}h_"}.get(context, "")
    c1, c2 = st.columns(2)
    c1.download_button("Download shown encounters (latest message each)", to_export_csv(queue),
                       f"{prefix}encounters.csv", "text/csv", width="stretch", disabled=queue.empty)
    c2.download_button("Download all scored messages for shown encounters", to_export_csv(scored_f),
                       f"{prefix}scored_messages.csv", "text/csv", width="stretch",
                       disabled=scored_f.empty)
    st.caption("Dashboard exports follow the scored-row contract with list fields as JSON strings. "
               "They are not the scoring submission file.")
    st.markdown(f"**Source:** {SOURCES[source]}  \n**Model version(s):** `{models}`  \n"
                f"**Rows loaded:** {len(scored):,}  \n**Encounters:** {len(latest_all):,}")
    with st.expander("Scored-row contract columns"):
        st.code(", ".join(SCORED_COLUMNS))
