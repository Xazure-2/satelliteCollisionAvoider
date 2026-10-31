"""Encounter Analysis: two-panel investigation workspace.

Left: relative R/T/N geometry. Right: answers to the operator's questions
(how dangerous, why, how long, how reliable, what next). Detail lives in tabs.
"""

from __future__ import annotations

import reflex as rx

from orbital.components.layout import shell
from orbital.components.widgets import panel, tier_badge
from orbital.state import AppState

E = AppState.event
D = AppState.decision


def _kv(label: str, value) -> rx.Component:
    return rx.fragment(rx.el.dt(label), rx.el.dd(value))


def _section(title: str, *children, question: str = "") -> rx.Component:
    return rx.el.div(
        rx.el.div(rx.el.span(title), rx.el.span(question, class_name="q") if question else rx.fragment(),
                  class_name="inv-sec-head"),
        *children,
        class_name="inv-sec",
    )


def _stat(label: str, value, sub, tag=None, tag_class="", testid: str = "") -> rx.Component:
    return rx.el.div(
        rx.el.div(label, rx.el.span(tag, class_name="inv-tag " + tag_class) if tag is not None else rx.fragment(),
                  class_name="inv-stat-label"),
        rx.el.div(value, class_name="inv-stat-value", data_testid=testid),
        rx.el.div(sub, class_name="inv-stat-sub"),
        class_name="inv-stat",
    )


def toolbar() -> rx.Component:
    return rx.el.div(
        rx.el.span("Encounter", class_name="oi-group-label"),
        rx.el.select(
            rx.foreach(AppState.event_options, lambda o: rx.el.option(o[1], value=o[0])),
            value=AppState.current_event_key, on_change=AppState.select_event, class_name="oi-select",
            data_testid="event-select",
        ),
        rx.el.button(rx.icon("chevron-left", size=15), class_name="oi-btn ghost", on_click=AppState.prev_event,
                     title="Previous encounter in queue order", data_testid="prev-event"),
        rx.el.button(rx.icon("chevron-right", size=15), class_name="oi-btn ghost", on_click=AppState.next_event,
                     title="Next encounter in queue order", data_testid="next-event"),
        rx.el.span(AppState.event_position, class_name="mono muted", style={"font_size": "12px"}),
        rx.el.span("Population", class_name="oi-group-label", style={"margin_left": "12px"}),
        rx.el.button("Test set", class_name=rx.cond(AppState.scope == "test", "oi-chip on", "oi-chip"),
                     on_click=AppState.set_scope("test")),
        rx.el.button("Held-out training", class_name=rx.cond(AppState.scope == "held_out", "oi-chip on", "oi-chip"),
                     on_click=AppState.set_scope("held_out"), data_testid="enc-scope-held"),
        rx.el.div(class_name="oi-spacer"),
        rx.el.span("Order and list follow the Collision Risk filters", class_name="oi-note"),
        class_name="oi-toolbar inv-toolbar",
    )


def geometry_panel() -> rx.Component:
    return rx.el.section(
        rx.el.div(
            rx.el.span("Relative encounter geometry"),
            rx.el.span("R/T/N frame centred on the primary object", class_name="sub"),
            rx.el.div(class_name="oi-spacer"),
            rx.el.span("NOT AN ORBITAL TRAJECTORY OR EARTH MAP", class_name="inv-caveat", data_testid="geometry-caveat"),
            class_name="oi-panel-head",
        ),
        rx.el.div(rx.plotly(data=AppState.geometry_figure, width="100%", height="680px"), data_testid="geometry-plot"),
        rx.el.div(
            rx.el.span(rx.icon("mouse-pointer-2", size=13), " Drag to rotate · scroll to zoom · right-drag to pan · "
                       "hover a point for its CDM"),
            rx.el.span("Each point is one CDM's estimate of where the secondary will be relative to the primary at "
                       "closest approach. The predicted miss-distance sphere shows a distance only: the model does "
                       "not predict a direction."),
            class_name="inv-footnote",
        ),
        class_name="oi-panel inv-left",
    )


def summary_panel() -> rx.Component:
    header = rx.el.div(
        rx.el.div(
            rx.el.div("ENCOUNTER", class_name="oi-side-tag"),
            rx.el.div("#", E["event_id"], class_name="inv-id", data_testid="event-id"),
        ),
        rx.el.div(
            rx.el.span(E["object_type"], class_name="oi-chip on", style={"cursor": "default"}),
            rx.el.span(E["split"], class_name="oi-chip", style={"cursor": "default"}),
            rx.cond(AppState.is_synthetic, rx.el.span("SYNTHETIC", class_name="oi-stamp"), rx.fragment()),
            style={"display": "flex", "gap": "6px", "align_items": "center", "flex_wrap": "wrap",
                   "justify_content": "flex-end"},
        ),
        class_name="inv-header",
    )
    severity = _section(
        "Predicted severity",
        rx.el.div(
            rx.el.div(rx.el.div("CURRENT OBSERVED", class_name="inv-mini"),
                      tier_badge(E["observed_tier"], E["observed_label"]), data_testid="side-observed"),
            rx.el.div(rx.icon("move-right", size=26), class_name="inv-arrow " + E["transition"].to(str)),
            rx.el.div(rx.el.div("PREDICTED FINAL", class_name="inv-mini"),
                      tier_badge(E["predicted_tier"], E["predicted_label"]), data_testid="side-predicted"),
            rx.el.div(E["transition"], class_name="inv-trend " + E["transition"].to(str), data_testid="transition"),
            class_name="inv-transition",
        ),
        rx.el.div(
            _stat("Time to closest approach", E["countdown"], rx.el.span("at latest eligible CDM (", E["cutoff_days"],
                  "), relative time only"), testid="countdown"),
            _stat("Predicted final Pc", E["pred_pc"], rx.el.span("log10 ", E["pred_log"]), "PREDICTED", "predicted",
                  "pred-pc"),
            _stat("Observed miss distance", E["obs_miss"], "latest eligible CDM", "OBSERVED", "observed", "obs-miss"),
            _stat("Final miss distance", E["pred_miss"], E["miss_basis"], E["miss_tag"], E["miss_tag_class"],
                  "pred-miss"),
            class_name="inv-stats",
        ),
        question="How dangerous, and how soon?",
    )

    def threshold_row(r):
        return rx.el.div(
            rx.el.div(rx.el.div(r["criterion"], class_name="inv-crit"), rx.el.div(r["rule"], class_name="inv-rule")),
            rx.el.div(r["value"], class_name="mono", style={"text_align": "right"}),
            rx.el.span(
                rx.match(r["level"], ("critical", "critical level"), ("elevated", "elevated level"),
                         ("none", "below thresholds"), "not provided"),
                class_name=rx.cond(r["triggered"] == "yes", "inv-hit on", "inv-hit"),
            ),
            class_name="inv-trow",
            data_testid="threshold-row",
        )

    why = _section(
        "Why this tier",
        rx.el.div(D["summary"], class_name="inv-why", data_testid="why-summary"),
        rx.foreach(AppState.threshold_rows, threshold_row),
        rx.cond(D["consistent"] == "no",
                rx.el.div(rx.icon("triangle-alert", size=14), " The tier supplied by the risk pipeline does not match "
                          "these values. Check the inputs before relying on it.", class_name="inv-warn"),
                rx.fragment()),
        question="Why was it classified this way?",
    )
    reliability = _section(
        "Reliability",
        rx.el.div(
            rx.el.span(D["reliability"], class_name="inv-rel " + D["reliability"].to(str).replace(" ", "-"),
                       data_testid="reliability"),
            rx.el.span("derived only from uncertainty the model supplied", class_name="oi-note"),
            style={"display": "flex", "gap": "10px", "align_items": "center"},
        ),
        rx.el.ul(rx.foreach(AppState.reliability_notes, lambda n: rx.el.li(n)), class_name="inv-notes"),
        question="How reliable is the prediction?",
    )
    step = _section(
        "Recommended next step",
        rx.el.div(
            rx.el.div(rx.match(D["step_key"], ("refer", rx.icon("user-check", size=18)),
                               ("request", rx.icon("satellite-dish", size=18)), rx.icon("eye", size=18)),
                      D["step_label"], class_name="inv-step-label", data_testid="next-step"),
            rx.el.ul(rx.foreach(AppState.next_step_reasons, lambda n: rx.el.li(n)), class_name="inv-notes"),
            class_name="inv-step " + D["step_key"].to(str),
        ),
        rx.el.div("Decision support only. The system does not decide whether a manoeuvre is needed or safe; "
                  "qualified operators make that call.", class_name="oi-note", style={"margin_top": "8px"}),
        question="What should I investigate next?",
    )
    return rx.el.section(header, severity, why, reliability, step, class_name="oi-panel inv-right",
                         data_testid="summary-panel")


def detail_tabs() -> rx.Component:
    def cdm_row(r):
        return rx.el.tr(*[rx.el.td(r[k], class_name="mono") for k in ("t", "log", "pc", "miss", "speed", "r", "tt", "n")],
                        rx.el.td(r["used"]), rx.el.td(r["flags"], class_name="muted"))

    cdm_table = rx.el.div(rx.el.table(
        rx.el.thead(rx.el.tr(*[rx.el.th(h) for h in ("time_to_tca (d)", "log10 Pc", "Pc", "miss (m)", "speed (m/s)",
                                                     "R (m)", "T (m)", "N (m)", "role", "flags")])),
        rx.el.tbody(rx.foreach(AppState.history_rows, cdm_row)), class_name="oi-table"),
        class_name="oi-table-wrap", style={"max_height": "420px"}, data_testid="cdm-table")

    quality = rx.el.div(
        rx.el.div(
            rx.el.div("MODEL OUTPUT", class_name="oi-side-tag"),
            rx.el.dl(_kv("P(critical)", E["p_crit"]), _kv("P(elevated)", E["p_elev"]), _kv("P(safe)", E["p_safe"]),
                     _kv("Predicted Pc range", E["uncertainty"]), _kv("At −30 floor (≈ 0)", E["floor"]),
                     _kv("Final miss-distance basis", E["miss_basis"]), _kv("CDMs used (≥ 2 d)", E["n_cdms"]),
                     _kv("Cutoff time_to_tca", E["cutoff_days"]), _kv("Model version", E["model_version"]),
                     class_name="oi-kv", style={"margin_top": "10px"}),
            rx.el.div("Fields shown as “—” or “not provided” were not supplied by the model; nothing is estimated "
                      "by the dashboard.", class_name="oi-note", style={"margin_top": "10px"}),
            class_name="oi-side",
        ),
        rx.el.div(
            rx.el.div("DATA QUALITY & CONTEXT", class_name="oi-side-tag"),
            rx.el.dl(_kv("Flags on prediction", E["flags"]), _kv("Input CDMs with flags", E["used_flags"]),
                     _kv("Pipeline alert triggers", E["reasons"]), _kv("Pipeline action", E["action"]),
                     class_name="oi-kv", style={"margin_top": "10px"}),
            rx.el.div(E["explanation"], class_name="oi-callout " + E["transition"].to(str),
                      style={"margin_top": "12px"}, data_testid="explanation"),
            rx.cond(E["actual_label"] != "",
                    rx.el.div(rx.el.b("Held-out truth (evaluation only, never a model input): "),
                              tier_badge(E["actual_tier"], E["actual_label"]), " actual final Pc ", E["actual_pc"],
                              class_name="oi-note", style={"margin_top": "12px"}),
                    rx.fragment()),
            rx.cond(AppState.data_warnings.length() > 0,
                    rx.el.ul(rx.foreach(AppState.data_warnings, lambda w: rx.el.li(w)), class_name="inv-notes",
                             style={"margin_top": "10px"}),
                    rx.fragment()),
            class_name="oi-side",
        ),
        class_name="oi-grid-2",
    )

    def tab(label, value, icon):
        return rx.tabs.trigger(rx.icon(icon, size=14), label, value=value, class_name="inv-tab",
                               data_testid=f"tab-{value}")

    return rx.el.section(
        rx.tabs.root(
            rx.tabs.list(
                tab("Risk evolution", "risk", "activity"),
                tab("Miss-distance evolution", "miss", "ruler"),
                tab("CDM history", "cdm", "table"),
                tab("Model uncertainty & data quality", "quality", "gauge"),
                class_name="inv-tablist",
            ),
            rx.tabs.content(rx.plotly(data=AppState.risk_figure, width="100%"), value="risk", class_name="inv-tabbody"),
            rx.tabs.content(rx.plotly(data=AppState.miss_figure, width="100%"), value="miss", class_name="inv-tabbody"),
            rx.tabs.content(cdm_table, value="cdm", class_name="inv-tabbody"),
            rx.tabs.content(quality, value="quality", class_name="inv-tabbody"),
            default_value="risk",
        ),
        class_name="oi-panel",
        data_testid="detail-tabs",
    )


def encounter() -> rx.Component:
    content = rx.cond(
        AppState.event.length() > 0,
        rx.fragment(
            rx.el.div(geometry_panel(), summary_panel(), class_name="inv-grid"),
            detail_tabs(),
        ),
        rx.el.div("No encounter matches the current filters.", class_name="oi-empty"),
    )
    return shell("/encounter", "Encounter Analysis", panel("Investigate encounter", toolbar()), content)
