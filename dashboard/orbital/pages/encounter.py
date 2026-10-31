"""Encounter Analysis: visual encounter workspace.

Left (~60%): Encounter View (illustrative, built on measured R/T/N positions) or
Measured Geometry (3-D R/T/N). Right (~40%): decision support. Bottom: evidence.
All values come from AppState; nothing is hard-coded here.
"""

from __future__ import annotations

import reflex as rx

from orbital.components.layout import shell
from orbital.components.scene import encounter_scene
from orbital.components.widgets import tier_badge
from orbital.state import AppState

E = AppState.event
D = AppState.decision
M = AppState.scene_meta
C = AppState.cdm_detail

VIEW_INFO = ("Encounter View: each CDM is drawn at its measured relative position (R/T/N, projected to 2-D). "
             "The Earth backdrop, satellite icons and approach arcs are illustrative and not to scale. "
             "Measured Geometry: the same R/T/N measurements in 3-D, without decoration.")


def header() -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.span("Encounter #", E["event_id"], class_name="enc-title", data_testid="event-id"),
            rx.el.span(E["object_type"], " · ", AppState.source_label, class_name="enc-sub"),
            rx.el.span("model ", rx.el.b(E["model_version"], class_name="mono"), class_name="enc-sub"),
            rx.cond(AppState.is_synthetic, rx.el.span("SYNTHETIC", class_name="oi-stamp", data_testid="enc-synthetic"),
                    rx.el.span("REAL DATA", class_name="oi-badge real")),
            class_name="enc-head-left",
        ),
        rx.el.div(
            rx.el.button("Encounter View", on_click=AppState.set_view_mode("illustrative"), data_testid="mode-illustrative",
                         class_name=rx.cond(AppState.view_mode == "illustrative", "seg on", "seg")),
            rx.el.button("Measured Geometry", on_click=AppState.set_view_mode("measured"), data_testid="mode-measured",
                         class_name=rx.cond(AppState.view_mode == "measured", "seg on", "seg")),
            class_name="enc-seg",
        ),
        rx.el.span(rx.icon("info", size=16), title=VIEW_INFO, class_name="enc-info"),
        rx.el.div(class_name="oi-spacer"),
        rx.el.div(
        rx.el.button("Test set", class_name=rx.cond(AppState.scope == "test", "oi-chip on", "oi-chip"),
                     on_click=AppState.set_scope("test")),
        rx.el.button("Held-out", class_name=rx.cond(AppState.scope == "held_out", "oi-chip on", "oi-chip"),
                     on_click=AppState.set_scope("held_out"), data_testid="enc-scope-held"),
        rx.el.select(
            rx.foreach(AppState.event_options, lambda o: rx.el.option(o[1], value=o[0])),
            value=AppState.current_event_key, on_change=AppState.select_event, class_name="oi-select enc-select",
            data_testid="event-select",
        ),
        rx.el.div(
            rx.el.button(rx.icon("chevron-left", size=15), on_click=AppState.prev_event, data_testid="prev-event",
                         title="Previous encounter in queue order"),
            rx.el.span(AppState.event_position, class_name="mono"),
            rx.el.button(rx.icon("chevron-right", size=15), on_click=AppState.next_event, data_testid="next-event",
                         title="Next encounter in queue order"),
            class_name="enc-pager",
        ),
        class_name="enc-head-right",
        ),
        class_name="enc-head",
    )


def visual_panel() -> rx.Component:
    scene = encounter_scene(
        points=AppState.scene_points, h_axis=M["h_axis"], v_axis=M["v_axis"], dropped_axis=M["dropped_axis"],
        predicted_miss_km=AppState.scene_ring_km, predicted_miss_label=M["ring_label"],
        separation_label=M["separation"], countdown=E["countdown"], selected_id=AppState.selected_cdm_id,
        reset_nonce=AppState.scene_reset, on_select=AppState.select_cdm,
    )
    measured = rx.el.div(
        rx.plotly(data=AppState.geometry_figure, width="100%", height="100%", on_click=AppState.select_cdm_point),
        rx.el.button(rx.icon("rotate-ccw", size=14), "Reset view", class_name="oi-btn ghost enc-reset",
                     on_click=AppState.reset_encounter_view, data_testid="geometry-reset"),
        class_name="enc-measured", data_testid="geometry-plot",
    )
    view = rx.cond(
        M["available"] == "yes",
        rx.cond(AppState.view_mode == "measured", measured, scene),
        rx.el.div("No relative-position telemetry for this encounter.", class_name="oi-empty enc-novis"),
    )
    inspector = rx.el.div(
        rx.el.span("SELECTED CDM", class_name="enc-insp-tag"),
        rx.el.span(
            rx.el.button(rx.icon("chevron-left", size=13), on_click=AppState.step_cdm(-1), title="Earlier CDM",
                         data_testid="cdm-prev"),
            rx.el.button(rx.icon("chevron-right", size=13), on_click=AppState.step_cdm(1), title="Later CDM",
                         data_testid="cdm-next"),
            class_name="enc-insp-nav",
        ),
        rx.el.span("CDM ", C["n"], " · ", C["countdown"], class_name="mono", data_testid="inspector-cdm"),
        rx.el.span("time_to_tca ", rx.el.b(C["t"])),
        rx.el.span("miss ", rx.el.b(C["miss"])),
        rx.el.span("R ", rx.el.b(C["r"]), " · T ", rx.el.b(C["tt"]), " · N ", rx.el.b(C["nn"])),
        rx.el.span("observed Pc ", rx.el.b(C["pc"])),
        rx.el.span(C["role"], class_name="enc-insp-role"),
        class_name="enc-inspector", data_testid="cdm-inspector",
    )
    return rx.el.section(
        rx.el.div(
            rx.cond(AppState.view_mode == "measured",
                    rx.el.span("MEASURED R/T/N GEOMETRY · relative to the primary, km · drag to rotate, scroll to zoom, "
                               "click a CDM"),
                    rx.el.span("ILLUSTRATIVE ENCOUNTER VIEW · ", M["plane_note"],
                               " Earth, icons and arcs are not to scale.")),
            class_name="enc-vis-note", data_testid="view-note",
        ),
        rx.el.div(view, class_name="enc-stage"),
        inspector,
        class_name="oi-panel enc-left",
    )


def _card(label: str, value, sub, tag, tag_class, testid: str) -> rx.Component:
    return rx.el.div(
        rx.el.div(rx.el.span(label), rx.el.span(tag, class_name="inv-tag " + tag_class), class_name="inv-stat-label"),
        rx.el.div(value, class_name="inv-stat-value", data_testid=testid),
        rx.el.div(sub, class_name="inv-stat-sub"),
        class_name="inv-stat",
    )


def risk_panel() -> rx.Component:
    severity = rx.el.div(
        rx.el.div(rx.el.span("PREDICTED SEVERITY"), rx.el.span("How dangerous, and is it escalating?", class_name="q"),
                  class_name="inv-sec-head"),
        rx.el.div(
            rx.el.div(rx.el.div("CURRENT OBSERVED", class_name="inv-mini"),
                      tier_badge(E["observed_tier"], E["observed_label"]), data_testid="side-observed",
                      class_name="enc-sev-box"),
            rx.el.div(rx.icon("move-right", size=30), class_name="inv-arrow " + E["transition"].to(str)),
            rx.el.div(rx.el.div("PREDICTED FINAL · AI", class_name="inv-mini"),
                      tier_badge(E["predicted_tier"], E["predicted_label"]), data_testid="side-predicted",
                      class_name="enc-sev-box predicted"),
            class_name="enc-sev",
        ),
        rx.el.div(D["escalation"], class_name="enc-esc " + D["escalation_kind"].to(str), data_testid="escalation"),
        class_name="inv-sec",
    )
    cards = rx.el.div(
        _card("Predicted final Pc", E["pred_pc"], rx.el.span("log10(Pc) ", E["pred_log"]), "PREDICTED", "predicted",
              "pred-pc"),
        _card("Time to closest approach", E["countdown"],
              rx.el.span("at latest eligible CDM (", E["cutoff_days"], "); relative time"), "OBSERVED", "observed",
              "countdown"),
        _card("Observed miss distance", E["obs_miss"], "latest eligible CDM", "OBSERVED", "observed", "obs-miss"),
        rx.cond(
            E["miss_basis"] == "model prediction",
            _card("Predicted final miss distance", E["pred_miss"], "model prediction", "PREDICTED", "predicted",
                  "pred-miss"),
            _card("Predicted final miss distance", "Not provided", "the model did not predict it", "N/A", "not",
                  "pred-miss"),
        ),
        class_name="inv-stats",
    )
    uncertainty = rx.el.div(
        rx.el.div(rx.el.span("MODEL UNCERTAINTY"), rx.el.span(D["reliability"], class_name="enc-rel"),
                  class_name="enc-unc-head"),
        rx.cond(
            AppState.uncertainty_probs.length() > 0,
            rx.el.div(rx.foreach(AppState.uncertainty_probs, lambda p: rx.el.div(
                rx.el.span(p["label"], class_name="enc-prob-l"),
                rx.el.div(rx.el.div(class_name="enc-prob-fill " + p["tier"], style={"width": p["width"]}),
                          class_name="enc-prob-bar"),
                rx.el.span(p["value"], class_name="mono enc-prob-v"), class_name="enc-prob")),
                class_name="enc-probs"),
            rx.fragment(),
        ),
        rx.cond(AppState.uncertainty_interval != "",
                rx.el.div("Predicted Pc range ", rx.el.b(AppState.uncertainty_interval, class_name="mono"),
                          class_name="oi-note"), rx.fragment()),
        rx.cond((AppState.uncertainty_probs.length() == 0) & (AppState.uncertainty_interval == ""),
                rx.el.div("Uncertainty not provided", class_name="enc-unc-none", data_testid="uncertainty-none"),
                rx.fragment()),
        class_name="enc-unc", data_testid="uncertainty",
    )

    def threshold_row(r):
        return rx.el.div(
            rx.el.div(rx.el.div(r["criterion"], class_name="inv-crit"), rx.el.div(r["rule"], class_name="inv-rule")),
            rx.el.div(r["value"], class_name="mono", style={"text_align": "right"}),
            rx.el.span(rx.match(r["level"], ("critical", "critical level"), ("elevated", "elevated level"),
                                ("none", "below thresholds"), "not provided"),
                       class_name=rx.cond(r["triggered"] == "yes", "inv-hit on", "inv-hit")),
            class_name="inv-trow", data_testid="threshold-row",
        )

    why = rx.el.details(
        rx.el.summary(rx.el.span("WHY THIS TIER?"), rx.icon("chevron-down", size=15, class_name="enc-chev")),
        rx.el.div(D["summary"], class_name="inv-why", data_testid="why-summary"),
        rx.foreach(AppState.threshold_rows, threshold_row),
        rx.cond(D["basis_note"] != "", rx.el.div(rx.icon("info", size=13), " ", D["basis_note"], class_name="enc-basis",
                                                 data_testid="basis-note"), rx.fragment()),
        rx.cond(AppState.stated_reasons.length() > 0,
                rx.el.div(rx.el.div("MODEL'S STATED REASONS", class_name="enc-reasons-tag"),
                          rx.el.ul(rx.foreach(AppState.stated_reasons, lambda r: rx.el.li(r)), class_name="inv-notes"),
                          class_name="enc-reasons", data_testid="stated-reasons"),
                rx.fragment()),
        rx.cond(D["consistent"] == "no",
                rx.el.div(rx.icon("triangle-alert", size=14), " The supplied tier is lower than the official thresholds "
                          "imply for these values. Request corrected information before relying on it.",
                          class_name="inv-warn", data_testid="under-warning"), rx.fragment()),
        open=True, class_name="enc-why",
    )
    step = rx.el.div(
        rx.el.div(
            rx.match(D["step_key"], ("refer", rx.icon("user-check", size=18)), ("prioritise", rx.icon("radar", size=18)),
                     ("data", rx.icon("database", size=18)), rx.icon("eye", size=18)),
            rx.el.div(rx.el.div("RECOMMENDED NEXT STEP", class_name="enc-step-tag"),
                      rx.el.div(D["step_label"], class_name="enc-step-label", data_testid="next-step")),
            class_name="enc-step-head",
        ),
        rx.el.ul(rx.foreach(AppState.next_step_reasons, lambda n: rx.el.li(n)), class_name="inv-notes"),
        rx.el.div("Suggested review action only. The system does not calculate, approve or execute manoeuvres.",
                  class_name="enc-step-foot"),
        class_name="enc-step " + D["step_key"].to(str),
    )
    return rx.el.section(severity, cards, uncertainty, why, step, class_name="oi-panel enc-right",
                         data_testid="summary-panel")


def evidence() -> rx.Component:
    def mini_row(r):
        return rx.el.tr(rx.el.td(r["t"], class_name="mono"), rx.el.td(r["log"], class_name="mono"),
                        rx.el.td(r["pc"], class_name="mono"), rx.el.td(r["miss"], class_name="mono"))

    def full_row(r):
        return rx.el.tr(*[rx.el.td(r[k], class_name="mono") for k in ("t", "log", "pc", "miss", "speed", "r", "tt", "n")],
                        rx.el.td(r["used"]), rx.el.td(r["flags"], class_name="muted"))

    def chart_panel(title, info, fig, testid):
        return rx.el.section(
            rx.el.div(rx.el.span(title), rx.el.span(rx.icon("info", size=13), title=info, class_name="enc-info"),
                      class_name="oi-panel-head"),
            rx.el.div(rx.plotly(data=fig, width="100%", height="280px"), class_name="enc-chart"),
            class_name="oi-panel", data_testid=testid,
        )

    return rx.fragment(
        rx.el.div(
            chart_panel("Risk evolution", "Observed log10(Pc) of each CDM against time_to_tca (days). The dashed line "
                        "is the 2-day prediction cutoff; the star is the model's predicted final value.",
                        AppState.risk_figure, "chart-risk"),
            chart_panel("Miss-distance evolution", "Observed miss distance (metres, log scale) against time_to_tca "
                        "(days). A predicted point appears only if the model supplies a final miss distance.",
                        AppState.miss_figure, "chart-miss"),
            rx.el.section(
                rx.el.div(rx.el.span("CDM history"), rx.el.span("latest 6", class_name="sub"), class_name="oi-panel-head"),
                rx.el.div(rx.el.table(
                    rx.el.thead(rx.el.tr(*[rx.el.th(h) for h in ("time_to_tca (d)", "log10 Pc", "Pc", "miss (m)")])),
                    rx.el.tbody(rx.foreach(AppState.latest_cdms, mini_row)), class_name="oi-table compact"),
                    class_name="oi-table-wrap"),
                class_name="oi-panel", data_testid="cdm-mini",
            ),
            class_name="enc-bottom",
        ),
        rx.el.details(
            rx.el.summary(rx.icon("table", size=14), " Full CDM telemetry (all messages, units in headers)"),
            rx.el.div(rx.el.table(
                rx.el.thead(rx.el.tr(*[rx.el.th(h) for h in ("time_to_tca (d)", "log10 Pc", "Pc", "miss (m)",
                                                             "rel. speed (m/s)", "R (m)", "T (m)", "N (m)", "role",
                                                             "flags")])),
                rx.el.tbody(rx.foreach(AppState.history_rows, full_row)), class_name="oi-table"),
                class_name="oi-table-wrap", style={"max_height": "380px"}, data_testid="cdm-table"),
            class_name="oi-panel enc-telemetry", data_testid="telemetry",
        ),
    )


def encounter() -> rx.Component:
    content = rx.cond(
        AppState.event.length() > 0,
        rx.fragment(header(), rx.el.div(visual_panel(), risk_panel(), class_name="enc-grid"), evidence()),
        rx.el.div("No encounter matches the current filters.", class_name="oi-empty"),
    )
    return shell("/encounter", "Encounter Analysis", content)
