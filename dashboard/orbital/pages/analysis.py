"""Model Validation and Reports pages."""

from __future__ import annotations

import reflex as rx

from orbital.components.layout import shell
from orbital.components.widgets import panel, queue_table
from orbital.data import adapters
from orbital.data.contract import EVENT_OPTIONAL, EVENT_REQUIRED, HISTORY_OPTIONAL, HISTORY_REQUIRED
from orbital.state import AppState

def _kv(label: str, value) -> rx.Component:
    return rx.fragment(rx.el.dt(label), rx.el.dd(value))


def validation() -> rx.Component:
    V = AppState.validation

    def metric(label, key, foot):
        return rx.el.div(rx.el.div(label, class_name="oi-kpi-label"), rx.el.div(V[key], class_name="oi-kpi-value"),
                         rx.el.div(foot, class_name="oi-kpi-foot"), class_name="oi-kpi", style={"--accent": "#38bdf8",
                                                                                                "cursor": "default"})

    def matrix_row(row):
        return rx.el.tr(
            rx.el.th(row[0]),
            rx.foreach(row[1:], lambda cell: rx.el.td(cell.split("|")[0], class_name=cell.split("|")[1])),
        )

    banner = rx.cond(
        AppState.is_synthetic,
        rx.el.div(rx.icon("flask-conical", size=15),
                  rx.el.span(rx.el.b("Demonstration only. "),
                             "These metrics are computed from SYNTHETIC held-out events and say nothing about the "
                             "real model's performance. Load real model outputs to see measured results."),
                  class_name="oi-synbar", data_testid="validation-demo"),
        rx.el.div(rx.icon("shield-check", size=15),
                  rx.el.span("Computed from held-out training events (", V["splits"], ") in the loaded model output. "
                             "Test-set events have no final labels and are never used here."),
                  class_name="oi-synbar", style={"color": "#86efac", "border_color": "rgba(34,197,94,0.45)",
                                                 "background": "rgba(34,197,94,0.06)"}),
    )
    return shell(
        "/validation", "Model Validation",
        banner,
        rx.el.div(
            metric("Evaluated events", "n", "held-out training events, split by event_id"),
            metric("Overall accuracy", "accuracy", "predicted final tier = actual"),
            metric("Critical recall", "recall", rx.el.span(V["tp"], " of ", V["critical_actual"], " actual critical caught")),
            metric("Critical precision", "precision", rx.el.span(V["fp"], " false alarms")),
            metric("Missed critical (FN)", "fn", "highest-consequence error"),
            class_name="oi-grid-kpi",
        ),
        rx.el.div(
            panel("Confusion matrix — actual final (rows) vs predicted final (columns)",
                  rx.el.table(
                      rx.el.thead(rx.el.tr(rx.el.th(""), *[rx.el.th(h) for h in ("Critical", "Elevated", "Safe", "Review")])),
                      rx.el.tbody(rx.foreach(AppState.matrix_rows, matrix_row)),
                      class_name="oi-matrix", data_testid="confusion-matrix"),
                  rx.el.div("Green: correct. Red: actual Critical predicted as something else (missed).",
                            class_name="oi-note", style={"margin_top": "8px"})),
            panel("Limitations and uncertainty",
                  rx.el.ul(
                      rx.el.li("The model estimates the supplied risk label; it is not an independent physical Pc "
                               "calculation and is not calibrated against observed collisions."),
                      rx.el.li("Critical events are rare, so recall and precision rest on few examples and swing a lot "
                               "between splits; report counts alongside percentages."),
                      rx.el.li("risk = −30 is a floor meaning ≈ 0 probability; errors there say little about real danger."),
                      rx.el.li("Only CDMs with time_to_tca ≥ 2 days are inputs; later changes in the encounter cannot be seen."),
                      rx.el.li("The test set has no final labels, so it is never used for these metrics."),
                      class_name="oi-note", style={"padding_left": "18px", "margin": 0, "display": "flex",
                                                   "flex_direction": "column", "gap": "6px"}),
                  ),
            class_name="oi-grid-2",
        ),
        panel("Missed critical encounters (held-out)",
              rx.cond(AppState.missed_critical.length() > 0,
                      queue_table(AppState.missed_critical, show_actual=True),
                      rx.el.div("No missed critical encounters in this held-out set.", class_name="oi-empty")),
              sub="actual final Critical, predicted lower; click to inspect", body_class=""),
    )


def reports() -> rx.Component:
    def li(text):
        return rx.el.li(text, class_name="mono", style={"font_size": "11.5px"})

    return shell(
        "/reports", "Reports",
        rx.el.div(
            panel("Loaded dataset",
                  rx.el.dl(_kv("Source", AppState.source_label), _kv("Model version", AppState.model_version),
                           _kv("Loaded at", AppState.loaded_at), class_name="oi-kv"),
                  rx.el.div("Files", class_name="oi-group-label", style={"margin_top": "12px", "display": "block"}),
                  rx.el.ul(rx.foreach(AppState.source_files, li), style={"padding_left": "18px"}),
                  rx.el.div(
                      rx.el.button(rx.icon("download", size=14), "Export filtered queue (CSV)", class_name="oi-btn",
                                   on_click=AppState.download_queue),
                      rx.el.button(rx.icon("refresh-cw", size=14), "Reload", class_name="oi-btn ghost",
                                   on_click=AppState.reload),
                      style={"display": "flex", "gap": "8px", "margin_top": "10px"})),
            panel("Data-quality warnings",
                  rx.cond(AppState.data_warnings.length() > 0,
                          rx.el.ul(rx.foreach(AppState.data_warnings, lambda w: rx.el.li(w)), class_name="oi-note",
                                   style={"padding_left": "18px", "margin": 0}),
                          rx.el.div("No warnings for this dataset.", class_name="oi-note"))),
            class_name="oi-grid-2",
        ),
        panel("Proposed data contract v2 (awaiting team approval)",
              rx.el.div("Event-level file outputs/event_predictions.csv", class_name="oi-group-label"),
              rx.el.p(rx.el.b("Required: "), ", ".join(EVENT_REQUIRED), class_name="oi-note mono"),
              rx.el.p(rx.el.b("Optional: "), ", ".join(EVENT_OPTIONAL), class_name="oi-note mono"),
              rx.el.div("Per-CDM file outputs/cdm_history.csv", class_name="oi-group-label"),
              rx.el.p(rx.el.b("Required: "), ", ".join(HISTORY_REQUIRED), class_name="oi-note mono"),
              rx.el.p(rx.el.b("Optional: "), ", ".join(HISTORY_OPTIONAL), class_name="oi-note mono"),
              rx.el.p("Column aliases accepted by the adapter: ",
                      ", ".join(f"{a}→{b}" for a, b in adapters.COLUMN_ALIASES["events"].items()),
                      class_name="oi-note mono")),
        panel("Globe data provenance",
              rx.el.p("The 3D globe uses a synthetic orbital population (orbital/data/synthetic_globe.py) with circular "
                      "orbits at one simulation epoch. It is not derived from the Airbus dataset, contains no real "
                      "objects, and no globe object is linked to an Airbus encounter. Earth textures: NASA Blue Marble "
                      "imagery (public domain), via the three-globe package examples.", class_name="oi-note")),
    )
