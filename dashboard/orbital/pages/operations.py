"""Mission Control, Global Tracking and Collision Risk pages."""

from __future__ import annotations

import reflex as rx

from orbital.components.globe_view import globe_panel
from orbital.components.layout import shell
from orbital.components.widgets import (
    filter_bar,
    kpi_row,
    panel,
    queue_table,
    transitions_panel,
    windows_panel,
)
from orbital.state import AppState


def mission_control() -> rx.Component:
    queue = panel(
        "Priority warning queue",
        queue_table(AppState.queue_top, compact=True),
        rx.el.div(
            rx.el.span(AppState.queue_count, " encounters match · test set, sorted by predicted final tier",
                       class_name="oi-note"),
            rx.el.a(rx.el.button("Open full queue", rx.icon("arrow-right", size=14), class_name="oi-btn"), href="/risk"),
            style={"display": "flex", "justify_content": "space-between", "align_items": "center", "padding": "10px 2px 0"},
        ),
        sub="click a row for analysis",
        body_class="oi-panel-body",
    )
    return shell(
        "/", "Mission Control",
        kpi_row(),
        rx.el.div(
            rx.el.div(windows_panel(), transitions_panel(),
                      panel("Decision policy",
                            rx.el.div("Tiers are produced by the risk pipeline, not by this interface:",
                                      class_name="oi-note"),
                            rx.el.ul(
                                rx.el.li(rx.el.b("Critical", style={"color": "#fca5a5"}),
                                         ": Pc ≥ 1e-4 or final miss distance < 1,000 m"),
                                rx.el.li(rx.el.b("Elevated", style={"color": "#fcd34d"}),
                                         ": Pc ≥ 1e-6 or miss distance 1,000–5,000 m"),
                                rx.el.li(rx.el.b("Safe", style={"color": "#86efac"}),
                                         ": Pc < 1e-6 and miss distance > 5,000 m"),
                                class_name="oi-note", style={"padding_left": "18px", "margin": "8px 0"},
                            ),
                            rx.el.div("Inputs: CDMs with time_to_tca ≥ 2 days only. Critical requests human review; "
                                      "no manoeuvre is executed automatically.", class_name="oi-note")),
                      class_name="oi-col"),
            globe_panel(show_filters=False, height="640px"),
            rx.el.div(queue, class_name="oi-col"),
            class_name="oi-overview",
        ),
    )


def global_tracking() -> rx.Component:
    return shell(
        "/tracking", "Global Tracking",
        globe_panel(show_filters=True, height="calc(100vh - 200px)"),
        rx.el.div(
            "The Airbus challenge data contains relative encounter geometry only: no absolute positions, TLEs or "
            "trajectories. This globe therefore shows a separate synthetic population with simple circular orbits "
            "at one simulation epoch. Positions, orbits and separations come from that simulation alone, and no "
            "object is linked to an Airbus encounter.",
            class_name="oi-note",
        ),
        needs_data=False,
    )


def collision_risk() -> rx.Component:
    return shell(
        "/risk", "Collision Risk",
        kpi_row(),
        panel("Filters", filter_bar()),
        panel(
            "Prioritised collision-warning queue",
            queue_table(AppState.queue),
            sub=rx.el.span(AppState.queue_count, " encounters"),
            actions=rx.el.button(rx.icon("download", size=14), "Export CSV", class_name="oi-btn ghost",
                                 on_click=AppState.download_queue, data_testid="export"),
            body_class="",
        ),
        rx.el.div(
            panel("Early-warning timeline", rx.plotly(data=AppState.window_figure, width="100%"),
                  sub="all tiers, current object/search filters"),
            transitions_panel(),
            class_name="oi-grid-2",
        ),
    )
