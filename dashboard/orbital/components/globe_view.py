"""Globe panel: WebGL globe plus legend, controls and object detail panel.

Everything on the globe is the SYNTHETIC orbital simulation and is labelled so.
It is never linked to Airbus encounter predictions.
"""

from __future__ import annotations

import reflex as rx

from orbital.components.globe import orbit_globe
from orbital.components.widgets import chip
from orbital.state import GLOBE_PROXIMITY, GLOBE_TYPES, AppState

TYPE_SWATCH = [("PAYLOAD", "#22d3ee", "circle", "Payload"), ("ROCKET BODY", "#a78bfa", "square", "Rocket body"),
               ("DEBRIS", "#cbd5e1", "diamond", "Debris"), ("UNKNOWN", "#7c8aa5", "ring", "Unknown")]


def legend() -> rx.Component:
    rows = [rx.el.div(rx.el.span(class_name=f"sw {shape}", style={"background": color, "color": color}), label,
                      class_name="oi-legend-row") for _, color, shape, label in TYPE_SWATCH]
    rows += [
        rx.el.div(rx.el.span(class_name="sw circle", style={"background": "#ef4444"}),
                  "Simulated separation < 1 km (pulsing ring)", class_name="oi-legend-row"),
        rx.el.div(rx.el.span(class_name="sw circle", style={"background": "#f59e0b"}),
                  "Simulated separation 1–5 km", class_name="oi-legend-row"),
        rx.el.div(rx.el.span(class_name="sw circle", style={"background": "#38bdf8"}), "Selected object",
                  class_name="oi-legend-row"),
        rx.el.div(rx.el.span(class_name="sw line"), "Synthetic orbit of selection", class_name="oi-legend-row"),
        rx.el.div("Altitudes above 2,000 km are compressed for display.", style={"font_size": "10px", "opacity": 0.8}),
    ]
    return rx.el.div(*rows, class_name="oi-legend compact", data_testid="globe-legend")


def object_panel(compact: bool = False) -> rx.Component:
    info = AppState.object_info

    def kv(label, key):
        return rx.fragment(rx.el.dt(label), rx.el.dd(info[key]))

    if compact:
        return rx.cond(
            AppState.selected_object != "",
            rx.el.div(
                rx.el.div(rx.icon("satellite", size=14), rx.el.span(info["id"], class_name="mono"),
                          rx.el.div(class_name="oi-spacer"),
                          rx.el.button(rx.icon("x", size=13), class_name="oi-btn ghost", on_click=AppState.reset_view,
                                       style={"padding": "3px 5px"}),
                          class_name="oi-panel-head"),
                rx.el.div(
                    rx.el.div("SYNTHETIC ORBITAL OBJECT", class_name="oi-stamp", style={"display": "inline-block",
                                                                                        "margin_bottom": "8px"}),
                    rx.el.dl(kv("Type", "type"), kv("Regime", "regime"), kv("Altitude", "alt"), class_name="oi-kv"),
                    rx.el.div(info["proximity_text"], class_name="oi-note", style={"margin_top": "6px"}),
                    rx.el.div(
                        rx.el.button(rx.icon("crosshair", size=13), "Focus", class_name="oi-btn",
                                     on_click=AppState.focus_object),
                        rx.el.button(rx.icon("globe", size=13), "Global view", class_name="oi-btn ghost",
                                     on_click=AppState.reset_view),
                        rx.el.a(rx.el.button("Details", rx.icon("arrow-right", size=13), class_name="oi-btn ghost"),
                                href="/tracking"),
                        style={"display": "flex", "flex_wrap": "wrap", "gap": "6px", "margin_top": "10px"},
                    ),
                    class_name="oi-panel-body",
                ),
                class_name="oi-objpanel compact",
                data_testid="object-panel",
            ),
            rx.fragment(),
        )

    return rx.cond(
        AppState.selected_object != "",
        rx.el.div(
            rx.el.div(
                rx.icon("satellite", size=15), rx.el.span(info["id"], class_name="mono"),
                rx.el.div(class_name="oi-spacer"),
                rx.el.button(rx.icon("x", size=14), class_name="oi-btn ghost", on_click=AppState.reset_view,
                             title="Close and return to global view", style={"padding": "4px 6px"}),
                class_name="oi-panel-head",
            ),
            rx.el.div(
                rx.el.div("SYNTHETIC ORBITAL OBJECT", class_name="oi-stamp", style={"display": "inline-block",
                                                                                    "margin_bottom": "10px"}),
                rx.el.dl(
                    kv("Object type", "type"), kv("Orbit regime", "regime"), kv("Altitude", "alt"),
                    kv("Inclination", "inc"), kv("RAAN", "raan"), kv("Sub-point lat", "lat"), kv("Sub-point lon", "lng"),
                    class_name="oi-kv",
                ),
                rx.el.div(info["proximity_text"], class_name="oi-note", style={"margin_top": "10px"}),
                rx.el.div(
                    rx.el.div(rx.el.b("Data source: "), "synthetic simulation (orbital/data/synthetic_globe.py)"),
                    rx.el.div(rx.el.b("Observation time: "), "none, simulation epoch T0"),
                    rx.el.div(rx.el.b("Related Airbus encounter: "),
                              "none. Synthetic objects are not linked to challenge events."),
                    class_name="oi-note", style={"margin_top": "8px", "display": "flex", "flex_direction": "column",
                                                 "gap": "3px"},
                ),
                rx.el.div(
                    rx.el.button(rx.icon("crosshair", size=14), "Focus on object", class_name="oi-btn",
                                 on_click=AppState.focus_object, data_testid="focus-object"),
                    rx.el.button(rx.icon("globe", size=14), "Global view", class_name="oi-btn ghost",
                                 on_click=AppState.reset_view, data_testid="reset-view"),
                    rx.cond(info["partner"] != "",
                            rx.el.button(rx.icon("git-compare-arrows", size=14), "Inspect partner", class_name="oi-btn ghost",
                                         on_click=AppState.select_partner),
                            rx.fragment()),
                    rx.el.button(rx.icon("link-2-off", size=14), "Related encounter", class_name="oi-btn ghost",
                                 disabled=True, title="No verifiable link between synthetic objects and Airbus encounters"),
                    style={"display": "flex", "flex_wrap": "wrap", "gap": "8px", "margin_top": "12px"},
                ),
                class_name="oi-panel-body",
            ),
            class_name="oi-objpanel",
            data_testid="object-panel",
        ),
        rx.fragment(),
    )


def controls(show_filters: bool) -> rx.Component:
    buttons = [
        rx.el.button(rx.icon("house", size=14), "Global view", class_name="oi-btn", on_click=AppState.reset_view,
                     data_testid="globe-home"),
        rx.el.button(rx.icon("rotate-3d", size=14), rx.cond(AppState.auto_rotate, "Rotation on", "Rotation off"),
                     class_name="oi-btn ghost", on_click=AppState.toggle_rotate),
        rx.el.button(rx.icon("orbit", size=14), rx.cond(AppState.show_orbits, "Orbits on", "Orbits off"),
                     class_name="oi-btn ghost", on_click=AppState.toggle_orbits),
    ]
    rows = [rx.el.div(*buttons, style={"display": "flex", "gap": "8px", "flex_wrap": "wrap"})]
    if show_filters:
        rows += [
            rx.el.div(
                rx.el.span("Type", class_name="oi-group-label"),
                *[chip(label, AppState.globe_types.contains(kind), AppState.toggle_globe_type(kind),
                       data_testid=f"gtype-{kind.replace(' ', '_')}") for kind, _, _, label in TYPE_SWATCH],
                class_name="oi-toolbar",
            ),
            rx.el.div(
                rx.el.span("Proximity", class_name="oi-group-label"),
                *[chip(level, AppState.globe_proximity.contains(level), AppState.toggle_globe_proximity(level),
                       data_testid=f"gprox-{level}") for level in GLOBE_PROXIMITY],
                rx.el.span("Population", class_name="oi-group-label"),
                rx.el.select(
                    *[rx.el.option(f"{n:,} objects", value=str(n)) for n in (1000, 2500, 6000, 12000)],
                    value=AppState.globe_size.to(str), on_change=AppState.set_globe_size, class_name="oi-select",
                    data_testid="globe-size",
                ),
                class_name="oi-toolbar",
            ),
        ]
    return rx.el.div(*rows, class_name="oi-controls" if show_filters else "",
                     style={"display": "flex", "flex_direction": "column", "gap": "8px"})


def globe_panel(show_filters: bool = False, height: str = "600px") -> rx.Component:
    assert set(GLOBE_TYPES) == {t[0] for t in TYPE_SWATCH}
    return rx.el.section(
        rx.el.div(
            rx.el.span("Orbital situational awareness"),
            rx.el.span("synthetic orbital simulation, not Airbus encounter data", class_name="sub"),
            rx.el.div(class_name="oi-spacer"),
            rx.el.span(AppState.globe_count, " objects shown", class_name="mono sub", data_testid="globe-count"),
            class_name="oi-panel-head",
        ),
        rx.el.div(
            orbit_globe(
                objects=AppState.globe_objects, visible_types=AppState.globe_types,
                visible_proximity=AppState.globe_proximity, selected_id=AppState.selected_object,
                focus_nonce=AppState.focus_nonce, reset_nonce=AppState.reset_nonce,
                auto_rotate=AppState.auto_rotate, show_orbits=AppState.show_orbits,
                on_select=AppState.select_object,
            ),
            rx.el.div(rx.el.div("SYNTHETIC ORBITAL VISUALISATION", class_name="oi-stamp", data_testid="globe-stamp"),
                      class_name="oi-overlay", style={"top": "14px", "left": "14px"}),
            rx.el.div(controls(show_filters), class_name="oi-overlay",
                      style={"top": "52px", "left": "14px", "max_width": "560px"}),
            rx.el.div(legend(), class_name="oi-overlay", style={"bottom": "14px", "left": "14px"}),
            rx.el.div("Drag to rotate · scroll to zoom · right-drag to pan · click a marker",
                      class_name="oi-overlay oi-note", style={"bottom": "14px", "right": "16px"}),
            object_panel(compact=not show_filters),
            class_name="oi-globe-stage",
            style={"min_height": height},
        ),
        class_name="oi-panel oi-globe-panel",
    )
