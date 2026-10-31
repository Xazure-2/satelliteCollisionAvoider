"""Application shell: collapsible sidebar, top bar and page frame."""

from __future__ import annotations

import reflex as rx

from orbital.components.widgets import error_block
from orbital.state import AppState

NAV = [
    ("/", "Mission Control", "layout-dashboard"),
    ("/tracking", "Global Tracking", "globe"),
    ("/risk", "Collision Risk", "triangle-alert"),
    ("/encounter", "Encounter Analysis", "crosshair"),
    ("/validation", "Model Validation", "shield-check"),
    ("/reports", "Reports", "file-text"),
]


def sidebar(active: str) -> rx.Component:
    def item(route, label, icon):
        return rx.el.a(
            rx.icon(icon, size=18),
            rx.el.span(label, class_name="oi-hide"),
            href=route,
            class_name="oi-nav-item active" if route == active else "oi-nav-item",
            title=label,
            data_testid=f"nav-{route.strip('/') or 'home'}",
        )

    return rx.el.aside(
        rx.el.div(
            rx.el.div(rx.icon("orbit", size=18), class_name="oi-brand-mark"),
            rx.el.div("ORBITAL", rx.el.br(), rx.el.span("INTELLIGENCE"), class_name="oi-brand-text oi-hide"),
            class_name="oi-brand",
        ),
        rx.el.nav(
            rx.el.div("OPERATIONS", class_name="oi-nav-label oi-hide"),
            *[item(*n) for n in NAV[:4]],
            rx.el.div("ASSURANCE", class_name="oi-nav-label oi-hide"),
            *[item(*n) for n in NAV[4:]],
            class_name="oi-nav",
        ),
        rx.el.button(
            rx.cond(AppState.sidebar_open, rx.icon("chevrons-left", size=16), rx.icon("chevrons-right", size=16)),
            rx.el.span("Collapse", class_name="oi-hide"),
            class_name="oi-collapse",
            on_click=AppState.toggle_sidebar,
            data_testid="collapse",
        ),
        class_name=rx.cond(AppState.sidebar_open, "oi-sidebar", "oi-sidebar collapsed"),
    )


def source_badge() -> rx.Component:
    return rx.cond(
        AppState.has_data,
        rx.el.span(rx.el.span(class_name="oi-dot"), "REAL MODEL DATA", class_name="oi-badge real",
                   data_testid="badge-real"),
        rx.el.span(rx.el.span(class_name="oi-dot"), "NO DATA", class_name="oi-badge error"),
    )


def topbar(title: str) -> rx.Component:
    return rx.el.header(
        rx.el.div(
            rx.el.div("ORBITAL INTELLIGENCE", class_name="oi-title"),
            rx.el.div("AI-Powered Satellite Collision Early-Warning System", class_name="oi-subtitle"),
        ),
        rx.el.div(rx.el.div("Current page", class_name="oi-meta"), rx.el.div(title, class_name="oi-crumb"),
                  style={"padding_left": "18px", "border_left": "1px solid var(--border)"}),
        rx.el.div(class_name="oi-spacer"),
        source_badge(),
        rx.el.div(
            rx.el.span("Model ", rx.el.b(AppState.model_version)),
            rx.el.span("Data loaded ", rx.el.b(AppState.loaded_at)),
            class_name="oi-meta",
        ),
        rx.el.button(rx.icon("refresh-cw", size=14), class_name="oi-btn ghost", on_click=AppState.reload,
                     title="Reload data from source"),
        class_name="oi-topbar",
    )


def shell(active: str, title: str, *content: rx.Component, needs_data: bool = True) -> rx.Component:
    body = rx.cond(AppState.has_data, rx.fragment(*content), error_block()) if needs_data else rx.fragment(*content)
    return rx.el.div(
        sidebar(active),
        rx.el.main(
            topbar(title),
            rx.el.div(body, class_name="oi-content"),
            class_name="oi-main",
        ),
        class_name="oi-shell",
    )
