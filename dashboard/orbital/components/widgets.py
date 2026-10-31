"""Reusable dashboard widgets. They render values from AppState only."""

from __future__ import annotations

import reflex as rx

from orbital.state import AppState

TIER_ACCENTS = {"critical": "#ef4444", "elevated": "#f59e0b", "safe": "#22c55e", "needs_review": "#a78bfa"}


def panel(title: str, *children, sub=None, actions=None, body_class: str = "oi-panel-body", **props) -> rx.Component:
    head = [rx.el.span(title)]
    if sub is not None:
        head.append(rx.el.span(sub, class_name="sub"))
    head.append(rx.el.div(class_name="oi-spacer"))
    if actions is not None:
        head.append(actions)
    return rx.el.section(
        rx.el.div(*head, class_name="oi-panel-head"),
        rx.el.div(*children, class_name=body_class),
        class_name="oi-panel " + props.pop("class_name", ""),
        **props,
    )


def tier_badge(tier, label) -> rx.Component:
    return rx.el.span(label, class_name="tier tier-" + tier.to(str))


def transition_badge(row) -> rx.Component:
    """Observed → predicted, the core early-warning signal."""
    return rx.el.span(
        tier_badge(row["observed_tier"], row["observed_label"]),
        rx.icon("chevrons-right", size=15, class_name="chev"),
        tier_badge(row["predicted_tier"], row["predicted_label"]),
        class_name="oi-arrow " + row["transition"].to(str),
        title="Current observed → predicted final",
    )


def kpi_card(label: str, key: str, accent: str, foot: str, icon: str) -> rx.Component:
    tier_key = "all" if key == "total" else key
    return rx.el.button(
        rx.el.div(rx.icon(icon, size=13), label, class_name="oi-kpi-label"),
        rx.el.div(AppState.kpis[key], class_name="oi-kpi-value"),
        rx.el.div(foot, class_name="oi-kpi-foot"),
        class_name=rx.cond(AppState.tier_filter == tier_key, "oi-kpi active", "oi-kpi"),
        style={"--accent": accent},
        on_click=AppState.set_tier_filter(tier_key),
        data_testid=f"kpi-{key}",
    )


def kpi_row() -> rx.Component:
    return rx.el.div(
        kpi_card("Encounters evaluated", "total", "#38bdf8", "click to show all", "radar"),
        kpi_card("Predicted critical", "critical", TIER_ACCENTS["critical"], "human manoeuvre-planning review", "octagon-alert"),
        kpi_card("Predicted elevated", "elevated", TIER_ACCENTS["elevated"], "analyst review", "triangle-alert"),
        kpi_card("Predicted safe", "safe", TIER_ACCENTS["safe"], "monitor", "shield-check"),
        kpi_card("Requires review", "needs_review", TIER_ACCENTS["needs_review"], "missing / unusable prediction", "circle-help"),
        class_name="oi-grid-kpi",
    )


def chip(label, on: rx.Var, on_click, **props) -> rx.Component:
    return rx.el.button(label, class_name=rx.cond(on, "oi-chip on", "oi-chip"), on_click=on_click, **props)


def filter_bar(show_scope: bool = True) -> rx.Component:
    windows = [("all", "All"), ("2-3", "2–3 d"), ("3-5", "3–5 d"), ("5-7", "5–7 d")]
    tiers = [("all", "All tiers"), ("critical", "Critical"), ("elevated", "Elevated"), ("safe", "Safe"),
             ("needs_review", "Review")]
    items = [
        rx.el.div(
            rx.icon("search", size=14),
            rx.el.input(placeholder="Event ID", value=AppState.search, on_change=AppState.set_search,
                        class_name="oi-input", data_testid="search"),
            class_name="oi-search",
        ),
        rx.el.span("Tier", class_name="oi-group-label"),
        *[chip(lbl, AppState.tier_filter == k,
               AppState.set_tier_filter(k) if k != "all" else AppState.set_tier_filter("all"),
               data_testid=f"tier-{k}") for k, lbl in tiers],
        rx.el.span("Time to TCA", class_name="oi-group-label"),
        *[chip(lbl, AppState.window_filter == k, AppState.set_window(k), data_testid=f"win-{k}") for k, lbl in windows],
        rx.el.span("Object", class_name="oi-group-label"),
        rx.foreach(AppState.object_types, lambda t: chip(t, AppState.type_filter.contains(t), AppState.toggle_type(t))),
    ]
    if show_scope:
        items += [
            rx.el.span("Population", class_name="oi-group-label"),
            chip("Test set (operational)", AppState.scope == "test", AppState.set_scope("test"), data_testid="scope-test"),
            chip("Held-out training", AppState.scope == "held_out", AppState.set_scope("held_out"), data_testid="scope-held"),
        ]
    items.append(rx.cond(AppState.filters_active,
                         rx.el.button(rx.icon("x", size=13), "Clear", class_name="oi-btn ghost",
                                      on_click=AppState.clear_filters), rx.fragment()))
    return rx.el.div(*items, class_name="oi-toolbar")


def _th(label: str, key: str | None = None) -> rx.Component:
    if key is None:
        return rx.el.th(label)
    arrow = rx.cond(AppState.sort_key == key, rx.cond(AppState.sort_desc, " ↓", " ↑"), "")
    return rx.el.th(label, arrow, class_name=rx.cond(AppState.sort_key == key, "sortable sorted", "sortable"),
                    on_click=AppState.set_sort(key), data_testid=f"sort-{key}")


def queue_table(rows, compact: bool = False, show_actual: bool = False) -> rx.Component:
    def row(r):
        cells = [
            rx.el.td(r["event_id"], class_name="mono"),
            rx.el.td(transition_badge(r)),
            rx.el.td(r["pred_pc"], class_name="mono"),
            rx.el.td(r["countdown"], class_name="mono"),
        ]
        if not compact:
            cells = [
                rx.el.td(r["event_id"], class_name="mono"),
                rx.el.td(r["object_type"]),
                rx.el.td(tier_badge(r["observed_tier"], r["observed_label"])),
                rx.el.td(rx.el.span(rx.icon("chevrons-right", size=15, class_name="chev"),
                                    class_name="oi-arrow " + r["transition"].to(str))),
                rx.el.td(tier_badge(r["predicted_tier"], r["predicted_label"])),
                *([rx.el.td(tier_badge(r["actual_tier"], r["actual_label"]))] if show_actual else []),
                rx.el.td(r["pred_pc"], class_name="mono"),
                rx.el.td(r["obs_miss"], class_name="mono"),
                rx.el.td(r["countdown"], class_name="mono"),
                rx.el.td(r["uncertainty"], class_name="mono muted"),
                rx.el.td(r["action"]),
            ]
        return rx.el.tr(*cells, on_click=AppState.open_event(r["key"]),
                        class_name=rx.cond(r["predicted_tier"] == "critical", "row-critical", ""),
                        data_testid="queue-row")

    if compact:
        head = [_th("Event", "event"), rx.el.th("Observed → Predicted final"), _th("Pred. Pc", "pc"), _th("TCA", "tca")]
    else:
        head = [_th("Event ID", "event"), rx.el.th("Object"), rx.el.th("Current observed"), rx.el.th(""),
                _th("Predicted final", "priority"), *([rx.el.th("Actual final (held-out)")] if show_actual else []),
                _th("Pred. final Pc", "pc"), _th("Obs. miss distance", "miss"),
                _th("Time to TCA", "tca"), rx.el.th("Uncertainty"), rx.el.th("Review status")]
    return rx.cond(
        rows.length() > 0,
        rx.el.div(rx.el.table(rx.el.thead(rx.el.tr(*head)), rx.el.tbody(rx.foreach(rows, row)),
                              class_name="oi-table compact" if compact else "oi-table"),
                  class_name="oi-table-wrap"),
        rx.el.div("No encounters match the current filters.", class_name="oi-empty", data_testid="queue-empty"),
    )


def windows_panel() -> rx.Component:
    def row(w):
        return rx.el.div(
            rx.el.span(w["label"], class_name="mono"),
            rx.el.div(
                rx.el.div(style={"width": w["critical_w"], "background": TIER_ACCENTS["critical"]}),
                rx.el.div(style={"width": w["elevated_w"], "background": TIER_ACCENTS["elevated"]}),
                rx.el.div(style={"width": w["review_w"], "background": TIER_ACCENTS["needs_review"]}),
                rx.el.div(style={"width": w["safe_w"], "background": "rgba(34,197,94,0.55)"}),
                class_name="oi-bar",
            ),
            rx.el.span(
                rx.el.span(w["critical"], style={"color": "#fca5a5"}), " / ",
                rx.el.span(w["elevated"], style={"color": "#fcd34d"}), class_name="mono", style={"text_align": "right"},
            ),
            class_name="oi-window",
            on_click=AppState.set_window(w["key"]),
            style={"cursor": "pointer"},
            data_testid="window-row",
        )

    return panel(
        "Early-warning windows",
        rx.foreach(AppState.windows, row),
        rx.el.div("Critical / Elevated predicted, by time_to_tca at the last eligible CDM (≥ 2 d). "
                  "Relative times only; no calendar dates exist in the data.", class_name="oi-note",
                  style={"margin_top": "8px"}),
        sub="click to filter",
    )


def transitions_panel() -> rx.Component:
    def stat(label, key, color):
        return rx.el.div(rx.el.span(label), rx.el.b(AppState.transitions[key], style={"color": color}), class_name="oi-stat")

    return panel(
        "Observed → predicted",
        stat("Escalations (early warnings)", "escalation", "#fca5a5"),
        stat("Unchanged tier", "unchanged", "#cfe6ff"),
        stat("De-escalations", "de-escalation", "#86efac"),
        stat("Observed tier unavailable", "unknown", "#7d91b6"),
        sub="tier change from last eligible CDM to predicted final",
    )


def error_block() -> rx.Component:
    return rx.el.div(
        rx.el.h3(rx.icon("circle-x", size=16), " ", AppState.error_title),
        rx.el.p(AppState.error_detail, class_name="oi-note"),
        rx.el.div(
            rx.el.button(rx.icon("refresh-cw", size=14), "Reload", class_name="oi-btn", on_click=AppState.reload),
            rx.el.button(rx.icon("flask-conical", size=14), "Show synthetic demo A instead", class_name="oi-btn ghost",
                         on_click=AppState.set_source("synthetic_a")),
            style={"display": "flex", "gap": "10px", "margin_top": "10px"},
        ),
        class_name="oi-error",
        data_testid="data-error",
    )
