"""Application state: connects UI events to the data layer.

Datasets are cached at module level (backend only); the browser only receives
the small, already-formatted lists and figures exposed as computed vars.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import reflex as rx

from orbital.components import charts
from orbital.data import adapters, analytics, encounter_view
from orbital.data.adapters import DataUnavailable, Dataset
from orbital.data.contract import HELD_OUT_SPLITS
from orbital.data.synthetic_globe import make_population

_DATASETS: dict[str, Dataset] = {}
_ERRORS: dict[str, DataUnavailable] = {}
_POPULATIONS: dict[int, list[dict]] = {}

TIER_KEYS = ["critical", "elevated", "safe", "needs_review"]
WINDOW_RANGES = {"all": (2.0, 7.0), "2-3": (2.0, 3.0), "3-5": (3.0, 5.0), "5-7": (5.0, 7.0)}
GLOBE_TYPES = ["PAYLOAD", "DEBRIS", "ROCKET BODY", "UNKNOWN"]
GLOBE_PROXIMITY = ["clear", "near", "close"]


def _dataset(key: str, version: int) -> Dataset | None:
    cache_key = f"{key}:{version}"
    if cache_key in _DATASETS:
        return _DATASETS[cache_key]
    if cache_key in _ERRORS:
        return None
    try:
        _DATASETS[cache_key] = adapters.load(key)
        return _DATASETS[cache_key]
    except DataUnavailable as exc:
        _ERRORS[cache_key] = exc
        return None


def _error(key: str, version: int) -> DataUnavailable | None:
    _dataset(key, version)
    return _ERRORS.get(f"{key}:{version}")


def _population(size: int) -> list[dict]:
    if size not in _POPULATIONS:
        _POPULATIONS[size] = make_population(size)
    return _POPULATIONS[size]


def _num(value) -> float | None:
    return float(value) if value is not None and isinstance(value, (int, float)) and math.isfinite(value) else None


class AppState(rx.State):
    source_key: str = "synthetic_a"
    data_version: int = 0
    sidebar_open: bool = True

    # Encounter filters (queue, cards, windows, charts all share them).
    tier_filter: str = "all"
    type_filter: list[str] = []
    window_filter: str = "all"
    search: str = ""
    scope: str = "test"
    sort_key: str = "priority"
    sort_desc: bool = False
    selected_key: str = ""

    # Encounter Analysis workspace.
    view_mode: str = "illustrative"
    selected_cdm: str = ""
    scene_reset: int = 0
    geometry_reset: int = 0

    # Globe (synthetic orbital population).
    globe_size: int = 2500
    globe_types: list[str] = []
    globe_proximity: list[str] = []
    selected_object: str = ""
    focus_nonce: int = 0
    reset_nonce: int = 0
    auto_rotate: bool = True
    show_orbits: bool = True

    # ---------------------------------------------------------------- dataset
    def _ds(self) -> Dataset | None:
        return _dataset(self.source_key, self.data_version)

    @rx.var
    def has_data(self) -> bool:
        return _dataset(self.source_key, self.data_version) is not None

    @rx.var
    def error_title(self) -> str:
        err = _error(self.source_key, self.data_version)
        return err.title if err else ""

    @rx.var
    def error_detail(self) -> str:
        err = _error(self.source_key, self.data_version)
        return err.detail if err else ""

    @rx.var
    def is_synthetic(self) -> bool:
        ds = _dataset(self.source_key, self.data_version)
        return bool(ds and ds.synthetic)

    @rx.var
    def source_label(self) -> str:
        return adapters.SOURCE_LABELS.get(self.source_key, self.source_key)

    @rx.var
    def model_version(self) -> str:
        ds = _dataset(self.source_key, self.data_version)
        return ds.model_version if ds else "—"

    @rx.var
    def loaded_at(self) -> str:
        ds = _dataset(self.source_key, self.data_version)
        return ds.loaded_at if ds else "not loaded"

    @rx.var
    def data_warnings(self) -> list[str]:
        ds = _dataset(self.source_key, self.data_version)
        return ds.warnings if ds else []

    @rx.var
    def source_files(self) -> list[str]:
        ds = _dataset(self.source_key, self.data_version)
        if not ds:
            return []
        return ds.files or ["Generated in memory by orbital/data/synthetic_encounters.py"]

    @rx.var
    def source_options(self) -> list[list[str]]:
        return [[k, v] for k, v in adapters.SOURCE_LABELS.items()]

    @rx.event
    def set_source(self, key: str):
        self.source_key = key
        self.selected_key = ""
        self.tier_filter = "all"
        self.type_filter = []

    @rx.event
    def reload(self):
        _DATASETS.clear()
        _ERRORS.clear()
        self.data_version += 1

    @rx.event
    def toggle_sidebar(self):
        self.sidebar_open = not self.sidebar_open

    # ---------------------------------------------------------------- filters
    def _filters(self, with_tier: bool = True) -> analytics.Filters:
        lo, hi = WINDOW_RANGES.get(self.window_filter, (2.0, 7.0))
        tiers = (self.tier_filter,) if with_tier and self.tier_filter != "all" else ()
        return analytics.Filters(tiers=tiers, object_types=tuple(self.type_filter), tca_min=lo, tca_max=hi,
                                 search=self.search, scope=self.scope)

    def _filtered(self, with_tier: bool = True) -> pd.DataFrame:
        ds = self._ds()
        if ds is None:
            return pd.DataFrame()
        return analytics.apply_filters(ds.events, self._filters(with_tier))

    @rx.event
    def set_tier_filter(self, tier: str):
        self.tier_filter = "all" if self.tier_filter == tier else tier

    @rx.event
    def toggle_type(self, kind: str):
        self.type_filter = [t for t in self.type_filter if t != kind] if kind in self.type_filter else self.type_filter + [kind]

    @rx.event
    def set_window(self, key: str):
        self.window_filter = key

    @rx.event
    def set_search(self, text: str):
        self.search = text

    @rx.event
    def set_scope(self, scope: str):
        self.scope = scope
        self.selected_key = ""

    @rx.event
    def set_sort(self, key: str):
        if self.sort_key == key:
            self.sort_desc = not self.sort_desc
        else:
            self.sort_key, self.sort_desc = key, key in ("pc",)

    @rx.event
    def clear_filters(self):
        self.tier_filter, self.type_filter, self.window_filter, self.search = "all", [], "all", ""

    @rx.var
    def object_types(self) -> list[str]:
        ds = _dataset(self.source_key, self.data_version)
        return sorted(ds.events["c_object_type"].unique().tolist()) if ds else []

    @rx.var
    def kpis(self) -> dict[str, int]:
        _ = (self.source_key, self.data_version, self.type_filter, self.window_filter, self.search, self.scope)
        df = self._filtered(with_tier=False)
        if df.empty:
            return {k: 0 for k in TIER_KEYS + ["total"]}
        return analytics.tier_counts(df)

    @rx.var
    def transitions(self) -> dict[str, int]:
        _ = (self.source_key, self.data_version, self.type_filter, self.window_filter, self.search, self.scope)
        df = self._filtered(with_tier=False)
        return analytics.transition_counts(df) if not df.empty else {}

    @rx.var
    def windows(self) -> list[dict[str, str]]:
        _ = (self.source_key, self.data_version, self.type_filter, self.search, self.scope)
        ds = self._ds()
        if ds is None:
            return []
        df = analytics.apply_filters(ds.events, analytics.Filters(object_types=tuple(self.type_filter),
                                                                  search=self.search, scope=self.scope))
        rows = analytics.warning_windows(df)
        peak = max([r["total"] for r in rows] + [1])
        out = []
        for r, key in zip(rows, ["2-3", "3-5", "5-7"]):
            out.append({k: str(v) for k, v in r.items()} | {
                "key": key,
                **{f"{t}_w": f"{100 * r[t] / peak:.1f}%" for t in ("critical", "elevated", "review", "safe")},
            })
        return out

    @rx.var
    def window_figure(self) -> go.Figure:
        rows = [{k: (int(v) if k not in ("label", "key") and not k.endswith("_w") else v) for k, v in r.items()}
                for r in self.windows]
        return charts.window_chart(rows) if rows else charts.empty("Warning windows", "No data")

    @rx.var
    def queue(self) -> list[dict[str, str]]:
        _ = (self.source_key, self.data_version, self.tier_filter, self.type_filter, self.window_filter,
             self.search, self.scope, self.sort_key, self.sort_desc)
        df = self._filtered()
        if df.empty:
            return []
        return analytics.queue_rows(analytics.sort_queue(df, self.sort_key, self.sort_desc))

    @rx.var
    def queue_top(self) -> list[dict[str, str]]:
        return self.queue[:12]

    @rx.var
    def queue_count(self) -> int:
        return len(self.queue)

    @rx.var
    def filters_active(self) -> bool:
        return self.tier_filter != "all" or bool(self.type_filter) or self.window_filter != "all" or bool(self.search)

    # ---------------------------------------------------------------- encounter detail
    @rx.event
    def open_event(self, key: str):
        self.selected_key = key
        ds = self._ds()
        if ds is not None:
            source, _, eid = key.partition("|")
            split = ds.events.loc[(ds.events["source"] == source) & (ds.events["event_id"].astype(str) == eid), "split"]
            if len(split):
                self.scope = "held_out" if split.iloc[0] in HELD_OUT_SPLITS else "test"
                self.tier_filter, self.window_filter, self.search, self.type_filter = "all", "all", "", []
        return rx.redirect("/encounter")

    @rx.event
    def select_event(self, key: str):
        self.selected_key = key

    def _event(self) -> tuple[dict | None, pd.DataFrame]:
        ds = self._ds()
        if ds is None:
            return None, pd.DataFrame()
        key = self.selected_key
        if not key:
            df = self._filtered()
            if df.empty:
                return None, pd.DataFrame()
            first = analytics.sort_queue(df, "priority", False).iloc[0]
            key = f"{first['source']}|{first['event_id']}"
        source, _, eid = key.partition("|")
        match = ds.events[(ds.events["source"] == source) & (ds.events["event_id"].astype(str) == eid)]
        if match.empty:
            return None, pd.DataFrame()
        ev = match.iloc[0].to_dict()
        hist = ds.history[(ds.history["source"] == source) & (ds.history["event_id"].astype(str) == eid)]
        return ev, hist

    @rx.var
    def event_options(self) -> list[list[str]]:
        return [[r["key"], f"#{r['event_id']} · {r['predicted_label']} · {r['object_type']}"] for r in self.queue[:400]]

    @rx.var
    def current_event_key(self) -> str:
        _ = (self.selected_key, self.queue)
        ev, _h = self._event()
        return f"{ev['source']}|{ev['event_id']}" if ev else ""

    @rx.var
    def event(self) -> dict[str, str]:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        ev, hist = self._event()
        if ev is None:
            return {}
        f = analytics
        kind = f.transition(ev["observed_tier"], ev["predicted_final_tier"])
        basis = ev.get("final_miss_distance_basis")
        held = ev["split"] in HELD_OUT_SPLITS and ev.get("actual_final_tier")
        return {
            "key": f"{ev['source']}|{ev['event_id']}",
            "event_id": str(ev["event_id"]),
            "source": ev["source"],
            "split": ev["split"],
            "object_type": ev["c_object_type"],
            "observed_tier": ev["observed_tier"] or "none",
            "observed_label": f.tier_label(ev["observed_tier"]),
            "predicted_tier": ev["predicted_final_tier"],
            "predicted_label": f.tier_label(ev["predicted_final_tier"]),
            "transition": kind,
            "countdown": f.fmt_countdown(ev["cutoff_time_to_tca_days"]),
            "cutoff_days": f"{ev['cutoff_time_to_tca_days']:.2f} d",
            "obs_pc": f.fmt_pc(ev["observed_log10_pc"]),
            "obs_log": f.fmt_log(ev["observed_log10_pc"]),
            "obs_miss": f.fmt_m(ev["observed_miss_distance_m"]),
            "pred_pc": f.fmt_pc(ev["predicted_final_log10_pc"], ev["predicted_final_at_floor"]),
            "pred_log": f.fmt_log(ev["predicted_final_log10_pc"]),
            "pred_miss": f.fmt_m(ev["predicted_final_miss_distance_m"]),
            "miss_basis": {"predicted": "model prediction", "observed_at_cutoff": "observed at cutoff (not predicted)"}.get(
                basis, "not provided"),
            "uncertainty": f.uncertainty_text(ev),
            "p_crit": f.fmt_prob(_num(ev.get("p_critical"))),
            "p_elev": f.fmt_prob(_num(ev.get("p_elevated"))),
            "p_safe": f.fmt_prob(_num(ev.get("p_safe"))),
            "n_cdms": str(int(ev["n_cdms_used"])) if _num(ev.get("n_cdms_used")) is not None else "—",
            "speed": f"{hist['relative_speed_mps'].dropna().iloc[-1]:,.0f} m/s" if not hist.empty and hist['relative_speed_mps'].notna().any() else "—",
            "reasons": " · ".join(ev["alert_reasons"]) or "No alert trigger reported",
            "flags": ", ".join(ev["quality_flags"]) or "none",
            "action": ev["recommended_action"] or "—",
            "model_version": ev["model_version"],
            "actual_label": f.tier_label(ev.get("actual_final_tier")) if held else "",
            "actual_tier": (ev.get("actual_final_tier") or "none") if held else "none",
            "actual_pc": f.fmt_pc(_num(ev.get("actual_final_log10_pc"))) if held else "",
            "explanation": self._explain(ev, kind),
            "obs_miss_basis": "latest eligible CDM",
            "miss_tag": "PREDICTED" if basis == "predicted" else ("OBSERVED" if basis == "observed_at_cutoff" else "N/A"),
            "miss_tag_class": "predicted" if basis == "predicted" else ("not" if basis == "observed_at_cutoff" else "observed"),
            "event_flags": str(len(ev["quality_flags"])),
            "used_flags": str(int(sum(1 for f in hist.loc[hist["used_for_prediction"], "quality_flags"] if f)))
            if not hist.empty else "0",
            "floor": "yes" if ev["predicted_final_at_floor"] else "no",
        }

    @staticmethod
    def _explain(ev: dict, kind: str) -> str:
        obs, pred = analytics.tier_label(ev["observed_tier"]), analytics.tier_label(ev["predicted_final_tier"])
        if ev["predicted_final_tier"] == "needs_review":
            return "No usable model output for this encounter. It needs a data-quality review; it is not treated as safe."
        if kind == "escalation":
            text = (f"ESCALATION: the latest eligible CDM is observed {obs}, but the model predicts the final CDM will be "
                    f"{pred}. This is the early-warning case: it flags the encounter before the observed risk rises.")
        elif kind == "de-escalation":
            text = (f"DE-ESCALATION: observed {obs} at the cutoff, predicted {pred} at the final CDM. Keep monitoring; "
                    "a lower prediction is not a guarantee.")
        elif kind == "unchanged":
            text = f"UNCHANGED: observed and predicted final tier are both {pred}."
        else:
            text = f"Predicted final tier {pred}; the observed tier at the cutoff is not available."
        if ev["predicted_final_tier"] == "critical":
            text += " A predicted Critical tier requests human manoeuvre-planning review; it never commands a manoeuvre."
        return text + " The model does not provide causal explanations; see the inputs and uncertainty shown."

    @rx.var
    def risk_figure(self) -> go.Figure:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        ev, hist = self._event()
        return charts.risk_evolution(hist, ev, 280) if ev else charts.empty("Risk evolution", "Select an encounter")

    @rx.var
    def miss_figure(self) -> go.Figure:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        ev, hist = self._event()
        return charts.miss_evolution(hist, ev, 280) if ev else charts.empty("Miss distance", "Select an encounter")

    @rx.var
    def geometry_figure(self) -> go.Figure:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue, self.geometry_reset)
        ev, hist = self._event()
        return (charts.encounter_geometry(hist, ev, self.geometry_reset, None) if ev
                else charts.empty("", "Select an encounter", 560))

    # ---------------------------------------------------------------- investigation decision support
    def _decision(self) -> tuple[dict, dict, dict] | None:
        ev, hist = self._event()
        if ev is None:
            return None
        used = hist[hist["used_for_prediction"]] if not hist.empty else hist
        flagged = int(sum(1 for f in used["quality_flags"] if f)) if not used.empty else 0
        rel = analytics.reliability(ev, flagged)
        checks = analytics.threshold_checks(ev)
        return checks, rel, analytics.next_step(ev, rel, checks["consistent"])

    @rx.var
    def threshold_rows(self) -> list[dict[str, str]]:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        d = self._decision()
        return d[0]["rows"] if d else []

    @rx.var
    def decision(self) -> dict[str, str]:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        d = self._decision()
        if d is None:
            return {}
        checks, rel, step = d
        ev, _h = self._event()
        kind = analytics.transition(ev["observed_tier"], ev["predicted_final_tier"])
        return {"summary": checks["summary"], "consistent": "yes" if checks["consistent"] else "no",
                "basis_note": checks["basis_note"],
                "escalation": {"escalation": "Escalation expected", "de-escalation": "De-escalation expected",
                               "unchanged": "No tier change expected"}.get(kind, "Escalation unknown (no observed tier)"),
                "escalation_kind": kind,
                "reliability": rel["level"], "probability": rel["probability"],
                "step_key": step["key"], "step_label": step["label"]}

    @rx.var
    def reliability_notes(self) -> list[str]:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        d = self._decision()
        return d[1]["notes"] if d else []

    @rx.var
    def next_step_reasons(self) -> list[str]:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        d = self._decision()
        return d[2]["reasons"] if d else []

    @rx.var
    def event_position(self) -> str:
        keys = [r["key"] for r in self.queue]
        current = self.current_event_key
        return f"{keys.index(current) + 1} of {len(keys)}" if current in keys else "not in current filter"

    def _step_event(self, delta: int):
        keys = [r["key"] for r in self.queue]
        if not keys:
            return
        current = self.current_event_key
        i = keys.index(current) if current in keys else -delta
        self.selected_key = keys[(i + delta) % len(keys)]

    @rx.event
    def next_event(self):
        self._step_event(1)

    @rx.event
    def prev_event(self):
        self._step_event(-1)


    # ---------------------------------------------------------------- encounter visualisation
    @rx.event
    def set_view_mode(self, mode: str):
        self.view_mode = mode

    @rx.event
    def select_cdm(self, row_id: str):
        self.selected_cdm = str(row_id)

    @rx.event
    def select_cdm_point(self, points: list[dict]):
        """Click on the measured 3-D view: map the clicked point back to its CDM."""
        if not points:
            return
        _ev, hist = self._event()
        row = encounter_view.nearest_cdm(hist, points[0])
        if row is not None:
            self.selected_cdm = str(row)

    @rx.event
    def step_cdm(self, delta: int):
        """Step the inspector through CDMs in time order (works even when markers overlap)."""
        _ev, hist = self._event()
        if hist.empty:
            return
        ids = encounter_view.ordered_history(hist)["row_id"].astype(str).tolist()
        current = self.cdm_detail.get("row_id", ids[-1])
        i = ids.index(current) if current in ids else len(ids) - 1
        self.selected_cdm = ids[(i + delta) % len(ids)]

    @rx.event
    def reset_encounter_view(self):
        if self.view_mode == "measured":
            self.geometry_reset += 1
        else:
            self.scene_reset += 1

    def _scene(self) -> dict:
        ev, hist = self._event()
        return encounter_view.scene(ev, hist) if ev else encounter_view.scene({}, pd.DataFrame())

    @rx.var
    def scene_points(self) -> list[dict]:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        return self._scene()["points"]

    @rx.var
    def scene_meta(self) -> dict[str, str]:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        sc = self._scene()
        return {"available": "yes" if sc["available"] else "no", "h_axis": sc["h_axis"], "v_axis": sc["v_axis"],
                "dropped_axis": sc["dropped_axis"], "plane_note": sc["plane_note"],
                "separation": sc["separation_label"], "ring_label": sc["predicted_miss_label"]}

    @rx.var
    def scene_ring_km(self) -> float | None:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        return self._scene()["predicted_miss_km"]

    @rx.var
    def cdm_detail(self) -> dict[str, str]:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue, self.selected_cdm)
        ev, hist = self._event()
        if ev is None or hist.empty:
            return {}
        h = encounter_view.ordered_history(hist)
        latest = encounter_view.latest_eligible_row(ev, h)
        pick = h[h["row_id"].astype(str) == self.selected_cdm]
        if pick.empty:
            pick = h[h["row_id"] == latest] if latest is not None else h.tail(1)
        return encounter_view.cdm_details(pick.iloc[0].to_dict(), latest)

    @rx.var
    def selected_cdm_id(self) -> str:
        return self.cdm_detail.get("row_id", "")

    @rx.var
    def latest_cdms(self) -> list[dict[str, str]]:
        """The six most recent CDMs, oldest first (compact evidence table)."""
        return self.history_rows[-6:]

    @rx.var
    def uncertainty_probs(self) -> list[dict[str, str]]:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        ev, _h = self._event()
        return encounter_view.uncertainty(ev)["probs"] if ev else []

    @rx.var
    def uncertainty_interval(self) -> str:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        ev, _h = self._event()
        return encounter_view.uncertainty(ev)["interval"] if ev else ""

    @rx.var
    def history_rows(self) -> list[dict[str, str]]:
        _ = (self.source_key, self.data_version, self.selected_key, self.queue)
        ev, hist = self._event()
        if ev is None or hist.empty:
            return []
        rows = []
        for r in hist.sort_values("time_to_tca_days", ascending=False).to_dict("records"):
            rows.append({
                "t": f"{r['time_to_tca_days']:.2f}", "log": analytics.fmt_log(_num(r["observed_log10_pc"])),
                "pc": analytics.fmt_pc(_num(r["observed_log10_pc"])), "miss": analytics.fmt_m(r["miss_distance_m"]),
                "speed": f"{r['relative_speed_mps']:,.0f}" if _num(r["relative_speed_mps"]) is not None else "—",
                "r": f"{r['relative_position_r_m']:,.0f}" if _num(r["relative_position_r_m"]) is not None else "—",
                "tt": f"{r['relative_position_t_m']:,.0f}" if _num(r["relative_position_t_m"]) is not None else "—",
                "n": f"{r['relative_position_n_m']:,.0f}" if _num(r["relative_position_n_m"]) is not None else "—",
                "used": "input" if r["used_for_prediction"] else "after cutoff",
                "flags": ", ".join(r["quality_flags"]) or "",
            })
        return rows

    # ---------------------------------------------------------------- validation
    @rx.var
    def validation(self) -> dict[str, str]:
        ds = _dataset(self.source_key, self.data_version)
        if ds is None:
            return {}
        m = analytics.validation_metrics(ds.events)
        pct = lambda v: f"{v:.1%}" if v is not None else "undefined"  # noqa: E731
        return {"n": str(m["n"]), "splits": m["splits"], "accuracy": pct(m["accuracy"]), "recall": pct(m["recall"]),
                "precision": pct(m["precision"]), "fn": str(m["fn"]), "fp": str(m["fp"]), "tp": str(m["tp"]),
                "critical_actual": str(m["critical_actual"])}

    @rx.var
    def matrix_rows(self) -> list[list[str]]:
        ds = _dataset(self.source_key, self.data_version)
        if ds is None:
            return []
        m = analytics.validation_metrics(ds.events)
        rows = []
        for actual, counts in zip(m["labels_true"], m["matrix"]):
            cells = []
            for pred, n in zip(m["labels_pred"], counts):
                cls = "diag" if pred == actual else ("miss" if actual == "critical" and n else "")
                cells.append(f"{n}|{cls}")
            rows.append([analytics.tier_label(actual)] + cells)
        return rows

    @rx.var
    def missed_critical(self) -> list[dict[str, str]]:
        ds = _dataset(self.source_key, self.data_version)
        if ds is None:
            return []
        missed = analytics.validation_metrics(ds.events)["missed"]
        return analytics.queue_rows(missed.head(12)) if len(missed) else []

    # ---------------------------------------------------------------- globe
    @rx.var
    def globe_objects(self) -> list[dict]:
        return _population(self.globe_size)

    @rx.var
    def globe_count(self) -> int:
        objs = _population(self.globe_size)
        types, prox = set(self.globe_types), set(self.globe_proximity)
        return sum(1 for o in objs if (not types or o["type"] in types) and (not prox or o["proximity"] in prox))

    @rx.var
    def globe_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in _population(self.globe_size):
            counts[o["type"]] = counts.get(o["type"], 0) + 1
        return counts

    @rx.var
    def object_info(self) -> dict[str, str]:
        if not self.selected_object:
            return {}
        obj = next((o for o in _population(self.globe_size) if o["id"] == self.selected_object), None)
        if obj is None:
            return {}
        return {
            "id": obj["id"], "type": obj["type"], "regime": obj["regime"],
            "alt": f"{obj['alt_km']:,.1f} km", "inc": f"{obj['inc_deg']:.2f}°", "raan": f"{obj['raan_deg']:.2f}°",
            "lat": f"{obj['lat']:.3f}°", "lng": f"{obj['lng']:.3f}°",
            "proximity": obj["proximity"],
            "proximity_text": (f"Simulated separation {obj['sep_km']} km from {obj['partner']}" if obj["partner"]
                               else "No simulated close approach"),
            "partner": obj["partner"] or "",
        }

    @rx.event
    def select_object(self, object_id: str):
        self.selected_object = object_id

    @rx.event
    def focus_object(self):
        if self.selected_object:
            self.focus_nonce += 1

    @rx.event
    def select_partner(self):
        info = self.object_info
        if info.get("partner"):
            self.selected_object = info["partner"]
            self.focus_nonce += 1

    @rx.event
    def reset_view(self):
        self.selected_object = ""
        self.reset_nonce += 1

    @rx.event
    def toggle_globe_type(self, kind: str):
        self.globe_types = [t for t in self.globe_types if t != kind] if kind in self.globe_types else self.globe_types + [kind]

    @rx.event
    def toggle_globe_proximity(self, level: str):
        self.globe_proximity = ([p for p in self.globe_proximity if p != level] if level in self.globe_proximity
                                else self.globe_proximity + [level])

    @rx.event
    def set_globe_size(self, value: str):
        self.globe_size = int(value)
        self.selected_object = ""

    @rx.event
    def toggle_rotate(self):
        self.auto_rotate = not self.auto_rotate

    @rx.event
    def toggle_orbits(self):
        self.show_orbits = not self.show_orbits

    # ---------------------------------------------------------------- export
    @rx.event
    def download_queue(self):
        ds = self._ds()
        if ds is None:
            return rx.toast.error("No dataset loaded.")
        df = analytics.sort_queue(self._filtered(), self.sort_key, self.sort_desc).copy()
        for col in ("alert_reasons", "quality_flags"):
            df[col] = df[col].map(lambda v: "; ".join(v))
        prefix = "SYNTHETIC_" if ds.synthetic else ""
        return rx.download(data=df.to_csv(index=False), filename=f"{prefix}encounter_queue.csv")
