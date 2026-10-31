"""Reflex wrapper for the illustrative encounter view in encounter_scene.jsx."""

from __future__ import annotations

import reflex as rx
from reflex.components.component import NoSSRComponent

_SCENE_JS = rx.asset("encounter_scene.jsx", shared=True)


class EncounterScene(NoSSRComponent):
    library = f"$/public{_SCENE_JS}"
    tag = "EncounterScene"

    points: rx.Var[list[dict]]
    h_axis: rx.Var[str]
    v_axis: rx.Var[str]
    dropped_axis: rx.Var[str]
    predicted_miss_km: rx.Var[float | None]
    predicted_miss_label: rx.Var[str]
    separation_label: rx.Var[str]
    countdown: rx.Var[str]
    selected_id: rx.Var[str]
    reset_nonce: rx.Var[int]

    on_select: rx.EventHandler[rx.event.passthrough_event_spec(str)]


encounter_scene = EncounterScene.create
