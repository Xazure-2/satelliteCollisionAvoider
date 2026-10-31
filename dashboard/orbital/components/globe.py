"""Reflex wrapper for the WebGL globe in orbit_globe.jsx."""

from __future__ import annotations

import reflex as rx
from reflex.components.component import NoSSRComponent

_GLOBE_JS = rx.asset("orbit_globe.jsx", shared=True)


class OrbitGlobe(NoSSRComponent):
    library = f"$/public{_GLOBE_JS}"
    lib_dependencies: list[str] = ["react-globe.gl@2.38.0", "three@0.183.2"]
    tag = "OrbitGlobe"

    objects: rx.Var[list[dict]]
    visible_types: rx.Var[list[str]]
    visible_proximity: rx.Var[list[str]]
    selected_id: rx.Var[str]
    focus_nonce: rx.Var[int]
    reset_nonce: rx.Var[int]
    auto_rotate: rx.Var[bool]
    show_orbits: rx.Var[bool]

    on_select: rx.EventHandler[rx.event.passthrough_event_spec(str)]


orbit_globe = OrbitGlobe.create
