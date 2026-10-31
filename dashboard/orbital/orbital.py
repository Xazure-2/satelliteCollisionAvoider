"""ORBITAL INTELLIGENCE — Reflex app entry point. Run from dashboard/:  reflex run"""

import reflex as rx

from orbital.pages.analysis import reports, validation
from orbital.pages.encounter import encounter
from orbital.pages.operations import collision_risk, global_tracking, mission_control

app = rx.App(
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
        "&family=JetBrains+Mono:wght@400;500;600&display=swap",
        "/orbital.css",
    ],
)

for route, page, title in [
    ("/", mission_control, "Mission Control"),
    ("/tracking", global_tracking, "Global Tracking"),
    ("/risk", collision_risk, "Collision Risk"),
    ("/encounter", encounter, "Encounter Analysis"),
    ("/validation", validation, "Model Validation"),
    ("/reports", reports, "Reports"),
]:
    app.add_page(page, route=route, title=f"{title} · Orbital Intelligence")
