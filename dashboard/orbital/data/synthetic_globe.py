"""SYNTHETIC orbital population for the 3D globe — a simulation, not Airbus data.

The Airbus challenge data has no absolute positions, TLEs or trajectories, so
nothing here is derived from it, and no object is linked to an Airbus
encounter. Objects get simple synthetic circular orbits (altitude,
inclination, right ascension of the ascending node, argument of latitude) at
a single simulation epoch "T0". Positions are computed from those synthetic
elements only.

A few deliberately placed synthetic close pairs demonstrate how proximity
could be highlighted. Their "simulated separation" is part of the simulation
and has nothing to do with Airbus collision probabilities.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0

# (object type, share, altitude band km)
POPULATION = [
    ("PAYLOAD", 0.30, [(400, 600, 0.55), (600, 1200, 0.3), (19000, 23500, 0.07), (35786, 35786, 0.08)]),
    ("DEBRIS", 0.48, [(500, 1000, 0.75), (1000, 2000, 0.25)]),
    ("ROCKET BODY", 0.14, [(450, 1200, 0.7), (1200, 2500, 0.2), (19000, 36000, 0.1)]),
    ("UNKNOWN", 0.08, [(400, 2000, 1.0)]),
]


def _regime(alt_km: float) -> str:
    if alt_km < 2000:
        return "LEO"
    if alt_km < 35000:
        return "MEO"
    return "GEO"


def _latlng(inc_deg: float, raan_deg: float, u_deg: float) -> tuple[float, float]:
    inc, raan, u = np.radians([inc_deg, raan_deg, u_deg])
    x = np.cos(raan) * np.cos(u) - np.sin(raan) * np.sin(u) * np.cos(inc)
    y = np.sin(raan) * np.cos(u) + np.cos(raan) * np.sin(u) * np.cos(inc)
    z = np.sin(u) * np.sin(inc)
    return float(np.degrees(np.arcsin(z))), float(np.degrees(np.arctan2(y, x)))


def make_population(n: int = 2500, seed: int = 4242, n_pairs: int = 14) -> list[dict]:
    rng = np.random.default_rng(seed)
    types = [p[0] for p in POPULATION]
    shares = np.array([p[1] for p in POPULATION])
    objects = []
    for i in range(n):
        kind = types[int(rng.choice(len(types), p=shares / shares.sum()))]
        bands = next(p[2] for p in POPULATION if p[0] == kind)
        lo, hi, _ = bands[int(rng.choice(len(bands), p=np.array([b[2] for b in bands])))]
        alt = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
        if alt > 35000:
            inc = float(rng.uniform(0, 3))
        elif alt > 15000:
            inc = float(rng.normal(55, 3))
        else:
            inc = float(rng.choice([rng.uniform(40, 60), rng.uniform(85, 99), rng.uniform(0, 30)], p=[0.35, 0.5, 0.15]))
        raan, u = float(rng.uniform(0, 360)), float(rng.uniform(0, 360))
        lat, lng = _latlng(inc, raan, u)
        objects.append({
            "id": f"SIM-{i:05d}", "type": kind, "regime": _regime(alt),
            "alt_km": round(alt, 1), "inc_deg": round(inc % 180, 2), "raan_deg": round(raan, 2),
            "u_deg": round(u, 2), "lat": round(lat, 4), "lng": round(lng, 4),
            "proximity": "clear", "sep_km": None, "partner": None,
        })

    # Synthetic close pairs: copy a LEO object's orbit with a tiny phase offset.
    leo = [o for o in objects if o["regime"] == "LEO"]
    for k, base in enumerate(rng.choice(len(leo), size=n_pairs, replace=False)):
        a = leo[int(base)]
        sep_km = float(rng.uniform(0.3, 0.95)) if k % 2 == 0 else float(rng.uniform(1.2, 4.8))
        du = np.degrees(sep_km / (EARTH_RADIUS_KM + a["alt_km"]))
        lat, lng = _latlng(a["inc_deg"], a["raan_deg"], a["u_deg"] + du)
        partner = {
            **a, "id": f"SIM-P{k:03d}", "type": "DEBRIS", "u_deg": round(a["u_deg"] + du, 4),
            "lat": round(lat, 4), "lng": round(lng, 4),
        }
        level = "close" if sep_km < 1 else "near"
        for obj, other in ((a, partner), (partner, a)):
            obj.update(proximity=level, sep_km=round(sep_km, 2), partner=other["id"])
        objects.append(partner)
    return objects
