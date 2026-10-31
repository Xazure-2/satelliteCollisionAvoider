"""Colours, labels and small HTML components shared by the dashboard."""

from __future__ import annotations

import html

TIER_COLORS = {
    "critical": "#D64545",
    "needs_review": "#8E6CC9",
    "elevated": "#E39B2D",
    "safe": "#3A9D6A",
}

# Colour is always paired with a text label (and an icon) for accessibility.
TIER_LABELS = {
    "critical": "🔴 Critical",
    "needs_review": "🟣 Needs review",
    "elevated": "🟠 Elevated",
    "safe": "🟢 Safe",
}

TIER_TITLES = {
    "critical": "Critical — Maneuver Required",
    "needs_review": "Needs Review",
    "elevated": "Elevated",
    "safe": "Safe",
}

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
.sca-card {
  border-radius: 10px; padding: 0.8rem 1rem; height: 100%;
  border: 1px solid rgba(128,128,128,0.25); border-left-width: 6px;
  background: rgba(128,128,128,0.06);
}
.sca-card .label {font-size: 0.82rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.04em;}
.sca-card .value {font-size: 2.1rem; font-weight: 700; line-height: 1.15;}
.sca-card .sub {font-size: 0.8rem; opacity: 0.75;}
.sca-card {margin-bottom: 0.6rem;}
.sca-banner {
  border-radius: 8px; padding: 0.6rem 0.9rem; margin-bottom: 0.8rem;
  border: 1px dashed currentColor; font-size: 0.92rem;
}
.sca-banner.synthetic {color: #B7791F; background: rgba(227,155,45,0.10);}
.sca-banner.real {color: #2F855A; background: rgba(58,157,106,0.10); border-style: solid;}
.sca-banner.replay {color: #2B6CB0; background: rgba(66,153,225,0.10);}
.sca-banner b {font-weight: 700;}
</style>
"""


def card(label: str, value, color: str, sub: str = "") -> str:
    return (
        f'<div class="sca-card" style="border-left-color:{color}">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div>'
        f'<div class="sub">{html.escape(sub)}</div></div>'
    )


def banner(kind: str, text_html: str) -> str:
    return f'<div class="sca-banner {kind}">{text_html}</div>'
