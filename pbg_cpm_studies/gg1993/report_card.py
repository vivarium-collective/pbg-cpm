"""Render a per-study *validation report card* — a compact PASS/FAIL panel that
states the paper behaviour, the acceptance criterion, and what the reproduced
run actually measured. Cards are written into ``studies/<slug>/charts/`` as
``00_report_card.png`` (+ meta.json) so they render first in the investigation
report and the dashboard chart panel.

Driven entirely by :mod:`pbg_cpm_studies.gg1993.validate` — the PASS/FAIL badge
is the test result, never a hand-set verdict.
"""

from __future__ import annotations

import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # selectable text in the SVG
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from . import run as runmod
from .validate import validate_all

STU_DIR = os.path.join(runmod.WS_ROOT, "workspace", "studies")

_PASS = "#0d9488"   # teal (validated colourblind-safe palette)
_FAIL = "#e11d48"   # red
_INK = "#1f2937"
_MUTED = "#6b7280"
_CARD = "#ffffff"
_LINE = "#e5e7eb"


def _wrap(s, w):
    return "\n".join(textwrap.wrap(s, w))


def render_card_png(slug, r, title):
    ok = r["passed"]
    accent = _PASS if ok else _FAIL
    fig = plt.figure(figsize=(8.6, 3.5), dpi=150)
    fig.patch.set_facecolor(_CARD)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # accent bar + card border
    ax.add_patch(FancyBboxPatch((0.012, 0.03), 0.976, 0.94,
                 boxstyle="round,pad=0.006,rounding_size=0.02",
                 linewidth=1.2, edgecolor=_LINE, facecolor=_CARD, zorder=1))
    ax.add_patch(plt.Rectangle((0.012, 0.03), 0.012, 0.94, color=accent, zorder=2))

    # header: title + badge
    ax.text(0.05, 0.86, title, fontsize=15, fontweight="bold", color=_INK, va="center")
    badge = "PASS" if ok else "FAIL"
    ax.add_patch(FancyBboxPatch((0.80, 0.80), 0.155, 0.12,
                 boxstyle="round,pad=0.004,rounding_size=0.03",
                 linewidth=0, facecolor=accent, zorder=3))
    ax.text(0.8775, 0.86, "✓ " + badge if ok else "✕ " + badge, fontsize=13,
            fontweight="bold", color="white", ha="center", va="center", zorder=4)

    rows = [
        ("Paper behaviour", r["paper_ref"]),
        ("Acceptance test", r["name"]),
        ("Criterion", r["expected"]),
        ("Measured (reproduced run)", r["measured"]),
    ]
    y = 0.68
    for label, val in rows:
        ax.text(0.05, y, label, fontsize=9.5, fontweight="bold", color=_MUTED, va="top")
        ax.text(0.33, y, _wrap(val, 62), fontsize=9.5, color=_INK, va="top")
        y -= 0.155

    ax.text(0.05, 0.055, "Validated against Glazier & Graner, Phys. Rev. E 47, 2128 (1993) · "
            "pbg-cpm reproduction — every value computed from our own simulation run",
            fontsize=7.5, color=_MUTED, va="center")

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", facecolor=_CARD)
    plt.close(fig)
    return buf.getvalue()


def _study_title(slug):
    import yaml
    p = os.path.join(STU_DIR, slug, "study.yaml")
    if os.path.exists(p):
        d = yaml.safe_load(open(p)) or {}
        return d.get("title") or slug
    return slug


def write_cards(results=None):
    """Render every study's report card into ``studies/<slug>/charts/``."""
    if results is None:
        results = validate_all()
    n = 0
    for slug, r in results.items():
        charts = os.path.join(STU_DIR, slug, "charts")
        os.makedirs(charts, exist_ok=True)
        svg = render_card_png(slug, r, _study_title(slug))
        with open(os.path.join(charts, "00_report_card.svg"), "wb") as f:
            f.write(svg)
        with open(os.path.join(charts, "00_report_card.meta.json"), "w") as f:
            json.dump({
                "title": ("✓ Validation PASS" if r["passed"] else "✕ Validation FAIL")
                         + " — reproduces the paper's behaviour",
                "caption": r["name"],
                "simulations": ("Validation report card. The PASS/FAIL badge is the "
                                "result of an automated acceptance test "
                                "(pbg_cpm_studies.gg1993.validate) evaluated against this "
                                "study's own reproduced metric time series — not a hand-set "
                                "verdict. " + r["expected"]),
                "interpretation": "Measured: " + r["measured"],
            }, f, indent=2)
        n += 1
    return n


if __name__ == "__main__":
    print(f"wrote {write_cards()} report cards")
