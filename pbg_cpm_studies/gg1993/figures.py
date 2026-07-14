"""Assemble the Glazier & Graner (1993) figures from saved run results.

Reproduces the paper's figure archetypes:
  * pattern-panel grids (Figs 7, 12, 18, 20, 22 and the single-panel dispersal/
    cavity figs 25-28, plus wall views 3, 4)
  * time-series line plots on log or linear MCS axes (Figs 2, 5, 8, 9, 13, 14,
    15, 16, 19, 21, 23, 24)
  * moment tables (Tables I, II, III)
"""

from __future__ import annotations

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import render
from .types import LIGHT, DARK, MEDIUM

# Resolution: pattern grids are raster (cell lattices) → high-DPI PNG; line
# plots and tables are vector → SVG, so they stay razor-sharp at any browser
# zoom (the report inlines SVG verbatim).
PNG_DPI = 220

# Validated colourblind-safe categorical palette (see the dataviz skill).
_PALETTE = ["#4f46e5", "#0d9488", "#e11d48", "#d97706", "#7c3aed", "#0891b2",
            "#2563eb", "#65a30d"]
_INK = "#1f2937"
_MUTED = "#6b7280"
_GRID = "#e5e7eb"

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": _MUTED,
    "axes.labelcolor": _INK,
    "axes.titlecolor": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "text.color": _INK,
    "svg.fonttype": "none",   # keep text as selectable text in the SVG
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _style_axes(ax):
    """Recessive grid + clean spines (dataviz house style)."""
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
        ax.spines[side].set_linewidth(0.8)


# ---------------------------------------------------------------------------
# loading saved results
# ---------------------------------------------------------------------------

def load_frames(results_dir, run_slug):
    d = np.load(os.path.join(results_dir, f"{run_slug}_frames.npz"))
    mcs = [int(m) for m in d["frame_mcs"]]
    frames = {}
    for m in mcs:
        frames[m] = (d[f"owner_{m}"], d[f"type_{m}"])
    return frames


def load_series(results_dir, run_slug):
    with open(os.path.join(results_dir, f"{run_slug}_series.json")) as f:
        j = json.load(f)
    return j["series_mcs"], j["series"], j.get("params", {})


def _col(series, key):
    return np.array([s.get(key, np.nan) for s in series], dtype=float)


# ---------------------------------------------------------------------------
# pattern panels
# ---------------------------------------------------------------------------

def pattern_grid(frames, path, order=None, cols=2, style="types",
                 titles=None, title=None, size=3.2):
    """Grid of pattern panels. style: 'types' (filled) or 'walls' (outlines)."""
    keys = order if order is not None else sorted(frames)
    n = len(keys)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows),
                             squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // cols][i % cols]
        owner, tg = frames[k]
        owner, tg = render.center_on_aggregate(owner, tg)
        if style == "walls":
            rgb = np.ones((*owner.shape, 3))
            rgb[render._wall_mask(owner)] = (0, 0, 0)
        else:
            rgb = render._rgb_from_types(tg)
            rgb[render._wall_mask(owner)] = (0, 0, 0)
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        lab = titles[i] if titles else f"{k} MCS"
        ax.set_title(lab, fontsize=11, color=_INK)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# time-series line plots
# ---------------------------------------------------------------------------

_MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _plot_curves(ax, x, curves):
    for i, (label, y) in enumerate(curves):
        c = _PALETTE[i % len(_PALETTE)]
        ax.plot(x, y, marker=_MARK[i % len(_MARK)], ms=5, lw=2.0,
                color=c, markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=0.6, label=label, zorder=3)


def series_plot(path, mcs, curves, ylabel, xlabel="Time (MCS)",
                xlog=True, ylog=False, title=None, size=(5.4, 4.2), ylim=None,
                markersize=5):
    """curves: list of (label, yarray). One axes. Saved as SVG when path ends .svg."""
    fig, ax = plt.subplots(figsize=size)
    x = np.asarray(mcs, dtype=float)
    _plot_curves(ax, x, curves)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    _style_axes(ax)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=8)
    if any(lbl for lbl, _ in curves):
        ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def multi_panel(path, panels, title=None, size=4.4):
    """panels: list of dicts {mcs, curves, ylabel, xlog, ylog, ylim, panel_label}.
    Lays out in a near-square grid. Saved as SVG when path ends .svg."""
    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * 0.82),
                             squeeze=False)
    for i, p in enumerate(panels):
        ax = axes[i // cols][i % cols]
        x = np.asarray(p["mcs"], dtype=float)
        _plot_curves(ax, x, p["curves"])
        if p.get("xlog", True):
            ax.set_xscale("log")
        if p.get("ylog", False):
            ax.set_yscale("log")
        if p.get("ylim"):
            ax.set_ylim(*p["ylim"])
        ax.set_xlabel(p.get("xlabel", "Time (MCS)"))
        ax.set_ylabel(p["ylabel"])
        _style_axes(ax)
        if p.get("panel_label"):
            ax.set_title(p["panel_label"], fontsize=11, fontweight="bold",
                         loc="left", color=_INK)
        if any(lbl for lbl, _ in p["curves"]):
            ax.legend(fontsize=8.5, frameon=False)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# moment tables
# ---------------------------------------------------------------------------

def moments_table(path, rows, col0_label="T"):
    """rows: list of (label, n, mu2, mu3, mu4). Writes a styled vector table + md."""
    header = ["⟨n⟩", "μ₂", "μ₃", "μ₄"]
    header = [col0_label] + header
    cells = [[f"{r[0]}", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.3f}"]
             for r in rows]
    ncol = len(header)
    fig, ax = plt.subplots(figsize=(0.95 * ncol + 0.4, 0.42 * len(rows) + 0.9))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    _HEAD = "#4f46e5"
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ffffff")
        cell.set_linewidth(1.2)
        if r == 0:  # header row
            cell.set_facecolor(_HEAD)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8fafc" if r % 2 else "#eef2ff")
            cell.set_text_props(color=_INK)
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    md = path.rsplit(".", 1)[0] + ".md"
    with open(md, "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} |\n")
