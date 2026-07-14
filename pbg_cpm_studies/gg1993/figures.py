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
        ax.set_title(lab, fontsize=10)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# time-series line plots
# ---------------------------------------------------------------------------

_MARK = ["o", "s", "^", "D", "v", "x", "+", "*"]


def series_plot(path, mcs, curves, ylabel, xlabel="Time (MCS)",
                xlog=True, ylog=False, title=None, size=(5, 4), ylim=None,
                markersize=3):
    """curves: list of (label, yarray). One axes."""
    fig, ax = plt.subplots(figsize=size)
    x = np.asarray(mcs, dtype=float)
    for i, (label, y) in enumerate(curves):
        ax.plot(x, y, marker=_MARK[i % len(_MARK)], ms=markersize, lw=1.0,
                label=label, markerfacecolor="none" if i % 2 else None)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    if any(lbl for lbl, _ in curves):
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def multi_panel(path, panels, title=None, size=4.0):
    """panels: list of dicts {mcs, curves, ylabel, xlog, ylog, ylim, panel_label}.
    Lays out in a near-square grid."""
    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * 0.85),
                             squeeze=False)
    for i, p in enumerate(panels):
        ax = axes[i // cols][i % cols]
        x = np.asarray(p["mcs"], dtype=float)
        for j, (label, y) in enumerate(p["curves"]):
            ax.plot(x, y, marker=_MARK[j % len(_MARK)], ms=3, lw=1.0, label=label,
                    markerfacecolor="none" if j % 2 else None)
        if p.get("xlog", True):
            ax.set_xscale("log")
        if p.get("ylog", False):
            ax.set_yscale("log")
        if p.get("ylim"):
            ax.set_ylim(*p["ylim"])
        ax.set_xlabel(p.get("xlabel", "Time (MCS)"))
        ax.set_ylabel(p["ylabel"])
        if p.get("panel_label"):
            ax.set_title(p["panel_label"], fontsize=10, loc="left")
        if any(lbl for lbl, _ in p["curves"]):
            ax.legend(fontsize=7, frameon=False)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# moment tables
# ---------------------------------------------------------------------------

def moments_table(path, rows, col0_label="T"):
    """rows: list of (label, n, mu2, mu3, mu4). Writes a PNG table + markdown."""
    fig, ax = plt.subplots(figsize=(5, 0.4 * len(rows) + 1))
    ax.axis("off")
    header = [col0_label, "<n>", "mu2", "mu3", "mu4"]
    cells = [[f"{r[0]}", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.3f}"]
             for r in rows]
    tbl = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    md = path.rsplit(".", 1)[0] + ".md"
    with open(md, "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} |\n")
