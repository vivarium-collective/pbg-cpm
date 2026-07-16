"""Assemble every paper figure for each study from saved run results.

Maps each study's `figures` list (from params.STUDIES) to concrete panel/curve
builders in `figures.py`.  Missing runs are skipped with a warning so partial
result sets still render what they can.
"""

from __future__ import annotations

import os
import numpy as np

from . import figures as F
from . import run as runmod
from .params import STUDIES, ORDER
from .types import LIGHT, DARK

RESULTS_DIR = os.path.join(runmod.DATA_DIR, "results")
FIG_ROOT = os.path.join(runmod.WS_ROOT, "workspace", "gg1993_figures")


def _rdir(slug):
    return os.path.join(RESULTS_DIR, slug)


def _has(slug, run_slug=None):
    return os.path.exists(os.path.join(_rdir(slug), f"{run_slug or slug}_series.json"))


def _frames(slug, run_slug=None):
    return F.load_frames(_rdir(slug), run_slug or slug)


def _series(slug, run_slug=None):
    return F.load_series(_rdir(slug), run_slug or slug)


def _sweep(slug, tag, values, fmt):
    """Yield (value, mcs, series) for each existing sweep run."""
    for v in values:
        rs = f"{slug}__{tag}{fmt(v)}"
        if _has(slug, rs):
            mcs, series, _ = _series(slug, rs)
            yield v, mcs, series


# ---------------------------------------------------------------------------
# per-figure builders
# ---------------------------------------------------------------------------

def build(slug, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    s = STUDIES[slug]
    figs = set(s["figures"])
    made = []

    def out(name):
        # Pattern-image grids stay raster (high-DPI PNG); line plots + moment
        # tables are vector SVG so they're crisp at any browser zoom.
        ext = "png" if ("pattern" in name or "wall" in name) else "svg"
        p = os.path.join(fig_dir, f"{name}.{ext}")
        made.append(p)
        return p

    # ---------------- pattern grids ----------------
    if "fig3" in figs:  # annealing cell walls
        fr = _frames(slug)
        F.pattern_grid(fr, out("fig3_annealing_walls"),
                       order=[0, 2, 20], cols=3, style="walls",
                       titles=["unannealed", "2 MCS (T=0)", "20 MCS (T=0)"],
                       title="Fig 3 — annealing of cell walls")
    if "fig4" in figs:  # equilibration walls
        fr = _frames(slug)
        F.pattern_grid(fr, out("fig4_equilibration_walls"),
                       order=[0, 400], cols=2, style="walls",
                       titles=["initial rectangular (0 MCS)", "rounded (400 MCS)"],
                       title="Fig 4 — global pattern equilibration")
    for fig, order, cols, ttl in [
        ("fig7", [10, 100, 1000, 2000], 2, "Fig 7 — checkerboard"),
        ("fig12", [0, 10, 100, 1000, 3000, 4000, 5000, 13500], 4, "Fig 12 — cell sorting"),
        ("fig18", [0, 1000, 5000, 10000, 20000], 3, "Fig 18 — engulfment"),
        ("fig20", [0, 50, 5000], 3, "Fig 20 — position reversal"),
        ("fig22", [10, 100, 1000, 2000], 2, "Fig 22 — partial cell sorting"),
    ]:
        if fig in figs:
            fr = _frames(slug)
            order = [m for m in order if m in fr]
            F.pattern_grid(fr, out(f"{fig}_patterns"), order=order, cols=cols,
                           style="types", title=ttl,
                           titles=[f"{m} MCS" for m in order])
    for fig, mcs, ttl in [("fig25", 480, "Fig 25 — light-cell sloughing"),
                          ("fig26", 2000, "Fig 26 — dispersal, clusters separate"),
                          ("fig27", 2000, "Fig 27 — dispersal, no separation"),
                          ("fig28", 200, "Fig 28 — vacancy nucleation (cavity)")]:
        if fig in figs:
            fr = _frames(slug)
            if mcs in fr:
                F.pattern_grid({mcs: fr[mcs]}, out(f"{fig}_pattern"),
                               order=[mcs], cols=1, style="types",
                               titles=[f"{mcs} MCS"], title=ttl)

    # ---------------- time series ----------------
    if "fig2" in figs:
        mcs, ser, _ = _series(slug)
        F.multi_panel(out("fig2_annealing_topology"), [
            dict(mcs=mcs, xlog=False, ylabel="Number of sides", panel_label="(a)",
                 curves=[("bulk <n>", F._col(ser, "n_bulk")),
                         ("total <n>", F._col(ser, "n_total"))]),
            dict(mcs=mcs, xlog=False, ylabel="Bulk moments", panel_label="(b)",
                 curves=[("mu2", F._col(ser, "mu2")), ("mu3", F._col(ser, "mu3")),
                         ("mu4", F._col(ser, "mu4"))]),
        ], title="Fig 2 — annealing convergence")
    if "fig5" in figs:
        mcs, ser, _ = _series(slug)
        F.multi_panel(out("fig5_equilibration_stats"), [
            dict(mcs=mcs, xlog=False, ylabel="Total boundary length", panel_label="(a)",
                 curves=[("", F._col(ser, "bl_total"))]),
            dict(mcs=mcs, xlog=False, ylabel="Bulk moments", panel_label="(b)",
                 curves=[("mu2", F._col(ser, "mu2")), ("mu3", F._col(ser, "mu3")),
                         ("mu4", F._col(ser, "mu4"))]),
            dict(mcs=mcs, xlog=False, ylabel="Light-Medium fractional length",
                 panel_label="(c)", curves=[("", F._col(ser, "frac_lM"))]),
        ], title="Fig 5 — global equilibration statistics")
    if "fig8" in figs:
        mcs, ser, _ = _series(slug)
        F.multi_panel(out("fig8_checkerboard_stats"), [
            dict(mcs=mcs, ylabel="Total boundary length", panel_label="(a)",
                 curves=[("", F._col(ser, "bl_total"))]),
            dict(mcs=mcs, ylabel="Fractional length", panel_label="(b)",
                 curves=[("l-l", F._col(ser, "frac_ll")), ("d-d", F._col(ser, "frac_dd")),
                         ("l-d", F._col(ser, "frac_ld"))]),
            dict(mcs=mcs, ylabel="Number of sides <n>", panel_label="(c)",
                 curves=[("", F._col(ser, "n_bulk"))]),
            dict(mcs=mcs, ylabel="Bulk moments", panel_label="(d)",
                 curves=[("mu2", F._col(ser, "mu2")), ("mu3", F._col(ser, "mu3")),
                         ("mu4", F._col(ser, "mu4"))]),
        ], title="Fig 8 — checkerboard topology")
    if "fig9" in figs:  # checkerboard temperature sweep
        panels = []
        for key, lbl, pl in [("frac_ll", "Light-light interface", "(a)"),
                             ("frac_ld", "Light-dark interface", "(b)"),
                             ("frac_lM", "Light-Medium interface", "(c)")]:
            curves = [(f"T={v}", F._col(ser, key))
                      for v, mcs, ser in _sweep(slug, "T", s["temps"], lambda x: x)]
            mcs0 = next((m for v, m, se in _sweep(slug, "T", s["temps"], lambda x: x)), [])
            panels.append(dict(mcs=mcs0, ylabel=lbl, panel_label=pl, curves=curves))
        F.multi_panel(out("fig9_checkerboard_temperature"), panels,
                      title="Fig 9 — checkerboard vs temperature")
    if "fig13" in figs:
        mcs, ser, _ = _series(slug)
        F.multi_panel(out("fig13_sorting_lengths"), [
            dict(mcs=mcs, ylabel="Total boundary length", panel_label="(a)",
                 curves=[("", F._col(ser, "bl_total"))]),
            dict(mcs=mcs, ylabel="Fractional contact w/ medium", panel_label="(b)",
                 curves=[("l-M", F._col(ser, "frac_lM")), ("d-M", F._col(ser, "frac_dM"))]),
            dict(mcs=mcs, ylabel="Fractional cell-cell length", panel_label="(c)",
                 curves=[("l-l", F._col(ser, "frac_ll")), ("d-d", F._col(ser, "frac_dd")),
                         ("l-d", F._col(ser, "frac_ld"))]),
            dict(mcs=mcs, ylabel="Type-type correlation", panel_label="(d)",
                 curves=[("l-l", F._col(ser, "corr_ll")), ("d-d", F._col(ser, "corr_dd")),
                         ("l-d", F._col(ser, "corr_ld"))]),
        ], title="Fig 13 — cell-sorting boundary lengths")
    if "fig14" in figs:
        mcs, ser, _ = _series(slug)
        F.multi_panel(out("fig14_sorting_topology"), [
            dict(mcs=mcs, ylabel="Number of sides <n>", panel_label="(a)",
                 curves=[("", F._col(ser, "n_bulk"))]),
            dict(mcs=mcs, ylabel="Bulk moments", panel_label="(b)",
                 curves=[("mu2", F._col(ser, "mu2")), ("mu3", F._col(ser, "mu3")),
                         ("mu4", F._col(ser, "mu4"))]),
        ], title="Fig 14 — cell-sorting topology")
    if "fig15" in figs:  # cell-sorting temperature sweep
        panels = []
        for key, lbl, pl in [("frac_ld", "Light-dark heterotypic interface", "(a)"),
                             ("frac_lM", "Light-Medium interface", "(b)"),
                             ("bl_total", "Total boundary length", "(c)")]:
            curves = [(f"T={v}", F._col(ser, key))
                      for v, mcs, ser in _sweep(slug, "T", s["temps"], lambda x: x)]
            mcs0 = next((m for v, m, se in _sweep(slug, "T", s["temps"], lambda x: x)), [])
            panels.append(dict(mcs=mcs0, ylabel=lbl, panel_label=pl, curves=curves))
        F.multi_panel(out("fig15_sorting_temperature"), panels,
                      title="Fig 15 — cell sorting vs temperature")
    if "fig16" in figs:  # cell-sorting lambda sweep
        fmt = lambda x: x
        panels = []
        for key, lbl, pl in [("bl_total", "Total boundary length", "(a)"),
                             ("frac_ld", "Light-dark heterotypic interface", "(b)"),
                             ("frac_dM", "Dark-Medium interface", "(c)")]:
            curves = [(f"lam={v}", F._col(ser, key))
                      for v, mcs, ser in _sweep(slug, "lam", s["lambdas"], fmt)]
            mcs0 = next((m for v, m, se in _sweep(slug, "lam", s["lambdas"], fmt)), [])
            panels.append(dict(mcs=mcs0, ylabel=lbl, panel_label=pl, curves=curves))
        F.multi_panel(out("fig16_sorting_lambda"), panels,
                      title="Fig 16 — cell sorting vs area constraint lambda")
    if "fig19" in figs:
        mcs, ser, _ = _series(slug)
        F.multi_panel(out("fig19_engulfment_lengths"), [
            dict(mcs=mcs, ylabel="Fractional homotypic length", panel_label="(a)",
                 curves=[("l-l", F._col(ser, "frac_ll")), ("d-d", F._col(ser, "frac_dd"))]),
            dict(mcs=mcs, ylabel="Fractional heterotypic length", panel_label="(b)",
                 curves=[("l-M", F._col(ser, "frac_lM")), ("d-M", F._col(ser, "frac_dM")),
                         ("l-d", F._col(ser, "frac_ld"))]),
        ], title="Fig 19 — engulfment boundary lengths")
    if "fig21" in figs:
        mcs, ser, _ = _series(slug)
        F.multi_panel(out("fig21_reversal_lengths"), [
            dict(mcs=mcs, ylabel="Fractional cell-cell length", panel_label="(a)",
                 curves=[("l-l", F._col(ser, "frac_ll")), ("d-d", F._col(ser, "frac_dd")),
                         ("l-d", F._col(ser, "frac_ld"))]),
            dict(mcs=mcs, ylabel="Medium type correlation", panel_label="(b)",
                 curves=[("l-M", F._col(ser, "corr_lM")), ("d-M", F._col(ser, "corr_dM"))]),
        ], title="Fig 21 — position reversal")
    if "fig23" in figs:
        mcs, ser, _ = _series(slug)
        F.multi_panel(out("fig23_partial_lengths"), [
            dict(mcs=mcs, ylabel="Fractional cell-cell length", panel_label="(a)",
                 curves=[("l-l", F._col(ser, "frac_ll")), ("d-d", F._col(ser, "frac_dd")),
                         ("l-d", F._col(ser, "frac_ld"))]),
            dict(mcs=mcs, ylabel="Fractional contact w/ medium", panel_label="(b)",
                 curves=[("l-M", F._col(ser, "frac_lM")), ("d-M", F._col(ser, "frac_dM"))]),
        ], title="Fig 23 — partial cell sorting")
    if "fig24" in figs:  # partial vs normal comparison
        if _has(slug) and _has("cell_sorting"):
            mp, sp, _ = _series(slug)
            mn, sn, _ = _series("cell_sorting")
            F.multi_panel(out("fig24_partial_vs_normal"), [
                dict(mcs=mp, ylabel="Light-dark cell-cell length", panel_label="(a)",
                     curves=[("partial", F._col(sp, "frac_ld")),
                             ("normal", np.interp(np.log10(np.maximum(mp, 1)),
                                                  np.log10(np.maximum(mn, 1)),
                                                  F._col(sn, "frac_ld")))]),
                dict(mcs=mp, ylabel="Light-Medium interface", panel_label="(b)",
                     curves=[("partial", F._col(sp, "frac_lM")),
                             ("normal", np.interp(np.log10(np.maximum(mp, 1)),
                                                  np.log10(np.maximum(mn, 1)),
                                                  F._col(sn, "frac_lM")))]),
            ], title="Fig 24 — partial vs normal cell sorting")

    # ---------------- moment tables ----------------
    if "table1" in figs:
        _moments_table(slug, "T", s["temps"], lambda x: x, out("table1_checkerboard_moments"))
    if "table2" in figs:
        _moments_table(slug, "T", s["temps"], lambda x: x, out("table2_sorting_moments_T"), "T")
    if "table3" in figs:
        _moments_table(slug, "lam", s["lambdas"], lambda x: x,
                       out("table3_sorting_moments_lambda"), "lambda")

    return made


def _moments_table(slug, tag, values, fmt, path, col0="T"):
    rows = []
    for v, mcs, ser in _sweep(slug, tag, values, fmt):
        if not ser:
            continue
        last = ser[-1]
        rows.append((v, last.get("n_bulk", float("nan")), last.get("mu2", float("nan")),
                     last.get("mu3", float("nan")), last.get("mu4", float("nan"))))
    if rows:
        F.moments_table(path, rows, col0_label=col0)


def build_all(fig_root=FIG_ROOT):
    index = {}
    for slug in ORDER:
        if not _has(slug) and not any(
                _has(slug, f"{slug}__T{t}") for t in (STUDIES[slug].get("temps") or [])):
            print(f"skip {slug} (no results)")
            continue
        fig_dir = os.path.join(fig_root, slug)
        made = build(slug, fig_dir)
        index[slug] = made
        print(f"{slug}: {len(made)} figures")
    return index


if __name__ == "__main__":
    build_all()
