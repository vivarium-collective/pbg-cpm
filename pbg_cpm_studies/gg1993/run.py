"""Run loop, canonical initial-condition generation, and result caching for the
Glazier & Graner (1993) reproduction."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from cpm import cpm_core
from . import engine, metrics
from .engine import WorldParams, energies_from_paper
from .types import MEDIUM, DARK, LIGHT

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pbg_cpm_studies/
WS_ROOT = os.path.dirname(PKG_DIR)
DATA_DIR = os.path.join(WS_ROOT, "workspace", "gg1993_data")
IC_PATH = os.path.join(DATA_DIR, "canonical_ic.npz")


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------------
# canonical equilibrated initial condition (Fig 4b): the rounded disk of ~1000
# light cells that every biological study reuses (Sec. II D 3).
# ---------------------------------------------------------------------------

def generate_canonical_ic(nx=300, ny=300, rect=(50, 50, 250, 250),
                          cell_w=5, cell_h=8, equil_mcs=400, temperature=5.0,
                          seed=17, save=True, parallel=True):
    """Build a rectangular brick tiling of light cells and equilibrate it at
    T=5 for `equil_mcs` paper-MCS (J_ll=2, J_lM=8, lambda=1), yielding the
    rounded aggregate used as the shared IC. Returns the owner-label 2D array."""
    labels = engine.brick_tiling_labels(nx, ny, rect, cell_w, cell_h)
    ncells = int(labels.max())
    wp = WorldParams(nx=nx, ny=ny, temperature=temperature, boundary="periodic")
    en = energies_from_paper(J_ll=2, J_dd=2, J_ld=2, J_lM=8, J_dM=8)
    type_of = {int(l): LIGHT for l in np.unique(labels) if l != 0}
    sim = engine.build_from_labels(labels, type_of, en, wp,
                                   target_by_type={LIGHT: 40.0, DARK: 40.0},
                                   lambda_volume=1.0, seed=seed)
    t0 = time.time()
    engine.step_paper_mcs(sim.world, mcs=equil_mcs, parallel=parallel)
    owner = sim.snapshot2d()
    dt = time.time() - t0
    print(f"canonical IC: {ncells} cells, {nx}x{ny}, {equil_mcs} MCS in {dt:.1f}s")
    if save:
        _ensure_dir(DATA_DIR)
        np.savez_compressed(IC_PATH, owner=owner, nx=nx, ny=ny)
    return owner


def load_canonical_ic():
    if not os.path.exists(IC_PATH):
        return generate_canonical_ic()
    d = np.load(IC_PATH)
    return d["owner"]


# ---------------------------------------------------------------------------
# type assignment for the shared IC
# ---------------------------------------------------------------------------

def assign_random(owner, frac_light=0.5, seed=0):
    rng = np.random.default_rng(seed)
    uniq = [int(l) for l in np.unique(owner) if l != 0]
    return {l: (LIGHT if rng.random() < frac_light else DARK) for l in uniq}


def assign_half_split(owner, light_on_top=True):
    """Upper half light, lower half dark (Engulfment IC, Fig 18a)."""
    ny, nx = owner.shape
    types_map = {}
    # cell centroid y decides the half
    for l in np.unique(owner):
        if l == 0:
            continue
        ys, xs = np.where(owner == l)
        cy = ys.mean()
        top = cy >= ny / 2.0
        is_light = top if light_on_top else (not top)
        types_map[int(l)] = LIGHT if is_light else DARK
    return types_map


def assign_all_light(owner):
    return {int(l): LIGHT for l in np.unique(owner) if l != 0}


# ---------------------------------------------------------------------------
# a study run: capture pattern frames at image timepoints + a metric time series
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    slug: str
    params: dict
    image_mcs: list                    # timepoints captured as patterns
    frames: dict                       # mcs -> (owner2d, type_grid2d) annealed
    series_mcs: list                   # timepoints of the metric series
    series: list                       # list of flat observable dicts
    meta: dict = field(default_factory=dict)


def _log_timepoints(max_mcs, per_decade=12, start=1):
    """Log-spaced integer timepoints from `start` to max_mcs (for Fig time-axes)."""
    if max_mcs <= start:
        return [max_mcs]
    import math
    n = int(per_decade * math.log10(max_mcs / start)) + 1
    pts = sorted({int(round(start * 10 ** (i / per_decade))) for i in range(n + 1)})
    pts = [p for p in pts if start <= p <= max_mcs]
    if pts[-1] != max_mcs:
        pts.append(max_mcs)
    return pts


def run_study(spec, owner_ic, types_map, seed=0, parallel=True,
              measure_series=True, verbose=True):
    """Run one study to completion, capturing pattern frames at spec.image_mcs
    and a (log-spaced) metric time series up to spec.series_max_mcs.

    spec must provide: J (dict of J_ll..), temperature, lam, target(dict),
    image_mcs(list), series_max_mcs(int), display_anneal_mcs, unannealed_display.
    """
    ny, nx = owner_ic.shape
    wp = WorldParams(nx=nx, ny=ny, temperature=spec["temperature"], boundary="periodic")
    en = energies_from_paper(**spec["J"])
    sim = engine.build_from_labels(owner_ic.ravel().astype(np.uint32), types_map,
                                   en, wp, target_by_type=spec["target"],
                                   lambda_volume=spec["lam"], seed=seed)

    image_set = sorted(set(spec["image_mcs"]))
    if not measure_series:
        series_pts = []
    elif spec.get("series_linear"):
        mx = spec["series_max_mcs"]
        step = max(1, mx // 30)
        series_pts = list(range(0, mx + 1, step))
        if series_pts[-1] != mx:
            series_pts.append(mx)
    else:
        series_pts = _log_timepoints(spec["series_max_mcs"])
    all_stops = sorted(set(image_set) | set(series_pts) | {0})

    frames, series, series_mcs = {}, [], []
    cur = 0
    anneal = spec.get("display_anneal_mcs", 2)
    unannealed = spec.get("unannealed_display", False)
    t0 = time.time()
    for stop in all_stops:
        if stop > cur:
            engine.step_paper_mcs(sim.world, mcs=stop - cur, parallel=parallel)
            cur = stop
        if stop in image_set:
            if unannealed:
                frames[stop] = sim.type_grid()
            else:
                frames[stop] = engine.annealed_grids(sim, anneal_mcs=anneal)
        if stop in series_pts:
            aowner, atg = (sim.type_grid() if unannealed
                           else engine.annealed_grids(sim, anneal_mcs=anneal))
            obs = metrics.measure_grid(aowner, atg)
            series.append(obs.flat())
            series_mcs.append(stop)
        if verbose:
            print(f"  {spec['slug']}: {stop}/{spec['series_max_mcs']} MCS "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return RunResult(slug=spec["slug"], params=spec, image_mcs=image_set,
                     frames=frames, series_mcs=series_mcs, series=series,
                     meta={"seed": seed, "runtime_s": time.time() - t0})


# ---------------------------------------------------------------------------
# result persistence
# ---------------------------------------------------------------------------

def save_result(res, out_dir):
    _ensure_dir(out_dir)
    # frames -> npz
    fkeys = sorted(res.frames)
    arrs = {}
    for k in fkeys:
        o, t = res.frames[k]
        arrs[f"owner_{k}"] = o.astype(np.int32)
        arrs[f"type_{k}"] = t.astype(np.int8)
    np.savez_compressed(os.path.join(out_dir, f"{res.slug}_frames.npz"),
                        frame_mcs=np.array(fkeys), **arrs)
    with open(os.path.join(out_dir, f"{res.slug}_series.json"), "w") as f:
        json.dump({"slug": res.slug, "series_mcs": res.series_mcs,
                   "series": res.series, "params": _jsonable(res.params),
                   "meta": res.meta}, f, indent=2)


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out
