"""Regression test for the Glazier & Graner display/measurement annealing.

`engine.annealed_grids` rebuilds a fresh T=0 world from the *evolved* lattice
snapshot to characterise the pattern without perturbing the running sim. That
rebuild must preserve each cell's type. A prior bug keyed the type map by the
original input label while `seed_from_labels` returns a *permuted* label->cellid
map, so the annealed pattern (used for BOTH the metric series and the figures)
had its light/dark assignment scrambled — every two-type study then measured a
near-random heterotypic fraction (~0.45) regardless of how well it had sorted.

This test evolves a small two-type aggregate until it visibly sorts, then
asserts the annealed-grid heterotypic fraction tracks the raw (live-world)
heterotypic fraction. On the buggy code these diverge (~0.45 vs ~0.15).
"""

import numpy as np

from pbg_cpm_studies.gg1993 import engine, metrics
from pbg_cpm_studies.gg1993.engine import WorldParams, energies_from_paper
from pbg_cpm_studies.gg1993.types import LIGHT, DARK


def _build_sorting_sim(seed=3):
    nx = ny = 80
    labels = engine.brick_tiling_labels(nx, ny, (10, 10, 70, 70), 5, 8)
    uniq = [int(l) for l in np.unique(labels) if l != 0]
    # Shuffle the label ids so numeric-label order != raster first-encounter
    # order. seed_from_labels assigns cell ids in first-encounter order, so this
    # forces label != cell-id — exactly the permutation the saved (equilibrated)
    # canonical IC has, which is what surfaced the type-scrambling bug. A raster
    # ordered tiling would be identity-mapped and hide the bug.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    remap = {old: int(new) for old, new in zip(uniq, perm)}
    labels = np.array([remap.get(int(l), 0) for l in labels], dtype=np.uint32)
    types = {int(l): (LIGHT if rng.random() < 0.5 else DARK) for l in perm}
    wp = WorldParams(nx=nx, ny=ny, temperature=10.0, boundary="periodic")
    en = energies_from_paper(J_ll=14, J_dd=2, J_ld=11, J_lM=16, J_dM=16)
    return engine.build_from_labels(
        labels, types, en, wp,
        target_by_type={LIGHT: 40.0, DARK: 40.0}, lambda_volume=1.0, seed=seed)


def _frac_ld(owner, type_grid):
    return metrics.measure_grid(owner, type_grid).flat()["frac_ld"]


def test_annealed_grid_preserves_cell_types():
    sim = _build_sorting_sim()
    engine.step_paper_mcs(sim.world, mcs=150, parallel=False)

    raw_ld = _frac_ld(*sim.type_grid())
    ann_ld = _frac_ld(*engine.annealed_grids(sim, anneal_mcs=2))

    # The aggregate must actually have sorted (heterotypic fraction well below
    # the ~0.5 of a random mix), else the test proves nothing.
    assert raw_ld < 0.35, f"sim did not sort (raw frac_ld={raw_ld:.3f})"

    # The annealed measurement must track the true (raw) state, not scramble it.
    assert abs(ann_ld - raw_ld) < 0.06, (
        f"annealed frac_ld={ann_ld:.3f} diverges from raw={raw_ld:.3f} "
        "-> cell types scrambled in the anneal rebuild")


def test_label_types_keyed_by_cell_id():
    """sim.label_types is documented as owner-id -> type; ensure the keys are
    the actual live cell ids, not the pre-seed input labels."""
    sim = _build_sorting_sim()
    live_ids = {int(i) for i in np.unique(sim.snapshot2d()) if i != 0}
    assert set(sim.label_types) == live_ids
    world_types = np.asarray(sim.world.cell_types())
    for cid, t in sim.label_types.items():
        assert int(world_types[cid]) == int(t)
