"""World construction, initial conditions and the run loop for the
Glazier & Graner (1993) reproduction.

Physics fidelity notes
----------------------
* Neighbourhood: order-2 (Moore-8 in 2D), the paper's second-nearest-neighbour
  square lattice.
* MCS convention: the paper defines **one MCS = 16 lattice-site sweeps**
  ("16 times as many time steps as there are lattice sites", Sec. II A 2).
  The engine's `step(n)` performs `n` single site-sweeps, so a paper-MCS maps to
  `MCS_SCALE (=16)` engine sweeps.  All `mcs` arguments below are *paper* MCS.
* Metropolis acceptance is the engine's standard `exp(-dH/T)`, with T=0 giving
  greedy zero-temperature dynamics (used for display annealing).
* Statistics/displays are taken after `anneal_mcs` (default 2) paper-MCS of
  T=0 annealing applied to a *copy* of the pattern, leaving the running spin
  array untouched (Sec. II D 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from cpm import cpm_core
from .types import MEDIUM, DARK, LIGHT

MCS_SCALE = 16  # engine sweeps per paper Monte-Carlo step


# ---------------------------------------------------------------------------
# energies
# ---------------------------------------------------------------------------

def energies_from_paper(J_ll, J_dd, J_ld, J_lM, J_dM, J_MM=0.0):
    """Return the type-pair contact-energy dict from the paper's J's.

    Types: light=2, dark=1, medium=0.  gamma surface tensions (for reference):
        gamma_ld = J_ld - (J_dd+J_ll)/2
        gamma_lM = J_lM - J_ll/2
        gamma_dM = J_dM - J_dd/2
    """
    return {
        (LIGHT, LIGHT): J_ll,
        (DARK, DARK): J_dd,
        (LIGHT, DARK): J_ld,
        (MEDIUM, LIGHT): J_lM,
        (MEDIUM, DARK): J_dM,
        (MEDIUM, MEDIUM): J_MM,
    }


def surface_tensions(J_ll, J_dd, J_ld, J_lM, J_dM):
    return dict(
        gamma_ld=J_ld - (J_dd + J_ll) / 2.0,
        gamma_lM=J_lM - J_ll / 2.0,
        gamma_dM=J_dM - J_dd / 2.0,
    )


# ---------------------------------------------------------------------------
# world parameters + a lightweight simulation handle
# ---------------------------------------------------------------------------

@dataclass
class Sim:
    """A finalized world plus the metadata needed to rebuild it for annealing."""
    world: object
    nx: int
    ny: int
    neighbor_order: int
    temperature: float
    boundary: str
    energies: dict
    lambda_volume: float
    # per-cell target volume, indexed for the anneal rebuild
    label_types: dict = field(default_factory=dict)   # owner-id -> type
    target_by_type: dict = field(default_factory=dict)  # type -> target volume

    def snapshot2d(self):
        arr = np.asarray(self.world.snapshot(), dtype=np.int64)
        return arr.reshape(self.ny, self.nx)

    def type_grid(self, recenter=True):
        owners = self.snapshot2d()
        types_of = np.asarray(self.world.cell_types(), dtype=np.int64)
        og, tg = owners, types_of[owners]
        if recenter and self.boundary == "periodic":
            from .metrics import recenter_periodic
            og, tg = recenter_periodic(og, tg)
        return og, tg


def _apply_contacts(world, energies):
    for (a, b), j in energies.items():
        world.set_contact(int(a), int(b), float(j))


# ---------------------------------------------------------------------------
# initial conditions
# ---------------------------------------------------------------------------

def brick_tiling_labels(nx, ny, rect, cell_w, cell_h, stagger=True):
    """Paint a rectangular region [x0:x1, y0:y1] with brick-wall cells.

    Returns a flat (ny*nx) uint32 label array; medium = 0, cells = 1..N.
    Rows are offset by half a cell (brick bond) when `stagger` is set, matching
    the tall-thin rectangular tiling of Fig. 4(a).
    """
    x0, y0, x1, y1 = rect
    labels = np.zeros((ny, nx), dtype=np.uint32)
    nid = 0
    for j, ry in enumerate(range(y0, y1, cell_h)):
        offset = (cell_w // 2) if (stagger and (j % 2)) else 0
        # start one shifted column back so the row still fills from x0
        cx = x0 - offset
        while cx < x1:
            nid += 1
            xa = max(cx, x0)
            xb = min(cx + cell_w, x1)
            ya = ry
            yb = min(ry + cell_h, y1)
            if xa < xb and ya < yb:
                labels[ya:yb, xa:xb] = nid
            cx += cell_w
    return labels.ravel()


def circular_mask(nx, ny, cx, cy, radius):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2


def build_from_labels(labels_flat, type_of_label, energies, wp,
                      target_by_type, lambda_volume, seed=0):
    """Build & finalize a World from an owner-label array.

    labels_flat   : flat uint32 array (len nx*ny), 0 = medium.
    type_of_label : dict owner-id -> type (light/dark).  Missing -> default light.
    energies      : type-pair J dict.
    wp            : WorldParams.
    target_by_type: dict type -> target volume (per-cell area constraint).
    Returns a finalized Sim.  Per-type target volumes are applied after seeding.
    """
    world = cpm_core.World((wp.nx, wp.ny, 1), wp.boundary,
                           wp.neighbor_order, wp.temperature)
    _apply_contacts(world, energies)
    labels_flat = np.asarray(labels_flat, dtype=np.uint32)
    types_map = {int(k): int(v) for k, v in type_of_label.items()}
    default_type = LIGHT
    # a nominal target for seeding; per-cell overrides applied below
    nominal_target = float(next(iter(target_by_type.values())))
    label_to_cid = world.seed_from_labels(
        labels_flat.tolist(), types_map, default_type,
        nominal_target, float(lambda_volume))
    # per-type target volume overrides
    for label, cid in label_to_cid.items():
        t = types_map.get(int(label), default_type)
        tv = target_by_type.get(t)
        if tv is not None and abs(tv - nominal_target) > 1e-9:
            world.set_target_volume(int(cid), float(tv))
    world.finalize(seed)
    return Sim(world=world, nx=wp.nx, ny=wp.ny,
               neighbor_order=wp.neighbor_order, temperature=wp.temperature,
               boundary=wp.boundary, energies=energies,
               lambda_volume=lambda_volume,
               label_types={int(l): types_map.get(int(l), default_type)
                            for l in np.unique(labels_flat) if l != 0},
               target_by_type=dict(target_by_type))


@dataclass
class WorldParams:
    nx: int
    ny: int
    temperature: float
    neighbor_order: int = 2
    boundary: str = "periodic"


# ---------------------------------------------------------------------------
# run loop
# ---------------------------------------------------------------------------

def step_paper_mcs(world, mcs, mcs_scale=MCS_SCALE, parallel=True, block=16):
    """Advance the world by `mcs` *paper* MCS (mcs_scale engine sweeps each)."""
    sweeps = int(mcs) * int(mcs_scale)
    if sweeps <= 0:
        return
    if parallel:
        world.step_parallel(sweeps, block)
    else:
        world.step(sweeps)


def annealed_grids(sim, anneal_mcs=2, mcs_scale=MCS_SCALE, seed=12345):
    """Return (owner, type_grid) after `anneal_mcs` paper-MCS of T=0 annealing
    applied to a *copy* of the current pattern (running sim untouched)."""
    owner = sim.snapshot2d()
    labels_flat = owner.ravel().astype(np.uint32)
    types_map = {int(k): int(v) for k, v in sim.label_types.items()}
    aworld = cpm_core.World((sim.nx, sim.ny, 1), sim.boundary,
                            sim.neighbor_order, 0.0)  # T = 0
    _apply_contacts(aworld, sim.energies)
    nominal_target = float(next(iter(sim.target_by_type.values())))
    label_to_cid = aworld.seed_from_labels(
        labels_flat.tolist(), types_map, LIGHT,
        nominal_target, float(sim.lambda_volume))
    for label, cid in label_to_cid.items():
        t = types_map.get(int(label), LIGHT)
        tv = sim.target_by_type.get(t)
        if tv is not None and abs(tv - nominal_target) > 1e-9:
            aworld.set_target_volume(int(cid), float(tv))
    aworld.finalize(seed)
    if anneal_mcs > 0:
        aworld.step(int(anneal_mcs) * int(mcs_scale))
    owners = np.asarray(aworld.snapshot(), dtype=np.int64).reshape(sim.ny, sim.nx)
    types_of = np.asarray(aworld.cell_types(), dtype=np.int64)
    og, tg = owners, types_of[owners]
    if sim.boundary == "periodic":
        from .metrics import recenter_periodic
        og, tg = recenter_periodic(og, tg)
    return og, tg
