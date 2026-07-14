"""Topological & boundary-length observables for the Glazier & Graner (1993)
Cellular Potts reproduction.

All measurements are taken from a lattice *snapshot* (per-site owner-label
array) plus the per-owner type array, exactly the quantities the paper
characterises patterns by (Sec. II D 1):

  * total boundary length      -- total # of mismatched (owner) bonds
  * fractional boundary length -- per type-pair fraction of the total:
        light-light (ll), dark-dark (dd), light-dark (ld) cell-cell contacts,
        light-Medium (lM), dark-Medium (dM) medium contacts
  * number of sides <n>        -- mean cell degree in the cell-adjacency graph
                                  ('bulk' = cells not touching the medium)
  * topological moments mu_l   -- mu_l = <(n - <n>)^l>,  l = 2,3,4  (Eq. 6)
  * type-type correlations     -- fraction of cell-cell adjacency edges of each
                                  type-pair, and medium-contact correlations

Bonds are counted on the second-nearest-neighbour (Moore-8) lattice, the same
neighbourhood the CPM Hamiltonian sums J over (`neighbor_order = 2`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .types import MEDIUM, DARK, LIGHT


# ---------------------------------------------------------------------------
# snapshot -> grids
# ---------------------------------------------------------------------------

def recenter_periodic(owner, type_grid):
    """Roll a periodic lattice so the non-medium aggregate's centre-of-mass sits
    at the grid centre (whole, away from the wrap seam). Uses the circular mean
    of occupied coordinates so it is correct across the periodic boundary."""
    ys, xs = np.where(owner != MEDIUM)
    if xs.size == 0:
        return owner, type_grid
    ny, nx = owner.shape
    ang_x = np.angle(np.exp(2j * np.pi * xs / nx).mean())
    ang_y = np.angle(np.exp(2j * np.pi * ys / ny).mean())
    cx = (ang_x / (2 * np.pi)) % 1.0 * nx
    cy = (ang_y / (2 * np.pi)) % 1.0 * ny
    sx = int(round(nx / 2 - cx))
    sy = int(round(ny / 2 - cy))
    return (np.roll(owner, (sy, sx), (0, 1)),
            np.roll(type_grid, (sy, sx), (0, 1)))


def grids(world):
    """Return (owner_grid, type_grid) as (ny, nx) int arrays for a 2D world."""
    nx, ny, nz = world.dims()
    if nz != 1:
        raise ValueError("metrics.grids expects a 2D world (nz == 1)")
    owners = np.asarray(world.snapshot(), dtype=np.int64).reshape(ny, nx)
    types_of = np.asarray(world.cell_types(), dtype=np.int64)  # indexed by owner id
    type_grid = types_of[owners]
    return owners, type_grid


# forward Moore directions so each unordered bond is visited exactly once
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))  # right, down, down-right, down-left


def _bond_pairs(owner):
    """Yield (a, b) flat owner-id arrays for every unordered Moore bond."""
    for dy, dx in _DIRS:
        if dy == 0:  # (0, 1) right
            a, b = owner[:, :-1], owner[:, 1:]
        elif dx == 0:  # (1, 0) down
            a, b = owner[:-1, :], owner[1:, :]
        elif dx == 1:  # (1, 1) down-right
            a, b = owner[:-1, :-1], owner[1:, 1:]
        else:  # (1, -1) down-left
            a, b = owner[:-1, 1:], owner[1:, :-1]
        yield a.ravel(), b.ravel()


# ---------------------------------------------------------------------------
# boundary lengths
# ---------------------------------------------------------------------------

def boundary_lengths(owner, type_grid):
    """Count mismatched Moore bonds, classified by type-pair.

    Returns a dict with raw bond counts:
        total, ll, dd, ld, lM, dM, cell_cell (= ll+dd+ld), medium (= lM+dM)
    'total' = every owner-mismatched bond (cell-cell of any type + cell-medium).
    Medium-medium bonds are never mismatched (single owner 0), so excluded.
    """
    c = dict(total=0, ll=0, dd=0, ld=0, lM=0, dM=0)
    types_of_owner = None  # not needed; type per site via type_grid
    tg = type_grid
    for (ao, bo), (at, bt) in zip(_bond_pairs(owner), _bond_pairs(tg)):
        diff = ao != bo               # owner mismatch = a boundary bond
        if not diff.any():
            continue
        ao, bo, at, bt = ao[diff], bo[diff], at[diff], bt[diff]
        c["total"] += ao.size
        a_med = at == MEDIUM
        b_med = bt == MEDIUM
        one_med = a_med ^ b_med       # exactly one side is medium
        both_cell = ~(a_med | b_med)
        # cell-medium bonds: classify by the non-medium side's type
        if one_med.any():
            nonmed_t = np.where(a_med, bt, at)[one_med]
            c["lM"] += int(np.count_nonzero(nonmed_t == LIGHT))
            c["dM"] += int(np.count_nonzero(nonmed_t == DARK))
        # cell-cell bonds
        if both_cell.any():
            cat, cbt = at[both_cell], bt[both_cell]
            same = cat == cbt
            c["ll"] += int(np.count_nonzero(same & (cat == LIGHT)))
            c["dd"] += int(np.count_nonzero(same & (cat == DARK)))
            c["ld"] += int(np.count_nonzero(~same))
    c["cell_cell"] = c["ll"] + c["dd"] + c["ld"]
    c["medium"] = c["lM"] + c["dM"]
    return c


def fractional_lengths(bl):
    """Per type-pair fraction of the total boundary length (Figs 5c,8,13,19...)."""
    tot = bl["total"] or 1
    return {k: bl[k] / tot for k in ("ll", "dd", "ld", "lM", "dM")}


# ---------------------------------------------------------------------------
# cell-adjacency graph -> <n>, moments, correlations
# ---------------------------------------------------------------------------

def owner_types(owner, type_grid):
    """Array `types_arr` indexed by owner id (owner 0 -> medium type 0)."""
    base = int(owner.max()) + 1
    types_arr = np.zeros(base, dtype=np.int64)
    types_arr[owner.ravel()] = type_grid.ravel()  # uniform per owner; any wins
    return types_arr


def adjacency(owner, type_grid):
    """Build the cell-adjacency graph from the lattice.

    Returns:
        neigh:     dict owner_id -> set of adjacent owner ids (incl. 0=medium)
        types_arr: np.ndarray, type per owner id
    Two distinct owners are 'adjacent' if they share any Moore bond.
    """
    base = int(owner.max()) + 1
    packed_all = []
    for ao, bo in _bond_pairs(owner):
        diff = ao != bo
        if not diff.any():
            continue
        u = np.minimum(ao[diff], bo[diff]).astype(np.int64)
        v = np.maximum(ao[diff], bo[diff]).astype(np.int64)
        packed_all.append(u * base + v)
    neigh: dict[int, set] = {}
    if packed_all:
        packed = np.unique(np.concatenate(packed_all))
        us = (packed // base).tolist()
        vs = (packed % base).tolist()
        for u, v in zip(us, vs):
            neigh.setdefault(u, set()).add(v)
            neigh.setdefault(v, set()).add(u)
    return neigh, owner_types(owner, type_grid)


def side_stats(owner, type_grid):
    """Number-of-sides <n> and topological moments for bulk & total cells.

    'bulk'  cells = non-medium cells NOT in contact with the medium.
    'total' cells = all non-medium cells, counting the medium as a single side.
    Moments mu_l = mean over bulk cells of (n - <n>_bulk)^l, l=2,3,4  (Eq. 6).
    Returns dict: n_bulk, n_total, mu2, mu3, mu4, n_bulk_cells, n_total_cells.
    """
    neigh, _types_arr = adjacency(owner, type_grid)
    bulk_sides = []
    total_sides = []
    for o, nb in neigh.items():
        if o == MEDIUM:
            continue
        touches_medium = MEDIUM in nb
        n_cell_neighbors = len(nb - {MEDIUM})
        # 'total' counts the medium as one extra side when present
        total_sides.append(n_cell_neighbors + (1 if touches_medium else 0))
        if not touches_medium:
            bulk_sides.append(n_cell_neighbors)
    out = dict(n_bulk=float("nan"), n_total=float("nan"),
               mu2=float("nan"), mu3=float("nan"), mu4=float("nan"),
               n_bulk_cells=len(bulk_sides), n_total_cells=len(total_sides))
    if total_sides:
        out["n_total"] = float(np.mean(total_sides))
    if bulk_sides:
        b = np.asarray(bulk_sides, dtype=float)
        nbar = b.mean()
        out["n_bulk"] = float(nbar)
        d = b - nbar
        out["mu2"] = float(np.mean(d ** 2))
        out["mu3"] = float(np.mean(d ** 3))
        out["mu4"] = float(np.mean(d ** 4))
    return out


def type_correlations(owner, type_grid):
    """Type-type correlations at the cell-adjacency-graph level (Figs 13d, 21b).

    cell-cell edge fractions (ll/dd/ld among cell-cell adjacency edges) and
    medium-contact fractions (fraction of medium-adjacent cells that are
    light / dark).  Returns dict: corr_ll, corr_dd, corr_ld, corr_lM, corr_dM.
    """
    neigh, types_arr = adjacency(owner, type_grid)
    ll = dd = ld = 0
    seen_edges = set()
    med_light = med_dark = 0
    for o, nb in neigh.items():
        if o == MEDIUM:
            continue
        if MEDIUM in nb:
            if types_arr[o] == LIGHT:
                med_light += 1
            elif types_arr[o] == DARK:
                med_dark += 1
        for m in nb:
            if m == MEDIUM or m == o:
                continue
            e = (min(o, m), max(o, m))
            if e in seen_edges:
                continue
            seen_edges.add(e)
            ta, tb = types_arr[o], types_arr[m]
            if ta == tb == LIGHT:
                ll += 1
            elif ta == tb == DARK:
                dd += 1
            else:
                ld += 1
    cc = (ll + dd + ld) or 1
    med = (med_light + med_dark) or 1
    return dict(corr_ll=ll / cc, corr_dd=dd / cc, corr_ld=ld / cc,
                corr_lM=med_light / med, corr_dM=med_dark / med)


# ---------------------------------------------------------------------------
# one-call bundle
# ---------------------------------------------------------------------------

@dataclass
class Observables:
    boundary: dict = field(default_factory=dict)
    fractional: dict = field(default_factory=dict)
    sides: dict = field(default_factory=dict)
    correlations: dict = field(default_factory=dict)

    def flat(self):
        d = {}
        d.update({f"bl_{k}": v for k, v in self.boundary.items()})
        d.update({f"frac_{k}": v for k, v in self.fractional.items()})
        d.update(self.sides)
        d.update(self.correlations)
        return d


def measure(world):
    """Compute the full Glazier-Graner observable bundle for a 2D world."""
    owner, type_grid = grids(world)
    bl = boundary_lengths(owner, type_grid)
    return Observables(
        boundary=bl,
        fractional=fractional_lengths(bl),
        sides=side_stats(owner, type_grid),
        correlations=type_correlations(owner, type_grid),
    )


def measure_grid(owner, type_grid):
    """Same as measure() but from pre-extracted (owner, type_grid) arrays."""
    bl = boundary_lengths(owner, type_grid)
    return Observables(
        boundary=bl,
        fractional=fractional_lengths(bl),
        sides=side_stats(owner, type_grid),
        correlations=type_correlations(owner, type_grid),
    )
