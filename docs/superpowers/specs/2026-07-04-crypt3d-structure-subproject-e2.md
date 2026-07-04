# Sub-project E2 — 3D Crypt Structure

**Status:** design approved 2026-07-04
**Umbrella:** [pbg-cpm design](2026-07-04-pbg-cpm-design.md), Sub-project E (tissue mechanics). Builds on E1 (connectivity constraint).

## Goal

Build a structurally-consistent, single-cell-thick 3D epithelial **crypt** — a
procedural test-tube shell (hollow cylinder closed by a hemispherical base) —
held together as a coherent monolayer by the E1 connectivity constraint plus
volume/surface/adhesion, with cells typed by axial position (stem niche at the
base). This is the geometry foundation for running the Sub-project D
differentiation Composite in 3D (spec **E2b**).

## Non-goals (this spec)

- Running the SBML/Boolean differentiation Composite in 3D — that is **E2b**.
- Junction springs, basement membrane, per-cell mechanical models — spec **E3**.
  E2 demonstrates short-run structural consistency; indefinite/under-growth
  monolayer stability needs E3's basement membrane and is out of scope.
- New viewer code — the existing 3D instanced-voxel renderer already displays
  the exported model.

## Design

### Geometry generator (`cpm/crypt3d.py`)

`build_crypt3d(radius=8, cyl_height=28, wall=3, cell_pitch=6, margin=4)` returns
`((nx, ny, nz), labels, seg_to_type, type_names)` — a 3D label grid ready for
`seed_from_labels`, plus the FTU-aligned type palette.

- **Shape.** Axis along `z` through the lattice centre `(cx, cy)`. The wall is
  the set of voxels within `wall/2` of the ideal surface: for the cylinder
  (`z >= z_base`) the surface is `r == radius` (`r = hypot(x-cx, y-cy)`); for the
  hemispherical cap (`z < z_base`) it is distance `radius` from the cap centre
  `(cx, cy, z_base)`. Interior of the surface is lumen (medium); exterior is
  medium too.
- **Axial parameter `a`.** Arc length along the profile from the basal pole:
  cap voxels use `a = radius * phi` where `phi` is the polar angle from the
  pole; cylinder voxels use `a = radius * (pi/2) + (z - z_base)`.
- **Tiling into cells.** Bin the shell by `(axial_bin, theta_bin)` where
  `axial_bin = floor(a / cell_pitch)` and the number of angular bins scales with
  the local circumference (`max(1, round(2*pi*r_local / cell_pitch))`) so cells
  stay ~equal-area and the pole gets few cells, not slivers. Each non-empty
  `(axial_bin, theta_bin)` is one cell with a unique consecutive label.
- **Typing by axial position.** Reuse the HRA crypt FTU type names
  (Epithelial Stem, Absorptive, Goblet, …). Basal band (cap + lowest cylinder)
  → Epithelial Stem; middle → Absorptive (progenitor); upper (toward the lumen
  opening) → a Goblet/Absorptive mix. Store the axial→type mapping so tests can
  assert ordering.

### CPM assembly (in the demo)

Seed from the generated labels; set cohesive cell-cell adhesion + costly medium
contact so cells stay packed on the shell; a firm volume constraint
(`lambda_volume`) with target = generated cell size; connectivity ON for all
cell types AND the medium (E1). Run a SHORT, low-temperature relaxation so the
seeded shell settles without collapsing. The connectivity + medium constraints
keep every cell whole and the lumen enclosed.

### Metrics (`cpm/metrics.py`, additions)

- `radial_thickness(world, cx, cy)` → for a sample of `(theta, z)` rays,
  the number of distinct cells crossed between lumen and exterior; returns the
  mean and max over samples (≈ 1 for a monolayer).
- Reuse `interior_medium_pockets` (E1): the lumen is an enclosed interior
  medium component, so a value `>= 1` means the wall still separates lumen from
  exterior (no breach); a wall breach connects lumen to the border and drops it.

## Validation (demo exits nonzero on any failed gate)

`demos/run_crypt3d.py` builds, relaxes, validates, and exports:

1. **Monolayer**: `radial_thickness` max `<= 2` (single-cell wall, allowing one
   cell of tolerance) and mean `< 1.5`.
2. **No fragmentation**: every cell is exactly one connected component (E1
   `connected_components == 1` for all).
3. **Lumen enclosed / no wall breach**: `interior_medium_pockets >= 1`
   throughout (the lumen never breaches to the exterior).
4. **Axial type order**: mean `z` of Epithelial-Stem cells < mean `z` of the
   differentiated (Goblet/Absorptive-upper) cells (stem niche is basal).
5. **Structure persists**: no cell vanishes over the run (all initial cells keep
   nonzero volume); cell count constant.

Exports 3D `voxels` frames (coloured by type) for the existing viewer.

## Testing

- `tests/test_crypt3d.py`:
  - geometry: `build_crypt3d` produces a plausible cell count (> 30), a thin
    shell (generated labels have `radial_thickness` max `<= 2` at t=0), Stem
    cells basal (axial type order holds on the generated labels), and lumen
    present (`interior_medium_pockets >= 1`).
  - the CPM round-trip: seeding from the labels yields `n_cells` == number of
    generated cells; a short relaxation with connectivity keeps every cell one
    component.
- Determinism: fixed seed; `build_crypt3d` is pure (no RNG) or seeded.

## Risks / open points

- **Monolayer collapse**: a thin shell tends to round up (surface tension) over
  a long run; E2 uses a short, low-temperature relaxation and strong volume +
  connectivity so it holds for the validated window. Indefinite stability under
  growth is E3 (basement membrane) — documented, not solved here.
- **Pole slivers**: circumference-scaled angular binning avoids degenerate tiny
  cells at the cap pole; the generator drops any cell that rasterises to `< 3`
  voxels (as the FTU rasteriser does).
- **Thickness metric sampling**: `radial_thickness` samples rays rather than
  every voxel; the sample density must be high enough to catch a thin breach —
  the demo uses a dense `(theta, z)` grid and the gate is the max over samples.
