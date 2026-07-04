# Sub-project E3a — Basement Membrane (basal anchor energy)

**Status:** design approved 2026-07-04
**Umbrella:** [pbg-cpm design](2026-07-04-pbg-cpm-design.md), Sub-project E (tissue
mechanics). Builds on E1 (connectivity) and E2 (3D crypt structure).

## Goal

Add a **basement-membrane anchor** to the Rust CPM core: a fixed membrane surface
that epithelial cells adhere to, implemented as a new energy term that keeps
anchored cells occupying a thin band hugging the membrane. This is the physics
E2 found missing — a free-standing single-cell monolayer curls, thickens, and
perforates because nothing holds the sheet to a substrate. The membrane anchor
lets the crypt monolayer survive a longer, hotter relaxation than E2 could,
staying a coherent monolayer on its basal surface.

This is the first of three E3 slices; E3b (junction springs) and E3c (per-cell
mechanical models wired through the subcellular framework) follow and are out of
scope here.

## Non-goals (this spec)

- Junction springs / cell-cell adhesion links — spec **E3b**.
- Per-cell mechanical models as Composite processes — spec **E3c**.
- A deformable/growing membrane. The membrane surface is FIXED (precomputed
  once); modelling membrane remodelling under growth is future work.
- Viewer changes beyond exporting the demo model (the 3D voxel renderer already
  displays it; the membrane surface may optionally be exported as context voxels).

## Background: why E2 was metastable

E2 showed (its own gates + the whole-branch review) that E1 connectivity keeps
every cell whole and forbids interior-medium pinch-off, but it does NOT resist:
(a) the sheet curling/thickening (adhesion-driven surface tension rounds a thin
shell up), or (b) a thin wall breaching (a breach only makes medium *more*
connected, which connectivity permits). Real epithelia resist both because they
are anchored to a **basement membrane**. E3a adds that anchor.

## Design

### The membrane surface

A membrane is a fixed set of lattice voxels — the **anchor surface** — supplied
by the caller (for the crypt: the outer/basal surface voxels of the seeded shell;
for a flat-sheet test: the substrate plane). The core precomputes, once, a
distance field `membrane_dist[v]` = distance from every lattice voxel `v` to the
nearest anchor voxel, via a **multi-source BFS** over the 6-neighbour lattice
graph (integer graph distance — an approximation to Euclidean, adequate because
the penalty is a tunable band; documented as such, RNG-free, deterministic given
sorted anchor seeds).

### The anchor energy

For each voxel `v` owned by an **anchored** cell, the membrane contributes

```
cost(v) = k * max(0, membrane_dist[v] - band)^2
```

- Within `band` of the membrane the cost is 0 — anchored cells relax freely in a
  shell of half-thickness `band` hugging the membrane.
- Beyond `band` the cost grows quadratically — an anchored cell cannot stack up
  (thicken away from the membrane) or drift/detach without paying energy.

The band is two-sided (unsigned distance), so it also penalises bulging to the
far side of the membrane; with `band ≈ wall thickness` this keeps a monolayer
pinned in the membrane shell. `k` sets the anchor stiffness.

Medium (cell 0) and unanchored cell types contribute 0.

### Incremental delta (hot loop)

A copy attempt reassigns site `s` from `target = owner(s)` to `source_owner`.
The membrane energy changes only at `s`:

```
delta_membrane(s, source_owner) = cost_for(s, source_owner) - cost_for(s, target)
cost_for(s, c) = 0                                   if c == 0 or type(c) not anchored
               = k * max(0, membrane_dist[s] - band)^2   otherwise
```

`Cpm::step` adds `delta_membrane` into the accept test's `dh`, alongside
`delta_hamiltonian` and `delta_chemotaxis`. When no membrane is set (or no type
is anchored) the term is skipped entirely (`any_membrane()` short-circuit), so
unmembraned runs pay nothing. Like connectivity, this is a pure energy term —
it changes only the accept probability; trackers/energy bookkeeping for
volume/surface/COM are untouched.

### Configuration and API

- **Rust (`World`)**: store `membrane_dist: Vec<f32>` (empty when unset),
  `membrane_k: f64`, `membrane_band: f64`, `membrane_types: Vec<bool>` (indexed
  by cell_type). Add:
  - `set_membrane(&mut self, anchors: &[usize], k: f64, band: f64)` — builds the
    distance field (multi-source BFS) and stores params.
  - `set_membrane_anchored(&mut self, cell_type: u16, on: bool)`
  - `any_membrane(&self) -> bool` (dist field non-empty AND some type anchored)
  - `delta_membrane(&self, site: usize, new_owner: CellId) -> f64`
- **`crates/cpm-core/src/membrane.rs` (new)**: the pure distance-field builder
  (`build_distance_field(dims, anchors) -> Vec<f32>`) and the pure `cost`
  function, kept separate from `world.rs` and testable in isolation. `World`
  delegates here. Wired via `mod membrane;` in `lib.rs`.
- **`crates/cpm-core/src/sweep.rs`**: add `+ self.world.delta_membrane(s, pick)`
  to the `dh` sum, guarded by `any_membrane()` for zero overhead when off.
- **pyo3 (`crates/cpm-py/src/lib.rs`)**: `set_membrane(anchors, k, band)` and
  `set_membrane_anchored(cell_type, on)`, callable before `finalize` and
  surviving the finalize handoff (mirror the connectivity/contacts pattern).
- **Schema (`cpm/schema.py`)**: optional `spec["membrane"] = {"anchors": [...],
  "k": ..., "band": ..., "types": [1, 2, 3]}`, read before `finalize`.
- **`cpm/metrics.py`**: `mean_membrane_distance(world, anchors, types)` — mean
  `membrane_dist` over the voxels of anchored cells (a Python re-derivation for
  tests/demos; the demo uses it to show anchored cells stay near the membrane).
  Reuse the existing `radial_cell_counts` / `interior_medium_pockets` for the
  monolayer + lumen gates.

## Demo and validation

`demos/run_membrane_demo.py` produces a viewer model AND validates (exits
nonzero on any failed gate), following the demo/manifest conventions
(`kind="membrane"`, `checks`, `index.json` order). It runs the E2 crypt under a
relaxation stronger than E2 survives (higher temperature and/or more MCS), twice
on the same seed:

1. **Control (no membrane)** — the monolayer detaches/thickens: gate asserts the
   control genuinely degrades (worst mean cells-per-ray climbs above the monolayer
   bound, OR the lumen breaches — proving the stress is real and the membrane is
   doing the work).
2. **With membrane** (outer basal shell anchored, epithelial types anchored):
   gates —
   - single-cell monolayer holds (worst-frame mean radial cells `< 1.5` and
     worst-frame p90 `<= 2`, per E2's robust measure);
   - lumen stays enclosed (`interior_medium_pockets >= 1` throughout);
   - anchored cells stay near the membrane (`mean_membrane_distance <= band`
     throughout — the anchor actually holds);
   - relaxation is non-trivial (per-step voxel churn `> 0` every step — the
     shell is genuinely relaxing, not frozen, exactly the E2 anti-vacuity gate);
   - no cell fragments (E1 invariant still holds under the new term).

Exports 3D `voxels` frames (coloured by type) plus, optionally, the membrane
anchor surface as context voxels of a distinct id, for the viewer.

## Testing

- **Rust unit (`membrane.rs`)**: `build_distance_field` — an anchor voxel has
  distance 0, a 6-neighbour has 1, a far voxel has the expected graph distance;
  a two-anchor field takes the min. `cost`/`delta`: a voxel within `band` costs
  0; beyond `band` costs `k*(d-band)^2`; moving a voxel from within-band to
  beyond-band yields positive `delta_membrane`; an unanchored type or medium
  yields 0; `any_membrane` false when unset or no type anchored.
- **Rust property (`tests/`)**: a cell anchored to a membrane under high
  temperature keeps `mean membrane distance` bounded (≤ band + small slack) over
  many MCS, while the SAME seed/stress with the membrane OFF drifts further
  (guards against a vacuous test). Determinism: membrane ON is reproducible under
  a fixed seed (distance field + delta consult no RNG/hash iteration order).
- **Python (`tests/test_membrane.py`)**: schema wires `spec["membrane"]`; an
  anchored stressed crypt stays a monolayer with bounded membrane distance, and
  on the same seed WITHOUT the membrane it degrades, using the metrics.

## Risks / open points

- **Graph-distance approximation**: 6-neighbour BFS overestimates diagonal
  distances vs true Euclidean. Acceptable because `band`/`k` are tunable and the
  anchor is a soft band, not a hard geometric constraint; documented. A true
  Euclidean distance transform is a future refinement if needed.
- **Two-sided band vs one-sided**: unsigned distance penalises both sides of the
  membrane equally. For a monolayer sitting on one side this is slightly
  stronger than physical (it also resists the cell poking through to the far
  side, which is harmless/desirable here). A signed (basal-only) anchor is a
  future refinement; not needed for E3a's goal.
- **Fixed membrane**: the surface does not remodel. Under strong growth the
  tissue could in principle grow along the membrane; membrane remodelling is out
  of scope and flagged for a later slice.
- **Stiffness tuning**: too-large `k` freezes the sheet (the E2 lesson — a frozen
  run is a vacuous "relaxation"); the demo's per-step-churn gate guards against
  that, and `k` is tuned so the monolayer relaxes while staying anchored.
