# Sub-project E3b — Cell–Cell Junctions (contact-conservation / anti-gap)

**Status:** design approved 2026-07-04
**Umbrella:** [pbg-cpm design](2026-07-04-pbg-cpm-design.md), Sub-project E (tissue
mechanics). Builds on E1 (connectivity), E2 (3D crypt), E3a (basement membrane).

## Goal

Add **cell–cell junctions** to the Rust CPM core: a new energy term that keeps
junction-enabled neighbouring cells knit together, so a gap (a thin medium film /
perforation) cannot open between them. This is the physics E2 and E3a leave
unsolved — E1 connectivity forbids a cell fragmenting or the medium pinching off
an interior pocket, and E3a anchors cells to the basal membrane, but neither
prevents a thin medium finger breaching the wall between two cells (a breach only
makes the medium *more* connected, which connectivity permits). Junctions seal
that, completing the "keep the structure consistent, without gaps, without
merging" goal from the crypt question.

This is the second of three E3 slices; E3c (per-cell mechanical models wired
through the subcellular framework) follows and is out of scope here.

## Non-goals (this spec)

- FocalPointPlasticity centre-of-mass springs (an alternative junction model that
  keeps neighbour *spacing* stable but does not directly seal gaps). Not chosen.
- Per-cell-pair dynamic link registries (which specific cell instances are
  bonded, link formation/breaking history). Junctions are enabled per cell TYPE
  (like connectivity), not per instance — a simple, state-function design.
- Per-cell mechanical models as Composite processes — spec **E3c**.
- Viewer changes beyond exporting the demo model.

## Design

### Why a *state* energy, not a transition penalty

The natural phrasing "penalise breaking a cell–cell junction into cell–medium"
is a per-*transition* rule, which is not a valid Hamiltonian (it has no state
function whose difference it is, so Metropolis detailed balance is undefined).
We use the equivalent **state** form: penalise medium that sits *pinched between
two different junction-enabled cells*. Opening a gap creates pinched medium (a
film between two cells); a perforation channel is pinched perpendicular to its
axis; closing the gap removes it. This is exactly "a junction was broken," but as
a well-defined energy.

Crucially this penalises ONLY medium-between-cells, never free surface (medium
with medium on the far side), so — unlike plain strong adhesion / a more negative
contact J — it does **not** add surface tension and does **not** cause the sheet
to curl or round up (the E2 failure mode).

### The junction energy

- Per-type flag `junction_types: Vec<bool>` (indexed by cell_type) + a global
  stiffness `lambda_junction: f64`.
- **State energy** `E_j = lambda_junction * P`, where `P` is the number of
  *pinches*: for every medium voxel `m` and every lattice axis (X, Y, and Z when
  3D), a pinch is counted when the two opposite axis-neighbours of `m` are both
  in-bounds, both non-medium, both of junction-enabled types, and belong to
  **different** cells. (Same-cell sandwiching is a cell hole, already forbidden
  by E1 connectivity, so "different cells" isolates true inter-cell gaps.)

A confluent tissue with cells touching directly has `P = 0`. A gap/film/breach
between two junctioned cells raises `P`, so it costs `lambda_junction` per pinched
face — the tissue resists opening gaps. The free apical/basal/exterior surface
(medium backed by medium) contributes nothing.

### Incremental delta (hot loop)

A flip reassigns site `s` from `old = owner(s)` to `new = source_owner`. Only
`s`'s membership changes, so `P` changes only at pinch-centres that involve `s`:
`s` itself (a pinch-centre only while it is medium) and each 6-neighbour of `s`
that is medium (for which `s` is one axis-neighbour). Define a local
`pinch_at(centre, s, new)` that counts pinches at a medium `centre` evaluating
`owner(s)` as `new` (and every other site normally). Then

```
delta_junction(s, new) = lambda_junction * (
      [pinch_at(s, s, new) if new==MEDIUM else 0] - [pinch_at(s, s, old) if old==MEDIUM else 0]
    + sum over 6-neighbours n of s with owner(n)==MEDIUM of
        (pinch_at(n, s, new) - pinch_at(n, s, old))
)
```

O(neighbourhood) per attempt, only when a junction is active. Added to the
Metropolis `dh`, guarded by `any_junction()` (some type enabled AND
`lambda_junction > 0`) for zero overhead when unused. Like connectivity/membrane
it is a pure energy term — it changes only the accept probability and never
touches volume/surface/COM trackers or `apply_flip`.

### Configuration and API

- **Rust (`World`)**: `junction_types: Vec<bool>`, `lambda_junction: f64`. Add
  `set_junction(cell_type, on)`, `set_junction_lambda(lambda)`,
  `any_junction() -> bool`, `delta_junction(site, new_owner) -> f64`, and a
  private `pinch_at(centre, s, new) -> u32` helper.
- **`crates/cpm-core/src/junction.rs` (new)**: the pure pinch predicate — given a
  medium centre's axis-neighbour owners+types, count pinched axes — kept separate
  and unit-testable; `World` delegates the per-site geometry to it.
- **`crates/cpm-core/src/sweep.rs`**: add the guarded `delta_junction` term to
  `dh`.
- **pyo3 (`crates/cpm-py/src/lib.rs`)**: `set_junction(cell_type, on)` (bump
  `max_type`, mirror `set_connectivity`) and `set_junction_lambda(lambda)`,
  surviving `finalize` via `world_mut()`.
- **Schema (`cpm/schema.py`)**: optional `spec["junctions"] = {"types": [1,2,3],
  "lambda": 8.0}`, read before `finalize`.
- **`cpm/metrics.py`**: `intercell_gap_faces(world, junction_types)` — a Python
  re-derivation counting pinched medium faces (the same quantity `P`), for
  tests/demos to show gaps opening WITHOUT junctions vs sealed WITH.

## Demo and validation

`demos/run_junction_demo.py` runs the E2 crypt under a relaxation hot enough that
the wall perforates WITHOUT junctions, twice on the same seed, and validates
(exits nonzero on any failed gate), following the manifest conventions
(`kind="junction"`, per-frame worst-case gates):

1. **Control (no junctions) develops gaps / breaches**: the lumen breaches
   (`interior_medium_pockets` drops to 0 at some frame) OR `intercell_gap_faces`
   rises materially — proving the stress really opens gaps and the junctions do
   the work.
2. **With junctions**: gaps stay sealed —
   - lumen stays enclosed (`interior_medium_pockets >= 1` every frame);
   - `intercell_gap_faces` stays low (near its initial value, and strictly below
     the control's worst) throughout;
   - the monolayer still holds (worst-frame mean radial cells `< 1.5`,
     worst-frame p90 `<= 2`) — junctions must seal gaps WITHOUT curling the sheet;
   - no cell fragments and all survive (E1 invariant intact);
   - relaxation is non-trivial (min per-step voxel churn `> 0` — sealed by
     junctions, not frozen).

Exports 3D `voxels` frames for the viewer + a `junction` BLURB entry.

## Testing

- **Rust unit (`junction.rs` / `world.rs`)**: the pinch predicate counts a pinch
  for medium flanked by two different junction-enabled cells and NOT for
  medium-backed-by-medium, an unanchored/none-enabled type, or a single cell on
  one side only; `delta_junction` sign — a flip that opens a one-voxel gap between
  two junctioned cells yields positive delta, closing it yields negative, and a
  flip on the free surface yields 0; `any_junction` false when lambda 0 or no
  type enabled.
- **Rust property (`tests/junction.rs`)**: two junctioned cells pressed together
  under high temperature keep `intercell_gap_faces == 0` (no gap opens) with
  junctions ON, while the SAME seed/stress with junctions OFF opens at least one
  gap face (guards against a vacuous test). Determinism under a fixed seed.
- **Python (`tests/test_junction.py`)**: schema wires `spec["junctions"]`; a
  stressed two-cell (or small tissue) config stays gap-free with junctions and
  develops gaps without, via `cpm.metrics.intercell_gap_faces`.

## Risks / open points

- **Pinch is a local 1-voxel test**: it catches thin films/perforations (the
  common gap mode) but a wide, multi-voxel cavity's interior medium is
  medium-backed and not itself pinched — only its walls are. Acceptable: the term
  resists gap *opening* (which always starts as a pinch at the cell–cell seam);
  E1 connectivity_medium already forbids fully-enclosed cavities pinching off.
  Documented.
- **Stiffness tuning**: too-large `lambda_junction` could over-stabilise and, with
  the volume term, approach a freeze — the demo's per-step-churn gate guards
  against a vacuous "sealed by freezing" result, exactly as in E3a.
- **Type-level, not instance-level**: any two adjacent junction-enabled cells are
  treated as bonded; we do not model selective/dynamic bonding. Sufficient for a
  confluent epithelium; instance-level FPP springs are a documented future
  alternative.
