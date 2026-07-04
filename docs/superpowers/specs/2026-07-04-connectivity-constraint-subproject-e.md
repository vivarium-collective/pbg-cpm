# Sub-project E, Spec 1 — CPM Connectivity Constraint

**Status:** design approved 2026-07-04
**Umbrella:** [pbg-cpm design](2026-07-04-pbg-cpm-design.md), Sub-project E (tissue mechanics / structural integrity)

## Goal

Add a **connectivity constraint** to the Rust CPM core that forbids any copy
attempt which would fragment a cell — or, optionally, the medium. This is the
foundational physics for structural integrity: cells cannot split into
disconnected blobs, cannot pinch off or "merge" by fragmenting, and (with the
medium constrained) cannot trap interior medium pockets, i.e. gaps. It is the
reusable primitive that makes a held-together 3D crypt (spec E2) and junction
springs / basement membrane (spec E3) viable.

## Non-goals (this spec)

- 3D crypt geometry generation — that is spec **E2**.
- Junction springs / focal-point plasticity, basement membrane, per-cell
  mechanical constraints wired through the subcellular framework — spec **E3**.
- A globally exact fragmentation guarantee. We implement the standard *local*
  connectivity test (cheap, per-attempt), which prevents the common
  fragmentation modes and matches CompuCell3D's practical approach. Pathological
  global cases are out of scope and documented.
- Viewer changes. The demos export the integrity metric in their validation
  `checks`; no new viewer UI is required.

## Background: where fragmentation comes from

In the Metropolis sweep (`crates/cpm-core/src/sweep.rs`, `Cpm::step`), a copy
attempt takes a site `s` (owned by `target = owner(s)`) and reassigns it to a
neighbouring cell `source_owner`. Repeatedly removing boundary pixels can, under
high temperature or strong de-adhesion, disconnect a cell into two blobs
(fragmentation) or let medium intrude between cells (gaps). Standard CPM has no
topology awareness; the connectivity constraint adds it.

## Design

### Hook point

In `Cpm::step`, immediately before `self.world.apply_flip(s, source_owner)`,
gate the flip:

```
if constraint active for target = owner(s):
    if removing s would locally disconnect target: reject (skip apply_flip)
if medium constrained and source_owner would extend medium / target is medium:
    apply the same test to the medium
```

A rejected attempt is simply not applied (the sweep continues), exactly like a
Metropolis rejection. The connectivity test runs only when a constraint is
active for the affected type, so unconstrained runs pay nothing.

### The local connectivity predicate

`World::would_stay_connected(s, target) -> bool` decides whether removing pixel
`s` from cell `target` keeps `target` locally connected. It is a local test over
the neighbourhood of `s` — O(neighbourhood size), never a global flood-fill:

1. Collect the neighbours of `s` (using the lattice's configured neighbourhood)
   that are owned by `target`. Call this set `N`.
2. If `|N| <= 1`, removal cannot disconnect `target` locally → return `true`.
3. Otherwise, build the adjacency graph among the members of `N` (two members
   are adjacent iff they are neighbours of each other in the lattice), and count
   its connected components **without routing through `s`**.
4. If there is exactly one component → `true` (safe). If more than one → `s` is
   a local articulation point of `target`; removing it may disconnect the cell →
   return `false` (reject the flip).

This is the Durand & Guesnet (2016) local test, equivalent in 2D to counting
contiguous arcs of `target` around the Moore ring (>1 arc ⇒ articulation) and in
3D to connected-components over the `target` voxels in the 26-neighbourhood.

### Medium connectivity (optional)

The medium is `CellId` 0. When `medium` connectivity is enabled, the same
predicate is applied to `target == 0`: a flip that would locally disconnect the
medium (pinch off an interior pocket) is rejected. This is what prevents interior
gaps forming in a confluent tissue. Enabling it does not forbid all gaps that
stay connected to the exterior medium — only the pinching-off of new interior
pockets — which is the correct, well-defined guarantee.

### Configuration and API

- **Rust (`World`)**: store a per-type boolean set `connectivity_types:
  Vec<bool>` (indexed by `cell_type`; sized to `max_type + 1`) plus a
  `connectivity_medium: bool`. Add:
  - `set_connectivity(&mut self, cell_type: u16, on: bool)`
  - `set_connectivity_medium(&mut self, on: bool)`
  - `would_stay_connected(&self, site: usize, target: CellId) -> bool`
  - a helper `any_connectivity(&self) -> bool` so `Cpm::step` can skip the check
    entirely when nothing is constrained.
  `Cpm::step` consults `world` before each `apply_flip`.
- **pyo3 (`crates/cpm-py/src/lib.rs`)**: `set_connectivity(cell_type, on)` and
  `set_connectivity_medium(on)`. These must be callable before `finalize`
  (stored on the core world like contacts) — decide storage so they survive the
  `finalize` handoff (mirror how contacts are collected then applied).
- **Schema (`cpm/schema.py`)**: read an optional
  `spec["connectivity"] = {"types": [1, 2], "medium": true}` after cells/fields
  and before `finalize`, calling the new setters.

### Interaction with existing constraints

Connectivity is a hard gate applied *after* the Metropolis accept decision, so
it only ever turns accepted flips into rejects — it cannot change energy
bookkeeping, and the incremental volume/surface/COM trackers are untouched
(a rejected flip calls neither `apply_flip` nor any tracker update). It composes
with volume/surface/contact/chemotaxis unchanged.

## Components / files

- `crates/cpm-core/src/connectivity.rs` (new) — the `would_stay_connected`
  predicate and its 2D/3D local component count, kept separate from `world.rs`
  so the topology logic is testable in isolation. `World` methods delegate here.
- `crates/cpm-core/src/world.rs` — connectivity state fields + setters +
  `any_connectivity`; `mod connectivity;` wired in `lib.rs`.
- `crates/cpm-core/src/sweep.rs` — the gate before `apply_flip`.
- `crates/cpm-py/src/lib.rs` — pyo3 setters (survive `finalize`).
- `cpm/schema.py` — `connectivity` spec section.
- `cpm/metrics.py` — a `connected_components(world, cell_id)` helper (global
  flood-fill, for TESTS/DEMOS only — not used in the hot loop) and an
  `interior_medium_pockets(world)` helper.
- `demos/run_connectivity_demos.py` (new) — the before/after integrity demos.
- Tests: `crates/cpm-core/tests/` (predicate + property), `tests/test_connectivity.py`.

## Demos and validation

`demos/run_connectivity_demos.py` produces viewer models AND validates (exits
nonzero on any failed gate), following the existing demo/manifest conventions
(`kind`, `checks`, `index.json` order):

1. **2D anti-fragmentation.** One cell in a high-temperature / strongly
   de-adhesive world that WITHOUT connectivity fragments. Gates:
   - without constraint: final `connected_components(cell) > 1` (it really does
     fragment — proves the stress is real);
   - with constraint: `connected_components(cell) == 1` at every captured frame.
2. **3D anti-fragmentation.** The same in 3D (26-neighbourhood). Same two gates.
3. **Confluent no-gap (medium).** A packed 2D tissue that WITHOUT medium
   connectivity develops interior medium pockets. Gates:
   - without: `interior_medium_pockets > 0`;
   - with medium connectivity: `interior_medium_pockets == 0` throughout.

Each demo captures frames for the viewer (reusing the 2D labels / 3D voxel
export already used by the other demos) so the integrity difference is visible.

## Testing

- **Rust unit (`connectivity.rs`)**: hand-built neighbourhoods where the removed
  site IS a local articulation point (predicate returns `false`) and where it is
  NOT (`true`), for both 2D Moore and 3D 26-neighbourhood, including the
  `|N| <= 1` early-out and a same-cell ring (single arc → `true`).
- **Rust property (`tests/`)**: run a small world with connectivity ON for a
  cell under high temperature for many MCS; after every MCS the cell has exactly
  one connected component (global flood-fill check). With connectivity OFF on
  the same seed/stress, at least one MCS produces >1 component (guards against a
  vacuous test / the stress being too weak).
- **Determinism**: connectivity ON is deterministic under a fixed seed (same
  final volumes across two runs) — the predicate must not consult any RNG or
  hash iteration order.
- **Python (`tests/test_connectivity.py`)**: schema wires the constraint
  (`spec["connectivity"]`); a stressed cell stays one component with it and (on
  the same seed) fragments without it, using `cpm.metrics.connected_components`.

## Risks / open points

- **Local-vs-global**: the local predicate rejects removals at local
  articulation points; a rare configuration could still disconnect a cell whose
  connectivity routes entirely outside the neighbourhood of `s`. This is the
  known limitation of the cheap local test (Durand & Guesnet) and is acceptable
  and documented; the property test measures the practical fragmentation rate
  (expected zero in the demos).
- **pyo3 finalize handoff**: the setters must record connectivity state such
  that it survives `finalize` (which moves the core world into `Cpm`). Mirror the
  `contacts` pattern (collect on the wrapper, apply into the core at `finalize`),
  or set directly on the core world if the setter is called post-`finalize` —
  pick one and keep it consistent with how contacts are handled.
- **Performance**: the predicate runs per accepted attempt only when a
  constraint is active; `any_connectivity()` short-circuits unconstrained runs to
  zero overhead. The 3D 26-neighbourhood component count is the heaviest path and
  should be measured, but micro-optimisation is out of scope here.
