# Basement Membrane (E3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a basement-membrane anchor to the Rust CPM core — a fixed membrane surface epithelial cells adhere to, as a new energy term keeping anchored cells in a thin band hugging the membrane — so the E2 crypt monolayer survives a longer/hotter relaxation without curling, thickening, or perforating.

**Architecture:** A pure distance-field + cost module (`membrane.rs`) feeds a new `delta_membrane` energy term on `World`, summed into the Metropolis accept test in `sweep.rs` (guarded by `any_membrane()` for zero overhead when off), exactly mirroring how E1 connectivity was added. pyo3 setters + a schema section + a Python metric + a before/after crypt demo complete the slice.

**Tech Stack:** Rust (cpm-core) + pyo3 (cpm-py, `maturin develop`) + Python (`cpm` pkg).

## Global Constraints

- The membrane term is a PURE energy term: it changes only the accept probability. It must NOT touch volume/surface/COM trackers or `apply_flip`. A rejected/expensive flip is handled exactly like any Metropolis rejection.
- Zero overhead when unused: `any_membrane()` (distance field non-empty AND ≥1 type anchored) short-circuits the term, like `any_connectivity()`.
- Deterministic, RNG-free: the distance field (multi-source BFS) and `delta_membrane` consult no RNG and no hash-iteration order. A fixed seed reproduces results.
- Setters must be callable before `finalize` and survive the finalize handoff — delegate through `world_mut()` (the connectivity/contacts pattern), since the core `World` (which holds the membrane state) is moved into `Cpm` at finalize.
- Anchor cost per voxel: `cost(d) = k * max(0, d - band)^2` where `d` = membrane distance; 0 within `band`, quadratic beyond. Medium (cell 0) and unanchored types cost 0.
- Build/test in the repo `.venv` (py3.12); rebuild the extension with `maturin develop -m crates/cpm-py/Cargo.toml` after Rust changes. Rust: `cargo test -p cpm-core`. pytest `pythonpath=["."]`.
- Cell type ids for the crypt demo follow E2: 1=Epithelial Stem, 2=Absorptive, 3=Goblet.

---

## File Structure

- `crates/cpm-core/src/membrane.rs` (new) — pure `build_distance_field` + `cost`.
- `crates/cpm-core/src/lib.rs` — `pub mod membrane;`.
- `crates/cpm-core/src/world.rs` — membrane state fields + setters + `any_membrane` + `delta_membrane`.
- `crates/cpm-core/src/sweep.rs` — add `delta_membrane` into the `dh` sum.
- `crates/cpm-core/tests/membrane.rs` (new) — behavioral property + determinism.
- `crates/cpm-py/src/lib.rs` — pyo3 `set_membrane` / `set_membrane_anchored`.
- `cpm/schema.py` — `spec["membrane"]` section.
- `cpm/metrics.py` — `membrane_distance_field` + `mean_membrane_distance`.
- `demos/run_membrane_demo.py` (new) — before/after crypt demo.
- `viewer/viewer.js` — a `membrane` BLURB entry.
- Tests: `tests/test_membrane.py`.

---

### Task 1: Membrane distance field, cost, and the `delta_membrane` energy term

**Files:**
- Create: `crates/cpm-core/src/membrane.rs`
- Modify: `crates/cpm-core/src/lib.rs`, `crates/cpm-core/src/world.rs`
- Test: unit tests inside `membrane.rs`

**Interfaces:**
- Produces: `cpm_core::membrane::build_distance_field(dims: [usize;3], anchors: &[usize]) -> Vec<f32>` and `cpm_core::membrane::cost(d: f32, k: f64, band: f64) -> f64`.
- Produces on `World`: `set_membrane(&mut self, anchors: &[usize], k: f64, band: f64)`, `set_membrane_anchored(&mut self, cell_type: u16, on: bool)`, `any_membrane(&self) -> bool`, `delta_membrane(&self, site: usize, new_owner: CellId) -> f64`.

- [ ] **Step 1: Write the failing unit tests**

Create `crates/cpm-core/src/membrane.rs`:

```rust
//! Basement-membrane anchor for the CPM sweep. `build_distance_field` is a pure
//! multi-source BFS (6-neighbour graph distance) from a fixed set of anchor
//! voxels; `cost` is the per-voxel anchor penalty (0 within `band`, quadratic
//! beyond). `World::delta_membrane` uses them to bias the Metropolis accept test
//! so anchored cells stay in a thin band hugging the membrane. RNG-free.
use std::collections::VecDeque;

/// Distance (6-neighbour graph steps) from every voxel to the nearest anchor.
/// Voxels unreachable from any anchor (only when `anchors` is empty) are +inf.
pub fn build_distance_field(dims: [usize; 3], anchors: &[usize]) -> Vec<f32> {
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let n = nx * ny * nz;
    let mut dist = vec![f32::INFINITY; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &a in anchors {
        if a < n && dist[a] != 0.0 {
            dist[a] = 0.0;
            queue.push_back(a);
        }
    }
    while let Some(v) = queue.pop_front() {
        let d = dist[v] + 1.0;
        let z = v / (nx * ny);
        let rem = v % (nx * ny);
        let y = rem / nx;
        let x = rem % nx;
        let mut nb: Vec<usize> = Vec::with_capacity(6);
        if x + 1 < nx { nb.push(v + 1); }
        if x >= 1 { nb.push(v - 1); }
        if y + 1 < ny { nb.push(v + nx); }
        if y >= 1 { nb.push(v - nx); }
        if z + 1 < nz { nb.push(v + nx * ny); }
        if z >= 1 { nb.push(v - nx * ny); }
        for w in nb {
            if d < dist[w] {
                dist[w] = d;
                queue.push_back(w);
            }
        }
    }
    dist
}

/// Anchor cost for a voxel at membrane distance `d`: 0 within `band`, quadratic
/// beyond, scaled by stiffness `k`.
#[inline]
pub fn cost(d: f32, k: f64, band: f64) -> f64 {
    let over = d as f64 - band;
    if over > 0.0 { k * over * over } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};
    use crate::world::World;

    #[test]
    fn bfs_line_distance() {
        assert_eq!(build_distance_field([5, 1, 1], &[0]), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bfs_two_anchors_take_min() {
        assert_eq!(build_distance_field([5, 1, 1], &[0, 4]), vec![0.0, 1.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn bfs_empty_anchors_all_infinite() {
        assert!(build_distance_field([3, 1, 1], &[]).iter().all(|x| x.is_infinite()));
    }

    #[test]
    fn cost_zero_within_band_quadratic_beyond() {
        assert_eq!(cost(1.0, 5.0, 2.0), 0.0);
        assert_eq!(cost(2.0, 5.0, 2.0), 0.0);
        assert_eq!(cost(4.0, 5.0, 2.0), 20.0); // 5 * (4-2)^2
    }

    #[test]
    fn delta_membrane_sign_and_gating() {
        // 5x1x1 line; membrane anchored at index 0 -> dist = [0,1,2,3,4].
        // band=1, k=1 so cost(d) = max(0, d-1)^2.
        let lat = Lattice::new([5, 1, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 5.0, 1.0, 0.0, 0.0);   // type 1
        let b = w.add_cell(2, 5.0, 1.0, 0.0, 0.0);   // type 2 (left unanchored)
        w.paint(0, a);
        w.set_membrane(&[0], 1.0, 1.0);
        w.set_membrane_anchored(1, true);            // only type 1 feels the anchor
        w.recompute_trackers();

        // Site 4 (dist 4) currently medium. Assigning it to `a` (anchored) costs
        // cost(4)-cost(medium) = (4-1)^2 - 0 = 9 > 0.
        assert!((w.delta_membrane(4, a) - 9.0).abs() < 1e-9);
        // Assigning site 4 to `b` (unanchored) costs 0.
        assert_eq!(w.delta_membrane(4, b), 0.0);
        // A site within band (dist 1) assigned to `a` costs 0.
        assert_eq!(w.delta_membrane(1, a), 0.0);
        // No type anchored -> any_membrane false.
        let mut w2 = w;
        w2.set_membrane_anchored(1, false);
        assert!(!w2.any_membrane());
    }
}
```

- [ ] **Step 2: Run, verify it fails**

Run: `cargo test -p cpm-core membrane`
Expected: FAIL — `membrane` module / `set_membrane` / `delta_membrane` not defined.

- [ ] **Step 3: Register the module**

In `crates/cpm-core/src/lib.rs`, add after `pub mod connectivity;`:

```rust
pub mod membrane;
```

- [ ] **Step 4: Add membrane state + methods to `World`**

In `crates/cpm-core/src/world.rs`, add four fields to the `World` struct (after `connectivity_medium: bool,`):

```rust
    pub membrane_dist: Vec<f32>,
    pub membrane_k: f64,
    pub membrane_band: f64,
    pub membrane_types: Vec<bool>,
```

In `World::new`, initialise them in the returned `World { ... }` literal (after `connectivity_medium: false,`):

```rust
            membrane_dist: Vec::new(),
            membrane_k: 0.0,
            membrane_band: 0.0,
            membrane_types: Vec::new(),
```

Add these methods in the `impl World` block (near `set_connectivity`):

```rust
    pub fn set_membrane(&mut self, anchors: &[usize], k: f64, band: f64) {
        let dims = [self.lattice.dims_x(), self.lattice.dims_y(), self.lattice.dims_z()];
        self.membrane_dist = crate::membrane::build_distance_field(dims, anchors);
        self.membrane_k = k;
        self.membrane_band = band;
    }

    pub fn set_membrane_anchored(&mut self, cell_type: u16, on: bool) {
        let t = cell_type as usize;
        if t >= self.membrane_types.len() {
            self.membrane_types.resize(t + 1, false);
        }
        self.membrane_types[t] = on;
    }

    pub fn any_membrane(&self) -> bool {
        !self.membrane_dist.is_empty() && self.membrane_types.iter().any(|&b| b)
    }

    fn membrane_type_anchored(&self, cell_type: u16) -> bool {
        self.membrane_types.get(cell_type as usize).copied().unwrap_or(false)
    }

    /// Membrane anchor energy change for reassigning `site` to `new_owner`.
    /// Only `site` changes membership, so this is cost(site,new) - cost(site,old).
    pub fn delta_membrane(&self, site: usize, new_owner: CellId) -> f64 {
        if self.membrane_dist.is_empty() {
            return 0.0;
        }
        let d = self.membrane_dist[site];
        let target = self.lattice.owner(site);
        let cost_for = |c: CellId| -> f64 {
            if c == crate::MEDIUM {
                return 0.0;
            }
            if self.membrane_type_anchored(self.cells[c as usize].cell_type) {
                crate::membrane::cost(d, self.membrane_k, self.membrane_band)
            } else {
                0.0
            }
        };
        cost_for(new_owner) - cost_for(target)
    }
```

- [ ] **Step 5: Run the tests, verify they pass**

Run: `cargo test -p cpm-core membrane`
Expected: PASS (5 tests). Then `cargo test -p cpm-core` — full core suite still green (no regressions).

- [ ] **Step 6: Commit**

```bash
git add crates/cpm-core/src/membrane.rs crates/cpm-core/src/lib.rs crates/cpm-core/src/world.rs
git commit -m "feat(membrane): distance-field + cost + delta_membrane energy term"
```

---

### Task 2: Wire the membrane term into the sweep + behavioral property test

**Files:**
- Modify: `crates/cpm-core/src/sweep.rs`
- Test: `crates/cpm-core/tests/membrane.rs` (new)

**Interfaces:**
- Consumes: `World::any_membrane`, `World::delta_membrane`, `cpm_core::membrane::build_distance_field` (Task 1).

- [ ] **Step 1: Write the failing property test**

Create `crates/cpm-core/tests/membrane.rs`:

```rust
use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::membrane::build_distance_field;
use cpm_core::sweep::Cpm;
use cpm_core::world::World;

const DIMS: [usize; 3] = [12, 12, 8];

fn anchors_z0() -> Vec<usize> {
    // the whole z=0 plane is the membrane surface
    let (nx, ny) = (DIMS[0], DIMS[1]);
    (0..nx * ny).collect()
}

fn mean_membrane_dist(cpm: &Cpm, cell: u32, dist: &[f32]) -> f64 {
    let n = cpm.world.lattice.n_sites();
    let (mut sum, mut cnt) = (0.0f64, 0u64);
    for i in 0..n {
        if cpm.world.lattice.owner(i) == cell {
            sum += dist[i] as f64;
            cnt += 1;
        }
    }
    if cnt == 0 { 0.0 } else { sum / cnt as f64 }
}

fn run(anchored: bool) -> f64 {
    let anchors = anchors_z0();
    let lat = Lattice::new(DIMS, [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 20.0); // hot
    // a 64-voxel type-1 cell seeded as a 4x4x4 block sitting a bit above z=0
    let a = w.add_cell(1, 64.0, 1.0, 0.0, 0.0);
    for z in 2..6 {
        for y in 4..8 {
            for x in 4..8 {
                let i = w.lattice.index(x, y, z);
                w.paint(i, a);
            }
        }
    }
    w.set_membrane(&anchors, 4.0, 2.0);
    if anchored {
        w.set_membrane_anchored(1, true);
    }
    w.recompute_trackers();
    let mut cpm = Cpm::new(w, 7);
    cpm.step(40);
    let dist = build_distance_field(DIMS, &anchors);
    mean_membrane_dist(&cpm, a, &dist)
}

#[test]
fn membrane_anchor_holds_cell_near_surface() {
    let anchored = run(true);
    let free = run(false);
    // z=0-plane anchors -> membrane distance == z. band=2, so an anchored cell
    // settles into z<=~2-3; a free hot cell wanders further up.
    assert!(anchored <= 3.5, "anchored cell drifted off the membrane: {anchored}");
    assert!(free > anchored + 0.5, "membrane made no difference: free {free} vs anchored {anchored}");
}

#[test]
fn membrane_run_is_deterministic() {
    assert_eq!(run(true), run(true));
}
```

- [ ] **Step 2: Run, verify the property test fails**

Run: `cargo test -p cpm-core --test membrane`
Expected: FAIL — the membrane term is not yet in the sweep, so `anchored` and `free` are ~equal and the assertion `free > anchored + 0.5` fails.

- [ ] **Step 3: Add the term to the sweep**

In `crates/cpm-core/src/sweep.rs`, change the `dh` computation:

```rust
                let dh = self.world.delta_hamiltonian(s, source_owner)
                    + self.world.delta_chemotaxis(s, pick, source_owner)
                    + if self.world.any_membrane() {
                        self.world.delta_membrane(s, source_owner)
                    } else {
                        0.0
                    };
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cargo test -p cpm-core --test membrane`
Expected: PASS (2 tests). Then `cargo test -p cpm-core` — full suite green.
If `membrane_anchor_holds_cell_near_surface` is flaky at the margin, the mechanism is still correct; nudge stiffness `k` up (e.g. 4.0 → 8.0) or band down — do NOT loosen the assertion so far it no longer distinguishes anchored from free.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/src/sweep.rs crates/cpm-core/tests/membrane.rs
git commit -m "feat(membrane): sum delta_membrane into the sweep + anchor property test"
```

---

### Task 3: pyo3 setters, schema section, and the Python metric

**Files:**
- Modify: `crates/cpm-py/src/lib.rs`, `cpm/schema.py`, `cpm/metrics.py`
- Test: `tests/test_membrane.py`

**Interfaces:**
- Produces (pyo3, on the `World` Python class): `set_membrane(anchors: list[int], k: float, band: float)` and `set_membrane_anchored(cell_type: int, on: bool)`.
- Produces (`cpm.metrics`): `membrane_distance_field(dims, anchors) -> list[float]` and `mean_membrane_distance(world, dist_field, anchored_types) -> float`.
- Consumes: the E1 schema pattern in `cpm/schema.py` (a `spec[...]` section applied before `finalize`).

- [ ] **Step 1: Write the failing Python test**

Create `tests/test_membrane.py`:

```python
import cpm_core
from cpm.schema import load_world
from cpm.metrics import membrane_distance_field, mean_membrane_distance, interior_medium_pockets


def _flat_sheet_labels(nx, ny, nz, z_sheet):
    # one 1-cell-thick type-1 sheet at height z_sheet, tiled into a few cells
    labels = [0] * (nx * ny * nz)
    seg_to_type = {}
    sid = 1
    for y in range(ny):
        for x in range(nx):
            # 2x2 patches -> distinct cells
            seg = 1 + (x // 2) + (y // 2) * (nx // 2)
            seg_to_type[seg] = 1
            labels[x + y * nx + z_sheet * nx * ny] = seg
    return labels, seg_to_type


def test_membrane_schema_wires_and_holds_sheet():
    nx = ny = 12
    nz = 8
    z_sheet = 4
    labels, seg_to_type = _flat_sheet_labels(nx, ny, nz, z_sheet)
    anchors = [x + y * nx + z_sheet * nx * ny for y in range(ny) for x in range(nx)]
    # NOTE: this is the ACTUAL cpm/schema.py shape — a "potts" wrapper (dims,
    # boundary, neighbor_order, temperature, seed); seed_labels with keys
    # labels/types/default_type/target_volume/lambda_volume; and contact as a
    # list of {a,b,j} dicts. The new membrane block is {anchors,k,band,types}.
    spec = {
        "potts": {"dims": [nx, ny, nz], "boundary": "noflux", "neighbor_order": 2,
                  "temperature": 18.0, "seed": 3},
        "seed_labels": {"labels": labels, "types": seg_to_type, "default_type": 1,
                        "target_volume": 4.0, "lambda_volume": 1.0},
        "contact": [{"a": 0, "b": 1, "j": 6.0}, {"a": 1, "b": 1, "j": 2.0}],
        "membrane": {"anchors": anchors, "k": 6.0, "band": 1.0, "types": [1]},
    }
    w = load_world(spec)
    dist = membrane_distance_field((nx, ny, nz), anchors)
    w.step(30)
    d_anchored = mean_membrane_distance(w, dist, {1})

    # same seed/stress WITHOUT the membrane
    import copy
    spec2 = copy.deepcopy(spec); spec2.pop("membrane")
    w2 = load_world(spec2)
    w2.step(30)
    d_free = mean_membrane_distance(w2, dist, {1})

    assert d_anchored < d_free, f"membrane did not hold the sheet: {d_anchored} vs {d_free}"
    assert d_anchored <= 1.5, f"anchored sheet drifted off the membrane: {d_anchored}"
```

Note: the membrane block shape (`anchors`/`k`/`band`/`types`) is the new contract; everything else in the spec mirrors the EXISTING `cpm/schema.py` keys verbatim (verify against the file while implementing Step 4).

- [ ] **Step 2: Run, verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_membrane.py -v`
Expected: FAIL — `membrane_distance_field`/`mean_membrane_distance` missing and/or `set_membrane` not on the world.

- [ ] **Step 3: Add the pyo3 setters**

In `crates/cpm-py/src/lib.rs`, in the `#[pymethods]` block near `set_connectivity`, add:

```rust
    fn set_membrane(&mut self, anchors: Vec<usize>, k: f64, band: f64) {
        self.world_mut().set_membrane(&anchors, k, band);
    }

    fn set_membrane_anchored(&mut self, cell_type: u16, on: bool) {
        self.world_mut().set_membrane_anchored(cell_type, on);
    }
```

Rebuild the extension:

```bash
maturin develop -m crates/cpm-py/Cargo.toml
```

- [ ] **Step 4: Add the schema section**

In `cpm/schema.py`, inside `load_world`, AFTER the connectivity section and BEFORE `finalize` (match the existing ordering/style for reading optional sections), add:

```python
    mem = spec.get("membrane")
    if mem:
        world.set_membrane(list(mem["anchors"]), float(mem["k"]), float(mem["band"]))
        for t in mem.get("types", []):
            world.set_membrane_anchored(int(t), True)
```

(Place this using the same `world` variable name and pre-finalize location the connectivity block uses — read the surrounding code and match it.)

- [ ] **Step 5: Add the Python metrics**

Append to `cpm/metrics.py`:

```python
def membrane_distance_field(dims, anchors):
    """Multi-source BFS (6-neighbour graph distance) from anchor voxels to every
    voxel. Mirrors cpm_core::membrane::build_distance_field for tests/demos."""
    from collections import deque
    nx, ny, nz = dims
    n = nx * ny * nz
    dist = [float("inf")] * n
    q = deque()
    for a in anchors:
        if 0 <= a < n and dist[a] != 0.0:
            dist[a] = 0.0
            q.append(a)
    while q:
        v = q.popleft()
        d = dist[v] + 1.0
        z, rem = divmod(v, nx * ny)
        y, x = divmod(rem, nx)
        nb = []
        if x + 1 < nx: nb.append(v + 1)
        if x >= 1: nb.append(v - 1)
        if y + 1 < ny: nb.append(v + nx)
        if y >= 1: nb.append(v - nx)
        if z + 1 < nz: nb.append(v + nx * ny)
        if z >= 1: nb.append(v - nx * ny)
        for w in nb:
            if d < dist[w]:
                dist[w] = d
                q.append(w)
    return dist


def mean_membrane_distance(world, dist_field, anchored_types):
    """Mean membrane distance over the voxels of cells whose type is in
    `anchored_types` (a set of type ids). ~<= band when the anchor holds."""
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    types = world.cell_types()
    total, count = 0.0, 0
    for i, c in enumerate(labels):
        if c != 0 and types[c] in anchored_types:
            total += dist_field[i]
            count += 1
    return total / count if count else 0.0
```

- [ ] **Step 6: Run the test, verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_membrane.py -v`
Expected: PASS. If the anchored sheet doesn't hold at `k=6.0`, raise `k` in the test spec (the mechanism is validated in Rust Task 2; here we only need the schema wiring + metric to demonstrate the difference) — but keep `d_anchored < d_free` as the load-bearing assertion.

- [ ] **Step 7: Commit**

```bash
git add crates/cpm-py/src/lib.rs cpm/schema.py cpm/metrics.py tests/test_membrane.py
git commit -m "feat(membrane): pyo3 setters + schema section + Python membrane metrics"
```

---

### Task 4: Before/after crypt demo + viewer blurb

**Files:**
- Create: `demos/run_membrane_demo.py`
- Modify: `viewer/viewer.js`

**Interfaces:**
- Consumes: `cpm.crypt3d.build_crypt3d` (E2), `cpm.metrics` (`radial_cell_counts`, `interior_medium_pockets`, `connected_components`, `membrane_distance_field`, `mean_membrane_distance`), the world API (`set_membrane`, `set_membrane_anchored`, `set_target_volume`, connectivity setters), and the 3D `voxels` export + manifest conventions from `demos/run_crypt3d.py`.
- Produces: `viewer/data/membrane.json` (`kind="membrane"`, `is3d=True`) + manifest entry; exits nonzero on any failed gate.

- [ ] **Step 1: Write the demo**

Create `demos/run_membrane_demo.py`:

```python
"""Basement membrane (E3a): the E2 crypt monolayer, run under a relaxation
STRONGER than E2 survives, twice on the same seed -- once WITHOUT a membrane
(control: it detaches/thickens or the lumen breaches) and once WITH the basal
shell anchored as a basement membrane (it stays a coherent monolayer hugging the
membrane). Shows the anchor is what holds the structure. Validates + exports.

Usage (repo root, venv active):  python demos/run_membrane_demo.py
"""
import json
import os
import sys

import cpm_core

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cpm.crypt3d import build_crypt3d
from cpm.metrics import (radial_cell_counts, interior_medium_pockets,
                         connected_components, membrane_distance_field,
                         mean_membrane_distance)

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
ANCHORED_TYPES = {1, 2, 3}   # all epithelial types feel the membrane
BAND = 2.0
K = 12.0
TEMP = 10.0                  # hotter than E2 (which used 4.0) -> control fails
N_FRAMES, MCS_PER_FRAME = 8, 4


def surface_3d(w):
    nx, ny, nz = w.dims()
    lab = w.snapshot()
    def owner(x, y, z):
        return lab[x + y * nx + z * nx * ny]
    out = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                c = owner(x, y, z)
                if c == 0:
                    continue
                for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    x2, y2, z2 = x+dx, y+dy, z+dz
                    if not (0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz) or owner(x2,y2,z2) != c:
                        out.append([x, y, z, c]); break
    return out


def shell_anchors(labels, dims):
    """Membrane = the seeded shell's own footprint (the basal lamina the
    epithelium sits on). Anchoring to it with band ~ wall keeps cells in the
    membrane shell."""
    return [i for i, v in enumerate(labels) if v != 0]


def build_world(dims, labels, seg_to_type, anchors, with_membrane):
    nx, ny, nz = dims
    w = cpm_core.World((nx, ny, nz), "noflux", 2, TEMP)
    w.seed_from_labels(labels, seg_to_type, 1, 20.0, 20.0)
    for t in range(1, 4):
        w.set_contact(0, t, 6.0)
        for u in range(t, 4):
            w.set_contact(t, u, 4.0)
        w.set_connectivity(t, True)
    w.set_connectivity_medium(True)
    if with_membrane:
        w.set_membrane(anchors, K, BAND)
        for t in ANCHORED_TYPES:
            w.set_membrane_anchored(t, True)
    w.finalize(1)
    vols0 = w.cell_volumes()
    for c in range(1, len(vols0)):
        if vols0[c] > 0:
            w.set_target_volume(c, float(vols0[c]))
    return w


def relax(w, dims, dist, capture):
    """Step the world, tracking worst-case structure over every frame. Returns a
    dict of aggregates (+ frames if capture)."""
    nx, ny, nz = dims
    frames = []
    prev = None
    min_pockets = 10 ** 9
    worst_mean, worst_p90, worst_mem = 0.0, 0, 0.0
    min_step_churn = None
    for f in range(N_FRAMES + 1):
        snap = w.snapshot()
        if prev is not None:
            churn = sum(1 for a, b in zip(prev, snap) if a != b)
            min_step_churn = churn if min_step_churn is None else min(min_step_churn, churn)
        prev = snap
        counts = radial_cell_counts(w, nx / 2.0, ny / 2.0)
        worst_mean = max(worst_mean, sum(counts) / len(counts))
        worst_p90 = max(worst_p90, counts[int(0.90 * len(counts))])
        worst_mem = max(worst_mem, mean_membrane_distance(w, dist, ANCHORED_TYPES))
        min_pockets = min(min_pockets, interior_medium_pockets(w))
        if capture:
            frames.append({"mcs": f * MCS_PER_FRAME, "voxels": surface_3d(w)})
        if f < N_FRAMES:
            w.step(MCS_PER_FRAME)
    types = w.cell_types(); vols = w.cell_volumes()
    frag = sum(1 for c in range(1, len(types)) if vols[c] > 0 and connected_components(w, c) != 1)
    alive = sum(1 for c in range(1, len(types)) if vols[c] > 0)
    return {"worst_mean": worst_mean, "worst_p90": worst_p90, "worst_mem": worst_mem,
            "min_pockets": min_pockets, "min_step_churn": min_step_churn or 0,
            "frag": frag, "alive": alive, "n0": w.n_cells(),
            "cell_types": list(types), "frames": frames}


def main():
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d(wall=3)
    dims = (nx, ny, nz)
    anchors = shell_anchors(labels, dims)
    dist = membrane_distance_field(dims, anchors)

    ctrl = relax(build_world(dims, labels, seg_to_type, anchors, False), dims, dist, capture=False)
    mem = relax(build_world(dims, labels, seg_to_type, anchors, True), dims, dist, capture=True)

    control_degrades = ctrl["min_pockets"] == 0 or ctrl["worst_mean"] >= 1.5
    checks = [
        (f"CONTROL (no membrane) degrades under the stress -- proves the anchor "
         f"does the work (min pockets {ctrl['min_pockets']}, worst mean {ctrl['worst_mean']:.2f})",
         control_degrades),
        (f"WITH membrane: single-cell monolayer holds (worst mean {mem['worst_mean']:.2f} < 1.5, "
         f"worst p90 {mem['worst_p90']} <= 2)", mem["worst_mean"] < 1.5 and mem["worst_p90"] <= 2),
        (f"WITH membrane: lumen stays enclosed (min interior pockets {mem['min_pockets']} >= 1)",
         mem["min_pockets"] >= 1),
        (f"WITH membrane: cells stay anchored (worst mean membrane distance "
         f"{mem['worst_mem']:.2f} <= band {BAND})", mem["worst_mem"] <= BAND),
        (f"WITH membrane: no cell fragments ({mem['frag']} split) and all survive "
         f"({mem['alive']}/{mem['n0']})", mem["frag"] == 0 and mem["alive"] == mem["n0"]),
        (f"WITH membrane: relaxation is non-trivial (min per-step churn "
         f"{mem['min_step_churn']} > 0 -- anchored, not frozen)", mem["min_step_churn"] > 0),
    ]

    data = {"name": "Basement Membrane (crypt)", "kind": "membrane", "dims": [nx, ny, nz],
            "is3d": True, "n_cells": mem["n0"], "cell_types": mem["cell_types"],
            "type_names": ["Medium"] + type_names, "frames": mem["frames"]}
    os.makedirs(DATA, exist_ok=True)
    json.dump(data, open(os.path.join(DATA, "membrane.json"), "w"))
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    manifest = [m for m in manifest if m["file"] != "membrane.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "membrane.json", "name": data["name"], "is3d": True,
                     "n_cells": data["n_cells"], "dims": data["dims"], "kind": "membrane",
                     "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json", "hra_ftu.json", "crypt_differentiation.json",
             "connectivity_2d.json", "connectivity_3d.json", "connectivity_gap.json",
             "crypt3d.json", "membrane.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (basement membrane) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the demo, tune the contrast**

Run: `source .venv/bin/activate && python demos/run_membrane_demo.py`
Expected: all 6 gates PASS, exit 0; writes `viewer/data/membrane.json` + manifest entry.

This demo has TWO tuning targets that must BOTH hold:
1. The **control must genuinely fail** (gate 1): the stress (TEMP, MCS) must be strong enough that the un-anchored crypt degrades. If gate 1 fails (control still looks fine), raise `TEMP` (e.g. 10 → 14) or `N_FRAMES`.
2. The **membrane must hold** (gates 2-6): raise `K` (stiffness) if the anchored monolayer thickens/breaches; but if `K` is so high the shell freezes (gate 6, min-step churn 0), lower it. There must be a window where the anchor holds the monolayer WHILE it still relaxes — find it. Do NOT weaken a gate or shrink the run to trivially pass.
Report the final TEMP/K/BAND/MCS you settled on and the control-vs-membrane numbers.

- [ ] **Step 3: Add the viewer blurb**

In `viewer/viewer.js`, add a `membrane` entry to the `BLURB` object (after the `crypt3d` entry):

```javascript
  membrane: "Basement membrane (Sub-project E3a) — the crypt monolayer anchored to " +
    "its basal surface by a new CPM energy term (anchored cells pay energy to leave a " +
    "thin band hugging the membrane). Under a relaxation the free-standing E2 shell " +
    "can't survive, the anchored monolayer stays coherent on its membrane — the physics " +
    "that keeps the structure consistent. Junctions (E3b) and per-cell mechanics (E3c) follow.",
```

- [ ] **Step 4: Commit**

```bash
git add demos/run_membrane_demo.py viewer/viewer.js viewer/data/membrane.json viewer/data/index.json
git commit -m "feat(demo): basement-membrane before/after crypt demo + viewer blurb"
```

---

## Self-Review

**Spec coverage:** distance field + cost (`membrane.rs`) → Task 1; `delta_membrane` energy term + World state/setters/`any_membrane` → Task 1; sweep integration with `any_membrane` short-circuit → Task 2; Rust unit (distance/cost/delta sign/gating) → Task 1; Rust property (anchored holds vs free drifts) + determinism → Task 2; pyo3 setters surviving finalize → Task 3; schema `spec["membrane"]` → Task 3; Python `membrane_distance_field` + `mean_membrane_distance` → Task 3; Python schema-wiring test (with vs without) → Task 3; before/after crypt demo with the six gates (control degrades; monolayer holds; lumen enclosed; cells anchored within band; no fragmentation + survival; non-trivial relaxation) + viewer export → Task 4.

**Placeholder scan:** no TBD/TODO. The two genuine tuning points (Rust property margin in Task 2; demo control-fails-AND-membrane-holds window in Task 4) are called out with concrete knobs (k/band/TEMP/MCS) and an explicit "do not weaken the gate" rule, not vague hand-waving.

**Type consistency:** `build_distance_field(dims: [usize;3], anchors: &[usize]) -> Vec<f32>`, `cost(d: f32, k: f64, band: f64) -> f64`, `delta_membrane(site, new_owner) -> f64`, `any_membrane() -> bool`, the four `World` fields, pyo3 `set_membrane(anchors, k, band)` / `set_membrane_anchored(cell_type, on)`, schema `{"anchors","k","band","types"}`, `membrane_distance_field(dims, anchors)`, `mean_membrane_distance(world, dist_field, anchored_types)`, and the demo `kind="membrane"` + `order` entry are consistent across Tasks 1–4 and match the E1/E2 idioms (per-type bool + `any_*` short-circuit; `world_mut()` finalize-surviving setters; per-frame worst-case gating).
```
