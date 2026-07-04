# Cell–Cell Junctions (E3b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cell–cell junction (contact-conservation / anti-gap) energy term to the Rust CPM core, so junction-enabled neighbouring cells cannot let a medium gap or perforation open between them — sealing the wall breaches E2/E3a leave open, without curling the sheet.

**Architecture:** A pure pinch predicate (`junction.rs`) feeds a `delta_junction` energy term on `World` — the state energy `E = λ·(pinched medium faces between two different junctioned cells)` — summed into the Metropolis accept test in `sweep.rs` (guarded by `any_junction()`), exactly mirroring E1 connectivity / E3a membrane. pyo3 setters + schema + a Python metric + a before/after crypt demo complete the slice.

**Tech Stack:** Rust (cpm-core) + pyo3 (cpm-py, `maturin develop`) + Python (`cpm` pkg).

## Global Constraints

- Pure energy term: changes only the accept probability; never touches volume/surface/COM trackers or `apply_flip`.
- Zero overhead when unused: `any_junction()` = (`lambda_junction > 0` AND ≥1 type enabled) short-circuits, like `any_connectivity()`/`any_membrane()`.
- Deterministic, RNG-free.
- Setters callable before `finalize` and surviving the handoff — delegate through `world_mut()`.
- **Pinch definition:** for a MEDIUM voxel and a lattice axis, a pinch is counted when the two opposite in-bounds axis-neighbours are both non-medium, both of junction-enabled types, and belong to DIFFERENT cells. `E_junction = lambda_junction * (total pinches)`. Free surface (medium backed by medium) contributes 0 — this is why it seals gaps without adding surface tension.
- Build/test in the repo `.venv` (py3.12); `maturin develop -m crates/cpm-py/Cargo.toml` after Rust changes. Rust: `cargo test -p cpm-core`. pytest `pythonpath=["."]`. Import the engine in Python as `from cpm import cpm_core`.
- Crypt demo cell type ids follow E2: 1=Epithelial Stem, 2=Absorptive, 3=Goblet.

---

## File Structure

- `crates/cpm-core/src/junction.rs` (new) — pure `axis_is_pinch`.
- `crates/cpm-core/src/lib.rs` — `pub mod junction;`.
- `crates/cpm-core/src/world.rs` — junction state + setters + `any_junction` + `pinch_at` + `delta_junction`.
- `crates/cpm-core/src/sweep.rs` — add `delta_junction` to the `dh` sum.
- `crates/cpm-core/tests/junction.rs` (new) — behavioural property + determinism.
- `crates/cpm-py/src/lib.rs` — pyo3 `set_junction` / `set_junction_lambda`.
- `cpm/schema.py` — `spec["junctions"]` section.
- `cpm/metrics.py` — `intercell_gap_faces`.
- `demos/run_junction_demo.py` (new) — before/after crypt demo.
- `viewer/viewer.js` — a `junction` BLURB entry.
- Tests: `tests/test_junction.py`.

---

### Task 1: Pinch predicate + `delta_junction` energy term

**Files:**
- Create: `crates/cpm-core/src/junction.rs`
- Modify: `crates/cpm-core/src/lib.rs`, `crates/cpm-core/src/world.rs`
- Test: unit tests inside `junction.rs`

**Interfaces:**
- Produces: `cpm_core::junction::axis_is_pinch(a: (u32, bool), b: (u32, bool)) -> bool`.
- Produces on `World`: `set_junction(&mut self, cell_type: u16, on: bool)`, `set_junction_lambda(&mut self, lambda: f64)`, `any_junction(&self) -> bool`, `delta_junction(&self, site: usize, new_owner: CellId) -> f64`.

- [ ] **Step 1: Write the failing unit tests**

Create `crates/cpm-core/src/junction.rs`:

```rust
//! Cell-cell junction (anti-gap) predicate for the CPM sweep. A "pinch" is a
//! medium voxel whose two opposite axis-neighbours are both junction-enabled
//! cells of DIFFERENT ids -- a thin medium film/gap between two bonded cells.
//! `axis_is_pinch` is the pure core; `World` supplies the per-site geometry.

/// One axis is pinched iff both sides are non-medium (`id != 0`) junction-enabled
/// (`true`) cells with different ids. Each side is (owner_id, is_junction_type);
/// medium is (0, false).
#[inline]
pub fn axis_is_pinch(a: (u32, bool), b: (u32, bool)) -> bool {
    let (ia, ja) = a;
    let (ib, jb) = b;
    ia != 0 && ib != 0 && ja && jb && ia != ib
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};
    use crate::world::World;

    #[test]
    fn axis_pinch_only_between_two_different_junction_cells() {
        assert!(axis_is_pinch((1, true), (2, true)));      // two diff junction cells
        assert!(!axis_is_pinch((1, true), (1, true)));     // same cell (a hole; E1's job)
        assert!(!axis_is_pinch((1, true), (0, false)));    // one side medium
        assert!(!axis_is_pinch((1, false), (2, true)));    // one side not junction-enabled
        assert!(!axis_is_pinch((0, false), (0, false)));   // free medium
    }

    #[test]
    fn delta_junction_opens_and_closes_gaps() {
        // 3x1x1: cell A | medium | cell B, both type 1 (junction-enabled), lambda 2.
        let lat = Lattice::new([3, 1, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 1.0, 1.0, 0.0, 0.0);
        let b = w.add_cell(1, 1.0, 1.0, 0.0, 0.0);
        w.paint(0, a);
        w.paint(2, b);                       // site 1 = medium -> a pinch exists
        w.set_junction(1, true);
        w.set_junction_lambda(2.0);
        w.recompute_trackers();

        // closing the gap (fill site 1 with A) removes the pinch -> favourable (-lambda)
        assert!((w.delta_junction(1, a) + 2.0).abs() < 1e-9);

        // fill it, then opening it back (A -> medium) creates the pinch -> +lambda
        w.paint(1, a);                       // A | A | B, no gap
        assert!((w.delta_junction(1, crate::MEDIUM) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn delta_junction_zero_on_free_surface() {
        // 3x1x1: cell A | medium | medium. Filling site 1 touches no pinch.
        let lat = Lattice::new([3, 1, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 1.0, 1.0, 0.0, 0.0);
        w.paint(0, a);                       // sites 1,2 medium
        w.set_junction(1, true);
        w.set_junction_lambda(2.0);
        w.recompute_trackers();
        assert_eq!(w.delta_junction(1, a), 0.0);   // no cell on the far side -> no pinch
    }

    #[test]
    fn any_junction_requires_lambda_and_a_type() {
        let lat = Lattice::new([3, 1, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        w.set_junction(1, true);
        assert!(!w.any_junction(), "lambda 0 -> off");
        w.set_junction_lambda(2.0);
        assert!(w.any_junction());
        w.set_junction(1, false);
        assert!(!w.any_junction(), "no type enabled -> off");
    }
}
```

- [ ] **Step 2: Run, verify it fails**

Run: `cargo test -p cpm-core junction`
Expected: FAIL — `junction` module / `set_junction` / `delta_junction` not defined.

- [ ] **Step 3: Register the module**

In `crates/cpm-core/src/lib.rs`, add after `pub mod membrane;`:

```rust
pub mod junction;
```

- [ ] **Step 4: Add junction state + methods to `World`**

In `crates/cpm-core/src/world.rs`, add two fields to the `World` struct (after the membrane fields):

```rust
    pub junction_types: Vec<bool>,
    pub lambda_junction: f64,
```

In `World::new`, initialise them in the returned literal (after the membrane inits):

```rust
            junction_types: Vec::new(),
            lambda_junction: 0.0,
```

Add these methods in the `impl World` block (near the membrane methods):

```rust
    pub fn set_junction(&mut self, cell_type: u16, on: bool) {
        let t = cell_type as usize;
        if t >= self.junction_types.len() {
            self.junction_types.resize(t + 1, false);
        }
        self.junction_types[t] = on;
    }

    pub fn set_junction_lambda(&mut self, lambda: f64) {
        self.lambda_junction = lambda;
    }

    pub fn any_junction(&self) -> bool {
        self.lambda_junction > 0.0 && self.junction_types.iter().any(|&b| b)
    }

    fn junction_type_enabled(&self, cell_type: u16) -> bool {
        self.junction_types.get(cell_type as usize).copied().unwrap_or(false)
    }

    // owner of `idx`, but treating site `s` as if it were `new`
    #[inline]
    fn owner_ov(&self, idx: usize, s: usize, new: CellId) -> CellId {
        if idx == s { new } else { self.lattice.owner(idx) }
    }

    // (owner_id, is_junction_type) for `idx` under the s->new override; medium = (0,false)
    #[inline]
    fn junction_tag(&self, idx: usize, s: usize, new: CellId) -> (u32, bool) {
        let o = self.owner_ov(idx, s, new);
        if o == crate::MEDIUM {
            (0, false)
        } else {
            (o, self.junction_type_enabled(self.cells[o as usize].cell_type))
        }
    }

    // number of pinched axes at medium centre `c`, under the s->new override
    fn pinch_at(&self, c: usize, s: usize, new: CellId) -> u32 {
        if self.owner_ov(c, s, new) != crate::MEDIUM {
            return 0; // only a medium voxel can be a pinch centre
        }
        let (nx, ny, nz) = (self.lattice.dims_x(), self.lattice.dims_y(), self.lattice.dims_z());
        let z = c / (nx * ny);
        let rem = c % (nx * ny);
        let y = rem / nx;
        let x = rem % nx;
        let mut n = 0u32;
        if x >= 1 && x + 1 < nx
            && crate::junction::axis_is_pinch(self.junction_tag(c - 1, s, new), self.junction_tag(c + 1, s, new))
        {
            n += 1;
        }
        if y >= 1 && y + 1 < ny
            && crate::junction::axis_is_pinch(self.junction_tag(c - nx, s, new), self.junction_tag(c + nx, s, new))
        {
            n += 1;
        }
        if nz > 1 && z >= 1 && z + 1 < nz
            && crate::junction::axis_is_pinch(
                self.junction_tag(c - nx * ny, s, new),
                self.junction_tag(c + nx * ny, s, new),
            )
        {
            n += 1;
        }
        n
    }

    /// Junction (anti-gap) energy change for reassigning `site` to `new_owner`.
    /// Only `site` changes owner, so pinch counts change only at `site` (a pinch
    /// centre while medium) and its medium 6-neighbours (for which `site` is one
    /// axis side). E = lambda_junction * total_pinches.
    pub fn delta_junction(&self, site: usize, new_owner: CellId) -> f64 {
        let old = self.lattice.owner(site);
        let mut d: i64 = self.pinch_at(site, site, new_owner) as i64
            - self.pinch_at(site, site, old) as i64;
        let (nx, ny, nz) = (self.lattice.dims_x(), self.lattice.dims_y(), self.lattice.dims_z());
        let z = site / (nx * ny);
        let rem = site % (nx * ny);
        let y = rem / nx;
        let x = rem % nx;
        let mut nb: Vec<usize> = Vec::with_capacity(6);
        if x + 1 < nx { nb.push(site + 1); }
        if x >= 1 { nb.push(site - 1); }
        if y + 1 < ny { nb.push(site + nx); }
        if y >= 1 { nb.push(site - nx); }
        if nz > 1 && z + 1 < nz { nb.push(site + nx * ny); }
        if nz > 1 && z >= 1 { nb.push(site - nx * ny); }
        for m in nb {
            if self.lattice.owner(m) == crate::MEDIUM {
                d += self.pinch_at(m, site, new_owner) as i64 - self.pinch_at(m, site, old) as i64;
            }
        }
        self.lambda_junction * d as f64
    }
```

- [ ] **Step 5: Run the tests, verify they pass**

Run: `cargo test -p cpm-core junction`
Expected: PASS (4 tests). Then `cargo test -p cpm-core` — full core suite still green.

- [ ] **Step 6: Commit**

```bash
git add crates/cpm-core/src/junction.rs crates/cpm-core/src/lib.rs crates/cpm-core/src/world.rs
git commit -m "feat(junction): pinch predicate + delta_junction anti-gap energy term"
```

---

### Task 2: Wire the junction term into the sweep + behavioural property test

**Files:**
- Modify: `crates/cpm-core/src/sweep.rs`
- Test: `crates/cpm-core/tests/junction.rs` (new)

**Interfaces:**
- Consumes: `World::any_junction`, `World::delta_junction` (Task 1).

- [ ] **Step 1: Write the failing property test**

Create `crates/cpm-core/tests/junction.rs`:

```rust
use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::sweep::Cpm;
use cpm_core::world::World;

const DIMS: [usize; 3] = [16, 16, 1];

// count pinched medium faces between two different junction-enabled cells (the P quantity)
fn gap_faces(w: &World, jt: &[bool]) -> u32 {
    let (nx, ny) = (DIMS[0], DIMS[1]);
    let enabled = |o: u32| o != 0 && *jt.get(w.cells[o as usize].cell_type as usize).unwrap_or(&false);
    let mut p = 0u32;
    for y in 0..ny {
        for x in 0..nx {
            let i = w.lattice.index(x, y, 0);
            if w.lattice.owner(i) != 0 {
                continue;
            }
            if x >= 1 && x + 1 < nx {
                let a = w.lattice.owner(i - 1);
                let b = w.lattice.owner(i + 1);
                if enabled(a) && enabled(b) && a != b { p += 1; }
            }
            if y >= 1 && y + 1 < ny {
                let a = w.lattice.owner(i - nx);
                let b = w.lattice.owner(i + nx);
                if enabled(a) && enabled(b) && a != b { p += 1; }
            }
        }
    }
    p
}

// two 3-wide cells pressed together as a 6x8 block, split down the middle
fn run(junctions: bool) -> u32 {
    let lat = Lattice::new(DIMS, [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 22.0); // hot -> boundary churns, gaps want to open
    let a = w.add_cell(1, 24.0, 1.0, 0.0, 0.0);
    let b = w.add_cell(1, 24.0, 1.0, 0.0, 0.0);
    for y in 4..12 {
        for x in 5..8 {
            w.paint(w.lattice.index(x, y, 0), a);
        }
        for x in 8..11 {
            w.paint(w.lattice.index(x, y, 0), b);
        }
    }
    // weak adhesion so the two cells don't strongly stick on their own
    let mut m = cpm_core::energy::ContactMatrix::new(2);
    m.set(0, 1, 4.0);
    m.set(1, 1, 4.0);
    w.set_contact_matrix(m);
    if junctions {
        w.set_junction(1, true);
        w.set_junction_lambda(12.0);
    }
    w.recompute_trackers();
    let jt = vec![false, junctions]; // index by cell_type: type 1 enabled iff junctions
    let mut cpm = Cpm::new(w, 5);
    let mut worst = 0u32;
    for _ in 0..30 {
        cpm.step(1);
        worst = worst.max(gap_faces(&cpm.world, &jt));
    }
    worst
}

#[test]
fn junctions_prevent_gaps_between_cells() {
    let with = run(true);
    let without = run(false);
    assert_eq!(with, 0, "junctions should keep the two cells gap-free, saw {with}");
    assert!(without > 0, "control must open a gap or the test is vacuous, saw {without}");
}

#[test]
fn junction_run_is_deterministic() {
    assert_eq!(run(true), run(true));
}
```

- [ ] **Step 2: Run, verify the property test fails**

Run: `cargo test -p cpm-core --test junction`
Expected: FAIL — the junction term is not in the sweep yet, so `with` opens gaps just like `without` and `assert_eq!(with, 0)` fails.

- [ ] **Step 3: Add the term to the sweep**

In `crates/cpm-core/src/sweep.rs`, extend the `dh` computation (it already sums `delta_hamiltonian` + `delta_chemotaxis` + the membrane term):

```rust
                let dh = self.world.delta_hamiltonian(s, source_owner)
                    + self.world.delta_chemotaxis(s, pick, source_owner)
                    + if self.world.any_membrane() {
                        self.world.delta_membrane(s, source_owner)
                    } else {
                        0.0
                    }
                    + if self.world.any_junction() {
                        self.world.delta_junction(s, source_owner)
                    } else {
                        0.0
                    };
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cargo test -p cpm-core --test junction`
Expected: PASS (2 tests). Then `cargo test -p cpm-core` — full suite green.
If `junctions_prevent_gaps_between_cells` is flaky at the margin (control occasionally shows 0 gaps at low heat, or the junction run shows a transient 1), raise the temperature (22 → 26) or `lambda_junction` (12 → 20) — do NOT loosen the `with == 0` / `without > 0` assertions so they stop distinguishing.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/src/sweep.rs crates/cpm-core/tests/junction.rs
git commit -m "feat(junction): sum delta_junction into the sweep + anti-gap property test"
```

---

### Task 3: pyo3 setters, schema section, and the Python metric

**Files:**
- Modify: `crates/cpm-py/src/lib.rs`, `cpm/schema.py`, `cpm/metrics.py`
- Test: `tests/test_junction.py`

**Interfaces:**
- Produces (pyo3, on the `World` Python class): `set_junction(cell_type: int, on: bool)` and `set_junction_lambda(lambda: float)`.
- Produces (`cpm.metrics`): `intercell_gap_faces(world, junction_types) -> int`.
- Consumes: the E1/E3a schema pattern (a `spec[...]` section applied before `finalize`).

- [ ] **Step 1: Write the failing Python test**

Create `tests/test_junction.py`:

```python
from cpm import cpm_core
from cpm.schema import load_world
from cpm.metrics import intercell_gap_faces


def _two_cell_block(nx, ny):
    # two type-1 cells pressed together (left half / right half of a block)
    labels = [0] * (nx * ny)
    seg_to_type = {1: 1, 2: 1}
    for y in range(4, ny - 4):
        for x in range(nx // 2 - 3, nx // 2):
            labels[x + y * nx] = 1
        for x in range(nx // 2, nx // 2 + 3):
            labels[x + y * nx] = 2
    return labels, seg_to_type


def _spec(nx, ny, labels, seg_to_type, junctions):
    spec = {
        "potts": {"dims": [nx, ny, 1], "boundary": "noflux", "neighbor_order": 2,
                  "temperature": 22.0, "seed": 5},
        "seed_labels": {"labels": labels, "types": seg_to_type, "default_type": 1,
                        "target_volume": 24.0, "lambda_volume": 1.0},
        "contact": [{"a": 0, "b": 1, "j": 4.0}, {"a": 1, "b": 1, "j": 4.0}],
    }
    if junctions:
        spec["junctions"] = {"types": [1], "lambda": 12.0}
    return spec


def test_junctions_wire_and_prevent_gaps():
    nx = ny = 16
    labels, seg_to_type = _two_cell_block(nx, ny)

    w = load_world(_spec(nx, ny, labels, seg_to_type, True))
    worst_with = 0
    for _ in range(30):
        w.step(1)
        worst_with = max(worst_with, intercell_gap_faces(w, {1}))

    w2 = load_world(_spec(nx, ny, labels, seg_to_type, False))
    worst_without = 0
    for _ in range(30):
        w2.step(1)
        worst_without = max(worst_without, intercell_gap_faces(w2, {1}))

    assert worst_with == 0, f"junctions should keep the cells gap-free, saw {worst_with}"
    assert worst_without > 0, f"control must open a gap (else vacuous), saw {worst_without}"
```

- [ ] **Step 2: Run, verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_junction.py -v`
Expected: FAIL — `intercell_gap_faces` missing and/or `set_junction` not on the world.

- [ ] **Step 3: Add the pyo3 setters**

In `crates/cpm-py/src/lib.rs`, in the `#[pymethods]` block near `set_membrane`, add:

```rust
    fn set_junction(&mut self, cell_type: u16, on: bool) {
        self.max_type = self.max_type.max(cell_type);
        self.world_mut().set_junction(cell_type, on);
    }

    fn set_junction_lambda(&mut self, lambda: f64) {
        self.world_mut().set_junction_lambda(lambda);
    }
```

Rebuild: `maturin develop -m crates/cpm-py/Cargo.toml`.

- [ ] **Step 4: Add the schema section**

In `cpm/schema.py`, AFTER the membrane section and BEFORE `finalize`, add:

```python
    jn = spec.get("junctions")
    if jn:
        for t in jn.get("types", []):
            world.set_junction(int(t), True)
        world.set_junction_lambda(float(jn.get("lambda", 0.0)))
```

- [ ] **Step 5: Add the Python metric**

Append to `cpm/metrics.py`:

```python
def intercell_gap_faces(world, junction_types):
    """Count pinched medium faces: a medium voxel whose two opposite axis-neighbours
    are both junction-enabled cells of DIFFERENT ids (a gap/film/perforation between
    two bonded cells). Mirrors the Rust E_junction quantity. `junction_types` is a
    set of type ids. 0 for a confluent tissue; rises as gaps open."""
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    types = world.cell_types()
    jt = set(junction_types)

    def enabled(o):
        return o != 0 and types[o] in jt

    total = 0
    for i, c in enumerate(labels):
        if c != 0:
            continue
        z, rem = divmod(i, nx * ny)
        y, x = divmod(rem, nx)
        if 1 <= x < nx - 1 and enabled(labels[i - 1]) and enabled(labels[i + 1]) \
                and labels[i - 1] != labels[i + 1]:
            total += 1
        if 1 <= y < ny - 1 and enabled(labels[i - nx]) and enabled(labels[i + nx]) \
                and labels[i - nx] != labels[i + nx]:
            total += 1
        if nz > 1 and 1 <= z < nz - 1 and enabled(labels[i - nx * ny]) and enabled(labels[i + nx * ny]) \
                and labels[i - nx * ny] != labels[i + nx * ny]:
            total += 1
    return total
```

- [ ] **Step 6: Run the test, verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_junction.py -v`
Expected: PASS. If the control doesn't open a gap at these params, raise the temperature in `_spec` — keep both assertions load-bearing.

- [ ] **Step 7: Commit**

```bash
git add crates/cpm-py/src/lib.rs cpm/schema.py cpm/metrics.py tests/test_junction.py
git commit -m "feat(junction): pyo3 setters + schema section + gap-faces metric"
```

---

### Task 4: Before/after crypt demo + viewer blurb

**Files:**
- Create: `demos/run_junction_demo.py`
- Modify: `viewer/viewer.js`

**Interfaces:**
- Consumes: `cpm.crypt3d.build_crypt3d` (E2), `cpm.metrics` (`radial_cell_counts`, `interior_medium_pockets`, `connected_components`, `intercell_gap_faces`), the world API (`set_junction`, `set_junction_lambda`, connectivity setters, `set_target_volume`), and the 3D `voxels` export + manifest conventions from `demos/run_membrane_demo.py`.
- Produces: `viewer/data/junction.json` (`kind="junction"`, `is3d=True`) + manifest entry; exits nonzero on any failed gate.

- [ ] **Step 1: Write the demo**

Create `demos/run_junction_demo.py`:

```python
"""Cell-cell junctions (E3b): the E2 crypt monolayer under a relaxation hot enough
that the wall perforates and the lumen breaches WITHOUT junctions, run twice on the
same seed -- once WITHOUT junctions (control: gaps open, the lumen breaches) and
once WITH junctions (the anti-gap energy seals the seams: no perforation, lumen
stays enclosed). This is the "no gaps, no merging" physics E3a's basal anchor
leaves open. Validates + exports for the viewer.

Junctions penalise medium pinched between two different junction-enabled cells, so
opening a gap/film/perforation costs energy -- WITHOUT adding surface tension (free
surface is never pinched), so the sheet does not curl.

Usage (repo root, venv active):  python demos/run_junction_demo.py
"""
import json
import os
import sys

from cpm import cpm_core

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cpm.crypt3d import build_crypt3d
from cpm.metrics import (radial_cell_counts, interior_medium_pockets,
                         connected_components, intercell_gap_faces)

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
JUNCTION_TYPES = {1, 2, 3}
LAMBDA_J = 16.0              # junction stiffness (tune so it seals without freezing)
TEMP = 10.0                 # hot enough that the control wall perforates
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


def build_world(dims, labels, seg_to_type, with_junctions):
    nx, ny, nz = dims
    w = cpm_core.World((nx, ny, nz), "noflux", 2, TEMP)
    w.seed_from_labels(labels, seg_to_type, 1, 20.0, 20.0)
    for t in range(1, 4):
        w.set_contact(0, t, 6.0)
        for u in range(t, 4):
            w.set_contact(t, u, 4.0)
        w.set_connectivity(t, True)
    w.set_connectivity_medium(True)
    if with_junctions:
        for t in JUNCTION_TYPES:
            w.set_junction(t, True)
        w.set_junction_lambda(LAMBDA_J)
    w.finalize(1)
    vols0 = w.cell_volumes()
    for c in range(1, len(vols0)):
        if vols0[c] > 0:
            w.set_target_volume(c, float(vols0[c]))
    return w


def relax(w, dims, capture):
    nx, ny, nz = dims
    frames = []
    prev = None
    min_pockets = 10 ** 9
    worst_mean, worst_p90, worst_gap = 0.0, 0, 0
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
        worst_gap = max(worst_gap, intercell_gap_faces(w, JUNCTION_TYPES))
        min_pockets = min(min_pockets, interior_medium_pockets(w))
        if capture:
            frames.append({"mcs": f * MCS_PER_FRAME, "voxels": surface_3d(w)})
        if f < N_FRAMES:
            w.step(MCS_PER_FRAME)
    types = w.cell_types(); vols = w.cell_volumes()
    frag = sum(1 for c in range(1, len(types)) if vols[c] > 0 and connected_components(w, c) != 1)
    alive = sum(1 for c in range(1, len(types)) if vols[c] > 0)
    return {"worst_mean": worst_mean, "worst_p90": worst_p90, "worst_gap": worst_gap,
            "min_pockets": min_pockets, "min_step_churn": min_step_churn or 0,
            "frag": frag, "alive": alive, "n0": w.n_cells(),
            "cell_types": list(types), "frames": frames}


def main():
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d(wall=3)
    dims = (nx, ny, nz)
    ctrl = relax(build_world(dims, labels, seg_to_type, False), dims, capture=False)
    jun = relax(build_world(dims, labels, seg_to_type, True), dims, capture=True)

    # Control must genuinely open gaps / breach; else the junction contrast is vacuous.
    control_degrades = ctrl["min_pockets"] == 0 or ctrl["worst_gap"] > jun["worst_gap"]
    checks = [
        (f"CONTROL (no junctions) opens gaps / breaches (worst gap faces {ctrl['worst_gap']}, "
         f"min pockets {ctrl['min_pockets']}) -- proves the junctions do the work",
         control_degrades),
        (f"WITH junctions: seams stay sealed (worst gap faces {jun['worst_gap']} < control "
         f"{ctrl['worst_gap']})", jun["worst_gap"] < ctrl["worst_gap"]),
        (f"WITH junctions: lumen stays enclosed (min interior pockets {jun['min_pockets']} >= 1)",
         jun["min_pockets"] >= 1),
        (f"WITH junctions: monolayer holds, no curling (worst mean {jun['worst_mean']:.2f} < 1.5, "
         f"worst p90 {jun['worst_p90']} <= 2)", jun["worst_mean"] < 1.5 and jun["worst_p90"] <= 2),
        (f"WITH junctions: no cell fragments ({jun['frag']} split) and all survive "
         f"({jun['alive']}/{jun['n0']})", jun["frag"] == 0 and jun["alive"] == jun["n0"]),
        (f"WITH junctions: relaxation is non-trivial (min per-step churn "
         f"{jun['min_step_churn']} > 0 -- sealed, not frozen)", jun["min_step_churn"] > 0),
    ]

    data = {"name": "Cell-Cell Junctions (crypt)", "kind": "junction", "dims": [nx, ny, nz],
            "is3d": True, "n_cells": jun["n0"], "cell_types": jun["cell_types"],
            "type_names": ["Medium"] + type_names, "frames": jun["frames"]}
    os.makedirs(DATA, exist_ok=True)
    json.dump(data, open(os.path.join(DATA, "junction.json"), "w"))
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    manifest = [m for m in manifest if m["file"] != "junction.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "junction.json", "name": data["name"], "is3d": True,
                     "n_cells": data["n_cells"], "dims": data["dims"], "kind": "junction",
                     "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json", "hra_ftu.json", "crypt_differentiation.json",
             "connectivity_2d.json", "connectivity_3d.json", "connectivity_gap.json",
             "crypt3d.json", "membrane.json", "junction.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (cell-cell junctions) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the demo, tune the contrast**

Run: `source .venv/bin/activate && python demos/run_junction_demo.py`
Expected: all 6 gates PASS, exit 0; writes `viewer/data/junction.json` + manifest entry.

Two tuning targets that must BOTH hold:
1. The control must genuinely open gaps / breach (gate 1). If it doesn't, raise `TEMP`.
2. Junctions must seal (gates 2-3) AND still let the sheet relax (gate 6, min-per-step churn > 0) without curling (gate 4). Raise `LAMBDA_J` if seams still gap; if it freezes (churn 0), lower it. Find the window.
Do NOT weaken a gate or trivially shorten the run. Report the final TEMP/LAMBDA_J and the control-vs-junction gap-face + pocket numbers. If junctions seal gaps but the lumen still breaches via a route the 1-voxel pinch can't catch, note it — but gate 3 (lumen enclosed) must pass for the demo to claim gap-sealing; tune `LAMBDA_J`/`TEMP` until it does.

- [ ] **Step 3: Add the viewer blurb**

In `viewer/viewer.js`, add a `junction` entry to the `BLURB` object (after the `membrane` entry):

```javascript
  junction: "Cell-cell junctions (Sub-project E3b) — an anti-gap energy that penalises " +
    "medium pinched between two bonded cells, so a gap or wall perforation can't open " +
    "between neighbours (without adding surface tension, so the sheet doesn't curl). " +
    "Under a stress that breaches the un-jointed control, the jointed crypt keeps its " +
    "seams sealed and its lumen enclosed — the 'no gaps, no merging' physics.",
```

- [ ] **Step 4: Commit**

```bash
git add demos/run_junction_demo.py viewer/viewer.js
git commit -m "feat(demo): cell-cell junction before/after crypt demo + viewer blurb"
```

---

## Self-Review

**Spec coverage:** pinch predicate (`junction.rs`) → Task 1; `delta_junction` state energy + `pinch_at` + World state/setters/`any_junction` → Task 1; sweep integration with `any_junction` short-circuit → Task 2; Rust unit (pinch predicate, delta sign open/close/free-surface, `any_junction` gating) → Task 1; Rust property (junctions keep two cells gap-free vs control opens a gap) + determinism → Task 2; pyo3 setters surviving finalize → Task 3; schema `spec["junctions"]` → Task 3; `intercell_gap_faces` mirroring the Rust P → Task 3; Python schema-wiring test (with vs without) → Task 3; before/after crypt demo with six gates (control degrades; seams sealed; lumen enclosed; monolayer holds no-curl; no fragmentation + survival; non-trivial relaxation) + viewer export → Task 4.

**Placeholder scan:** no TBD/TODO. The two genuine tuning points (Rust property margin in Task 2; demo control-degrades-AND-junctions-seal window in Task 4) are called out with concrete knobs (temperature/`lambda_junction`) and an explicit "do not weaken the gate" rule.

**Type consistency:** `axis_is_pinch((u32,bool),(u32,bool)) -> bool`, `delta_junction(site, new_owner) -> f64`, `any_junction() -> bool`, `set_junction(cell_type, on)` / `set_junction_lambda(lambda)`, the two `World` fields, pyo3 setters, schema `{"types","lambda"}`, `intercell_gap_faces(world, junction_types)`, and the demo `kind="junction"` + `order` entry are consistent across Tasks 1–4 and match the E1/E3a idioms (per-type bool + `any_*` short-circuit; `world_mut()` finalize-surviving setters; per-frame worst-case gating).
```
