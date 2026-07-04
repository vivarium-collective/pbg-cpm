# CPM Connectivity Constraint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local connectivity constraint to the Rust CPM core that forbids copy attempts which would fragment a cell (or the medium), preventing cell splitting/merging and interior gaps.

**Architecture:** A pure local graph predicate (`connectivity.rs`) counts connected components among a flip site's same-owner neighbours; `World` holds per-type + medium connectivity flags and a `would_stay_connected` method; `Cpm::step` gates `apply_flip` on it. Exposed through pyo3, the schema, Python metrics helpers, and before/after integrity demos.

**Tech Stack:** Rust (cpm-core) + pyo3/maturin (`cpm_core`), Python pkg `cpm`, existing three.js viewer export.

## Global Constraints

- Connectivity is a HARD gate applied AFTER the Metropolis accept decision — it only turns accepted flips into rejects; it never touches energy or the incremental volume/surface/COM trackers (a rejected flip calls neither `apply_flip` nor any tracker update).
- The predicate is LOCAL (O(neighbourhood²)), never a global flood-fill, and must not consult RNG or hash-iteration order (determinism).
- `any_connectivity()` short-circuits: unconstrained runs pay zero overhead.
- The constraint applies to the cell LOSING the pixel (`target = owner(s)`), and to the medium (CellId 0) only when medium connectivity is enabled. The gaining cell is never checked (adding a pixel cannot fragment).
- Backward compatible: no connectivity configured → identical behaviour to today.
- Build in the repo's own `.venv` (py3.12) with `maturin develop -m crates/cpm-py/Cargo.toml`; pytest resolves `cpm`/`cpm_core` via `pythonpath=["."]`.
- The Python `connected_components` / `interior_medium_pockets` metrics use Moore adjacency (8-conn 2D / 26-conn 3D, matching `neighbor_order=2`) and assume a bounded `noflux` domain; the demos use `noflux`.

---

## File Structure

- `crates/cpm-core/src/connectivity.rs` (new) — pure `count_components(members, adjacent)` graph function. No World dependency.
- `crates/cpm-core/src/lib.rs` — `pub mod connectivity;`.
- `crates/cpm-core/src/world.rs` — connectivity flag fields, `set_connectivity`/`set_connectivity_medium`/`any_connectivity`/`type_is_constrained`/`would_stay_connected`.
- `crates/cpm-core/src/sweep.rs` — the gate before `apply_flip`.
- `crates/cpm-py/src/lib.rs` — pyo3 `set_connectivity`/`set_connectivity_medium`.
- `cpm/schema.py` — `spec["connectivity"]` section.
- `cpm/metrics.py` — `connected_components(world, cell_id)` + `interior_medium_pockets(world)`.
- `demos/run_connectivity_demos.py` (new) — 2D + 3D anti-fragmentation + confluent no-gap demos.
- Tests: `crates/cpm-core/tests/property.rs` (append), `tests/test_connectivity.py` (new).

---

### Task 1: Local connectivity predicate (`connectivity.rs`)

**Files:**
- Create: `crates/cpm-core/src/connectivity.rs`
- Modify: `crates/cpm-core/src/lib.rs` (add `pub mod connectivity;`)

**Interfaces:**
- Produces: `pub fn count_components(members: &[usize], adjacent: &dyn Fn(usize, usize) -> bool) -> usize` — number of connected components among `members` (treated as opaque node ids), where nodes `a,b` share an edge iff `adjacent(a, b)` is true. Returns `members.len()` when `len <= 1`.

- [ ] **Step 1: Write the failing test**

Create `crates/cpm-core/src/connectivity.rs` with the function stub returning `0` and this test module:

```rust
//! Local connectivity test for the CPM sweep. `count_components` is a pure
//! graph routine over a tiny node set (the same-owner neighbours of a flip
//! site); it decides whether removing the site would locally disconnect a cell.

pub fn count_components(_members: &[usize], _adjacent: &dyn Fn(usize, usize) -> bool) -> usize {
    0 // stub
}

#[cfg(test)]
mod tests {
    use super::*;

    // adjacency from an explicit undirected edge list over node ids
    fn adj(edges: &'static [(usize, usize)]) -> impl Fn(usize, usize) -> bool {
        move |a, b| edges.iter().any(|&(u, v)| (u == a && v == b) || (u == b && v == a))
    }

    #[test]
    fn empty_and_single() {
        let none = adj(&[]);
        assert_eq!(count_components(&[], &none), 0);
        assert_eq!(count_components(&[7], &none), 1);
    }

    #[test]
    fn one_component_when_chained() {
        // 10-11-12 chain -> single component
        let a = adj(&[(10, 11), (11, 12)]);
        assert_eq!(count_components(&[10, 11, 12], &a), 1);
    }

    #[test]
    fn two_components_when_split() {
        // 10-11 and an isolated 12 -> two components (12 is only reachable via
        // the removed site, i.e. an articulation case)
        let a = adj(&[(10, 11)]);
        assert_eq!(count_components(&[10, 11, 12], &a), 2);
    }
}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cargo test -p cpm-core connectivity::`
Expected: FAIL (stub returns 0; the chain/split assertions fail).

- [ ] **Step 3: Implement**

Replace the stub:

```rust
pub fn count_components(members: &[usize], adjacent: &dyn Fn(usize, usize) -> bool) -> usize {
    let n = members.len();
    if n <= 1 {
        return n;
    }
    let mut seen = vec![false; n];
    let mut components = 0;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        components += 1;
        seen[start] = true;
        stack.push(start);
        while let Some(i) = stack.pop() {
            for j in 0..n {
                if !seen[j] && adjacent(members[i], members[j]) {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
    }
    components
}
```

- [ ] **Step 4: Wire the module + run tests**

In `crates/cpm-core/src/lib.rs`, add `pub mod connectivity;` after `pub mod init;`.
Run: `cargo test -p cpm-core connectivity::`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/src/connectivity.rs crates/cpm-core/src/lib.rs
git commit -m "feat(core): local connected-components predicate for connectivity"
```

---

### Task 2: `World` connectivity state + `would_stay_connected`

**Files:**
- Modify: `crates/cpm-core/src/world.rs`
- Test: `crates/cpm-core/tests/property.rs` (append)

**Interfaces:**
- Consumes: `crate::connectivity::count_components`; `self.lattice.neighbors(idx) -> Vec<usize>`; `self.lattice.owner(idx) -> CellId`.
- Produces on `World`:
  - `set_connectivity(&mut self, cell_type: u16, on: bool)` — grows an internal `Vec<bool>` as needed.
  - `set_connectivity_medium(&mut self, on: bool)`.
  - `any_connectivity(&self) -> bool`.
  - `type_is_constrained(&self, cell_type: u16) -> bool`.
  - `would_stay_connected(&self, site: usize, target: CellId) -> bool`.

- [ ] **Step 1: Write the failing test**

Append to `crates/cpm-core/tests/property.rs`:

```rust
#[test]
fn would_stay_connected_flags_local_articulation() {
    use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
    use cpm_core::world::World;
    // 2D Moore neighbourhood
    let lat = Lattice::new([7, 7, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 10.0);
    let c = w.add_cell(1, 9.0, 1.0, 0.0, 0.0);
    // dumbbell: two 1x3 arms joined by a single neck pixel at (3,3)
    for x in 1..3 { let i = w.lattice.index(x, 3, 0); w.paint(i, c); }   // left arm x=1,2
    let neck = w.lattice.index(3, 3, 0); w.paint(neck, c);              // neck x=3
    for x in 4..6 { let i = w.lattice.index(x, 3, 0); w.paint(i, c); }   // right arm x=4,5
    w.recompute_trackers();
    // removing the neck would split the cell -> not safe
    assert!(!w.would_stay_connected(neck, c));
    // removing a tip is safe (its only same-owner neighbour is one pixel)
    let tip = w.lattice.index(1, 3, 0);
    assert!(w.would_stay_connected(tip, c));

    // connectivity flags
    assert!(!w.any_connectivity());
    w.set_connectivity(1, true);
    assert!(w.any_connectivity());
    assert!(w.type_is_constrained(1));
    assert!(!w.type_is_constrained(2));
    w.set_connectivity(1, false);
    w.set_connectivity_medium(true);
    assert!(w.any_connectivity());
}
```

Note on the tip: with a Moore neighbourhood the tip at x=1 has same-owner neighbour only at x=2 (`|members| == 1`) → early-out `true`. The neck at x=3 has same-owner neighbours x=2 (left arm) and x=4 (right arm); x=2 and x=4 are not Moore-adjacent to each other (distance 2), so 2 components → `false`.

- [ ] **Step 2: Run it, verify it fails**

Run: `cargo test -p cpm-core would_stay_connected_flags`
Expected: FAIL — methods not found.

- [ ] **Step 3: Implement**

In `crates/cpm-core/src/world.rs`, add two fields to `struct World`:

```rust
    pub connectivity_types: Vec<bool>,
    pub connectivity_medium: bool,
```

Initialise them in `World::new` (in the returned struct literal, after `fields: Vec::new(),`):

```rust
            connectivity_types: Vec::new(),
            connectivity_medium: false,
```

Add methods inside `impl World` (near `set_contact_matrix`):

```rust
    pub fn set_connectivity(&mut self, cell_type: u16, on: bool) {
        let t = cell_type as usize;
        if t >= self.connectivity_types.len() {
            self.connectivity_types.resize(t + 1, false);
        }
        self.connectivity_types[t] = on;
    }

    pub fn set_connectivity_medium(&mut self, on: bool) {
        self.connectivity_medium = on;
    }

    pub fn any_connectivity(&self) -> bool {
        self.connectivity_medium || self.connectivity_types.iter().any(|&b| b)
    }

    pub fn type_is_constrained(&self, cell_type: u16) -> bool {
        self.connectivity_types
            .get(cell_type as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Would removing pixel `site` keep cell `target` locally connected?
    /// Local test over `site`'s same-owner neighbours; O(neighbourhood^2).
    pub fn would_stay_connected(&self, site: usize, target: CellId) -> bool {
        let members: Vec<usize> = self
            .lattice
            .neighbors(site)
            .into_iter()
            .filter(|&n| self.lattice.owner(n) == target)
            .collect();
        if members.len() <= 1 {
            return true;
        }
        let adj = |a: usize, b: usize| self.lattice.neighbors(a).contains(&b);
        crate::connectivity::count_components(&members, &adj) == 1
    }
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cargo test -p cpm-core`
Expected: PASS (existing tests + the new one). If any existing struct-literal construction of `World` outside `new()` fails to compile, it does not exist (all construction goes through `World::new`) — but if the compiler flags one, add the two new fields there too.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/src/world.rs crates/cpm-core/tests/property.rs
git commit -m "feat(core): World connectivity flags + would_stay_connected"
```

---

### Task 3: Sweep gate

**Files:**
- Modify: `crates/cpm-core/src/sweep.rs`
- Test: `crates/cpm-core/tests/property.rs` (append)

**Interfaces:**
- Consumes: `World::any_connectivity`, `type_is_constrained`, `connectivity_medium`, `would_stay_connected`, `lattice.owner`, `cells[..].cell_type`.
- Produces: `Cpm::step` rejects any accepted flip that would fragment a constrained `target`.

- [ ] **Step 1: Write the failing test**

Append to `crates/cpm-core/tests/property.rs`:

```rust
#[test]
fn connectivity_keeps_cell_in_one_piece_under_stress() {
    use cpm_core::energy::ContactMatrix;
    use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
    use cpm_core::sweep::Cpm;
    use cpm_core::world::World;

    // Count connected components of a cell's pixels via a global flood-fill
    // (Moore adjacency), for the test only.
    fn components(w: &World, cid: u32) -> usize {
        let [nx, ny, nz] = [w.lattice.dims_x(), w.lattice.dims_y(), w.lattice.dims_z()];
        let sites: Vec<usize> = (0..w.lattice.n_sites())
            .filter(|&i| w.lattice.owner(i) == cid)
            .collect();
        let inset: std::collections::HashSet<usize> = sites.iter().copied().collect();
        let mut seen = std::collections::HashSet::new();
        let mut comps = 0;
        for &s in &sites {
            if seen.contains(&s) { continue; }
            comps += 1;
            let mut stack = vec![s];
            seen.insert(s);
            while let Some(c) = stack.pop() {
                let cz = c / (nx * ny);
                let cy = (c % (nx * ny)) / nx;
                let cx = c % nx;
                for dz in -1i64..=1 { for dy in -1i64..=1 { for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    let (x2, y2, z2) = (cx as i64 + dx, cy as i64 + dy, cz as i64 + dz);
                    if x2 < 0 || y2 < 0 || z2 < 0 || x2 >= nx as i64 || y2 >= ny as i64 || z2 >= nz as i64 { continue; }
                    let n = x2 as usize + y2 as usize * nx + z2 as usize * nx * ny;
                    if inset.contains(&n) && !seen.contains(&n) { seen.insert(n); stack.push(n); }
                }}}
            }
        }
        comps
    }

    fn dumbbell(connectivity: bool) -> usize {
        let lat = Lattice::new([25, 9, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 30.0); // high temperature -> stress
        let c = w.add_cell(1, 40.0, 1.0, 0.0, 0.0);
        // two 5x5 blobs joined by a single 1px neck at x=12,y=4
        for y in 2..7 { for x in 5..10 { let i = w.lattice.index(x, y, 0); w.paint(i, c); } }
        for y in 2..7 { for x in 15..20 { let i = w.lattice.index(x, y, 0); w.paint(i, c); } }
        let neck = w.lattice.index(12, 4, 0); w.paint(neck, c);
        // bridge the neck to both blobs so it starts connected
        for x in 10..12 { let i = w.lattice.index(x, 4, 0); w.paint(i, c); }
        for x in 13..15 { let i = w.lattice.index(x, 4, 0); w.paint(i, c); }
        let mut m = ContactMatrix::new(2);
        m.set(0, 1, -2.0); // negative medium adhesion -> cell wants boundary -> shreds
        w.set_contact_matrix(m);
        w.recompute_trackers();
        if connectivity { w.set_connectivity(1, true); }
        let mut cpm = Cpm::new(w, 1);
        cpm.step(40);
        components(&cpm.world, c)
    }

    // Without the constraint the stressed dumbbell fragments; with it, it stays whole.
    assert!(dumbbell(false) > 1, "stress must actually fragment (guard against vacuous test)");
    assert_eq!(dumbbell(true), 1, "connectivity must keep the cell in one piece");
}
```

This test references `w.lattice.dims_x()/dims_y()/dims_z()`. If the `Lattice` does not already expose these, add trivial accessors in `crates/cpm-core/src/lattice.rs` inside `impl Lattice`:

```rust
    pub fn dims_x(&self) -> usize { self.dims[0] }
    pub fn dims_y(&self) -> usize { self.dims[1] }
    pub fn dims_z(&self) -> usize { self.dims[2] }
```

(Only add them if absent — check first with `grep -n "dims_x" crates/cpm-core/src/lattice.rs`.)

- [ ] **Step 2: Run it, verify it fails**

Run: `cargo test -p cpm-core connectivity_keeps_cell`
Expected: FAIL — with connectivity not yet gating the sweep, `dumbbell(true)` also fragments (`> 1`), so `assert_eq!(..., 1)` fails.

- [ ] **Step 3: Implement the gate**

In `crates/cpm-core/src/sweep.rs`, replace the acceptance block:

```rust
                let accept = dh <= 0.0 || self.rng.gen::<f64>() < (-dh / t).exp();
                if accept {
                    self.world.apply_flip(s, source_owner);
                }
```

with:

```rust
                let accept = dh <= 0.0 || self.rng.gen::<f64>() < (-dh / t).exp();
                if accept {
                    if self.world.any_connectivity() {
                        let target = self.world.lattice.owner(s);
                        let constrained = if target == 0 {
                            self.world.connectivity_medium
                        } else {
                            self.world
                                .type_is_constrained(self.world.cells[target as usize].cell_type)
                        };
                        if constrained && !self.world.would_stay_connected(s, target) {
                            continue; // reject: would fragment `target`
                        }
                    }
                    self.world.apply_flip(s, source_owner);
                }
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cargo test -p cpm-core`
Expected: PASS, including `connectivity_keeps_cell_in_one_piece_under_stress` and the existing `deterministic_under_seed`. If `dumbbell(false)` does NOT fragment (`> 1` fails), raise the temperature (e.g. `40.0`) or `cpm.step` count until it does — the WITHOUT case must genuinely fragment for the test to be meaningful.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/src/sweep.rs crates/cpm-core/src/lattice.rs crates/cpm-core/tests/property.rs
git commit -m "feat(core): gate the sweep on the connectivity constraint"
```

---

### Task 4: pyo3 setters

**Files:**
- Modify: `crates/cpm-py/src/lib.rs`
- Test: `tests/test_connectivity.py` (create — Python binding smoke)

**Interfaces:**
- Consumes: `World::set_connectivity`, `set_connectivity_medium`; the pyo3 wrapper's `world_mut()`.
- Produces: Python `World.set_connectivity(cell_type: int, on: bool)`, `World.set_connectivity_medium(on: bool)`. Connectivity state set on the core world persists through `finalize` (the core `World` is moved intact into `Cpm`), so the setters may be called before `finalize`.

- [ ] **Step 1: Expose the setters**

In `crates/cpm-py/src/lib.rs`, add inside `#[pymethods] impl World` (after `set_cell_type`):

```rust
    fn set_connectivity(&mut self, cell_type: u16, on: bool) {
        self.max_type = self.max_type.max(cell_type);
        self.world_mut().set_connectivity(cell_type, on);
    }

    fn set_connectivity_medium(&mut self, on: bool) {
        self.world_mut().set_connectivity_medium(on);
    }
```

- [ ] **Step 2: Rebuild the extension**

Run: `source .venv/bin/activate && maturin develop -m crates/cpm-py/Cargo.toml`
Expected: builds `cpm_core`.

- [ ] **Step 3: Write + run the binding test**

Create `tests/test_connectivity.py`:

```python
import cpm_core


def test_connectivity_setters_survive_finalize():
    w = cpm_core.World((25, 9, 1), "noflux", 2, 30.0)
    c = w.add_cell(1, 40.0, 1.0, 0.0, 0.0)
    w.set_contact(0, 1, -2.0)
    # dumbbell: two blobs + a 1px neck bridge
    for y in range(2, 7):
        for x in range(5, 10):
            w.seed_block(c, x, y, 0, x + 1, y + 1, 1)
        for x in range(15, 20):
            w.seed_block(c, x, y, 0, x + 1, y + 1, 1)
    for x in range(10, 15):
        w.seed_block(c, x, 4, 0, x + 1, 5, 1)
    w.set_connectivity(1, True)          # set BEFORE finalize
    w.finalize(1)
    w.step(40)
    # cell survived as nonzero volume; detailed single-component check is in
    # test_connectivity_metric below via cpm.metrics
    assert w.cell_volumes()[c] > 0
```

Run: `source .venv/bin/activate && pytest tests/test_connectivity.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/cpm-py/src/lib.rs tests/test_connectivity.py
git commit -m "feat(py): expose connectivity setters on the World binding"
```

---

### Task 5: Schema section + metrics helpers

**Files:**
- Modify: `cpm/schema.py`, `cpm/metrics.py`
- Test: `tests/test_connectivity.py` (append)

**Interfaces:**
- Consumes: pyo3 `World.set_connectivity`, `set_connectivity_medium`, `snapshot`, `dims`.
- Produces:
  - `load_world` reads `spec["connectivity"] = {"types": [int, ...], "medium": bool}` (both optional) BEFORE `finalize`.
  - `cpm.metrics.connected_components(world, cell_id) -> int` (Moore adjacency, bounded domain).
  - `cpm.metrics.interior_medium_pockets(world) -> int` — medium (cell 0) components not touching the lattice border.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_connectivity.py`:

```python
def _dumbbell_world(connectivity):
    from cpm.schema import load_world
    spec = {
        "potts": {"dims": [25, 9, 1], "boundary": "noflux",
                  "neighbor_order": 2, "temperature": 30.0, "seed": 1},
        "contact": [{"a": 0, "b": 1, "j": -2.0}],
        "cells": [{"type": 1, "target_volume": 40, "lambda_volume": 1.0,
                   "target_surface": 0, "lambda_surface": 0.0,
                   "seed_block": [5, 2, 0, 20, 7, 1]}],   # solid bar (stays connected only if protected)
    }
    if connectivity:
        spec["connectivity"] = {"types": [1], "medium": False}
    return load_world(spec)


def test_connected_components_metric_and_constraint():
    from cpm.metrics import connected_components
    # WITH the constraint the cell stays one connected component
    w_on = _dumbbell_world(True)
    w_on.step(40)
    c = 1
    assert connected_components(w_on, c) == 1
    # WITHOUT it, the same stressed bar fragments into >1 component
    w_off = _dumbbell_world(False)
    w_off.step(40)
    assert connected_components(w_off, c) > 1


def test_interior_medium_pockets_metric():
    from cpm.metrics import interior_medium_pockets
    # a 6x6 lattice: fill all but one interior medium pixel -> exactly 1 pocket
    w = cpm_core_world_all_cells_with_hole()
    assert interior_medium_pockets(w) == 1
```

Add the helper builder to the same test file:

```python
import cpm_core

def cpm_core_world_all_cells_with_hole():
    # 6x6, fill with one big cell except pixel (3,3) left as medium -> 1 interior pocket
    w = cpm_core.World((6, 6, 1), "noflux", 2, 1.0)
    c = w.add_cell(1, 35.0, 1.0, 0.0, 0.0)
    w.set_contact(0, 1, 4.0)
    for y in range(6):
        for x in range(6):
            if (x, y) == (3, 3):
                continue
            w.seed_block(c, x, y, 0, x + 1, y + 1, 1)
    w.finalize(1)
    return w
```

- [ ] **Step 2: Run it, verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_connectivity.py -k "metric" -v`
Expected: FAIL — `spec["connectivity"]` unhandled (no fragmentation difference) and/or `connected_components`/`interior_medium_pockets` not defined.

- [ ] **Step 3: Implement the schema section**

In `cpm/schema.py` `load_world`, after the `fields` loop and BEFORE `world.finalize(...)`, add:

```python
    conn = spec.get("connectivity")
    if conn is not None:
        for t in conn.get("types", []):
            world.set_connectivity(int(t), True)
        if conn.get("medium", False):
            world.set_connectivity_medium(True)
```

- [ ] **Step 4: Implement the metrics**

Append to `cpm/metrics.py`:

```python
def _neighbors_moore(cx, cy, cz, nx, ny, nz):
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                x2, y2, z2 = cx + dx, cy + dy, cz + dz
                if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
                    yield x2 + y2 * nx + z2 * nx * ny


def connected_components(world, cell_id):
    """Number of connected components of `cell_id`'s pixels (Moore adjacency,
    bounded/no-wrap domain)."""
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    sites = {i for i, v in enumerate(labels) if v == cell_id}
    seen, comps = set(), 0
    for s in sites:
        if s in seen:
            continue
        comps += 1
        stack = [s]
        seen.add(s)
        while stack:
            c = stack.pop()
            cz, rem = divmod(c, nx * ny)
            cy, cx = divmod(rem, nx)
            for n in _neighbors_moore(cx, cy, cz, nx, ny, nz):
                if n in sites and n not in seen:
                    seen.add(n)
                    stack.append(n)
    return comps


def interior_medium_pockets(world):
    """Count medium (cell 0) connected components that do NOT touch the lattice
    border — interior gap pockets. Assumes a bounded (noflux) domain."""
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    sites = {i for i, v in enumerate(labels) if v == 0}
    seen, pockets = set(), 0
    for s in sites:
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        touches_border = False
        while stack:
            c = stack.pop()
            cz, rem = divmod(c, nx * ny)
            cy, cx = divmod(rem, nx)
            if cx == 0 or cx == nx - 1 or cy == 0 or cy == ny - 1 or \
               (nz > 1 and (cz == 0 or cz == nz - 1)):
                touches_border = True
            for n in _neighbors_moore(cx, cy, cz, nx, ny, nz):
                if n in sites and n not in seen:
                    seen.add(n)
                    stack.append(n)
        if not touches_border:
            pockets += 1
    return pockets
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_connectivity.py -v`
Expected: PASS. If `test_connected_components_metric_and_constraint`'s WITHOUT case does not fragment (a solid bar is harder to fragment than a dumbbell), change the `cells` seed in `_dumbbell_world` to the explicit dumbbell shape (two `seed_block` blobs + a 1px neck bridge, as in Task 4's binding test) so the unprotected bar genuinely breaks; keep both branches identical except for the `connectivity` key.

- [ ] **Step 6: Commit**

```bash
git add cpm/schema.py cpm/metrics.py tests/test_connectivity.py
git commit -m "feat(schema+metrics): connectivity spec section + integrity metrics"
```

---

### Task 6: Integrity demos

**Files:**
- Create: `demos/run_connectivity_demos.py`

**Interfaces:**
- Consumes: `cpm_core.World`, `cpm.metrics.connected_components`/`interior_medium_pockets`, the viewer data-dir + `index.json` conventions from `demos/run_cc3d_demos.py`/`demos/run_hra_ftu.py` (2D `labels` frames, 3D `voxels` frames, `kind`, `checks`, `order` sort).
- Produces: `viewer/data/connectivity_2d.json`, `connectivity_3d.json`, `connectivity_gap.json` + manifest entries `kind="integrity"`; exits nonzero on any failed gate.

**Design notes (deterministic scenarios — no flaky tuning):**
- **2D / 3D anti-fragmentation**: seed a DUMBBELL (two blobs joined by a 1-pixel neck), negative medium adhesion (`set_contact(0, 1, -2.0)`), high temperature (30). Without connectivity the neck erodes → `connected_components == 2+`; with connectivity the neck pixel is a local articulation point → always protected → stays 1. Capture frames both ways; export the WITH run for the viewer.
- **Confluent no-gap**: seed a horseshoe/C-shaped cell enclosing a medium bay connected to the exterior by a 1-pixel mouth, positive medium adhesion (`set_contact(0,1,6.0)`) so the cell tends to close the mouth. Without medium connectivity the mouth closes → the bay pinches off → `interior_medium_pockets >= 1`; with `set_connectivity_medium(True)` the mouth-closing flip disconnects medium → rejected → `interior_medium_pockets == 0`.

- [ ] **Step 1: Write the demo**

Create `demos/run_connectivity_demos.py`:

```python
"""Structural-integrity demos for the CPM connectivity constraint.

Each demo runs the SAME stressed configuration twice -- once WITHOUT the
constraint (structure breaks) and once WITH it (structure holds) -- and
validates the difference, exiting nonzero on any failed gate. The WITH run is
exported for the viewer.

  * connectivity_2d : a dumbbell cell; the 1px neck erodes without the
                      constraint (fragments) and is protected with it.
  * connectivity_3d : the same in 3D (two cubes + a 1-voxel bridge).
  * connectivity_gap: a horseshoe cell around a medium bay; the mouth closes
                      and traps an interior pocket without medium connectivity,
                      and cannot with it.

Usage (repo root, venv active):  python demos/run_connectivity_demos.py
"""
import json
import os

import cpm_core
from cpm.metrics import connected_components, interior_medium_pockets

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))


def labels_2d(w):
    return list(w.snapshot())


def surface_3d(w):
    # boundary voxels [x, y, z, cellId] for the viewer (matches other 3D demos)
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
                boundary = False
                for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    x2, y2, z2 = x+dx, y+dy, z+dz
                    if not (0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz) or owner(x2,y2,z2) != c:
                        boundary = True; break
                if boundary:
                    out.append([x, y, z, c])
    return out


def _paint_dumbbell_2d(w, c):
    for y in range(2, 7):
        for x in range(5, 10):
            w.seed_block(c, x, y, 0, x + 1, y + 1, 1)
        for x in range(15, 20):
            w.seed_block(c, x, y, 0, x + 1, y + 1, 1)
    for x in range(10, 15):
        w.seed_block(c, x, 4, 0, x + 1, 5, 1)


def run_dumbbell_2d(connectivity):
    w = cpm_core.World((25, 9, 1), "noflux", 2, 30.0)
    c = w.add_cell(1, 55.0, 1.0, 0.0, 0.0)
    w.set_contact(0, 1, -2.0)
    _paint_dumbbell_2d(w, c)
    if connectivity:
        w.set_connectivity(1, True)
    w.finalize(1)
    frames = []
    for f in range(13):
        frames.append({"mcs": f * 4, "labels": labels_2d(w)})
        if f < 12:
            w.step(4)
    return w, frames


def _paint_dumbbell_3d(w, c):
    for z in range(2, 6):
        for y in range(2, 6):
            for x in range(3, 7):
                w.seed_block(c, x, y, z, x + 1, y + 1, z + 1)
            for x in range(11, 15):
                w.seed_block(c, x, y, z, x + 1, y + 1, z + 1)
    for x in range(7, 11):   # 1-voxel-thick bridge along x at y=3,z=3
        w.seed_block(c, x, 3, 3, x + 1, 4, 4)


def run_dumbbell_3d(connectivity):
    w = cpm_core.World((18, 8, 8), "noflux", 2, 24.0)
    c = w.add_cell(1, 130.0, 1.0, 0.0, 0.0)
    w.set_contact(0, 1, -1.5)
    _paint_dumbbell_3d(w, c)
    if connectivity:
        w.set_connectivity(1, True)
    w.finalize(1)
    frames = []
    for f in range(13):
        frames.append({"mcs": f * 3, "voxels": surface_3d(w)})
        if f < 12:
            w.step(3)
    return w, frames


def _paint_horseshoe(w, c):
    # C-shape around a medium bay in a 15x15 lattice; mouth (1px) on the +x side
    # walls: left column, top row, bottom row spanning x=3..11, y=3..11
    for y in range(3, 12):
        w.seed_block(c, 3, y, 0, 4, y + 1, 1)          # left wall
    for x in range(3, 12):
        w.seed_block(c, x, 3, 0, x + 1, 4, 1)          # bottom wall
        w.seed_block(c, x, 11, 0, x + 1, 12, 1)        # top wall
    # right wall with a 1px mouth at y=7 (leave (11,7) as medium)
    for y in range(3, 12):
        if y == 7:
            continue
        w.seed_block(c, 11, y, 0, 12, y + 1, 1)


def run_horseshoe(medium_connectivity):
    w = cpm_core.World((15, 15, 1), "noflux", 2, 12.0)
    c = w.add_cell(1, 60.0, 2.0, 0.0, 0.0)
    w.set_contact(0, 1, 6.0)   # positive medium adhesion -> cell closes the mouth
    _paint_horseshoe(w, c)
    if medium_connectivity:
        w.set_connectivity_medium(True)
    w.finalize(1)
    frames = []
    max_pockets = 0
    for f in range(16):
        frames.append({"mcs": f * 3, "labels": labels_2d(w)})
        max_pockets = max(max_pockets, interior_medium_pockets(w))
        if f < 15:
            w.step(3)
    return w, frames, max_pockets


def emit(slug, data, checks, manifest, results):
    with open(os.path.join(DATA, slug + ".json"), "w") as fh:
        json.dump(data, fh)
    ok = all(p for _, p in checks)
    results.append((data["name"], checks, ok))
    manifest[:] = [m for m in manifest if m["file"] != slug + ".json"] + [{
        "file": slug + ".json", "name": data["name"], "is3d": data["is3d"],
        "n_cells": data["n_cells"], "dims": data["dims"], "kind": "integrity",
        "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]}]
    print(f"  {'PASS' if ok else 'FAIL'} {data['name']}")


def main():
    os.makedirs(DATA, exist_ok=True)
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    results = []

    # 2D anti-fragmentation
    w_off, _ = run_dumbbell_2d(False)
    w_on, frames_on = run_dumbbell_2d(True)
    checks = [
        (f"without constraint the dumbbell fragments (components "
         f"{connected_components(w_off, 1)} > 1)", connected_components(w_off, 1) > 1),
        (f"with constraint it stays one connected cell (components "
         f"{connected_components(w_on, 1)} == 1)", connected_components(w_on, 1) == 1),
    ]
    emit("connectivity_2d", {"name": "Connectivity — 2D Anti-Fragmentation",
         "kind": "integrity", "dims": list(w_on.dims()), "is3d": False,
         "n_cells": w_on.n_cells(), "cell_types": list(w_on.cell_types()),
         "frames": frames_on}, checks, manifest, results)

    # 3D anti-fragmentation
    w3_off, _ = run_dumbbell_3d(False)
    w3_on, frames3 = run_dumbbell_3d(True)
    checks = [
        (f"without constraint the 3D dumbbell fragments (components "
         f"{connected_components(w3_off, 1)} > 1)", connected_components(w3_off, 1) > 1),
        (f"with constraint it stays one connected cell (components "
         f"{connected_components(w3_on, 1)} == 1)", connected_components(w3_on, 1) == 1),
    ]
    emit("connectivity_3d", {"name": "Connectivity — 3D Anti-Fragmentation",
         "kind": "integrity", "dims": list(w3_on.dims()), "is3d": True,
         "n_cells": w3_on.n_cells(), "cell_types": list(w3_on.cell_types()),
         "frames": frames3}, checks, manifest, results)

    # confluent no-gap (medium connectivity)
    _, _, pockets_off = run_horseshoe(False)
    w_h, frames_h, pockets_on = run_horseshoe(True)
    checks = [
        (f"without medium connectivity the mouth closes and traps a gap "
         f"(interior pockets {pockets_off} >= 1)", pockets_off >= 1),
        (f"with medium connectivity no interior gap forms "
         f"(interior pockets {pockets_on} == 0)", pockets_on == 0),
    ]
    emit("connectivity_gap", {"name": "Connectivity — No Interior Gaps",
         "kind": "integrity", "dims": list(w_h.dims()), "is3d": False,
         "n_cells": w_h.n_cells(), "cell_types": list(w_h.cell_types()),
         "frames": frames_h}, checks, manifest, results)

    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json", "hra_ftu.json", "crypt_differentiation.json",
             "connectivity_2d.json", "connectivity_3d.json", "connectivity_gap.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (connectivity) ===========")
    allok = True
    for name, checks, ok in results:
        print(f"\n{name}: {'PASS' if ok else 'FAIL'}")
        for t, p in checks:
            print(f"   [{'PASS' if p else 'FAIL'}] {t}")
        allok = allok and ok
    print("\nALL CONNECTIVITY DEMOS VALIDATED" if allok else "\nSOME FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the demo**

Run: `source .venv/bin/activate && python demos/run_connectivity_demos.py`
Expected: all three demos PASS, exit 0; writes the three JSONs + manifest entries. If a WITHOUT case does not break (fragment / trap a pocket), increase temperature or step count, or narrow the neck/mouth — the WITHOUT case must fail structurally for the demo to be meaningful; do NOT weaken the WITH gate.

- [ ] **Step 3: Commit**

```bash
git add demos/run_connectivity_demos.py
git commit -m "feat(demo): connectivity structural-integrity demos (2D/3D/gap)"
```

---

## Self-Review

**Spec coverage:** local predicate → Task 1; World state + `would_stay_connected` + setters + `any_connectivity` → Task 2; sweep gate (cells + medium, after accept, trackers untouched) → Task 3; pyo3 setters surviving `finalize` → Task 4; schema `connectivity` section + `connected_components`/`interior_medium_pockets` metrics → Task 5; 2D + 3D anti-fragmentation + confluent no-gap demos with before/after gates → Task 6. Determinism requirement → covered by the existing `deterministic_under_seed` test (unchanged path when unconstrained) and the predicate using no RNG/hash iteration (Tasks 1–3). Viewer: no changes (spec non-goal); demos export `kind="integrity"` frames the existing viewer already renders (2D labels / 3D voxels).

**Placeholder scan:** no TBD/TODO; every code step carries complete code. The two runtime-tuning points (WITHOUT case must genuinely break) are called out explicitly with how to adjust, not left vague. The `dims_x/y/z` accessor addition is gated on a grep check.

**Type consistency:** `set_connectivity(cell_type, on)`, `set_connectivity_medium(on)`, `any_connectivity`, `type_is_constrained`, `would_stay_connected(site, target)`, `count_components(members, adjacent)`, `connected_components(world, cell_id)`, `interior_medium_pockets(world)` are used identically across tasks. The sweep gate reads `owner(s)` as the losing cell and checks medium via `connectivity_medium` — consistent with Task 2's fields. Demo manifest `kind="integrity"` and the `order` list match the existing demo conventions.
