# CPM Core Engine (Sub-project A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fast Rust Cellular Potts (CPM) core — 2D/3D lattice, incremental trackers, energy plugins, Metropolis sweep — exposed to Python via pyo3 and wrapped as a process-bigraph Process, proven by a schema-driven 2D+3D cell-sorting regression test.

**Architecture:** A dimension-generic Rust core (`cpm-core`) owns a flat label lattice and per-cell aggregate trackers (volume, surface, center-of-mass) updated incrementally on every accepted pixel flip. Energy terms (Volume, Surface, Contact) compute ΔH for a candidate copy without mutating state. A seeded Metropolis sweep drives the dynamics. pyo3 bindings (`cpm_core`) expose world construction, sweeping, and readback; a thin Python `cpm` package wraps it as a process-bigraph `Process` and loads a minimal schema.

**Tech Stack:** Rust 2021 (cargo workspace), `rand` (SmallRng), `rayon` (later, not this sub-project), pyo3 + maturin, Python 3.12, `process-bigraph`, `pytest`, `uv` for the venv.

## Global Constraints

- Repo root: `/Users/eranagmon/code/pbg-cpm` (git already initialized; spec committed).
- `CellId = u32`; `MEDIUM: CellId = 0` is the ECM/medium sentinel and is exempt from Volume and Surface energy.
- Dimension-generic: 2D is 3D with `dims[2] == 1`. Never write a separate 2D code path.
- Lattice index order: `idx = x + y*nx + z*nx*ny`.
- Surface definition (used everywhere): `S_c` = number of ordered (site `p`, neighbor `q`) pairs where `owner(p) == c` and `owner(q) != c`, using the configured neighborhood.
- One Monte Carlo Step (MCS) = `nx*ny*nz` copy attempts.
- All randomness via a seeded `SmallRng` so every run is reproducible.
- Rust module built by maturin is named `cpm_core`; the pure-Python package is `cpm` (imports `cpm_core`). Do not conflate them.
- Commit after every task. Rust tests: `cargo test`. Python tests: `pytest`.

---

## File Structure

```
pbg-cpm/
  Cargo.toml                      # workspace
  pyproject.toml                  # maturin, module cpm_core, python-source "."
  crates/
    cpm-core/
      Cargo.toml
      src/lib.rs                  # re-exports; CellId, MEDIUM
      src/lattice.rs              # Lattice, Boundary, Neighborhood
      src/world.rs                # Cell, World, trackers, apply_flip, deltas
      src/energy.rs               # ContactMatrix, delta_hamiltonian
      src/sweep.rs                # metropolis_sweep
    cpm-py/
      Cargo.toml                  # cdylib, pyo3
      src/lib.rs                  # #[pymodule] cpm_core
    cpm-bench/
      Cargo.toml
      src/main.rs                 # 3D sweeps/sec baseline
  cpm/
    __init__.py
    schema.py                     # load_world(spec_dict) -> cpm_core.World
    pack.py                       # write_pack(world, path)
    processes/__init__.py
    processes/cpm_process.py      # CPMProcess(process_bigraph.Process)
  demos/
    cell_sorting_2d.json
    cell_sorting_3d.json
  tests/
    test_bindings.py
    test_process.py
    test_cell_sorting.py
```

---

### Task 1: Cargo workspace + `cpm-core` skeleton

**Files:**
- Create: `Cargo.toml` (workspace)
- Create: `crates/cpm-core/Cargo.toml`
- Create: `crates/cpm-core/src/lib.rs`

**Interfaces:**
- Produces: crate `cpm_core` (lib) exporting `pub type CellId = u32;` and `pub const MEDIUM: CellId = 0;`.

- [ ] **Step 1: Write the failing test**

In `crates/cpm-core/src/lib.rs`:

```rust
pub type CellId = u32;
pub const MEDIUM: CellId = 0;

pub mod lattice;
pub mod world;
pub mod energy;
pub mod sweep;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn medium_is_zero() {
        assert_eq!(MEDIUM, 0 as CellId);
    }
}
```

Create empty module files so it compiles: `crates/cpm-core/src/lattice.rs`, `world.rs`, `energy.rs`, `sweep.rs` each containing only `// placeholder`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/pbg-cpm && cargo test -p cpm-core`
Expected: FAIL — no workspace/manifest yet ("could not find `Cargo.toml`").

- [ ] **Step 3: Write minimal implementation**

`Cargo.toml` (workspace root):

```toml
[workspace]
resolver = "2"
members = ["crates/cpm-core", "crates/cpm-py", "crates/cpm-bench"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[workspace.dependencies]
cpm-core = { path = "crates/cpm-core" }
rand = "0.8"
```

`crates/cpm-core/Cargo.toml`:

```toml
[package]
name = "cpm-core"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
name = "cpm_core"

[dependencies]
rand = { workspace = true }
```

Create placeholder crates so the workspace resolves: `crates/cpm-py/Cargo.toml` and `crates/cpm-bench/Cargo.toml` with minimal stubs plus a `src/lib.rs`/`src/main.rs`:

`crates/cpm-py/Cargo.toml`:
```toml
[package]
name = "cpm-py"
version.workspace = true
edition.workspace = true
[lib]
name = "cpm_py_placeholder"
path = "src/lib.rs"
```
`crates/cpm-py/src/lib.rs`: `// placeholder, replaced in Task 9`

`crates/cpm-bench/Cargo.toml`:
```toml
[package]
name = "cpm-bench"
version.workspace = true
edition.workspace = true
[[bin]]
name = "cpm-bench"
path = "src/main.rs"
```
`crates/cpm-bench/src/main.rs`: `fn main() {}`

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p cpm-core`
Expected: PASS — `test tests::medium_is_zero ... ok`.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/
git commit -m "feat(core): cargo workspace + cpm-core skeleton"
```

---

### Task 2: Lattice — indexing, neighborhoods, boundary conditions

**Files:**
- Modify: `crates/cpm-core/src/lattice.rs`
- Test: inline `#[cfg(test)]` in `lattice.rs`

**Interfaces:**
- Produces:
  - `pub enum Boundary { NoFlux, Periodic }`
  - `pub struct Neighborhood { offsets: Vec<[i64; 3]> }` with `Neighborhood::new(is_3d: bool, order: u8) -> Neighborhood` (order = max Manhattan distance: 1 → 4/6 neighbors, 2 → 8/18, 3 → 26) and `pub fn offsets(&self) -> &[[i64;3]]`.
  - `pub struct Lattice { pub dims: [usize;3], pub boundary: [Boundary;3], site: Vec<CellId>, pub nbr: Neighborhood }`
  - `Lattice::new(dims:[usize;3], boundary:[Boundary;3], nbr:Neighborhood) -> Lattice`
  - `pub fn index(&self, x:usize,y:usize,z:usize) -> usize`
  - `pub fn owner(&self, idx:usize) -> CellId` / `pub fn set_owner(&mut self, idx:usize, c:CellId)`
  - `pub fn coords(&self, idx:usize) -> [usize;3]`
  - `pub fn neighbors(&self, idx:usize) -> Vec<usize>` — boundary-aware (NoFlux drops out-of-range; Periodic wraps).
  - `pub fn n_sites(&self) -> usize`

- [ ] **Step 1: Write the failing test**

Append to `crates/cpm-core/src/lattice.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_roundtrip() {
        let lat = Lattice::new([4, 3, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let idx = lat.index(2, 1, 0);
        assert_eq!(lat.coords(idx), [2, 1, 0]);
        assert_eq!(lat.n_sites(), 12);
    }

    #[test]
    fn moore_2d_interior_has_8_neighbors() {
        let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let center = lat.index(2, 2, 0);
        assert_eq!(lat.neighbors(center).len(), 8);
    }

    #[test]
    fn noflux_corner_drops_neighbors() {
        let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let corner = lat.index(0, 0, 0);
        assert_eq!(lat.neighbors(corner).len(), 3); // E, N, NE
    }

    #[test]
    fn periodic_corner_wraps_to_full() {
        let lat = Lattice::new([5, 5, 1], [Boundary::Periodic; 3], Neighborhood::new(false, 2));
        let corner = lat.index(0, 0, 0);
        assert_eq!(lat.neighbors(corner).len(), 8);
    }

    #[test]
    fn von_neumann_3d_has_6() {
        let lat = Lattice::new([5, 5, 5], [Boundary::NoFlux; 3], Neighborhood::new(true, 1));
        let c = lat.index(2, 2, 2);
        assert_eq!(lat.neighbors(c).len(), 6);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p cpm-core lattice`
Expected: FAIL — `Lattice`/`Neighborhood` not found.

- [ ] **Step 3: Write minimal implementation**

Replace `crates/cpm-core/src/lattice.rs` (keep the test module above at the end):

```rust
use crate::CellId;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Boundary {
    NoFlux,
    Periodic,
}

pub struct Neighborhood {
    offsets: Vec<[i64; 3]>,
}

impl Neighborhood {
    /// `order` = maximum Manhattan distance of an offset.
    /// 2D order 1 -> 4 (von Neumann), order 2 -> 8 (Moore).
    /// 3D order 1 -> 6, order 2 -> 18, order 3 -> 26.
    pub fn new(is_3d: bool, order: u8) -> Neighborhood {
        let zr: i64 = if is_3d { 1 } else { 0 };
        let mut offsets = Vec::new();
        for dz in -zr..=zr {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let manhattan = dx.abs() + dy.abs() + dz.abs();
                    if manhattan <= order as i64 {
                        offsets.push([dx, dy, dz]);
                    }
                }
            }
        }
        Neighborhood { offsets }
    }

    pub fn offsets(&self) -> &[[i64; 3]] {
        &self.offsets
    }
}

pub struct Lattice {
    pub dims: [usize; 3],
    pub boundary: [Boundary; 3],
    site: Vec<CellId>,
    pub nbr: Neighborhood,
}

impl Lattice {
    pub fn new(dims: [usize; 3], boundary: [Boundary; 3], nbr: Neighborhood) -> Lattice {
        let n = dims[0] * dims[1] * dims[2];
        Lattice { dims, boundary, site: vec![crate::MEDIUM; n], nbr }
    }

    pub fn n_sites(&self) -> usize {
        self.site.len()
    }

    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }

    pub fn coords(&self, idx: usize) -> [usize; 3] {
        let nx = self.dims[0];
        let ny = self.dims[1];
        let x = idx % nx;
        let y = (idx / nx) % ny;
        let z = idx / (nx * ny);
        [x, y, z]
    }

    #[inline]
    pub fn owner(&self, idx: usize) -> CellId {
        self.site[idx]
    }

    #[inline]
    pub fn set_owner(&mut self, idx: usize, c: CellId) {
        self.site[idx] = c;
    }

    /// Resolve one axis coordinate + offset under this axis' boundary.
    /// Returns None if NoFlux and out of range.
    #[inline]
    fn wrap(&self, coord: usize, off: i64, dim: usize, axis: usize) -> Option<usize> {
        let v = coord as i64 + off;
        match self.boundary[axis] {
            Boundary::NoFlux => {
                if v < 0 || v >= dim as i64 {
                    None
                } else {
                    Some(v as usize)
                }
            }
            Boundary::Periodic => Some(((v % dim as i64 + dim as i64) % dim as i64) as usize),
        }
    }

    pub fn neighbors(&self, idx: usize) -> Vec<usize> {
        let [x, y, z] = self.coords(idx);
        let mut out = Vec::with_capacity(self.nbr.offsets().len());
        for off in self.nbr.offsets() {
            let nx = match self.wrap(x, off[0], self.dims[0], 0) { Some(v) => v, None => continue };
            let ny = match self.wrap(y, off[1], self.dims[1], 1) { Some(v) => v, None => continue };
            let nz = match self.wrap(z, off[2], self.dims[2], 2) { Some(v) => v, None => continue };
            out.push(self.index(nx, ny, nz));
        }
        out
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p cpm-core lattice`
Expected: PASS — all 5 lattice tests ok.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/src/lattice.rs
git commit -m "feat(core): lattice indexing, neighborhoods, boundary conditions"
```

---

### Task 3: World + cells + volume/COM trackers (full recompute)

**Files:**
- Modify: `crates/cpm-core/src/world.rs`
- Test: inline in `world.rs`

**Interfaces:**
- Produces:
  - `pub struct Cell { pub id: CellId, pub cell_type: u16, pub volume: i64, pub surface: i64, pub com_sum: [f64;3], pub target_volume: f64, pub lambda_volume: f64, pub target_surface: f64, pub lambda_surface: f64 }`
  - `pub struct World { pub lattice: Lattice, pub cells: Vec<Cell>, pub temperature: f64 }`
  - `World::new(lattice: Lattice, temperature: f64) -> World` (creates `cells[0]` = medium sentinel).
  - `pub fn add_cell(&mut self, cell_type: u16, target_volume: f64, lambda_volume: f64, target_surface: f64, lambda_surface: f64) -> CellId`
  - `pub fn paint(&mut self, idx: usize, c: CellId)` — sets a site's owner (used only during setup, before trackers are computed).
  - `pub fn recompute_trackers(&mut self)` — full recompute of volume, surface, com_sum for all cells from the lattice.
  - `pub fn com(&self, c: CellId) -> [f64;3]` — `com_sum / volume` (returns `[0.0;3]` if volume 0).
- Consumes: `Lattice`, `Neighborhood`, `Boundary` from Task 2.

- [ ] **Step 1: Write the failing test**

Append to `crates/cpm-core/src/world.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};

    fn small_world() -> World {
        let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        World::new(lat, 10.0)
    }

    #[test]
    fn add_cell_and_paint_volume() {
        let mut w = small_world();
        let a = w.add_cell(1, 9.0, 1.0, 12.0, 1.0);
        // paint a 3x3 block -> volume 9
        for y in 1..4 {
            for x in 1..4 {
                let idx = w.lattice.index(x, y, 0);
                w.paint(idx, a);
            }
        }
        w.recompute_trackers();
        assert_eq!(w.cells[a as usize].volume, 9);
        assert_eq!(w.com(a), [2.0, 2.0, 0.0]);
    }

    #[test]
    fn surface_of_isolated_3x3_moore_is_correct() {
        // 3x3 block, Moore neighborhood: each of the 8 border sites has some
        // neighbors outside the cell; center has 0. Total unlike faces = 40.
        let mut w = small_world();
        let a = w.add_cell(1, 9.0, 1.0, 12.0, 1.0);
        for y in 1..4 {
            for x in 1..4 {
                let idx = w.lattice.index(x, y, 0);
                w.paint(idx, a);
            }
        }
        w.recompute_trackers();
        assert_eq!(w.cells[a as usize].surface, 40);
    }
}
```

(The 40 comes from: center 0; the 4 edge-midpoints have 5 unlike Moore-neighbors each = 20; the 4 corners have 5 unlike each = 20; total 40.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p cpm-core world`
Expected: FAIL — `World` not found.

- [ ] **Step 3: Write minimal implementation**

Replace `crates/cpm-core/src/world.rs` (keep the test module at the end):

```rust
use crate::lattice::Lattice;
use crate::{CellId, MEDIUM};

#[derive(Clone, Debug)]
pub struct Cell {
    pub id: CellId,
    pub cell_type: u16,
    pub volume: i64,
    pub surface: i64,
    pub com_sum: [f64; 3],
    pub target_volume: f64,
    pub lambda_volume: f64,
    pub target_surface: f64,
    pub lambda_surface: f64,
}

pub struct World {
    pub lattice: Lattice,
    pub cells: Vec<Cell>,
    pub temperature: f64,
}

impl World {
    pub fn new(lattice: Lattice, temperature: f64) -> World {
        let medium = Cell {
            id: MEDIUM,
            cell_type: 0,
            volume: 0,
            surface: 0,
            com_sum: [0.0; 3],
            target_volume: 0.0,
            lambda_volume: 0.0,
            target_surface: 0.0,
            lambda_surface: 0.0,
        };
        World { lattice, cells: vec![medium], temperature }
    }

    pub fn add_cell(
        &mut self,
        cell_type: u16,
        target_volume: f64,
        lambda_volume: f64,
        target_surface: f64,
        lambda_surface: f64,
    ) -> CellId {
        let id = self.cells.len() as CellId;
        self.cells.push(Cell {
            id,
            cell_type,
            volume: 0,
            surface: 0,
            com_sum: [0.0; 3],
            target_volume,
            lambda_volume,
            target_surface,
            lambda_surface,
        });
        id
    }

    pub fn paint(&mut self, idx: usize, c: CellId) {
        self.lattice.set_owner(idx, c);
    }

    pub fn recompute_trackers(&mut self) {
        for cell in self.cells.iter_mut() {
            cell.volume = 0;
            cell.surface = 0;
            cell.com_sum = [0.0; 3];
        }
        let n = self.lattice.n_sites();
        for idx in 0..n {
            let owner = self.lattice.owner(idx);
            let [x, y, z] = self.lattice.coords(idx);
            {
                let cell = &mut self.cells[owner as usize];
                cell.volume += 1;
                cell.com_sum[0] += x as f64;
                cell.com_sum[1] += y as f64;
                cell.com_sum[2] += z as f64;
            }
            let mut unlike = 0i64;
            for nidx in self.lattice.neighbors(idx) {
                if self.lattice.owner(nidx) != owner {
                    unlike += 1;
                }
            }
            self.cells[owner as usize].surface += unlike;
        }
    }

    pub fn com(&self, c: CellId) -> [f64; 3] {
        let cell = &self.cells[c as usize];
        if cell.volume == 0 {
            return [0.0; 3];
        }
        let v = cell.volume as f64;
        [cell.com_sum[0] / v, cell.com_sum[1] / v, cell.com_sum[2] / v]
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p cpm-core world`
Expected: PASS — both world tests ok.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/src/world.rs
git commit -m "feat(core): World, cells, full-recompute volume/surface/COM trackers"
```

---

### Task 4: Flip deltas + incremental `apply_flip`

**Files:**
- Modify: `crates/cpm-core/src/world.rs`
- Test: inline in `world.rs`

**Interfaces:**
- Produces (on `World`):
  - `pub fn surface_deltas(&self, site: usize, new_owner: CellId) -> Vec<(CellId, i64)>` — per-cell surface change if `site` were reassigned from its current owner to `new_owner`. Uses the Surface definition from Global Constraints.
  - `pub fn apply_flip(&mut self, site: usize, new_owner: CellId)` — reassigns `site` and updates volume, surface, com_sum for all affected cells incrementally. No-op if `new_owner == current owner`.
- Consumes: Task 3 `World`, `Cell`.

Delta derivation (implement exactly this):
- Let `A = owner(site)` (old), `B = new_owner`. Neighbors `N = lattice.neighbors(site)`.
- **Site term:** `A` loses site's contribution `= count_{q in N}[owner(q) != A]`; `B` gains `count_{q in N}[owner(q) != B]`.
- **Neighbor term:** for each `q in N` with `C = owner(q)`: `ΔS_C += [B != C] - [A != C]`.
- Accumulate into a map keyed by CellId.

- [ ] **Step 1: Write the failing test**

Append to `world.rs` test module:

```rust
#[test]
fn apply_flip_matches_full_recompute() {
    use crate::lattice::{Boundary, Lattice, Neighborhood};
    let lat = Lattice::new([6, 6, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 10.0);
    let a = w.add_cell(1, 9.0, 1.0, 12.0, 1.0);
    let b = w.add_cell(2, 9.0, 1.0, 12.0, 1.0);
    for y in 1..4 { for x in 1..4 { let i = w.lattice.index(x, y, 0); w.paint(i, a); } }
    for y in 1..4 { for x in 3..5 { let i = w.lattice.index(x, y, 0); w.paint(i, b); } }
    // fix overlap: column x=3 belongs to b above; repaint cleanly
    for y in 1..4 { let i = w.lattice.index(3, y, 0); w.paint(i, b); }
    w.recompute_trackers();

    // flip site (2,2) from a to b, then compare to a fresh full recompute
    let site = w.lattice.index(2, 2, 0);
    w.apply_flip(site, b);

    let mut ref_w = World::new(
        Lattice::new([6, 6, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2)),
        10.0,
    );
    ref_w.add_cell(1, 9.0, 1.0, 12.0, 1.0);
    ref_w.add_cell(2, 9.0, 1.0, 12.0, 1.0);
    for idx in 0..w.lattice.n_sites() {
        ref_w.paint(idx, w.lattice.owner(idx));
    }
    ref_w.recompute_trackers();

    for c in 0..w.cells.len() {
        assert_eq!(w.cells[c].volume, ref_w.cells[c].volume, "volume cell {c}");
        assert_eq!(w.cells[c].surface, ref_w.cells[c].surface, "surface cell {c}");
        for k in 0..3 {
            assert!((w.cells[c].com_sum[k] - ref_w.cells[c].com_sum[k]).abs() < 1e-9);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p cpm-core apply_flip`
Expected: FAIL — `apply_flip` not found.

- [ ] **Step 3: Write minimal implementation**

Add these methods inside `impl World` in `world.rs`:

```rust
    pub fn surface_deltas(&self, site: usize, new_owner: CellId) -> Vec<(CellId, i64)> {
        let a = self.lattice.owner(site);
        let b = new_owner;
        let mut acc: std::collections::HashMap<CellId, i64> = std::collections::HashMap::new();
        if a == b {
            return Vec::new();
        }
        let neighbors = self.lattice.neighbors(site);
        // Site term
        let mut unlike_a = 0i64;
        let mut unlike_b = 0i64;
        for &q in &neighbors {
            let c = self.lattice.owner(q);
            if c != a { unlike_a += 1; }
            if c != b { unlike_b += 1; }
        }
        *acc.entry(a).or_insert(0) -= unlike_a;
        *acc.entry(b).or_insert(0) += unlike_b;
        // Neighbor term
        for &q in &neighbors {
            let c = self.lattice.owner(q);
            let delta = (if b != c { 1 } else { 0 }) - (if a != c { 1 } else { 0 });
            *acc.entry(c).or_insert(0) += delta;
        }
        acc.into_iter().collect()
    }

    pub fn apply_flip(&mut self, site: usize, new_owner: CellId) {
        let a = self.lattice.owner(site);
        let b = new_owner;
        if a == b {
            return;
        }
        let deltas = self.surface_deltas(site, b);
        for (c, d) in deltas {
            self.cells[c as usize].surface += d;
        }
        let [x, y, z] = self.lattice.coords(site);
        // volume + com
        {
            let ca = &mut self.cells[a as usize];
            ca.volume -= 1;
            ca.com_sum[0] -= x as f64;
            ca.com_sum[1] -= y as f64;
            ca.com_sum[2] -= z as f64;
        }
        {
            let cb = &mut self.cells[b as usize];
            cb.volume += 1;
            cb.com_sum[0] += x as f64;
            cb.com_sum[1] += y as f64;
            cb.com_sum[2] += z as f64;
        }
        self.lattice.set_owner(site, b);
    }
```

Also add `use std::collections::HashMap;` is inlined via full path above, so no extra import needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p cpm-core apply_flip`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/src/world.rs
git commit -m "feat(core): incremental flip deltas + apply_flip (volume/surface/COM)"
```

---

### Task 5: Energy — Contact matrix + ΔH for a candidate copy

**Files:**
- Modify: `crates/cpm-core/src/energy.rs`
- Modify: `crates/cpm-core/src/world.rs` (add `contact: ContactMatrix` field to `World`)
- Test: inline in `energy.rs`

**Interfaces:**
- Produces:
  - `pub struct ContactMatrix { n_types: usize, j: Vec<f64> }` with `ContactMatrix::new(n_types: usize) -> Self` (all zeros), `pub fn set(&mut self, a: u16, b: u16, j: f64)` (symmetric), `pub fn get(&self, a: u16, b: u16) -> f64`.
  - On `World`: field `pub contact: ContactMatrix`; `World::new` initializes it to `ContactMatrix::new(1)` and a setter `pub fn set_contact_matrix(&mut self, m: ContactMatrix)`.
  - `pub fn delta_hamiltonian(&self, site: usize, new_owner: CellId) -> f64` on `World` — total ΔH = ΔVolume + ΔSurface + ΔContact for the candidate copy, computed WITHOUT mutating state.
- Consumes: Task 3/4 `World`, `surface_deltas`.

Energy formulas (implement exactly):
- **Volume:** per non-medium cell `c`, `E_v = λ_v (V_c - Vt_c)^2`. On flip, `A: V→V-1`, `B: V→V+1`. Sum the change over `{A, B}` skipping MEDIUM.
- **Surface:** per non-medium cell `c`, `E_s = λ_s (S_c - St_c)^2`. Use `surface_deltas(site, B)`; for each affected non-medium cell apply `λ_s((S+ΔS-St)^2 - (S-St)^2)`.
- **Contact:** `ΔContact = Σ_{q in N(site)} [ J(τ_B, τ(owner(q)))·(owner(q)≠B) − J(τ_A, τ(owner(q)))·(owner(q)≠A) ]`, where `τ(c)` is `cells[c].cell_type`.

- [ ] **Step 1: Write the failing test**

Replace `crates/cpm-core/src/energy.rs`:

```rust
use crate::world::World;
use crate::CellId;

pub struct ContactMatrix {
    n_types: usize,
    j: Vec<f64>,
}

impl ContactMatrix {
    pub fn new(n_types: usize) -> ContactMatrix {
        ContactMatrix { n_types: n_types.max(1), j: vec![0.0; n_types.max(1) * n_types.max(1)] }
    }
    pub fn set(&mut self, a: u16, b: u16, val: f64) {
        let (a, b) = (a as usize, b as usize);
        self.j[a * self.n_types + b] = val;
        self.j[b * self.n_types + a] = val;
    }
    pub fn get(&self, a: u16, b: u16) -> f64 {
        self.j[a as usize * self.n_types + b as usize]
    }
}

impl World {
    pub fn delta_hamiltonian(&self, site: usize, new_owner: CellId) -> f64 {
        let a = self.lattice.owner(site);
        let b = new_owner;
        if a == b {
            return 0.0;
        }
        // Volume
        let mut d = 0.0;
        for (c, dv) in [(a, -1i64), (b, 1i64)] {
            if c == crate::MEDIUM {
                continue;
            }
            let cell = &self.cells[c as usize];
            let before = cell.volume as f64;
            let after = (cell.volume + dv) as f64;
            d += cell.lambda_volume
                * ((after - cell.target_volume).powi(2) - (before - cell.target_volume).powi(2));
        }
        // Surface
        for (c, ds) in self.surface_deltas(site, b) {
            if c == crate::MEDIUM {
                continue;
            }
            let cell = &self.cells[c as usize];
            let before = cell.surface as f64;
            let after = (cell.surface + ds) as f64;
            d += cell.lambda_surface
                * ((after - cell.target_surface).powi(2) - (before - cell.target_surface).powi(2));
        }
        // Contact
        let ta = self.cells[a as usize].cell_type;
        let tb = self.cells[b as usize].cell_type;
        for q in self.lattice.neighbors(site) {
            let c = self.lattice.owner(q);
            let tc = self.cells[c as usize].cell_type;
            let after = if c != b { self.contact.get(tb, tc) } else { 0.0 };
            let before = if c != a { self.contact.get(ta, tc) } else { 0.0 };
            d += after - before;
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};

    #[test]
    fn contact_matrix_symmetric() {
        let mut m = ContactMatrix::new(3);
        m.set(1, 2, 5.0);
        assert_eq!(m.get(1, 2), 5.0);
        assert_eq!(m.get(2, 1), 5.0);
    }

    #[test]
    fn delta_h_volume_penalizes_growth_away_from_target() {
        // one cell at target volume; growing it by 1 raises volume energy.
        let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 9.0, 1.0, 0.0, 0.0); // target vol 9, no surface term
        for y in 1..4 { for x in 1..4 { let i = w.lattice.index(x,y,0); w.paint(i, a); } }
        w.recompute_trackers();
        assert_eq!(w.cells[a as usize].volume, 9);
        // medium site adjacent to the cell tries to become the cell -> volume 9->10
        let site = w.lattice.index(4, 2, 0); // medium, neighbor of (3,2)
        let dh = w.delta_hamiltonian(site, a);
        // ΔVolume = 1*((10-9)^2 - (9-9)^2) = 1 ; contact from medium term = 0 (J all zero)
        assert!((dh - 1.0).abs() < 1e-9, "dh was {dh}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p cpm-core energy`
Expected: FAIL — `World` has no `contact` field yet.

- [ ] **Step 3: Write minimal implementation**

In `world.rs`, add to `struct World` the field `pub contact: crate::energy::ContactMatrix,`. In `World::new`, initialize `contact: crate::energy::ContactMatrix::new(1),`. Add method:

```rust
    pub fn set_contact_matrix(&mut self, m: crate::energy::ContactMatrix) {
        self.contact = m;
    }
```

(The `energy.rs` body from Step 1 already defines everything else.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p cpm-core energy`
Expected: PASS — both energy tests ok.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/src/energy.rs crates/cpm-core/src/world.rs
git commit -m "feat(core): contact matrix + delta-Hamiltonian (volume/surface/contact)"
```

---

### Task 6: Metropolis sweep

**Files:**
- Modify: `crates/cpm-core/src/sweep.rs`
- Test: inline in `sweep.rs`

**Interfaces:**
- Produces:
  - `pub struct Cpm { pub world: World, rng: rand::rngs::SmallRng }`
  - `Cpm::new(world: World, seed: u64) -> Cpm`
  - `pub fn step(&mut self, mcs: u64)` — runs `mcs` Monte Carlo steps; one MCS = `n_sites` copy attempts. Each attempt: pick random target site `s`; pick a random neighbor `n` of `s`; if `owner(n) == owner(s)` skip; else `dh = delta_hamiltonian(s, owner(n))`; accept if `dh <= 0` or `rng.gen::<f64>() < exp(-dh / T)`; on accept `apply_flip(s, owner(n))`.
- Consumes: Task 5 `World`, `delta_hamiltonian`, `apply_flip`.

- [ ] **Step 1: Write the failing test**

Add `rand` dep is already in `cpm-core/Cargo.toml`. Ensure `SmallRng` feature: set `rand = { workspace = true, features = ["small_rng"] }` in `crates/cpm-core/Cargo.toml`.

Replace `crates/cpm-core/src/sweep.rs`:

```rust
use crate::world::World;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub struct Cpm {
    pub world: World,
    rng: SmallRng,
}

impl Cpm {
    pub fn new(world: World, seed: u64) -> Cpm {
        Cpm { world, rng: SmallRng::seed_from_u64(seed) }
    }

    pub fn step(&mut self, mcs: u64) {
        let n = self.world.lattice.n_sites();
        let t = self.world.temperature;
        for _ in 0..mcs {
            for _ in 0..n {
                let s = self.rng.gen_range(0..n);
                let neighbors = self.world.lattice.neighbors(s);
                if neighbors.is_empty() {
                    continue;
                }
                let pick = neighbors[self.rng.gen_range(0..neighbors.len())];
                let source_owner = self.world.lattice.owner(pick);
                if source_owner == self.world.lattice.owner(s) {
                    continue;
                }
                let dh = self.world.delta_hamiltonian(s, source_owner);
                let accept = dh <= 0.0 || self.rng.gen::<f64>() < (-dh / t).exp();
                if accept {
                    self.world.apply_flip(s, source_owner);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::ContactMatrix;
    use crate::lattice::{Boundary, Lattice, Neighborhood};

    #[test]
    fn deterministic_under_seed() {
        fn run(seed: u64) -> Vec<i64> {
            let lat = Lattice::new([20, 20, 1], [Boundary::Periodic; 3], Neighborhood::new(false, 2));
            let mut w = World::new(lat, 10.0);
            let a = w.add_cell(1, 25.0, 2.0, 20.0, 0.0);
            for y in 5..10 { for x in 5..10 { let i = w.lattice.index(x,y,0); w.paint(i, a); } }
            let mut m = ContactMatrix::new(2);
            m.set(0, 1, 16.0);
            w.set_contact_matrix(m);
            w.recompute_trackers();
            let mut cpm = Cpm::new(w, seed);
            cpm.step(5);
            cpm.world.cells.iter().map(|c| c.volume).collect()
        }
        assert_eq!(run(42), run(42));
    }

    #[test]
    fn single_cell_relaxes_toward_target_volume() {
        let lat = Lattice::new([30, 30, 1], [Boundary::Periodic; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 15.0);
        // start far above target: 100 sites, target 49
        let a = w.add_cell(1, 49.0, 5.0, 28.0, 0.0);
        for y in 5..15 { for x in 5..15 { let i = w.lattice.index(x,y,0); w.paint(i, a); } }
        let mut m = ContactMatrix::new(2);
        m.set(0, 1, 6.0);
        w.set_contact_matrix(m);
        w.recompute_trackers();
        let start = w.cells[a as usize].volume;
        let mut cpm = Cpm::new(w, 7);
        cpm.step(200);
        let end = cpm.world.cells[a as usize].volume;
        assert!(end < start, "volume should shrink toward target: {start} -> {end}");
        assert!((end - 49).abs() < 30, "should be near target 49, got {end}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p cpm-core sweep`
Expected: FAIL — `Cpm` not found (and possibly a `small_rng` feature error until Cargo.toml is updated).

- [ ] **Step 3: Write minimal implementation**

The `sweep.rs` body is written in Step 1. Ensure `crates/cpm-core/Cargo.toml` dependency line reads:

```toml
rand = { workspace = true, features = ["small_rng"] }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p cpm-core sweep`
Expected: PASS — both sweep tests ok.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/src/sweep.rs crates/cpm-core/Cargo.toml
git commit -m "feat(core): seeded Metropolis sweep (Cpm::step)"
```

---

### Task 7: Property test — incremental trackers equal full recompute

**Files:**
- Modify: `crates/cpm-core/src/sweep.rs` (add test) OR new `crates/cpm-core/tests/property.rs`
- Test: `crates/cpm-core/tests/property.rs`

**Interfaces:**
- Consumes: `Cpm`, `World`, `recompute_trackers`. No new production code — this is the correctness gate for Global Constraints' incremental-tracker invariant.

- [ ] **Step 1: Write the failing test**

Create `crates/cpm-core/tests/property.rs`:

```rust
use cpm_core::energy::ContactMatrix;
use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::sweep::Cpm;
use cpm_core::world::World;

#[test]
fn incremental_equals_full_recompute_after_random_sweeps() {
    let lat = Lattice::new([25, 25, 1], [Boundary::Periodic; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 12.0);
    for k in 0..6 {
        let a = w.add_cell((1 + k % 2) as u16, 20.0, 2.0, 18.0, 0.5);
        let ox = 2 + (k % 3) * 7;
        let oy = 2 + (k / 3) * 10;
        for y in oy..oy + 4 {
            for x in ox..ox + 4 {
                let i = w.lattice.index(x, y, 0);
                w.paint(i, a);
            }
        }
    }
    let mut m = ContactMatrix::new(3);
    m.set(0, 1, 10.0);
    m.set(0, 2, 10.0);
    m.set(1, 2, 14.0);
    w.set_contact_matrix(m);
    w.recompute_trackers();

    let mut cpm = Cpm::new(w, 99);
    cpm.step(50);

    // snapshot incremental trackers
    let inc: Vec<(i64, i64, [f64; 3])> =
        cpm.world.cells.iter().map(|c| (c.volume, c.surface, c.com_sum)).collect();

    // full recompute and compare
    cpm.world.recompute_trackers();
    for (i, c) in cpm.world.cells.iter().enumerate() {
        assert_eq!(inc[i].0, c.volume, "volume drift cell {i}");
        assert_eq!(inc[i].1, c.surface, "surface drift cell {i}");
        for k in 0..3 {
            assert!((inc[i].2[k] - c.com_sum[k]).abs() < 1e-6, "com drift cell {i}");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails or errors**

Run: `cargo test -p cpm-core --test property`
Expected: initially may FAIL to compile if `pub mod` visibility is missing. Ensure `lib.rs` has `pub mod lattice; pub mod world; pub mod energy; pub mod sweep;` (already set in Task 1). If trackers are correct it should PASS; if there's any drift bug it FAILS with "drift" — fix the incremental code until green.

- [ ] **Step 3: Fix any drift**

If the test fails, the bug is in `surface_deltas`/`apply_flip` (Task 4). Re-derive against the Surface definition and fix. Do not weaken the test.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p cpm-core --test property`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-core/tests/property.rs
git commit -m "test(core): property test — incremental trackers == full recompute"
```

---

### Task 8: pyo3 bindings — `cpm_core` module

**Files:**
- Modify: `crates/cpm-py/Cargo.toml`
- Modify: `crates/cpm-py/src/lib.rs`
- Create: `pyproject.toml`
- Test: `tests/test_bindings.py`

**Interfaces:**
- Produces a Python module `cpm_core` exposing one class `World`:
  - `World(dims: tuple[int,int,int], boundary: str, neighbor_order: int, temperature: float)` — `boundary` in `{"noflux","periodic"}` (applied to all axes).
  - `.add_cell(cell_type: int, target_volume: float, lambda_volume: float, target_surface: float, lambda_surface: float) -> int`
  - `.set_contact(type_a: int, type_b: int, j: float)`
  - `.seed_block(cell_id: int, x0:int,y0:int,z0:int, x1:int,y1:int,z1:int)` — paint an axis-aligned block `[x0,x1) × [y0,y1) × [z0,z1)`.
  - `.finalize(seed: int)` — call `recompute_trackers` and build the internal `Cpm` with `seed`. Must be called after all seeding, before `step`.
  - `.step(mcs: int)`
  - `.cell_volumes() -> list[int]`, `.cell_surfaces() -> list[int]`, `.cell_types() -> list[int]`, `.cell_coms() -> list[tuple[float,float,float]]`
  - `.snapshot() -> list[int]` — flat lattice owner array, length `nx*ny*nz`, index order per Global Constraints.
  - `.dims() -> tuple[int,int,int]`
- Consumes: `cpm-core` crate.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bindings.py`:

```python
import cpm_core


def test_single_cell_shrinks_to_target():
    w = cpm_core.World((30, 30, 1), "periodic", 2, 15.0)
    a = w.add_cell(1, 49.0, 5.0, 28.0, 0.0)
    w.set_contact(0, 1, 6.0)
    w.seed_block(a, 5, 5, 0, 15, 15, 1)   # 100 sites
    w.finalize(7)
    start = w.cell_volumes()[a]
    w.step(200)
    end = w.cell_volumes()[a]
    assert end < start
    assert abs(end - 49) < 30


def test_snapshot_shape_and_determinism():
    def run():
        w = cpm_core.World((20, 20, 1), "periodic", 2, 10.0)
        a = w.add_cell(1, 25.0, 2.0, 20.0, 0.0)
        w.set_contact(0, 1, 16.0)
        w.seed_block(a, 5, 5, 0, 10, 10, 1)
        w.finalize(42)
        w.step(5)
        return w.snapshot()
    s = run()
    assert len(s) == 400
    assert run() == s
```

- [ ] **Step 2: Set up the build + run test to verify it fails**

`crates/cpm-py/Cargo.toml`:

```toml
[package]
name = "cpm-py"
version.workspace = true
edition.workspace = true

[lib]
name = "cpm_core"
crate-type = ["cdylib"]

[dependencies]
cpm-core = { workspace = true }
pyo3 = { version = "0.22", features = ["extension-module"] }
```

`pyproject.toml` (repo root):

```toml
[build-system]
requires = ["maturin>=1.5,<2"]
build-backend = "maturin"

[project]
name = "pbg-cpm"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["process-bigraph"]

[tool.maturin]
manifest-path = "crates/cpm-py/Cargo.toml"
module-name = "cpm_core"
python-source = "."
```

Build the environment:

```bash
cd /Users/eranagmon/code/pbg-cpm
uv venv --python 3.12
source .venv/bin/activate
uv pip install maturin pytest process-bigraph
maturin develop
```

Run: `pytest tests/test_bindings.py -v`
Expected: FAIL/ERROR — `cpm_core` has no `World` (placeholder lib) until Step 3.

- [ ] **Step 3: Write the binding**

Replace `crates/cpm-py/src/lib.rs`:

```rust
use cpm_core::energy::ContactMatrix;
use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::sweep::Cpm;
use cpm_core::world::World as CoreWorld;
use pyo3::prelude::*;

#[pyclass]
struct World {
    // Before finalize(): hold the core world. After: hold the Cpm driver.
    core: Option<CoreWorld>,
    cpm: Option<Cpm>,
    dims: [usize; 3],
    max_type: u16,
    contacts: Vec<(u16, u16, f64)>,
}

impl World {
    fn world_ref(&self) -> &CoreWorld {
        if let Some(c) = &self.cpm { &c.world } else { self.core.as_ref().unwrap() }
    }
}

#[pymethods]
impl World {
    #[new]
    fn new(dims: (usize, usize, usize), boundary: &str, neighbor_order: u8, temperature: f64) -> PyResult<Self> {
        let b = match boundary {
            "noflux" => Boundary::NoFlux,
            "periodic" => Boundary::Periodic,
            other => return Err(pyo3::exceptions::PyValueError::new_err(format!("bad boundary {other}"))),
        };
        let dims = [dims.0, dims.1, dims.2];
        let is_3d = dims[2] > 1;
        let lat = Lattice::new(dims, [b; 3], Neighborhood::new(is_3d, neighbor_order));
        let core = CoreWorld::new(lat, temperature);
        Ok(World { core: Some(core), cpm: None, dims, max_type: 0, contacts: Vec::new() })
    }

    fn add_cell(&mut self, cell_type: u16, target_volume: f64, lambda_volume: f64, target_surface: f64, lambda_surface: f64) -> PyResult<u32> {
        let core = self.core.as_mut().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("add_cell after finalize"))?;
        self.max_type = self.max_type.max(cell_type);
        Ok(core.add_cell(cell_type, target_volume, lambda_volume, target_surface, lambda_surface))
    }

    fn set_contact(&mut self, type_a: u16, type_b: u16, j: f64) {
        self.max_type = self.max_type.max(type_a).max(type_b);
        self.contacts.push((type_a, type_b, j));
    }

    fn seed_block(&mut self, cell_id: u32, x0: usize, y0: usize, z0: usize, x1: usize, y1: usize, z1: usize) -> PyResult<()> {
        let core = self.core.as_mut().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("seed after finalize"))?;
        for z in z0..z1 {
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = core.lattice.index(x, y, z);
                    core.paint(idx, cell_id);
                }
            }
        }
        Ok(())
    }

    fn finalize(&mut self, seed: u64) -> PyResult<()> {
        let mut core = self.core.take().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("already finalized"))?;
        let mut m = ContactMatrix::new((self.max_type as usize) + 1);
        for (a, b, j) in &self.contacts {
            m.set(*a, *b, *j);
        }
        core.set_contact_matrix(m);
        core.recompute_trackers();
        self.cpm = Some(Cpm::new(core, seed));
        Ok(())
    }

    fn step(&mut self, mcs: u64) -> PyResult<()> {
        let cpm = self.cpm.as_mut().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("call finalize() first"))?;
        cpm.step(mcs);
        Ok(())
    }

    fn cell_volumes(&self) -> Vec<i64> {
        self.world_ref().cells.iter().map(|c| c.volume).collect()
    }
    fn cell_surfaces(&self) -> Vec<i64> {
        self.world_ref().cells.iter().map(|c| c.surface).collect()
    }
    fn cell_types(&self) -> Vec<u16> {
        self.world_ref().cells.iter().map(|c| c.cell_type).collect()
    }
    fn cell_coms(&self) -> Vec<(f64, f64, f64)> {
        let w = self.world_ref();
        w.cells.iter().map(|c| {
            let com = w.com(c.id);
            (com[0], com[1], com[2])
        }).collect()
    }
    fn snapshot(&self) -> Vec<u32> {
        let w = self.world_ref();
        (0..w.lattice.n_sites()).map(|i| w.lattice.owner(i)).collect()
    }
    fn dims(&self) -> (usize, usize, usize) {
        (self.dims[0], self.dims[1], self.dims[2])
    }
}

#[pymodule]
fn cpm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    Ok(())
}
```

Rebuild: `maturin develop`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bindings.py -v`
Expected: PASS — both binding tests ok.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-py/ pyproject.toml tests/test_bindings.py
git commit -m "feat(py): pyo3 cpm_core bindings (World construct/seed/step/readback)"
```

---

### Task 9: `cpm.pack` export seam

**Files:**
- Create: `cpm/__init__.py`
- Create: `cpm/pack.py`
- Test: `tests/test_pack.py`

**Interfaces:**
- Produces: `cpm.pack.write_pack(world: "cpm_core.World", path: str, frame: int = 0) -> dict` — writes a JSON `cpm.pack` v1 file and returns the dict. Schema:
  ```json
  {"format": "cpm.pack.v1", "dims": [nx,ny,nz], "frame": 0,
   "cell_types": [int,...], "labels": [cellid,...]}
  ```
  `labels` is the flat lattice snapshot (index order per Global Constraints). This is the seam the Sub-project F viewer will consume; surface extraction happens later in F, not here.
- Consumes: `cpm_core.World` from Task 8.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pack.py`:

```python
import json
import cpm_core
from cpm.pack import write_pack


def test_write_pack_roundtrip(tmp_path):
    w = cpm_core.World((8, 8, 1), "noflux", 2, 10.0)
    a = w.add_cell(1, 9.0, 1.0, 12.0, 1.0)
    w.seed_block(a, 2, 2, 0, 5, 5, 1)
    w.finalize(1)
    p = tmp_path / "frame.cpm.json"
    d = write_pack(w, str(p))
    assert d["format"] == "cpm.pack.v1"
    assert d["dims"] == [8, 8, 1]
    assert len(d["labels"]) == 64
    on_disk = json.loads(p.read_text())
    assert on_disk == d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pack.py -v`
Expected: FAIL — `cpm` package / `write_pack` not found.

- [ ] **Step 3: Write minimal implementation**

`cpm/__init__.py`:
```python
"""pbg-cpm: process-bigraph Cellular Potts framework (Python layer)."""
```

`cpm/pack.py`:
```python
import json


def write_pack(world, path, frame=0):
    nx, ny, nz = world.dims()
    data = {
        "format": "cpm.pack.v1",
        "dims": [nx, ny, nz],
        "frame": frame,
        "cell_types": list(world.cell_types()),
        "labels": list(world.snapshot()),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return data
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pack.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cpm/__init__.py cpm/pack.py tests/test_pack.py
git commit -m "feat(py): cpm.pack v1 export seam for the viewer"
```

---

### Task 10: Schema loader — build a world from a spec dict

**Files:**
- Create: `cpm/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: `cpm.schema.load_world(spec: dict) -> cpm_core.World` (finalized and ready to `step`). Minimal schema (a slice of Sub-project C):
  ```json
  {
    "potts": {"dims": [nx,ny,nz], "boundary": "periodic",
              "neighbor_order": 2, "temperature": 10.0, "seed": 42},
    "cell_types": [{"name":"medium","id":0},
                   {"name":"dark","id":1},{"name":"light","id":2}],
    "contact": [{"a":0,"b":1,"j":16.0}, {"a":0,"b":2,"j":16.0},
                {"a":1,"b":2,"j":11.0}, {"a":1,"b":1,"j":2.0}, {"a":2,"b":2,"j":2.0}],
    "cells": [{"type":1,"target_volume":25,"lambda_volume":2.0,
               "target_surface":20,"lambda_surface":0.0,
               "seed_block":[x0,y0,z0,x1,y1,z1]}, ...]
  }
  ```
  `load_world` builds the `cpm_core.World`, adds cells, sets contacts, seeds blocks, calls `finalize(seed)`.
- Consumes: `cpm_core.World` (Task 8).

- [ ] **Step 1: Write the failing test**

Create `tests/test_schema.py`:

```python
from cpm.schema import load_world


def test_load_world_builds_and_steps():
    spec = {
        "potts": {"dims": [24, 24, 1], "boundary": "periodic",
                  "neighbor_order": 2, "temperature": 12.0, "seed": 5},
        "cell_types": [{"name": "medium", "id": 0}, {"name": "a", "id": 1}],
        "contact": [{"a": 0, "b": 1, "j": 8.0}, {"a": 1, "b": 1, "j": 2.0}],
        "cells": [
            {"type": 1, "target_volume": 16, "lambda_volume": 2.0,
             "target_surface": 16, "lambda_surface": 0.0,
             "seed_block": [4, 4, 0, 10, 10, 1]},
        ],
    }
    w = load_world(spec)
    v0 = w.cell_volumes()[1]
    w.step(50)
    v1 = w.cell_volumes()[1]
    assert v0 == 36           # 6x6 seed block
    assert v1 != v0           # dynamics ran
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL — `load_world` not found.

- [ ] **Step 3: Write minimal implementation**

`cpm/schema.py`:
```python
import cpm_core


def load_world(spec):
    p = spec["potts"]
    dims = tuple(p["dims"])
    world = cpm_core.World(dims, p["boundary"], int(p["neighbor_order"]), float(p["temperature"]))
    for c in spec.get("cells", []):
        cid = world.add_cell(
            int(c["type"]),
            float(c["target_volume"]),
            float(c["lambda_volume"]),
            float(c["target_surface"]),
            float(c["lambda_surface"]),
        )
        c["_id"] = cid  # remember assigned id
    for pair in spec.get("contact", []):
        world.set_contact(int(pair["a"]), int(pair["b"]), float(pair["j"]))
    for c in spec.get("cells", []):
        x0, y0, z0, x1, y1, z1 = c["seed_block"]
        world.seed_block(c["_id"], x0, y0, z0, x1, y1, z1)
    world.finalize(int(p["seed"]))
    return world
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cpm/schema.py tests/test_schema.py
git commit -m "feat(py): minimal schema loader (potts/cell_types/contact/cells)"
```

---

### Task 11: `CPMProcess` — process-bigraph wrapper

**Files:**
- Create: `cpm/processes/__init__.py`
- Create: `cpm/processes/cpm_process.py`
- Test: `tests/test_process.py`

**Interfaces:**
- Produces: `cpm.processes.cpm_process.CPMProcess(process_bigraph.Process)`:
  - `config_schema` includes `spec` (the schema dict from Task 10) and `mcs_per_update` (int, default 10).
  - `inputs()` → `{}` (self-contained for A; later versions read target_volume/type from the store).
  - `outputs()` → `{"cell_volumes": "list", "cell_surfaces": "list", "cell_coms": "list"}`.
  - `update(inputs, interval)` → steps `mcs_per_update` MCS and returns the three lists.
- Consumes: `cpm.schema.load_world` (Task 10).

**NOTE for implementer:** before writing, confirm the exact `process_bigraph.Process` base-class method names in the installed version:
`python -c "import process_bigraph, inspect; print([m for m in dir(process_bigraph.Process) if not m.startswith('__')])"`
The plan uses `config_schema` (class attr) + `inputs`/`outputs`/`update`. If the installed API differs (e.g. `initial_state`, or `update(self, state, interval)` signature), adapt the method names to match — the behavior (load world in `__init__`, step in `update`, return readback lists) stays the same.

- [ ] **Step 1: Write the failing test**

Create `tests/test_process.py`:

```python
from cpm.processes.cpm_process import CPMProcess


SPEC = {
    "potts": {"dims": [24, 24, 1], "boundary": "periodic",
              "neighbor_order": 2, "temperature": 12.0, "seed": 3},
    "cell_types": [{"name": "medium", "id": 0}, {"name": "a", "id": 1}],
    "contact": [{"a": 0, "b": 1, "j": 8.0}, {"a": 1, "b": 1, "j": 2.0}],
    "cells": [{"type": 1, "target_volume": 16, "lambda_volume": 2.0,
               "target_surface": 16, "lambda_surface": 0.0,
               "seed_block": [4, 4, 0, 10, 10, 1]}],
}


def test_process_update_returns_readback():
    proc = CPMProcess({"spec": SPEC, "mcs_per_update": 5})
    out = proc.update({}, 1.0)
    assert "cell_volumes" in out
    assert len(out["cell_volumes"]) == 2   # medium + 1 cell
    assert out["cell_volumes"][1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_process.py -v`
Expected: FAIL — `CPMProcess` not found.

- [ ] **Step 3: Write minimal implementation**

`cpm/processes/__init__.py`: empty file.

`cpm/processes/cpm_process.py`:
```python
from process_bigraph import Process
from cpm.schema import load_world


class CPMProcess(Process):
    config_schema = {
        "spec": "tree",
        "mcs_per_update": {"_type": "integer", "_default": 10},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self.world = load_world(self.config["spec"])
        self.mcs = int(self.config["mcs_per_update"])

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "cell_volumes": "list",
            "cell_surfaces": "list",
            "cell_coms": "list",
        }

    def update(self, inputs, interval):
        self.world.step(self.mcs)
        return {
            "cell_volumes": list(self.world.cell_volumes()),
            "cell_surfaces": list(self.world.cell_surfaces()),
            "cell_coms": [list(c) for c in self.world.cell_coms()],
        }
```

If Step 3's `super().__init__` signature or `config_schema` type names mismatch the installed process-bigraph (verified in the NOTE), adjust to match; keep behavior identical.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_process.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cpm/processes/ tests/test_process.py
git commit -m "feat(py): CPMProcess process-bigraph wrapper"
```

---

### Task 12: 2D cell-sorting demo + regression test

**Files:**
- Create: `demos/cell_sorting_2d.json`
- Create: `tests/test_cell_sorting.py`
- Create: `cpm/metrics.py` (heterotypic boundary length helper)
- Test: `tests/test_cell_sorting.py`

**Interfaces:**
- Produces: `cpm.metrics.heterotypic_boundary(world) -> int` — counts lattice faces between two cells of DIFFERENT non-medium types (the cell-sorting order parameter). Uses `snapshot()` + `dims()` + `cell_types()`; counts each unordered face once, 4-neighbor (von Neumann) faces only, cells only (skip medium-involving faces).
- Consumes: `cpm_core.World`, `cpm.schema.load_world`.

The demo: two cell types with strong self-adhesion (low J within type) and weak cross-adhesion (high J between types) sort into segregated domains → heterotypic boundary shrinks over time.

- [ ] **Step 1: Write the failing test**

Create `demos/cell_sorting_2d.json`. Generate the `cells` list programmatically is easier, but the demo must be a static schema file. Use a 60×60 lattice with a checkerboard of ~50 small cells of two types. To keep the file human-authored yet compact, place a 5×5 grid of 10-wide cells (25 cells), alternating type by parity:

```json
{
  "potts": {"dims": [60, 60, 1], "boundary": "periodic",
            "neighbor_order": 2, "temperature": 10.0, "seed": 17},
  "cell_types": [{"name": "medium", "id": 0},
                 {"name": "dark", "id": 1}, {"name": "light", "id": 2}],
  "contact": [{"a": 0, "b": 1, "j": 16.0}, {"a": 0, "b": 2, "j": 16.0},
              {"a": 1, "b": 1, "j": 2.0}, {"a": 2, "b": 2, "j": 2.0},
              {"a": 1, "b": 2, "j": 11.0}],
  "cells": []
}
```

Leave `cells` empty in the file and fill it in the test via a helper so the file stays small, OR (preferred) commit a generated file. Use this test which builds the cell grid, writes the demo file, then asserts sorting:

```python
import json
import os
from cpm.schema import load_world
from cpm.metrics import heterotypic_boundary

DEMO = os.path.join(os.path.dirname(__file__), "..", "demos", "cell_sorting_2d.json")


def _build_spec():
    with open(DEMO) as f:
        spec = json.load(f)
    cells = []
    n = 5           # 5x5 grid
    w = 10          # cell width
    for gy in range(n):
        for gx in range(n):
            t = 1 + ((gx + gy) % 2)
            x0, y0 = 2 + gx * w, 2 + gy * w
            cells.append({
                "type": t, "target_volume": 64, "lambda_volume": 2.0,
                "target_surface": 40, "lambda_surface": 0.0,
                "seed_block": [x0, y0, 0, x0 + 8, y0 + 8, 1],
            })
    spec["cells"] = cells
    return spec


def test_cell_sorting_reduces_heterotypic_boundary():
    spec = _build_spec()
    world = load_world(spec)
    start = heterotypic_boundary(world)
    world.step(400)
    end = heterotypic_boundary(world)
    assert end < start, f"sorting should reduce heterotypic boundary: {start} -> {end}"
    assert end < 0.85 * start


def test_cell_sorting_deterministic():
    spec = _build_spec()
    w1 = load_world(spec); w1.step(50)
    w2 = load_world(spec); w2.step(50)
    assert w1.snapshot() == w2.snapshot()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cell_sorting.py -v`
Expected: FAIL — `cpm.metrics` not found.

- [ ] **Step 3: Write minimal implementation**

`cpm/metrics.py`:
```python
def heterotypic_boundary(world):
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    types = world.cell_types()

    def idx(x, y, z):
        return x + y * nx + z * nx * ny

    count = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                a = labels[idx(x, y, z)]
                # +x and +y faces only (each unordered face once); periodic wrap
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    if dz and nz == 1:
                        continue
                    nxp, nyp, nzp = (x + dx) % nx, (y + dy) % ny, (z + dz) % nz
                    b = labels[idx(nxp, nyp, nzp)]
                    if a == b:
                        continue
                    if a == 0 or b == 0:
                        continue
                    if types[a] != types[b]:
                        count += 1
    return count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cell_sorting.py -v`
Expected: PASS — both tests ok. (If `end < 0.85*start` is flaky, raise MCS to 600 or lower cross-adhesion J; do NOT delete the assertion.)

- [ ] **Step 5: Commit**

```bash
git add demos/cell_sorting_2d.json cpm/metrics.py tests/test_cell_sorting.py
git commit -m "feat(demo): 2D cell sorting — schema + heterotypic-boundary regression test"
```

---

### Task 13: 3D cell-sorting demo + test

**Files:**
- Create: `demos/cell_sorting_3d.json`
- Modify: `tests/test_cell_sorting.py` (add a 3D test)

**Interfaces:**
- Consumes: `load_world`, `heterotypic_boundary` (both already dimension-generic).

- [ ] **Step 1: Write the failing test**

Create `demos/cell_sorting_3d.json`:
```json
{
  "potts": {"dims": [30, 30, 30], "boundary": "periodic",
            "neighbor_order": 2, "temperature": 10.0, "seed": 23},
  "cell_types": [{"name": "medium", "id": 0},
                 {"name": "dark", "id": 1}, {"name": "light", "id": 2}],
  "contact": [{"a": 0, "b": 1, "j": 16.0}, {"a": 0, "b": 2, "j": 16.0},
              {"a": 1, "b": 1, "j": 2.0}, {"a": 2, "b": 2, "j": 2.0},
              {"a": 1, "b": 2, "j": 11.0}],
  "cells": []
}
```

Add to `tests/test_cell_sorting.py`:
```python
DEMO3D = os.path.join(os.path.dirname(__file__), "..", "demos", "cell_sorting_3d.json")


def _build_spec_3d():
    with open(DEMO3D) as f:
        spec = json.load(f)
    cells = []
    n = 3          # 3x3x3 grid of cells
    w = 9
    for gz in range(n):
        for gy in range(n):
            for gx in range(n):
                t = 1 + ((gx + gy + gz) % 2)
                x0, y0, z0 = 1 + gx * w, 1 + gy * w, 1 + gz * w
                cells.append({
                    "type": t, "target_volume": 343, "lambda_volume": 1.0,
                    "target_surface": 0, "lambda_surface": 0.0,
                    "seed_block": [x0, y0, z0, x0 + 7, y0 + 7, z0 + 7],
                })
    spec["cells"] = cells
    return spec


def test_cell_sorting_3d_runs_and_sorts():
    spec = _build_spec_3d()
    world = load_world(spec)
    start = heterotypic_boundary(world)
    world.step(60)
    end = heterotypic_boundary(world)
    assert end <= start, f"3D sorting should not increase heterotypic boundary: {start} -> {end}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cell_sorting.py::test_cell_sorting_3d_runs_and_sorts -v`
Expected: FAIL — `demos/cell_sorting_3d.json` missing until created (or KeyError). Create the file, rerun.

- [ ] **Step 3: Confirm it runs**

The demo file is the implementation. Run the test; if the 3D run is slow, reduce `world.step(60)` to `world.step(30)` — but it must complete in well under a minute (this is also the informal 3D-perf sanity check). Keep `end <= start`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cell_sorting.py -v`
Expected: PASS — all cell-sorting tests (2D + 3D) ok.

- [ ] **Step 5: Commit**

```bash
git add demos/cell_sorting_3d.json tests/test_cell_sorting.py
git commit -m "feat(demo): 3D cell sorting demo + test"
```

---

### Task 14: `cpm-bench` — 3D single-threaded sweeps/sec baseline

**Files:**
- Modify: `crates/cpm-bench/Cargo.toml`
- Modify: `crates/cpm-bench/src/main.rs`

**Interfaces:**
- Produces: a CLI binary that builds a 3D world (default 50×50×50, ~100 cells), runs a fixed number of MCS, and prints `MCS/sec` and `pixel-copy-attempts/sec`. This is the "better in 3D" scorecard baseline; no threading yet.
- Consumes: `cpm-core`.

- [ ] **Step 1: Write a smoke assertion (bench is a bin, so gate via a run)**

`crates/cpm-bench/Cargo.toml`:
```toml
[package]
name = "cpm-bench"
version.workspace = true
edition.workspace = true

[[bin]]
name = "cpm-bench"
path = "src/main.rs"

[dependencies]
cpm-core = { workspace = true }
```

`crates/cpm-bench/src/main.rs`:
```rust
use cpm_core::energy::ContactMatrix;
use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::sweep::Cpm;
use cpm_core::world::World;
use std::time::Instant;

fn main() {
    let dim = 50usize;
    let mcs = 20u64;
    let lat = Lattice::new([dim, dim, dim], [Boundary::Periodic; 3], Neighborhood::new(true, 2));
    let mut w = World::new(lat, 10.0);
    // ~125 cells in a 5x5x5 grid of 8^3 blocks
    let step = dim / 5;
    for gz in 0..5 {
        for gy in 0..5 {
            for gx in 0..5 {
                let t = 1 + ((gx + gy + gz) % 2) as u16;
                let a = w.add_cell(t, 512.0, 1.0, 0.0, 0.0);
                let (x0, y0, z0) = (gx * step, gy * step, gz * step);
                for z in z0..z0 + 8 {
                    for y in y0..y0 + 8 {
                        for x in x0..x0 + 8 {
                            let i = w.lattice.index(x, y, z);
                            w.paint(i, a);
                        }
                    }
                }
            }
        }
    }
    let mut m = ContactMatrix::new(3);
    m.set(0, 1, 16.0);
    m.set(0, 2, 16.0);
    m.set(1, 2, 11.0);
    w.set_contact_matrix(m);
    w.recompute_trackers();

    let n_sites = w.lattice.n_sites();
    let mut cpm = Cpm::new(w, 1);
    let t0 = Instant::now();
    cpm.step(mcs);
    let secs = t0.elapsed().as_secs_f64();
    let attempts = mcs as f64 * n_sites as f64;
    println!("3D bench {dim}^3, {mcs} MCS: {:.2} MCS/s, {:.2e} copy-attempts/s",
             mcs as f64 / secs, attempts / secs);
}
```

- [ ] **Step 2: Run it (verify it builds and prints)**

Run: `cargo run -p cpm-bench --release`
Expected: prints a line like `3D bench 50^3, 20 MCS: X.XX MCS/s, Y.YYe6 copy-attempts/s`. Record the number in the commit message.

- [ ] **Step 3: (no separate impl step — the bin is the deliverable)**

- [ ] **Step 4: Run the full test suite green**

Run: `cargo test && pytest -q`
Expected: all Rust + Python tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/cpm-bench/
git commit -m "feat(bench): 3D single-threaded sweeps/sec baseline"
```

---

## Self-Review

**Spec coverage (Sub-project A acceptance):**
- 2D cell sorting from a schema file, monotonic-ish decrease + threshold + deterministic → Task 12. ✓
- 3D cell sorting runs and sorts → Task 13. ✓
- Property test incremental == full recompute → Task 7. ✓
- `cpm-bench` 3D single-threaded baseline → Task 14. ✓
- Energy plugins Volume/Surface/Contact → Task 5. ✓ (Chemotaxis/Connectivity correctly deferred to B per spec.)
- `cpm.pack` export seam → Task 9. ✓
- pyo3 binding + CPMProcess pbg wrapper + schema loader → Tasks 8, 10, 11. ✓
- Dimension-generic (2D = 3D with nz=1) → enforced throughout (Neighborhood `is_3d`, block seeding). ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. The one external-API uncertainty (process-bigraph Process method names) is handled with an explicit verification command in Task 11, not a placeholder.

**Type consistency:** `CellId=u32` and `MEDIUM=0` consistent across all crates. `Neighborhood::new(is_3d, order)`, `Lattice::neighbors`, `World::{add_cell,paint,recompute_trackers,surface_deltas,apply_flip,delta_hamiltonian,set_contact_matrix}`, `Cpm::{new,step}`, and the pyo3 `World` method names all match between definition (Tasks 2–8) and use (Tasks 9–14). Python `load_world`/`heterotypic_boundary`/`write_pack`/`CPMProcess` signatures consistent across Tasks 9–13.

**Scope:** Single sub-project (CPM core), one testable deliverable per task, 14 tasks. Fields/subcellular/viewer/parallelism all correctly out of scope.
