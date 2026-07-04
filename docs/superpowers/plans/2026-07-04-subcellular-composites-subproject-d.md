# Sub-project D — Per-cell Subcellular Composites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give CPM cells per-cell subcellular models (SBML ODE + Boolean network) that sense the local environment and drive phenotype, all assembled and run as one process-bigraph `Composite`, demonstrated as crypt differentiation on the HRA colonic-crypt FTU.

**Architecture:** One `process_bigraph.Composite` holds the Rust-backed `CPMProcess` plus per-cell subcellular processes wired to it. A differentiation event is a per-cell `cell_type` switch (adhesion/secretion/chemotaxis follow type), so the engine gains one new method `set_cell_type`. `SBMLSubcell` wraps `pbg_tellurium.TelluriumProcess`; `BooleanSubcell` is a small new process.

**Tech Stack:** Rust (cpm-core) + pyo3/maturin (`cpm_core`), Python pkg `cpm`, `process_bigraph` (Composite engine), `pbg_tellurium`/`libroadrunner`, three.js viewer.

## Global Constraints

- The whole simulation MUST be assembled as a `process_bigraph.Composite` and advanced with `Composite.run(...)` — no batched integrator that bypasses the bigraph. Every subcellular model is a genuine `process_bigraph.Process`.
- A differentiation event is a per-cell `cell_type` switch. The ONLY new Rust engine methods are `set_cell_type(cell_id, new_type)` and `set_target_volume(cell_id, v)`.
- Per-cell composites at crypt scale (~118 cells) now; batching for thousands of cells is OUT OF SCOPE.
- Reuse `pbg_tellurium.TelluriumProcess` for SBML integration — no fork.
- Determinism: seeded RNG everywhere; Boolean tie-breaks by cell id.
- Build in the repo's own `.venv` (py3.12) with `maturin develop`; pytest resolves `cpm` via `pyproject` `pythonpath=["."]`.
- Process address form in Composite documents: `"local:!<module.path>.<ClassName>"`. Per-cell readback outputs use `overwrite[...]` types so Composite merges replace rather than concatenate.
- No division in this sub-project (differentiation is driven by the Wnt gradient, not proliferation).

---

## File Structure

- `crates/cpm-core/src/world.rs` — add `set_cell_type`, `set_target_volume` (+ property test).
- `crates/cpm-py/src/lib.rs` — expose both on the pyo3 `World`.
- `cpm/schema.py` — extend `load_world` to build fields (add_field + per-type secretion/chemotaxis).
- `cpm/processes/cpm_process.py` — coupled `CPMProcess`: per-cell outputs + `fates` input applied via `set_cell_type`.
- `cpm/subcellular/__init__.py`, `cpm/subcellular/boolean.py`, `cpm/subcellular/sbml.py` — the two backends.
- `cpm/ftu.py` — FTU-illustration → CPM label rasterizer (lifted from `demos/run_hra_ftu.py`).
- `cpm/composites/__init__.py`, `cpm/composites/crypt.py` — assemble the crypt `Composite`.
- `demos/run_crypt_differentiation.py` — run the Composite, export frames + per-cell state, validate.
- `viewer/viewer.js` — add `subcell` kind + color-by-`state`.
- Tests: `tests/test_set_cell_type.py`, `tests/test_cpm_coupling.py`, `tests/test_boolean_subcell.py`, `tests/test_sbml_subcell.py`, `tests/test_crypt_composite.py`; Rust `crates/cpm-core/tests/property.rs` gains a retype-invariance test.

---

### Task 1: Rust `set_cell_type` + `set_target_volume`

**Files:**
- Modify: `crates/cpm-core/src/world.rs` (add methods to `impl World`)
- Modify: `crates/cpm-py/src/lib.rs` (expose on pyo3 `World`)
- Test: `crates/cpm-core/tests/property.rs` (append), `tests/test_set_cell_type.py` (create)

**Interfaces:**
- Consumes: existing `World { pub cells: Vec<Cell> }`, `Cell { pub cell_type: u16, pub target_volume: f64 }`, pyo3 `World.world_mut()`.
- Produces: Rust `World::set_cell_type(&mut self, cell_id: CellId, new_type: u16)`, `World::set_target_volume(&mut self, cell_id: CellId, v: f64)`; Python `World.set_cell_type(cell_id: int, new_type: int)`, `World.set_target_volume(cell_id: int, v: float)`.

- [ ] **Step 1: Write the failing Rust property test**

Append to `crates/cpm-core/tests/property.rs`:

```rust
#[test]
fn set_cell_type_relabels_without_disturbing_trackers() {
    use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
    use cpm_core::world::World;
    let lat = Lattice::new([6, 6, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 10.0);
    let a = w.add_cell(1, 9.0, 1.0, 0.0, 0.0);
    for y in 1..4 { for x in 1..4 { let i = w.lattice.index(x, y, 0); w.paint(i, a); } }
    w.recompute_trackers();
    let (v0, s0, com0) = (w.cells[a as usize].volume, w.cells[a as usize].surface, w.com(a));
    w.set_cell_type(a, 7);
    assert_eq!(w.cells[a as usize].cell_type, 7);
    // relabel must not touch volume/surface/COM trackers
    assert_eq!(w.cells[a as usize].volume, v0);
    assert_eq!(w.cells[a as usize].surface, s0);
    assert_eq!(w.com(a), com0);
    w.set_target_volume(a, 42.0);
    assert_eq!(w.cells[a as usize].target_volume, 42.0);
}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cargo test -p cpm-core set_cell_type_relabels`
Expected: FAIL — `no method named set_cell_type`.

- [ ] **Step 3: Implement in `world.rs`**

Add inside `impl World` (near `add_cell`):

```rust
/// Relabel a live cell's type. Type affects only contact energy, never the
/// volume/surface/COM trackers, so this is a pure O(1) label write.
pub fn set_cell_type(&mut self, cell_id: CellId, new_type: u16) {
    self.cells[cell_id as usize].cell_type = new_type;
}

/// Set a cell's target volume (per-cell growth control).
pub fn set_target_volume(&mut self, cell_id: CellId, v: f64) {
    self.cells[cell_id as usize].target_volume = v;
}
```

- [ ] **Step 4: Run the Rust test, verify it passes**

Run: `cargo test -p cpm-core set_cell_type_relabels`
Expected: PASS.

- [ ] **Step 5: Expose on pyo3 `World`**

In `crates/cpm-py/src/lib.rs`, add inside `#[pymethods] impl World` (after `grow`):

```rust
fn set_cell_type(&mut self, cell_id: u32, new_type: u16) {
    self.world_mut().set_cell_type(cell_id, new_type);
}

fn set_target_volume(&mut self, cell_id: u32, v: f64) {
    self.world_mut().set_target_volume(cell_id, v);
}
```

- [ ] **Step 6: Rebuild the extension**

Run: `source .venv/bin/activate && maturin develop -m crates/cpm-py/Cargo.toml`
Expected: builds `cpm_core` into the venv.

- [ ] **Step 7: Write + run the Python binding test**

Create `tests/test_set_cell_type.py`:

```python
import cpm_core


def test_set_cell_type_changes_only_type():
    w = cpm_core.World((10, 10, 1), "periodic", 2, 10.0)
    c = w.add_cell(1, 16.0, 2.0, 0.0, 0.0)
    w.set_contact(0, 1, 8.0); w.set_contact(0, 2, 8.0); w.set_contact(1, 1, 2.0)
    w.seed_block(c, 2, 2, 0, 6, 6, 1)
    w.finalize(1)
    vol_before = w.cell_volumes()[c]
    w.set_cell_type(c, 2)
    assert w.cell_types()[c] == 2
    assert w.cell_volumes()[c] == vol_before  # relabel does not move mass
    w.set_target_volume(c, 30.0)              # smoke: no error, cell still alive
    w.step(1)
    assert w.cell_volumes()[c] > 0
```

Run: `source .venv/bin/activate && pytest tests/test_set_cell_type.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/cpm-core/src/world.rs crates/cpm-py/src/lib.rs crates/cpm-core/tests/property.rs tests/test_set_cell_type.py
git commit -m "feat(core): per-cell set_cell_type + set_target_volume"
```

---

### Task 2: Fields in the spec loader

**Files:**
- Modify: `cpm/schema.py` (extend `load_world`)
- Test: `tests/test_fields.py` (append) or `tests/test_schema.py` (append) — use `tests/test_schema.py`

**Interfaces:**
- Consumes: existing `load_world(spec)` building `cpm_core.World`; pyo3 `World.add_field(name, d, decay)->idx`, `set_secretion(idx, type, rate)`, `set_chemotaxis(idx, type, lambda)`.
- Produces: `load_world` also reads `spec["fields"]` — a list of `{"name": str, "d": float, "decay": float, "secretion": [{"type": int, "rate": float}], "chemotaxis": [{"type": int, "lambda": float}]}`. Fields are added BEFORE `finalize`. Returns the same `World`. Field index = order in the list (0-based).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schema.py`:

```python
def test_load_world_builds_fields():
    from cpm.schema import load_world
    spec = {
        "potts": {"dims": [20, 20, 1], "boundary": "noflux",
                  "neighbor_order": 2, "temperature": 10.0, "seed": 1},
        "contact": [{"a": 0, "b": 1, "j": 8.0}, {"a": 1, "b": 1, "j": 2.0}],
        "cells": [{"type": 1, "target_volume": 16, "lambda_volume": 2.0,
                   "target_surface": 0, "lambda_surface": 0.0,
                   "seed_block": [6, 6, 0, 12, 12, 1]}],
        "fields": [{"name": "Wnt", "d": 0.1, "decay": 0.05,
                    "secretion": [{"type": 1, "rate": 5.0}],
                    "chemotaxis": []}],
    }
    w = load_world(spec)
    w.step(3)
    conc = w.field_conc(0)
    assert max(conc) > 0.0            # the type-1 cell secreted Wnt
```

- [ ] **Step 2: Run it, verify it fails**

Run: `pytest tests/test_schema.py::test_load_world_builds_fields -v`
Expected: FAIL — field 0 is empty / `field_conc` index error (no field added).

- [ ] **Step 3: Implement**

In `cpm/schema.py`, inside `load_world`, AFTER cells+contact are set and BEFORE `world.finalize(...)`, insert:

```python
    for fi, f in enumerate(spec.get("fields", [])):
        idx = world.add_field(f["name"], float(f["d"]), float(f["decay"]))
        # idx equals fi by construction; keep them in sync
        for s in f.get("secretion", []):
            world.set_secretion(idx, int(s["type"]), float(s["rate"]))
        for c in f.get("chemotaxis", []):
            world.set_chemotaxis(idx, int(c["type"]), float(c["lambda"]))
```

(`add_field` must be called before `finalize`, which the existing ordering satisfies.)

- [ ] **Step 4: Run the test, verify it passes**

Run: `pytest tests/test_schema.py::test_load_world_builds_fields -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cpm/schema.py tests/test_schema.py
git commit -m "feat(schema): build chemical fields from spec"
```

---

### Task 3: Coupled `CPMProcess`

**Files:**
- Modify: `cpm/processes/cpm_process.py`
- Test: `tests/test_cpm_coupling.py` (create)

**Interfaces:**
- Consumes: `load_world`, pyo3 `World` methods `set_cell_type`, `field_mean_at_cell(idx, cell_id)`, `cell_coms`, `cell_volumes`, `cell_types`, `snapshot`, `dims`, `n_cells`.
- Produces: `CPMProcess` with:
  - `config_schema` adds `n_fields` (int, default 0).
  - `outputs()`: `{"volumes": "overwrite[list]", "types": "overwrite[list]", "positions": "overwrite[list]", "field_at_cell": "overwrite[list]", "neighbor_secretory": "overwrite[list]"}`. Each list is indexed by cell id (index 0 = medium). `field_at_cell[cid]` is field 0's mean at the cell. `neighbor_secretory[cid]` is the count of face-adjacent cells whose type is in `self.secretory_types` (a config set), for lateral inhibition.
  - `inputs()`: `{"fates": "maybe[list]"}` — list indexed by cell id; a nonzero entry `t` triggers `set_cell_type(cid, t)` applied at the START of `update`, before stepping.
  - config also carries `secretory_types` (list[int], default `[]`) so the process knows which types count as secretory neighbors.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cpm_coupling.py`:

```python
import process_bigraph as pb
from cpm.processes.cpm_process import CPMProcess

SPEC = {
    "potts": {"dims": [24, 24, 1], "boundary": "periodic",
              "neighbor_order": 2, "temperature": 10.0, "seed": 3},
    "contact": [{"a": 0, "b": 1, "j": 8.0}, {"a": 0, "b": 2, "j": 8.0},
                {"a": 1, "b": 1, "j": 2.0}, {"a": 2, "b": 2, "j": 2.0}],
    "cells": [{"type": 1, "target_volume": 25, "lambda_volume": 2.0,
               "target_surface": 0, "lambda_surface": 0.0,
               "seed_block": [8, 8, 0, 14, 14, 1]}],
    "fields": [{"name": "L", "d": 0.1, "decay": 0.02,
                "secretion": [{"type": 1, "rate": 3.0}], "chemotaxis": []}],
}


def test_outputs_expose_per_cell_readouts():
    core = pb.allocate_core()
    proc = CPMProcess({"spec": SPEC, "mcs_per_update": 3, "n_fields": 1}, core=core)
    out = proc.update({}, 1.0)
    assert len(out["types"]) == 2 and out["types"][1] == 1
    assert out["field_at_cell"][1] >= 0.0
    assert len(out["positions"]) == 2 and len(out["positions"][1]) == 3


def test_fates_input_switches_cell_type():
    core = pb.allocate_core()
    proc = CPMProcess({"spec": SPEC, "mcs_per_update": 1, "n_fields": 1}, core=core)
    proc.update({}, 1.0)
    # cell 1 -> type 2
    out = proc.update({"fates": [0, 2]}, 1.0)
    assert out["types"][1] == 2
```

- [ ] **Step 2: Run it, verify it fails**

Run: `pytest tests/test_cpm_coupling.py -v`
Expected: FAIL — `KeyError: 'types'` (outputs not implemented) / fates ignored.

- [ ] **Step 3: Implement the coupled process**

Replace `cpm/processes/cpm_process.py` with:

```python
from process_bigraph import Process

from cpm.schema import load_world


class CPMProcess(Process):
    """process-bigraph wrapper around the Rust CPM engine, with a per-cell
    coupling surface so subcellular processes can read the local environment
    and drive differentiation via a cell-type switch."""

    config_schema = {
        "spec": "tree",
        "mcs_per_update": {"_type": "integer", "_default": 10},
        "n_fields": {"_type": "integer", "_default": 0},
        "secretory_types": {"_type": "list", "_default": []},
    }

    def initialize(self, config):
        self.world = load_world(self.config["spec"])
        self.mcs = int(self.config["mcs_per_update"])
        self.n_fields = int(self.config["n_fields"])
        self.secretory = set(int(t) for t in self.config["secretory_types"])
        self.dims = self.world.dims()

    def inputs(self):
        return {"fates": "maybe[list]"}

    def outputs(self):
        return {
            "volumes": "overwrite[list]",
            "types": "overwrite[list]",
            "positions": "overwrite[list]",
            "field_at_cell": "overwrite[list]",
            "neighbor_secretory": "overwrite[list]",
        }

    def _neighbor_secretory_counts(self, types):
        """Count, per cell, face-adjacent cells whose type is secretory."""
        nx, ny, nz = self.dims
        lab = self.world.snapshot()
        n = len(types)
        counts = [0] * n
        seen = [set() for _ in range(n)]
        def owner(x, y, z):
            return lab[x + y * nx + z * nx * ny]
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    a = owner(x, y, z)
                    if a == 0:
                        continue
                    for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                        xx, yy, zz = x + dx, y + dy, z + dz
                        if xx >= nx or yy >= ny or zz >= nz:
                            continue
                        b = owner(xx, yy, zz)
                        if b == a or b == 0:
                            continue
                        if types[b] in self.secretory and b not in seen[a]:
                            counts[a] += 1; seen[a].add(b)
                        if types[a] in self.secretory and a not in seen[b]:
                            counts[b] += 1; seen[b].add(a)
        return counts

    def update(self, state, interval):
        fates = (state or {}).get("fates")
        if fates:
            for cid, t in enumerate(fates):
                if t and cid > 0:
                    self.world.set_cell_type(cid, int(t))
        self.world.step(self.mcs)

        types = list(self.world.cell_types())
        n = len(types)
        field_at = [0.0] * n
        if self.n_fields > 0:
            for cid in range(1, n):
                field_at[cid] = self.world.field_mean_at_cell(0, cid)
        return {
            "volumes": list(self.world.cell_volumes()),
            "types": types,
            "positions": [list(c) for c in self.world.cell_coms()],
            "field_at_cell": field_at,
            "neighbor_secretory": self._neighbor_secretory_counts(types),
        }
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `pytest tests/test_cpm_coupling.py tests/test_process.py -v`
Expected: `test_process.py` may reference old outputs (`cell_volumes`). If it fails, update its assertions to the new `volumes` key in the SAME commit (it is the process's own test):

```python
    out = proc.update({}, 1.0)
    assert "volumes" in out
    assert len(out["volumes"]) == 2
    assert out["volumes"][1] > 0
```

Expected after fix: PASS.

- [ ] **Step 5: Commit**

```bash
git add cpm/processes/cpm_process.py tests/test_cpm_coupling.py tests/test_process.py
git commit -m "feat(process): per-cell coupling surface + fate switching on CPMProcess"
```

---

### Task 4: `BooleanSubcell` process

**Files:**
- Create: `cpm/subcellular/__init__.py`, `cpm/subcellular/boolean.py`
- Test: `tests/test_boolean_subcell.py`

**Interfaces:**
- Consumes: `process_bigraph.Process`.
- Produces: `BooleanSubcell(Process)` with config `{stemness_threshold: float, goblet_type: int, absorptive_type: int}`. `inputs()`: `{"state": "float", "neighbor_secretory": "float"}`. `outputs()`: `{"fate": "overwrite[integer]"}`. Rule in `update`: if `state >= threshold` → `fate = 0` (stay). Else lateral inhibition: `fate = goblet_type` if `neighbor_secretory == 0` else `absorptive_type`. Deterministic.

- [ ] **Step 1: Write the failing test**

Create `tests/test_boolean_subcell.py`:

```python
import process_bigraph as pb
from cpm.subcellular.boolean import BooleanSubcell

CFG = {"stemness_threshold": 0.4, "goblet_type": 3, "absorptive_type": 2}


def _proc():
    return BooleanSubcell(CFG, core=pb.allocate_core())


def test_high_stemness_stays():
    assert _proc().update({"state": 0.9, "neighbor_secretory": 0}, 1.0)["fate"] == 0


def test_low_stemness_no_secretory_neighbor_becomes_goblet():
    assert _proc().update({"state": 0.1, "neighbor_secretory": 0}, 1.0)["fate"] == 3


def test_low_stemness_with_secretory_neighbor_becomes_absorptive():
    assert _proc().update({"state": 0.1, "neighbor_secretory": 2}, 1.0)["fate"] == 2
```

- [ ] **Step 2: Run it, verify it fails**

Run: `pytest tests/test_boolean_subcell.py -v`
Expected: FAIL — module `cpm.subcellular.boolean` not found.

- [ ] **Step 3: Implement**

Create `cpm/subcellular/__init__.py` (empty). Create `cpm/subcellular/boolean.py`:

```python
from process_bigraph import Process


class BooleanSubcell(Process):
    """A minimal synchronous Boolean fate switch with Notch-style lateral
    inhibition. When stemness falls below threshold the cell differentiates:
    it becomes secretory (goblet) unless a neighbour already committed
    secretory, in which case it becomes absorptive. Deterministic."""

    config_schema = {
        "stemness_threshold": {"_type": "float", "_default": 0.4},
        "goblet_type": {"_type": "integer", "_default": 3},
        "absorptive_type": {"_type": "integer", "_default": 2},
    }

    def initialize(self, config):
        self.thresh = float(config["stemness_threshold"])
        self.goblet = int(config["goblet_type"])
        self.absorptive = int(config["absorptive_type"])

    def inputs(self):
        return {"state": "float", "neighbor_secretory": "float"}

    def outputs(self):
        return {"fate": "overwrite[integer]"}

    def update(self, state, interval):
        s = float((state or {}).get("state", 1.0))
        if s >= self.thresh:
            return {"fate": 0}
        neigh = float((state or {}).get("neighbor_secretory", 0))
        return {"fate": self.goblet if neigh == 0 else self.absorptive}
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `pytest tests/test_boolean_subcell.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cpm/subcellular/__init__.py cpm/subcellular/boolean.py tests/test_boolean_subcell.py
git commit -m "feat(subcell): BooleanSubcell fate switch with lateral inhibition"
```

---

### Task 5: `SBMLSubcell` process (reuses `TelluriumProcess`)

**Files:**
- Create: `cpm/subcellular/sbml.py`
- Test: `tests/test_sbml_subcell.py`
- Modify: `pyproject.toml` (add deps)

**Interfaces:**
- Consumes: `pbg_tellurium.TelluriumProcess` (config `model`, `species_overrides`; `update(state, interval)` accepts `{"species": {...}}` and returns `{"species": {...}, ...}`).
- Produces: `SBMLSubcell(Process)` with config `{model: str (antimony), ligand_species: str, state_species: str, ligand_scale: float}`. `inputs()`: `{"ligand": "float"}`. `outputs()`: `{"state": "overwrite[float]"}`. Internally holds a `TelluriumProcess`; each `update` pushes `{ligand_species: ligand*ligand_scale}` as the Tellurium `species` input, integrates over `interval`, and returns the `state_species` value as `state`.

- [ ] **Step 1: Add dependencies**

In `pyproject.toml` `[project] dependencies`, add `"libroadrunner>=2.9"`, `"tellurium>=2.2"`, `"pbg-tellurium"` (if not resolvable from PyPI, install the local checkout editable). Then:

Run: `source .venv/bin/activate && pip install libroadrunner tellurium && pip install -e /Users/eranagmon/code/pbg-tellurium`
Expected: `python -c "import roadrunner, pbg_tellurium"` succeeds.

- [ ] **Step 2: Write the failing test**

Create `tests/test_sbml_subcell.py`:

```python
import process_bigraph as pb
from cpm.subcellular.sbml import SBMLSubcell

# S is produced at a Wnt-gated rate and decays; Wnt is a held floating species
MODEL = """
J1: -> S; k_on*Wnt^4/(K^4 + Wnt^4);
J2: S -> ; k_off*S;
species S, Wnt;
S = 1.0; Wnt = 0.0;
k_on = 0.8; K = 0.3; k_off = 0.4;
"""

CFG = {"model": MODEL, "ligand_species": "Wnt", "state_species": "S", "ligand_scale": 1.0}


def _proc():
    return SBMLSubcell(CFG, core=pb.allocate_core())


def test_low_ligand_lets_stemness_decay():
    p = _proc()
    s_last = 1.0
    for _ in range(20):
        s_last = p.update({"ligand": 0.0}, 1.0)["state"]
    assert s_last < 0.2            # no Wnt -> S decays toward 0


def test_high_ligand_sustains_stemness():
    p = _proc()
    s_last = 1.0
    for _ in range(20):
        s_last = p.update({"ligand": 1.0}, 1.0)["state"]
    assert s_last > 0.8            # saturating Wnt -> S held high
```

- [ ] **Step 3: Run it, verify it fails**

Run: `pytest tests/test_sbml_subcell.py -v`
Expected: FAIL — module `cpm.subcellular.sbml` not found.

- [ ] **Step 4: Implement**

Create `cpm/subcellular/sbml.py`:

```python
from process_bigraph import Process
from pbg_tellurium.processes import TelluriumProcess


class SBMLSubcell(Process):
    """Per-cell SBML ODE via pbg-tellurium. The environmental ligand is pushed
    into a held floating species each step; a chosen species is published as
    the cell's scalar ``state`` (e.g. stemness)."""

    config_schema = {
        "model": {"_type": "string", "_default": ""},
        "ligand_species": {"_type": "string", "_default": "Wnt"},
        "state_species": {"_type": "string", "_default": "S"},
        "ligand_scale": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.ligand_species = config["ligand_species"]
        self.state_species = config["state_species"]
        self.scale = float(config["ligand_scale"])
        self._tp = TelluriumProcess({"model": config["model"]}, core=self.core)

    def inputs(self):
        return {"ligand": "float"}

    def outputs(self):
        return {"state": "overwrite[float]"}

    def update(self, state, interval):
        ligand = float((state or {}).get("ligand", 0.0)) * self.scale
        out = self._tp.update({"species": {self.ligand_species: ligand}}, interval)
        return {"state": float(out["species"][self.state_species])}
```

- [ ] **Step 5: Run the tests, verify they pass**

Run: `pytest tests/test_sbml_subcell.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add cpm/subcellular/sbml.py tests/test_sbml_subcell.py pyproject.toml
git commit -m "feat(subcell): SBMLSubcell wrapping pbg-tellurium"
```

---

### Task 6: FTU rasterizer module + crypt `Composite`

**Files:**
- Create: `cpm/ftu.py` (lift rasterizer from `demos/run_hra_ftu.py`)
- Create: `cpm/composites/__init__.py`, `cpm/composites/crypt.py`
- Test: `tests/test_crypt_composite.py`

**Interfaces:**
- Consumes: `cpm.ftu.load_crypt_labels() -> ((nx, ny), labels, seg_to_type, type_names, median_size)`; `CPMProcess`, `SBMLSubcell`, `BooleanSubcell`; `process_bigraph.Composite`; process address form `"local:!cpm.processes.cpm_process.CPMProcess"` etc.
- Produces: `cpm.composites.crypt.build_crypt_composite(core, *, downscale=1.0, mcs_per_update=8, subcell_every=1) -> (composite, meta)` where `composite` is a `process_bigraph.Composite` and `meta` carries `{"dims", "type_names", "stem_type", "goblet_type", "absorptive_type", "wnt_field", "n_subcells", "seg_to_cell"}`. Each Epithelial-Stem cell gets an `SBMLSubcell` + a `BooleanSubcell` wired: CPM `field_at_cell[cid]` → SBML `ligand`; SBML `state` → Boolean `state` and into a shared `cells/{cid}/state` store; CPM `neighbor_secretory[cid]` → Boolean `neighbor_secretory`; Boolean `fate` → CPM `fates[cid]`.

**Design notes (constants):**
- Type ids reuse the FTU order: `Absorptive=1, Enteroendocrine=2, Epithelial Stem=3, Goblet=4, Tuft=5` (from `cpm/ftu.py`, which returns `type_names` index-aligned). Set `stem_type=3, goblet_type=4, absorptive_type=1`.
- Wnt field: `secretion` by `stem_type` (rate 6.0), `d=0.12, decay=0.15` → base-localized gradient. `secretory_types=[goblet_type]` on the CPMProcess (goblet neighbors inhibit).
- The stemness model is the `MODEL` Antimony string from Task 5 (embed a copy in `crypt.py`).

- [ ] **Step 1: Create `cpm/ftu.py`**

Move the SVG download + `load_ftu_cells` + `rasterize` logic out of `demos/run_hra_ftu.py` into `cpm/ftu.py`, exposing:

```python
def load_crypt_labels(target_maxdim=500, margin=6):
    """Return ((nx, ny), labels, seg_to_type, type_names, median_size) for the
    HRA colonic-crypt FTU. Downloads the SVG on first use."""
    cells, type_names = load_ftu_cells()
    (nx, ny), labels, seg_to_type, median = rasterize(cells, target_maxdim, margin)
    return (nx, ny), labels, seg_to_type, type_names, median
```

Keep `SVG_URL`, `SVG_FILE`, `load_ftu_cells`, `rasterize` (parameterize `rasterize(cells, target_maxdim, margin)`). Then edit `demos/run_hra_ftu.py` to import from `cpm.ftu` instead of defining them locally (keep the demo's behavior identical). Run the existing demo once to confirm no regression:

Run: `source .venv/bin/activate && python demos/run_hra_ftu.py`
Expected: still prints `PASS` with 118 cells.

- [ ] **Step 2: Write the failing composite test**

Create `tests/test_crypt_composite.py`:

```python
import process_bigraph as pb
from cpm.composites.crypt import build_crypt_composite


def test_crypt_composite_runs_and_differentiates():
    core = pb.allocate_core()
    comp, meta = build_crypt_composite(core, downscale=0.5, mcs_per_update=6,
                                       subcell_every=1)
    assert meta["n_subcells"] > 10           # per-stem-cell processes exist
    types0 = list(comp.state["types"]) if "types" in comp.state else None
    comp.run(40.0)
    types1 = list(comp.state["types"])
    stem = meta["stem_type"]
    goblet, absorp = meta["goblet_type"], meta["absorptive_type"]
    # at least one stem cell differentiated to a non-stem epithelial fate
    differentiated = sum(1 for t in types1 if t in (goblet, absorp))
    assert differentiated >= 1
```

- [ ] **Step 3: Run it, verify it fails**

Run: `pytest tests/test_crypt_composite.py -v`
Expected: FAIL — module `cpm.composites.crypt` not found.

- [ ] **Step 4: Implement `cpm/composites/crypt.py`**

Create `cpm/composites/__init__.py` (empty) and `cpm/composites/crypt.py`:

```python
import process_bigraph as pb

from cpm.ftu import load_crypt_labels

STEMNESS_MODEL = """
J1: -> S; k_on*Wnt^4/(K^4 + Wnt^4);
J2: S -> ; k_off*S;
species S, Wnt;
S = 1.0; Wnt = 0.0;
k_on = 0.8; K = 0.3; k_off = 0.4;
"""

CPM_ADDR = "local:!cpm.processes.cpm_process.CPMProcess"
SBML_ADDR = "local:!cpm.subcellular.sbml.SBMLSubcell"
BOOL_ADDR = "local:!cpm.subcellular.boolean.BooleanSubcell"


def build_crypt_composite(core, *, downscale=1.0, mcs_per_update=8, subcell_every=1):
    (nx, ny), labels, seg_to_type, type_names, median = load_crypt_labels(
        target_maxdim=int(500 * downscale))
    # type ids from the FTU order (1-based)
    names = ["Medium"] + type_names
    stem = names.index("Epithelial Stem Cells")
    goblet = names.index("Goblet Cells")
    absorp = names.index("Absorptive Cells")

    # Build a CPM spec by seeding from the FTU labels through a temporary world
    # is not available in schema.load_world; instead we seed via seed_from_labels
    # inside a bespoke spec understood by a small loader shim (below).
    spec = {
        "potts": {"dims": [nx, ny, 1], "boundary": "noflux",
                  "neighbor_order": 2, "temperature": 8.0, "seed": 1},
        "seed_labels": {"labels": labels, "types": seg_to_type,
                        "default_type": stem, "target_volume": float(median),
                        "lambda_volume": 2.0},
        "contact": _crypt_contacts(len(names) - 1, stem, goblet, absorp),
        "fields": [{"name": "Wnt", "d": 0.12, "decay": 0.15,
                    "secretion": [{"type": stem, "rate": 6.0}], "chemotaxis": []}],
    }

    # figure out which cell ids are stem cells (they get subcell composites)
    seg_to_cell = {seg: i + 1 for i, seg in enumerate(sorted(seg_to_type))}
    stem_cell_ids = [cid for seg, cid in seg_to_cell.items()
                     if seg_to_type[seg] == stem]

    state = {
        "cpm": {
            "_type": "process", "address": CPM_ADDR,
            "config": {"spec": spec, "mcs_per_update": mcs_per_update,
                       "n_fields": 1, "secretory_types": [goblet]},
            "inputs": {"fates": ["fates"]},
            "outputs": {"volumes": ["volumes"], "types": ["types"],
                        "positions": ["positions"], "field_at_cell": ["field_at_cell"],
                        "neighbor_secretory": ["neighbor_secretory"]},
        },
    }
    for cid in stem_cell_ids:
        state[f"sbml_{cid}"] = {
            "_type": "process", "address": SBML_ADDR,
            "config": {"model": STEMNESS_MODEL, "ligand_species": "Wnt",
                       "state_species": "S", "ligand_scale": 1.0},
            "interval": float(subcell_every),
            "inputs": {"ligand": ["field_at_cell", cid]},
            "outputs": {"state": ["cell_state", cid]},
        }
        state[f"bool_{cid}"] = {
            "_type": "process", "address": BOOL_ADDR,
            "config": {"stemness_threshold": 0.4, "goblet_type": goblet,
                       "absorptive_type": absorp},
            "interval": float(subcell_every),
            "inputs": {"state": ["cell_state", cid],
                       "neighbor_secretory": ["neighbor_secretory", cid]},
            "outputs": {"fate": ["fates", cid]},
        }

    comp = pb.Composite({"state": state}, core=core)
    meta = {"dims": [nx, ny, 1], "type_names": names, "stem_type": stem,
            "goblet_type": goblet, "absorptive_type": absorp, "wnt_field": 0,
            "n_subcells": len(stem_cell_ids), "seg_to_cell": seg_to_cell}
    return comp, meta


def _crypt_contacts(k, stem, goblet, absorp):
    pairs = []
    for t in range(1, k + 1):
        pairs.append({"a": 0, "b": t, "j": 14.0})       # medium costly -> packed
        for u in range(t, k + 1):
            pairs.append({"a": t, "b": u, "j": 6.0})     # uniform cell-cell
    return pairs
```

Note: `load_world` must also handle the `seed_labels` spec branch. Add to `cpm/schema.py` `load_world`, before contacts/finalize, a branch:

```python
    sl = spec.get("seed_labels")
    if sl is not None:
        world.seed_from_labels(list(sl["labels"]), {int(k): int(v) for k, v in sl["types"].items()},
                               int(sl["default_type"]), float(sl["target_volume"]),
                               float(sl["lambda_volume"]))
```

(Keep the existing `cells`/`seed_block` path working for other specs — the two are mutually exclusive; if `seed_labels` present, skip the `cells` loop.)

- [ ] **Step 5: Run the composite test, verify it passes**

Run: `pytest tests/test_crypt_composite.py -v`
Expected: PASS (a downscaled crypt runs under the Composite engine and at least one cell differentiates). If no differentiation occurs, lower `stemness_threshold` or raise Wnt `decay` so upper cells see low Wnt — tune in `crypt.py`/config, re-run.

- [ ] **Step 6: Commit**

```bash
git add cpm/ftu.py cpm/composites/ cpm/schema.py demos/run_hra_ftu.py tests/test_crypt_composite.py
git commit -m "feat(composite): crypt differentiation Composite from HRA FTU"
```

---

### Task 7: Crypt differentiation demo + validation + export

**Files:**
- Create: `demos/run_crypt_differentiation.py`

**Interfaces:**
- Consumes: `build_crypt_composite`, `process_bigraph.allocate_core`, `Composite.run`, `comp.state` (`types`, `positions`, `cell_state`), the viewer data dir + `index.json` conventions from `demos/run_hra_ftu.py`.
- Produces: `viewer/data/crypt_differentiation.json` (frames with `labels`, per-cell `state`, `types`) + manifest entry `kind="subcell"`; exits nonzero on any failed gate.

- [ ] **Step 1: Write the demo (captures frames each outer step, then validates)**

Create `demos/run_crypt_differentiation.py`:

```python
"""Crypt differentiation as a process-bigraph Composite.

Assembles the HRA colonic-crypt FTU as a CPM plus one SBML stemness ODE and a
Boolean fate switch PER stem cell, all run by the process_bigraph Composite
engine. Basal Wnt keeps basal cells stem; cells in low-Wnt regions differentiate
to Goblet / Absorptive via the ODE->Boolean->cell_type coupling. Validates the
expected biology and exits nonzero on failure.

Usage (repo root, venv active):  python demos/run_crypt_differentiation.py
"""
import json
import os
from statistics import mean

import process_bigraph as pb

from cpm.composites.crypt import build_crypt_composite

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))


def _corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs) or 1e-9
    syy = sum((y - my) ** 2 for y in ys) or 1e-9
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def main(n_frames=24, mcs_per_frame=8):
    core = pb.allocate_core()
    comp, meta = build_crypt_composite(core, downscale=1.0,
                                       mcs_per_update=mcs_per_frame, subcell_every=1)
    stem, goblet, absorp = meta["stem_type"], meta["goblet_type"], meta["absorptive_type"]
    nx, ny, _ = meta["dims"]

    # world handle for snapshots (labels) lives on the CPM process instance
    cpm_proc = comp.state["cpm"]["instance"] if "instance" in comp.state.get("cpm", {}) else None

    frames, diff_frac = [], []
    for f in range(n_frames + 1):
        types = list(comp.state.get("types", []))
        cell_state = list(comp.state.get("cell_state", []))
        labels = list(_snapshot(comp))
        frames.append({"mcs": f * mcs_per_frame, "labels": labels,
                       "types": types, "state": cell_state})
        n_diff = sum(1 for t in types[1:] if t in (goblet, absorp))
        n_tot = sum(1 for t in types[1:] if t != 0)
        diff_frac.append(n_diff / max(1, n_tot))
        if f < n_frames:
            comp.run(float(mcs_per_frame))

    types = list(comp.state["types"])
    pos = list(comp.state["positions"])
    field = list(comp.state["field_at_cell"])
    state = list(comp.state["cell_state"])
    stem_y = [pos[c][1] for c in range(1, len(types)) if types[c] == stem]
    diff_y = [pos[c][1] for c in range(1, len(types)) if types[c] in (goblet, absorp)]
    n_goblet = sum(1 for t in types[1:] if t == goblet)
    n_absorp = sum(1 for t in types[1:] if t == absorp)
    n_diff = n_goblet + n_absorp
    sec_frac = n_goblet / max(1, n_diff)
    # causality: stemness vs local Wnt across stem-lineage cells that have state
    lineage = [c for c in range(1, len(types)) if c < len(state) and state[c] > 0]
    r = _corr([field[c] for c in lineage], [state[c] for c in lineage]) if lineage else 0.0

    checks = [
        (f"stem cells stay basal (mean stem y {mean(stem_y):.0f} < diff y {mean(diff_y):.0f})"
         if stem_y and diff_y else "stem/diff populations present",
         bool(stem_y) and bool(diff_y) and mean(stem_y) < mean(diff_y)),
        (f"differentiation progresses ({diff_frac[0]:.2f} -> {diff_frac[-1]:.2f})",
         diff_frac[-1] > diff_frac[0] + 0.1),
        (f"both fates present, secretory frac {sec_frac:.2f} in [0.1,0.6] "
         f"(goblet {n_goblet}, absorptive {n_absorp})",
         n_goblet > 0 and n_absorp > 0 and 0.1 <= sec_frac <= 0.6),
        (f"stemness tracks local Wnt (corr {r:.2f} > 0.4)", r > 0.4),
        (f"ran under process_bigraph.Composite with {meta['n_subcells']} subcell processes",
         isinstance(comp, pb.Composite) and meta["n_subcells"] > 10),
    ]

    data = {"name": "Crypt Differentiation (subcellular)", "kind": "subcell",
            "dims": meta["dims"], "is3d": False, "n_cells": len(types) - 1,
            "cell_types": types, "type_names": meta["type_names"],
            "frames": frames}
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "crypt_differentiation.json"), "w") as fh:
        json.dump(data, fh)
    _merge_manifest(data, checks)

    print("\n=========== VALIDATION (crypt differentiation) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    ok = all(p for _, p in checks)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _snapshot(comp):
    proc = comp.state["cpm"]["instance"]
    return proc.world.snapshot()


def _merge_manifest(data, checks):
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    manifest = [m for m in manifest if m["file"] != "crypt_differentiation.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "crypt_differentiation.json", "name": data["name"],
                     "is3d": False, "n_cells": data["n_cells"], "dims": data["dims"],
                     "kind": "subcell", "validated": ok,
                     "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json", "hra_ftu.json", "crypt_differentiation.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
```

Note on `comp.state["cpm"]["instance"]`: process-bigraph stores the live process instance under the node's `instance` key after composition. If the key differs in this pb version, discover it once (`print(comp.state["cpm"].keys())`) and use the actual key; the world snapshot is required for `labels`.

- [ ] **Step 2: Run the demo**

Run: `source .venv/bin/activate && python demos/run_crypt_differentiation.py`
Expected: prints five checks, all PASS, exit 0; writes `viewer/data/crypt_differentiation.json` and updates the manifest. If a biology gate fails, tune (in `crypt.py`): Wnt `decay` up sharpens the gradient; `stemness_threshold` down keeps more stem; `k_off` up speeds differentiation. Re-run until all five pass.

- [ ] **Step 3: Commit**

```bash
git add demos/run_crypt_differentiation.py
git commit -m "feat(demo): crypt differentiation Composite demo + validation"
```

---

### Task 8: Viewer — color-by subcellular state

**Files:**
- Modify: `viewer/viewer.js`

**Interfaces:**
- Consumes: `viewer/data/crypt_differentiation.json` frames carrying per-frame `state` (list indexed by cell id) and `types`; existing `heat()`, `cellRGB`, color-mode plumbing, `BLURB`.
- Produces: a new `state` color mode (only offered when the loaded model's frames carry `state`) + `subcell` blurb; hover shows the cell's `state`.

- [ ] **Step 1: Add the `subcell` blurb**

In `viewer/viewer.js` `BLURB`, add:

```js
  subcell: "Per-cell subcellular models run as a process-bigraph Composite: each " +
    "stem cell carries an SBML stemness ODE (pbg-tellurium) and a Boolean fate switch. " +
    "Basal Wnt keeps cells stem; in low-Wnt regions they differentiate (colour = stemness).",
```

- [ ] **Step 2: Add a `state` color mode**

In `index.html`'s `#colormode` select, add `<option value="state">stemness (subcell)</option>`. In `viewer.js`, extend `cellRGB` so that when `colorMode === "state"` and `current.frameState` is set, it maps `current.frameState[id]` (0..1) through `heat(...)`. In `showFrame`/`render`, set `current.frameState = model.frames[fi].state` when present. In `loadModel`, enable the `state` option only when `model.frames[0].state` exists (else disable like the volume option is disabled for 3D).

```js
// in cellRGB, before the final else:
  } else if (colorMode === "state" && c.frameState) {
    const h = heat(c.frameState[id] || 0);
    out[0]=h[0]; out[1]=h[1]; out[2]=h[2];
```

- [ ] **Step 3: Hover shows state**

In the hover handler, when `current.model.frames[frameIdx].state` exists, append `· stemness ${state[id].toFixed(2)}` to the tooltip.

- [ ] **Step 4: Manual verification**

Run: `cd viewer && python3 -m http.server 8899` (if not already running), open `http://127.0.0.1:8899`, select "Crypt Differentiation (subcellular)", switch color mode to "stemness", scrub — basal cells stay bright (stem), upper cells dim and change type. Hover shows stemness.
Expected: differentiation wave visible; no console errors.

- [ ] **Step 5: Commit**

```bash
git add viewer/index.html viewer/viewer.js
git commit -m "feat(viewer): color-by subcellular stemness for the crypt demo"
```

---

## Self-Review

**Spec coverage:** architecture (one Composite) → Tasks 3/6; fate-as-type-switch + `set_cell_type` → Task 1; coupling surface → Task 3; `SBMLSubcell` reusing TelluriumProcess → Task 5; `BooleanSubcell` → Task 4; crypt model (Wnt + stemness ODE + Boolean lateral inhibition) → Tasks 5/6; validation gates incl. causality + Composite integrity → Task 7; viewer color-by-state → Task 8; deps → Task 5; fields → Task 2. `SubcellularAdapter` from the spec is intentionally dropped (YAGNI — normalization folded into `ligand_scale` and the Boolean threshold); noted here so a reviewer does not flag its absence as a gap. `set_target_volume` is built (Task 1) though unused by the crypt demo — kept because the spec lists it and it is one line.

**Placeholder scan:** no TBD/TODO; every code step carries complete code; the one runtime-discovery point (`comp.state["cpm"]["instance"]` key name) is called out explicitly with how to resolve it rather than left vague.

**Type consistency:** `fates`, `field_at_cell`, `neighbor_secretory`, `types`, `positions`, `cell_state`/`state`, `cell_type` ids (`stem`/`goblet`/`absorptive`) are used identically across Tasks 3–8. Process addresses match module paths. Output types are `overwrite[...]` per the Global Constraints.
