# Per-cell Mechanotransduction (E3c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a per-cell mechanotransduction model into the CPM as a process-bigraph `Composite` process (the Sub-project D pattern, driving mechanics not fate): each Composite step every cell reads whether it is compressed and adjusts its own target volume + contractility, giving **contact-inhibited growth** run entirely through `Composite.run`.

**Architecture:** One tiny engine setter (`set_lambda_volume`); two new backward-compatible per-cell input ports on `CPMProcess` (`target_volumes`, `lambda_volumes`); a `MechanoProcess` subcellular process whose pure rule (`mechano_step`) arrests growth + stiffens compressed cells; a `build_mechano_composite` wiring CPM↔Mechano in a feedback loop; and a before/after (control vs contact-inhibited) demo.

**Tech Stack:** Rust (cpm-core) + pyo3 (cpm-py) + Python (`cpm` pkg) + process-bigraph.

## Global Constraints

- Must run through the engine: the demo/tests advance state ONLY via `Composite.run` (no direct `world.step` driving the science); reach the CPM world via `comp.state["cpm"]["instance"].world`.
- Backward compatible: the new `CPMProcess` input ports default to no-op when unwired, so every existing D composite/test still passes. Verify by running the full suite (the crypt-differentiation composite must stay green).
- Map ports keyed by `str(cid)`; pre-seed the `target_volumes`/`lambda_volumes` map stores with a key per live cell (D lesson: absent map keys are dropped by the `overwrite[map]` apply).
- Engine imported in Python as `from cpm import cpm_core`; build with `maturin develop -m crates/cpm-py/Cargo.toml`. Rust `cargo test -p cpm-core`; pytest `pythonpath=["."]`.
- The per-cell mechanical knobs are `target_volume` (exists) and `lambda_volume` (this slice adds `set_lambda_volume`). No new energy terms.

---

## File Structure

- `crates/cpm-core/src/world.rs` — add `set_lambda_volume(cell_id, lambda)`.
- `crates/cpm-py/src/lib.rs` — pyo3 `set_lambda_volume`.
- `crates/cpm-core/tests/mechano.rs` (new) — setter unit test.
- `cpm/processes/cpm_process.py` — two new input ports + apply-before-step.
- `cpm/subcellular/mechano.py` (new) — `mechano_step` pure rule + `MechanoProcess`.
- `cpm/composites/mechano.py` (new) — `build_mechano_composite`.
- `demos/run_mechano_demo.py` (new) — control-vs-contact-inhibited demo.
- `viewer/viewer.js` — a `mechano` BLURB entry.
- Tests: `tests/test_mechano.py`.

---

### Task 1: Engine `set_lambda_volume` + CPMProcess per-cell mechanics ports

**Files:**
- Modify: `crates/cpm-core/src/world.rs`, `crates/cpm-py/src/lib.rs`, `cpm/processes/cpm_process.py`
- Test: `crates/cpm-core/tests/mechano.rs` (new), `tests/test_mechano.py` (new; first test)

**Interfaces:**
- Produces: `World::set_lambda_volume(&mut self, cell_id: CellId, lambda: f64)`; pyo3 `set_lambda_volume(cell_id, lambda)`.
- Produces (CPMProcess): input ports `target_volumes: map[float]`, `lambda_volumes: map[float]`, applied via `set_target_volume`/`set_lambda_volume` before stepping.

- [ ] **Step 1: Write the failing Rust setter test**

Create `crates/cpm-core/tests/mechano.rs`:

```rust
use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::world::World;

#[test]
fn set_lambda_volume_updates_the_cell() {
    let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 10.0);
    let a = w.add_cell(1, 20.0, 1.0, 0.0, 0.0); // lambda_volume seeded 1.0
    w.set_lambda_volume(a, 7.5);
    assert_eq!(w.cells[a as usize].lambda_volume, 7.5);
}
```

- [ ] **Step 2: Run, verify it fails**

Run: `cargo test -p cpm-core --test mechano`
Expected: FAIL — `set_lambda_volume` not defined.

- [ ] **Step 3: Add `set_lambda_volume` (Rust + pyo3)**

In `crates/cpm-core/src/world.rs`, right after `set_target_volume` (which is `self.cells[cell_id as usize].target_volume = v;`), add:

```rust
    pub fn set_lambda_volume(&mut self, cell_id: CellId, lambda: f64) {
        self.cells[cell_id as usize].lambda_volume = lambda;
    }
```

In `crates/cpm-py/src/lib.rs`, right after the pyo3 `set_target_volume`, add:

```rust
    fn set_lambda_volume(&mut self, cell_id: u32, lambda: f64) {
        self.world_mut().set_lambda_volume(cell_id, lambda);
    }
```

Run `cargo test -p cpm-core --test mechano` (PASS), then rebuild the extension:

```bash
maturin develop -m crates/cpm-py/Cargo.toml
```

- [ ] **Step 4: Add the CPMProcess ports (backward-compatible)**

In `cpm/processes/cpm_process.py`, extend `inputs()`:

```python
    def inputs(self):
        # fates is a map keyed by string cell id ... (existing comment kept)
        return {
            "fates": "map[integer]",
            # per-cell mechanics driven by a mechanotransduction process; keyed
            # by str(cid). Absent/empty when unwired -> no-op (backward compatible).
            "target_volumes": "map[float]",
            "lambda_volumes": "map[float]",
        }
```

In `update()`, immediately AFTER the existing `fates` loop and BEFORE `self.world.step(self.mcs)`, add:

```python
        for cid_key, v in ((state or {}).get("target_volumes") or {}).items():
            cid = int(cid_key)
            if cid > 0:
                self.world.set_target_volume(cid, float(v))
        for cid_key, lam in ((state or {}).get("lambda_volumes") or {}).items():
            cid = int(cid_key)
            if cid > 0:
                self.world.set_lambda_volume(cid, float(lam))
```

- [ ] **Step 5: Write + run the CPMProcess port test**

Create `tests/test_mechano.py` (first test):

```python
import process_bigraph as pb

CPM_ADDR = "local:!cpm.processes.cpm_process.CPMProcess"


def _one_cell_spec(nx, ny):
    labels = [0] * (nx * ny)
    for y in range(6, 13):
        for x in range(6, 13):          # 7x7 = 49 voxels
            labels[x + y * nx] = 1
    return {
        "potts": {"dims": [nx, ny, 1], "boundary": "noflux", "neighbor_order": 2,
                  "temperature": 5.0, "seed": 1},
        "seed_labels": {"labels": labels, "types": {1: 1}, "default_type": 1,
                        "target_volume": 49.0, "lambda_volume": 2.0},
        "contact": [{"a": 0, "b": 1, "j": 6.0}],
    }


def test_cpmprocess_applies_target_volumes():
    core = pb.allocate_core()
    nx = ny = 24
    state = {
        "target_volumes": {"1": 20.0},   # drive the resting-49 cell to a small target
        "cpm": {
            "_type": "process", "address": CPM_ADDR,
            "config": {"spec": _one_cell_spec(nx, ny), "mcs_per_update": 40},
            "inputs": {"target_volumes": ["target_volumes"]},
            "outputs": {"volumes": ["volumes"]},
        },
    }
    comp = pb.Composite({"state": state}, core=core)
    v0 = comp.state["cpm"]["instance"].world.cell_volumes()[1]
    comp.run(1.0)
    v1 = comp.state["cpm"]["instance"].world.cell_volumes()[1]
    assert v1 < v0, f"applying a smaller target_volume should shrink the cell: {v0} -> {v1}"
```

Run: `source .venv/bin/activate && pytest tests/test_mechano.py -v`
Expected: PASS (the CPMProcess consumed `target_volumes` and the cell shrank).
Then run the FULL suite `pytest -q` to confirm no D/regression breakage.

- [ ] **Step 6: Commit**

```bash
git add crates/cpm-core/src/world.rs crates/cpm-py/src/lib.rs crates/cpm-core/tests/mechano.rs cpm/processes/cpm_process.py tests/test_mechano.py
git commit -m "feat(mechano): per-cell set_lambda_volume + CPMProcess target/lambda ports"
```

---

### Task 2: MechanoProcess (contact-inhibited growth rule)

**Files:**
- Create: `cpm/subcellular/mechano.py`
- Test: `tests/test_mechano.py` (add the rule test)

**Interfaces:**
- Produces: `cpm.subcellular.mechano.mechano_step(volumes, targets, *, grow_rate, tol, max_target, lambda_base, stiffen_gain, contact_inhibited) -> (new_targets, lambdas)` (lists indexed by cell id, 0 = medium).
- Produces: `MechanoProcess` — input `volumes: list`; outputs `target_volumes: overwrite[map[float]]`, `lambda_volumes: overwrite[map[float]]` (keyed `str(cid)`).

- [ ] **Step 1: Write the failing rule test**

Add to `tests/test_mechano.py`:

```python
from cpm.subcellular.mechano import mechano_step


def test_mechano_step_contact_inhibition():
    targets = [0.0, 20.0, 20.0]  # index 0 = medium
    # cell 1 compressed (V=15 < 20*(1-0.1)=18); cell 2 has room (V=20)
    nt, lam = mechano_step([0, 15, 20], targets, grow_rate=2.0, tol=0.1,
                           max_target=100.0, lambda_base=1.0, stiffen_gain=3.0,
                           contact_inhibited=True)
    assert nt[1] == 20.0 and lam[1] == 4.0   # compressed: growth arrested + stiffened (1*(1+3))
    assert nt[2] == 22.0 and lam[2] == 1.0   # room: grew by grow_rate

    # control mode grows even when compressed
    nt2, lam2 = mechano_step([0, 15, 20], targets, grow_rate=2.0, tol=0.1,
                             max_target=100.0, lambda_base=1.0, stiffen_gain=3.0,
                             contact_inhibited=False)
    assert nt2[1] == 22.0 and lam2[1] == 1.0
```

Run: `pytest tests/test_mechano.py::test_mechano_step_contact_inhibition -v` → FAIL (no module).

- [ ] **Step 2: Implement `cpm/subcellular/mechano.py`**

```python
"""Per-cell mechanotransduction as a process-bigraph subcellular process.

Each Composite step, every cell reads whether it is COMPRESSED (its actual volume
is below its current target because neighbours crowd it). Under contact inhibition
a compressed cell stops growing and stiffens (raises its volume lambda to resist
further compression); a cell with room keeps growing its target. Run WITHOUT
contact inhibition it grows unconditionally -- the control that over-grows past the
available space. Pure rule in `mechano_step`; `MechanoProcess` wraps it for the
Composite engine, applying the per-cell rule over the whole population.
"""
from process_bigraph import Process


def mechano_step(volumes, targets, *, grow_rate, tol, max_target,
                 lambda_base, stiffen_gain, contact_inhibited):
    """Per-cell contact-inhibited growth. `volumes`/`targets` are indexed by cell
    id (index 0 = medium). Returns (new_targets, lambdas), same indexing."""
    n = len(targets)
    new_targets = list(targets)
    lambdas = [lambda_base] * n
    for cid in range(1, n):
        v = volumes[cid] if cid < len(volumes) else 0
        t = targets[cid]
        if v <= 0:                       # dead / absent cell
            continue
        compressed = v < t * (1.0 - tol)
        if contact_inhibited and compressed:
            lambdas[cid] = lambda_base * (1.0 + stiffen_gain)   # arrest + stiffen
        else:
            new_targets[cid] = min(max_target, t + grow_rate)   # room to grow
    return new_targets, lambdas


class MechanoProcess(Process):
    config_schema = {
        "resting_targets": "list",
        "grow_rate": {"_type": "float", "_default": 1.0},
        "tol": {"_type": "float", "_default": 0.1},
        "max_target": {"_type": "float", "_default": 1.0e9},
        "lambda_base": {"_type": "float", "_default": 1.0},
        "stiffen_gain": {"_type": "float", "_default": 2.0},
        "contact_inhibited": {"_type": "boolean", "_default": True},
    }

    def initialize(self, config):
        self.targets = [float(x) for x in config["resting_targets"]]
        self.params = dict(
            grow_rate=float(config["grow_rate"]), tol=float(config["tol"]),
            max_target=float(config["max_target"]), lambda_base=float(config["lambda_base"]),
            stiffen_gain=float(config["stiffen_gain"]),
            contact_inhibited=bool(config["contact_inhibited"]),
        )

    def inputs(self):
        return {"volumes": "list"}

    def outputs(self):
        return {"target_volumes": "overwrite[map[float]]",
                "lambda_volumes": "overwrite[map[float]]"}

    def update(self, state, interval):
        volumes = (state or {}).get("volumes") or []
        if not volumes:
            return {}
        while len(self.targets) < len(volumes):   # keep targets sized to the population
            self.targets.append(self.targets[-1] if self.targets else 0.0)
        new_targets, lambdas = mechano_step(volumes, self.targets, **self.params)
        self.targets = new_targets
        n = len(new_targets)
        return {
            "target_volumes": {str(cid): new_targets[cid] for cid in range(1, n)},
            "lambda_volumes": {str(cid): lambdas[cid] for cid in range(1, n)},
        }
```

- [ ] **Step 3: Run the rule test, verify it passes**

Run: `pytest tests/test_mechano.py -v` → PASS (both the port test and the rule test).

- [ ] **Step 4: Commit**

```bash
git add cpm/subcellular/mechano.py tests/test_mechano.py
git commit -m "feat(mechano): MechanoProcess contact-inhibited growth rule + tests"
```

---

### Task 3: Mechano Composite + before/after demo + viewer blurb

**Files:**
- Create: `cpm/composites/mechano.py`, `demos/run_mechano_demo.py`
- Modify: `viewer/viewer.js`
- Test: `tests/test_mechano.py` (add the composite smoke test)

**Interfaces:**
- Produces: `cpm.composites.mechano.build_mechano_composite(core, spec, resting_targets, *, contact_inhibited, mcs_per_update=8, grow_rate=1.0, tol=0.1, max_target=1e9, lambda_base=2.0, stiffen_gain=3.0) -> (comp, meta)`.
- Consumes: `CPMProcess` ports (Task 1), `MechanoProcess` (Task 2), `cpm_core.World` via `comp.state["cpm"]["instance"].world`.

- [ ] **Step 1: Implement the composite**

Create `cpm/composites/mechano.py`:

```python
"""Per-cell mechanotransduction as a process-bigraph Composite: one CPMProcess
coupled to one MechanoProcess in a feedback loop (CPM volumes -> mechano ->
per-cell target/lambda -> CPM), advanced by Composite.run. Contact inhibition is
a config switch so the demo can run control vs treatment on the same seed."""
import process_bigraph as pb

CPM_ADDR = "local:!cpm.processes.cpm_process.CPMProcess"
MECHANO_ADDR = "local:!cpm.subcellular.mechano.MechanoProcess"


def build_mechano_composite(core, spec, resting_targets, *, contact_inhibited,
                            mcs_per_update=8, grow_rate=1.0, tol=0.1,
                            max_target=1.0e9, lambda_base=2.0, stiffen_gain=3.0):
    n = len(resting_targets)                     # index 0 = medium
    live = [cid for cid in range(1, n) if resting_targets[cid] > 0]
    state = {
        # pre-seed the per-cell mechanics maps so overwrite[map] writes land
        "target_volumes": {str(cid): float(resting_targets[cid]) for cid in live},
        "lambda_volumes": {str(cid): float(lambda_base) for cid in live},
        "cpm": {
            "_type": "process", "address": CPM_ADDR,
            "config": {"spec": spec, "mcs_per_update": mcs_per_update},
            "inputs": {"target_volumes": ["target_volumes"],
                       "lambda_volumes": ["lambda_volumes"]},
            "outputs": {"volumes": ["volumes"], "types": ["types"],
                        "positions": ["positions"]},
        },
        "mechano": {
            "_type": "process", "address": MECHANO_ADDR,
            "config": {"resting_targets": list(resting_targets), "grow_rate": grow_rate,
                       "tol": tol, "max_target": max_target, "lambda_base": lambda_base,
                       "stiffen_gain": stiffen_gain, "contact_inhibited": contact_inhibited},
            "inputs": {"volumes": ["volumes"]},
            "outputs": {"target_volumes": ["target_volumes"],
                        "lambda_volumes": ["lambda_volumes"]},
        },
    }
    comp = pb.Composite({"state": state}, core=core)
    meta = {"dims": spec["potts"]["dims"], "live_ids": live,
            "contact_inhibited": contact_inhibited}
    return comp, meta
```

- [ ] **Step 2: Add the composite smoke test**

Add to `tests/test_mechano.py`:

```python
from cpm.composites.mechano import build_mechano_composite


def _packed_spec(nx, ny, cell, gap, seed_vol_side):
    # a grid of small square cells with costly medium so they stay confined
    labels = [0] * (nx * ny)
    seg = {}
    sid = 1
    for gy in range(2):
        for gx in range(4):
            ox = 2 + gx * (cell + gap)
            oy = 2 + gy * (cell + gap)
            for y in range(oy, oy + seed_vol_side):
                for x in range(ox, ox + seed_vol_side):
                    labels[x + y * nx] = sid
            seg[sid] = 1
            sid += 1
    spec = {
        "potts": {"dims": [nx, ny, 1], "boundary": "noflux", "neighbor_order": 2,
                  "temperature": 6.0, "seed": 1},
        "seed_labels": {"labels": labels, "types": seg, "default_type": 1,
                        "target_volume": float(seed_vol_side ** 2), "lambda_volume": 2.0},
        "contact": [{"a": 0, "b": 1, "j": 14.0}, {"a": 1, "b": 1, "j": 6.0}],
    }
    n_cells = sid - 1
    return spec, n_cells


def _frustration(comp):
    w = comp.state["cpm"]["instance"].world
    vols = w.cell_volumes()
    tvs = comp.state.get("target_volumes") or {}
    fr, cnt = 0.0, 0
    for cid_key, t in tvs.items():
        cid = int(cid_key)
        if cid < len(vols) and vols[cid] > 0 and t > 0:
            fr += max(0.0, (float(t) - vols[cid]) / float(t)); cnt += 1
    return fr / cnt if cnt else 0.0


def test_mechano_composite_contact_inhibition_reduces_frustration():
    nx = ny = 26
    spec, n_cells = _packed_spec(nx, ny, cell=5, gap=1, seed_vol_side=3)  # 9-voxel seeds
    resting = [0.0] + [40.0] * n_cells   # want to grow to 40 but the box is packed

    def run(ci):
        core = pb.allocate_core()
        comp, _ = build_mechano_composite(core, spec, resting, contact_inhibited=ci,
                                          mcs_per_update=8, grow_rate=3.0, tol=0.1)
        assert isinstance(comp, pb.Composite)
        for _ in range(10):
            comp.run(1.0)
        return comp

    f_ctrl = _frustration(run(False))
    f_treat = _frustration(run(True))
    assert f_treat < f_ctrl, f"contact inhibition should lower frustration: {f_treat} vs {f_ctrl}"
```

Run: `pytest tests/test_mechano.py -v` → PASS (all three tests). If `f_treat < f_ctrl` is marginal, raise `grow_rate` (the control over-grows harder) — do NOT flip the assertion.

- [ ] **Step 3: Write the demo**

Create `demos/run_mechano_demo.py`:

```python
"""Per-cell mechanotransduction (E3c): a confined cluster of cells that want to
grow, run as a process-bigraph Composite (CPMProcess <-> MechanoProcess feedback)
TWICE on the same seed -- WITHOUT contact inhibition (control: targets grow past
the box's capacity, cells end up badly frustrated) and WITH it (cells stop growing
when compressed, targets track achievable volume, low frustration). The per-cell
mechanics are driven entirely through Composite.run. Validates + exports.

Usage (repo root, venv active):  python demos/run_mechano_demo.py
"""
import json
import os
import sys

import process_bigraph as pb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cpm.composites.mechano import build_mechano_composite
from cpm.metrics import connected_components

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
NX = NY = 26
SEED_SIDE = 3
RESTING = 40.0
GROW_RATE = 3.0
MCS_PER_UPDATE = 8
N_STEPS = 10


def _spec_and_cells():
    labels = [0] * (NX * NY)
    seg = {}
    sid = 1
    for gy in range(2):
        for gx in range(4):
            ox = 2 + gx * 6
            oy = 2 + gy * 6
            for y in range(oy, oy + SEED_SIDE):
                for x in range(ox, ox + SEED_SIDE):
                    labels[x + y * NX] = sid
            seg[sid] = 1
            sid += 1
    spec = {
        "potts": {"dims": [NX, NY, 1], "boundary": "noflux", "neighbor_order": 2,
                  "temperature": 6.0, "seed": 1},
        "seed_labels": {"labels": labels, "types": seg, "default_type": 1,
                        "target_volume": float(SEED_SIDE ** 2), "lambda_volume": 2.0},
        "contact": [{"a": 0, "b": 1, "j": 14.0}, {"a": 1, "b": 1, "j": 6.0}],
    }
    return spec, sid - 1


def _labels_frame(comp, mcs):
    w = comp.state["cpm"]["instance"].world
    return {"mcs": mcs, "labels": list(w.snapshot())}


def _run(spec, resting, ci, capture):
    core = pb.allocate_core()
    comp, meta = build_mechano_composite(core, spec, resting, contact_inhibited=ci,
                                         mcs_per_update=MCS_PER_UPDATE, grow_rate=GROW_RATE, tol=0.1)
    w0 = comp.state["cpm"]["instance"].world
    seed_mean = sum(v for v in w0.cell_volumes()[1:] if v > 0) / max(1, len(meta["live_ids"]))
    frames = []
    for i in range(N_STEPS + 1):
        if capture:
            frames.append(_labels_frame(comp, i * MCS_PER_UPDATE))
        if i < N_STEPS:
            comp.run(1.0)
    w = comp.state["cpm"]["instance"].world
    vols = w.cell_volumes()
    tvs = comp.state.get("target_volumes") or {}
    fr, cnt, sum_t = 0.0, 0, 0.0
    for cid_key, t in tvs.items():
        cid = int(cid_key)
        if cid < len(vols) and vols[cid] > 0 and t > 0:
            fr += max(0.0, (float(t) - vols[cid]) / float(t)); cnt += 1; sum_t += float(t)
    alive = sum(1 for v in vols[1:] if v > 0)
    frag = sum(1 for cid in range(1, len(vols)) if vols[cid] > 0 and connected_components(w, cid) != 1)
    final_mean = sum(v for v in vols[1:] if v > 0) / max(1, alive)
    return {"comp": comp, "isinst": isinstance(comp, pb.Composite),
            "frustration": fr / cnt if cnt else 0.0, "sum_target": sum_t,
            "alive": alive, "n0": len(meta["live_ids"]), "frag": frag,
            "seed_mean": seed_mean, "final_mean": final_mean, "frames": frames}


def main():
    spec, n_cells = _spec_and_cells()
    resting = [0.0] + [RESTING] * n_cells
    ctrl = _run(spec, resting, False, capture=False)
    treat = _run(spec, resting, True, capture=True)

    checks = [
        (f"ran through the engine (both are process_bigraph.Composite, advanced by "
         f"Composite.run)", ctrl["isinst"] and treat["isinst"]),
        (f"CONTROL over-grows -> frustrated (mean frustration {ctrl['frustration']:.2f} high)",
         ctrl["frustration"] > 0.3),
        (f"contact inhibition works (treatment frustration {treat['frustration']:.2f} < "
         f"0.5 x control {ctrl['frustration']:.2f}, and < 0.15)",
         treat["frustration"] < 0.5 * ctrl["frustration"] and treat["frustration"] < 0.15),
        (f"growth self-limited (treatment total target {treat['sum_target']:.0f} < control "
         f"{ctrl['sum_target']:.0f})", treat["sum_target"] < ctrl["sum_target"]),
        (f"cells grew from seed then held (treatment mean vol {treat['final_mean']:.1f} > seed "
         f"{treat['seed_mean']:.1f})", treat["final_mean"] > treat["seed_mean"]),
        (f"integrity: all cells survive ({treat['alive']}/{treat['n0']}), none fragmented "
         f"({treat['frag']})", treat["alive"] == treat["n0"] and treat["frag"] == 0),
    ]

    w = treat["comp"].state["cpm"]["instance"].world
    data = {"name": "Mechanotransduction (contact inhibition)", "kind": "mechano",
            "dims": [NX, NY, 1], "is3d": False, "n_cells": treat["n0"],
            "cell_types": list(w.cell_types()), "type_names": ["Medium", "Cell"],
            "frames": treat["frames"]}
    os.makedirs(DATA, exist_ok=True)
    json.dump(data, open(os.path.join(DATA, "mechano.json"), "w"))
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    manifest = [m for m in manifest if m["file"] != "mechano.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "mechano.json", "name": data["name"], "is3d": False,
                     "n_cells": data["n_cells"], "dims": data["dims"], "kind": "mechano",
                     "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json", "hra_ftu.json", "crypt_differentiation.json",
             "connectivity_2d.json", "connectivity_3d.json", "connectivity_gap.json",
             "crypt3d.json", "membrane.json", "junction.json", "mechano.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (mechanotransduction) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the demo, tune**

Run: `source .venv/bin/activate && python demos/run_mechano_demo.py`
Expected: all 6 gates PASS, exit 0; writes `viewer/data/mechano.json` + manifest entry.
Tuning: the CONTROL must over-grow (gate 2, frustration high) — raise `GROW_RATE` or `N_STEPS` if not. The TREATMENT must plateau with low frustration (gate 3) AND its cells must have grown from seed (gate 5) — if treatment never grew (arrested at seed), lower the initial packing / raise the box so there IS room to grow before compression. Do NOT weaken a gate.

- [ ] **Step 5: Add the viewer blurb**

In `viewer/viewer.js`, add a `mechano` entry to `BLURB` (after `junction`):

```javascript
  mechano: "Per-cell mechanotransduction (Sub-project E3c) — each cell runs a mechanical " +
    "model as a process-bigraph Composite process: it senses when it's compressed (can't " +
    "reach its target volume) and stops growing + stiffens. Confined cells grow until they " +
    "touch, then arrest (contact inhibition) — versus a control that over-grows past the " +
    "available space. The per-cell mechanics are driven through the Composite engine.",
```

- [ ] **Step 6: Commit**

```bash
git add cpm/composites/mechano.py demos/run_mechano_demo.py viewer/viewer.js tests/test_mechano.py
git commit -m "feat(mechano): Composite + contact-inhibition demo + viewer blurb"
```

---

## Self-Review

**Spec coverage:** engine `set_lambda_volume` → Task 1; CPMProcess `target_volumes`/`lambda_volumes` ports (backward compatible) → Task 1; `MechanoProcess` + `mechano_step` contact-inhibited-growth rule → Task 2; `build_mechano_composite` feedback loop wired + map-store pre-seeding → Task 3; before/after demo with the gates (ran-through-engine, control over-grows, contact inhibition lowers frustration, growth self-limited, grew-from-seed, integrity) + viewer → Task 3; tests (Rust setter, CPMProcess port application, rule unit, composite smoke) across Tasks 1–3.

**Placeholder scan:** no TBD/TODO; complete code in every step. The demo's tuning point (control over-grows AND treatment grows-then-plateaus) is called out with concrete knobs (GROW_RATE/N_STEPS/packing) and a no-weaken rule.

**Type consistency:** `set_lambda_volume(cell_id, lambda)` (Rust + pyo3), CPMProcess ports `target_volumes`/`lambda_volumes` (`map[float]`, keyed str(cid)), `mechano_step(volumes, targets, *, grow_rate, tol, max_target, lambda_base, stiffen_gain, contact_inhibited) -> (new_targets, lambdas)`, `MechanoProcess` output `overwrite[map[float]]`, `build_mechano_composite(core, spec, resting_targets, *, contact_inhibited, ...)`, and the demo `kind="mechano"` + `order` entry are consistent across Tasks 1–3 and match the D Composite idioms (`local:!` addresses, `comp.state["cpm"]["instance"].world`, pre-seeded map stores).
```
