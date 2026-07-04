# Sub-project E3c — Per-cell Mechanotransduction (Composite-wired)

**Status:** design approved 2026-07-04
**Umbrella:** [pbg-cpm design](2026-07-04-pbg-cpm-design.md), Sub-project E (tissue
mechanics). Builds on E1/E2/E3a/E3b and reuses the Sub-project D subcellular
Composite pattern.

## Goal

Wire a **per-cell mechanical model** into the CPM as a process-bigraph `Composite`
process — exactly as Sub-project D wired per-cell SBML/Boolean models to drive
cell *fate*, but here driving per-cell *mechanics*. Each Composite step every cell
reads its own mechanical state (is it compressed — unable to reach its target
volume because its neighbours crowd it?) and adjusts its own CPM mechanics
(target volume, and contractility λ_volume) via a **mechanotransduction feedback
loop**, run entirely through the `Composite.run` engine (no bypass).

The demonstrable behaviour is **contact-inhibited growth**: cells grow until they
feel compression, then stop — a textbook mechanotransduction result. This closes
the E-arc: E1 keeps cells whole, E2 builds 3D structure, E3a anchors to a
membrane, E3b seals gaps, and E3c makes the cells' *own mechanics responsive*.

## Non-goals (this spec)

- New CPM energy terms (E3a/E3b added those). E3c only *drives existing* per-cell
  mechanics (target volume, λ_volume) from a Composite process.
- Coupling to the D biochemistry models (differentiation-driven mechanics) — a
  possible future combination, not this slice.
- Per-cell *adhesion*/junction strength (those are per-TYPE in the core); the
  per-cell knobs are target volume and λ_volume.

## Design

### Engine addition (one setter)

The core already exposes `set_target_volume(cell_id, v)` (E1). Add the companion
`set_lambda_volume(cell_id, lambda)` (per-cell contractility/stiffness) on `World`
+ pyo3 — a trivial mirror. These are the two per-cell mechanical knobs.

### CPMProcess: two new per-cell input ports

Extend `cpm/processes/cpm_process.py` `CPMProcess` (backward-compatibly) with two
optional input ports, applied to the engine BEFORE stepping, exactly like the
existing `fates` port:

- `target_volumes: map[float]` — keyed by `str(cid)`; each applied via
  `set_target_volume(cid, v)`.
- `lambda_volumes: map[float]` — keyed by `str(cid)`; each applied via
  `set_lambda_volume(cid, lam)`.

Absent/empty (as in every existing D composite, which never wires them) → no-op,
so D and all current tests are unaffected. `volumes` (already an output) is the
mechanical readout the mechano process consumes.

### MechanoProcess (subcellular, single instance)

`cpm/subcellular/mechano.py` `MechanoProcess(Process)` — one process embodying the
per-cell mechanotransduction rule over the whole cell population (a single process
reading the `volumes` list and writing the per-cell maps is the clean form; the
rule is per-cell). It holds each cell's current target across steps (stateful).

- **config:** `resting_targets: list` (per-cell resting target, index = cell id),
  `grow_rate` (target increase per step when there is room), `tol` (compression
  tolerance), `max_target` (safety cap), `lambda_base`, `stiffen_gain`,
  `contact_inhibited: bool` (the treatment/control switch).
- **input:** `volumes: list` (from CPMProcess; index = cell id, 0 = medium).
- **output:** `target_volumes: overwrite[map[float]]`, `lambda_volumes:
  overwrite[map[float]]` (keyed `str(cid)`).
- **rule (per cell `cid ≥ 1`):** let `V = volumes[cid]`, `T = target[cid]`.
  `compressed = V < T * (1 - tol)` (the cell cannot reach its target — its
  neighbours are squeezing it).
  - If `contact_inhibited` AND `compressed`: **hold** the target (growth arrest)
    and **stiffen** — `lam = lambda_base * (1 + stiffen_gain)` (resist further
    compression).
  - Else (room to grow, or control mode): **grow** —
    `target[cid] = min(max_target, T + grow_rate)`, `lam = lambda_base`.
  - Emit `target[cid]` and `lam` for `cid`.

Control mode (`contact_inhibited = False`) grows every cell's target
unconditionally, so targets run away far past the confined space's capacity and
cells end up badly frustrated (actual volume ≪ target). Treatment mode arrests
growth at confluence, so targets track achievable volume and frustration stays low.

### Composite

`cpm/composites/mechano.py` `build_mechano_composite(core, *, contact_inhibited)`
wires ONE `CPMProcess` + ONE `MechanoProcess`:
`cpm.volumes → mechano.volumes`; `mechano.target_volumes → cpm.target_volumes`;
`mechano.lambda_volumes → cpm.lambda_volumes`. Pre-seed the `target_volumes` /
`lambda_volumes` map stores with a `str(cid)` key per live cell (so the map
`overwrite` writes land, per the D lesson). Advanced by `Composite.run`.

## Demo and validation

`demos/run_mechano_demo.py` seeds a small confined cluster of cells (costly medium
so they stay packed in a fixed box, seeded near a modest resting target so the box
is ~at capacity) and runs the Composite TWICE — `contact_inhibited=False` (control)
and `True` (treatment) — then validates (exits nonzero on any failed gate):

- **Ran through the engine**: both are `pb.Composite`s advanced by `Composite.run`
  (assert the CPM world is reached via `comp.state["cpm"]["instance"]`).
- **Control over-grows (frustrated)**: control mean frustration
  `F = mean_cells max(0, (target − V) / target)` is high (targets ran past
  capacity) — proves the stress is real.
- **Contact inhibition works**: treatment frustration is materially lower than
  control (e.g. `F_treat < 0.5 · F_control`) AND low in absolute terms
  (`F_treat < 0.15`) — cells stopped growing when compressed.
- **Growth self-limited**: treatment total target volume plateaus near the box
  capacity while control's total target far exceeds it (report both; gate that
  `Σtarget_treat < Σtarget_control`).
- **Integrity**: all cells survive; no cell fragments (E1 metric); the run is
  non-trivial (cells actually grew from their seed before arrest).

Exports the treatment run's frames (2D labels, coloured by per-cell target via a
new `state`-like channel, or plain type) for the viewer + a `mechano` BLURB.

## Testing

- **Rust unit**: `set_lambda_volume(cid, lam)` sets the cell's `lambda_volume`
  (read back via a getter or by observing behaviour); survives `finalize`.
- **Python (`tests/test_mechano.py`)**:
  - `MechanoProcess` rule unit test (no engine): a compressed cell with
    `contact_inhibited=True` holds its target and stiffens; an uncompressed cell
    grows; `contact_inhibited=False` grows even when compressed.
  - CPMProcess applies `target_volumes`/`lambda_volumes` (wire a one-cell map,
    step, assert the engine used it — e.g. the cell's volume moves toward the new
    target).
  - Composite smoke: `build_mechano_composite` runs via `Composite.run` and the
    treatment run ends with lower frustration than the control on the same seed.
- Determinism: fixed seed reproduces frustration numbers.

## Risks / open points

- **Backward compatibility**: the new CPMProcess input ports must default to
  no-op so every existing D composite/test is unaffected — verified by running
  the full suite (the crypt-differentiation composite must still pass).
- **Single-process vs per-cell-instance**: a single MechanoProcess applying the
  per-cell rule is cleaner than N per-cell instances and still "wired through the
  Composite engine"; documented. Per-cell-instance wiring (à la D's SBML) is a
  future scaling variant.
- **Tuning**: the confined-box geometry + grow_rate must make the control
  genuinely over-grow while the treatment plateaus with cells still having grown
  from seed (not frozen at seed); the demo's growth + frustration gates guard
  both, tuned empirically.
- **`overwrite[map[float]]` store pre-seeding**: as in D, the map stores must be
  pre-seeded with a key per cell or per-key writes are dropped.
