# Sub-project D — Per-cell Subcellular Composites (SBML ODE + Boolean)

**Status:** design approved 2026-07-04
**Umbrella:** [pbg-cpm design](2026-07-04-pbg-cpm-design.md), Sub-project D (subcellular plugins)

## Goal

Give CPM cells internal state and behavior by attaching **per-cell subcellular
models** — SBML ODEs (via `pbg-tellurium`/RoadRunner) and Boolean networks —
that read the cell's local environment and drive its phenotype (differentiation,
adhesion, secretion, growth). The whole simulation is assembled and run as a
single **process-bigraph `Composite`**, with every subcellular model a genuine
`process_bigraph.Process` scheduled by the Composite engine (no batched
side-integrator that bypasses the bigraph). First demonstrator: **crypt
differentiation** on the HRA colonic-crypt FTU geometry.

## Non-goals (this sub-project)

- Performance work for thousands of ODE-bearing cells. Per-cell composites are
  correct and idiomatic at crypt scale (~118 cells, stem subset). A batched
  pbg process that vectorizes internally while still living in the Composite is
  a **later** optimization, explicitly out of scope here.
- New ontology-schema layer (Sub-project C). We reuse `cell_type` as the
  Cell-Ontology-aligned coupling variable but do not build the schema layer.
- MaBoSS integration. The Boolean backend is a small self-contained process.

## Key design decision: fate is a per-cell type switch

The CPM coupling surface in the Rust core is **keyed by cell type**, not per
cell: `set_secretion(field, type, rate)`, `set_chemotaxis(field, type, lambda)`,
`grow(type, rate)`, and the contact/adhesion matrix are all per-type. Only
`field_mean_at_cell(field, cell_id)` is per-cell (a read).

Therefore a **differentiation event is just a per-cell `cell_type` switch**, and
adhesion + secretion + chemotaxis all follow the new type automatically. This is
both minimal (the engine needs essentially one new per-cell method) and
ontology-aligned (`cell_type` is a Cell-Ontology term). The subcellular model's
job reduces to: sense local environment → decide fate → write the new type.

## Architecture

One `process_bigraph.Composite` document contains:

1. **`CPMProcess`** — the Rust spatial engine, extended with a coupling surface.
2. **Per-cell subcellular composites** under a `cells/{id}/subcell` store — each
   an `SBMLSubcell` (wraps `TelluriumProcess`) and/or a `BooleanSubcell`.
3. **Wires** connecting CPM per-cell outputs → subcell inputs, and subcell fate
   outputs → CPM inputs.

The Composite scheduler runs all processes with per-process intervals: the CPM
on its MCS clock (`mcs_per_update`), the subcellular models on a slower
biological `dt` (`subcell_interval`, a multiple of the CPM interval) to reflect
timescale separation.

### Coupling surface

| Direction | Per-cell payload | Mechanism |
|---|---|---|
| CPM → subcell | local field conc(s) (`field_mean_at_cell`), COM position, volume, current type | new `CPMProcess.outputs()` returning per-cell arrays |
| subcell → CPM | **fate** = new `cell_type` (0 = no change), optional `target_volume` | new `CPMProcess.inputs()`, applied via new Rust `set_cell_type(cell_id, type)` (+ optional `set_target_volume(cell_id, v)`) |

`CPMProcess.update` applies pending inputs at the **start** of its step (apply
fates → run MCS → read back), so the order each Composite tick is:
`subcell decides → CPM applies + sweeps → CPM publishes new readouts`.

### Engine change (Rust)

Add to `crates/cpm-core/src/world.rs` + `crates/cpm-py/src/lib.rs`:
- `set_cell_type(cell_id: u32, new_type: u16)` — retype a live cell; updates the
  cell's type field so subsequent energy/secretion/chemotaxis use the new type.
  Must NOT corrupt volume/surface/COM trackers (only the type label changes).
- `set_target_volume(cell_id: u32, v: f64)` — optional per-cell growth control.

Both are O(1) label writes. A property test asserts trackers are unchanged by a
retype (only `cell_types()[id]` differs; volumes/surfaces/coms identical).

## Components

### `cpm/subcellular/base.py` — the abstraction
A common contract both backends satisfy, expressed as pbg process
inputs/outputs:
- **inputs** (read from the bigraph): `ligand` (float, wired from CPM
  field-at-cell), `position_y` (float), `volume` (float), `cell_type` (int).
- **outputs** (write to the bigraph): `fate` (int cell_type, 0 = unchanged),
  `state` (float, e.g. stemness, for visualization/validation).

### `cpm/subcellular/sbml.py` — `SBMLSubcell`
Wraps `pbg_tellurium.TelluriumProcess`. Config: `model` (SBML/Antimony string or
path), `inputs_map` (bigraph input → SBML species/param), `outputs_map` (SBML
species → bigraph output), `fate_rule` (threshold + target type). Integrates the
ODE over `interval`, maps the resulting species to `state`, and applies
`fate_rule` to emit `fate`.

### `cpm/subcellular/boolean.py` — `BooleanSubcell`
A small synchronous Boolean-network `Process`. Config: `nodes` (list),
`rules` (node → boolean expression over nodes + inputs), `update` (`sync`),
`fate_map` (node → cell_type). Evaluates one Boolean step per `interval`,
supports reading neighbor state via an input for lateral inhibition, emits
`fate`. Deterministic given a seed (tie-breaking by cell id).

### `cpm/processes/cpm_process.py` — coupled `CPMProcess`
Extend the existing process:
- `outputs()` gains per-cell `field_at_cell` (dict/list keyed by field),
  `positions`, `volumes`, `types`.
- `inputs()` gains per-cell `fates` (list of new types, 0 = unchanged) and
  optional `target_volumes`.
- `update()` applies fates via `set_cell_type` before sweeping.
Backward compatible: with no inputs wired, behaves as today.

### `cpm/composites/crypt.py` — the demonstrator composite
Builds the full Composite dict: loads the crypt FTU into a CPM spec (reusing the
FTU rasterizer from `demos/run_hra_ftu.py`, refactored into `cpm/`), adds a Wnt
field high at the base, instantiates one subcell composite per Epithelial-Stem
cell, and wires them. Returns a `process_bigraph.Composite` ready to `run`.

### `cpm/subcellular/adapter.py` — `SubcellularAdapter` (pbg `Step`)
Optional unit-translation Step between raw CPM readouts and model variables
(e.g. normalize Wnt conc, threshold neighbor counts). Kept separate so wiring
stays declarative.

## The crypt differentiation model (concrete)

- **Wnt field**: `add_field("Wnt", ...)`; a basal niche type (or the lowest crypt
  cells) secretes it, establishing a base-high gradient via diffusion+decay.
- **Stem ODE (SBML/Antimony)**: bistable stemness
  `S' = k_on * Wnt^2/(K^2 + Wnt^2) - k_off * S`, `S(0)=1`. High basal Wnt keeps
  `S` high; cells pushed up the crypt see falling Wnt → `S` collapses.
- **Boolean fate**: when `S < S_thresh`, a Notch-style Boolean picks fate with
  lateral inhibition — a cell becomes **Secretory (Goblet)** unless a neighbor
  already committed Secretory, else **Absorptive**. Writes `fate = GOBLET` or
  `ABSORPTIVE`.
- **CPM applies** the type switch → adhesion follows the contact matrix; Goblet
  type has a mucus `set_secretion` so secretory cells secrete on differentiation.

Cell types (Cell-Ontology aligned, reuse FTU type ids): Epithelial Stem,
Absorptive, Goblet, Enteroendocrine, Tuft (+ a basal niche/Paneth-like source).

## Data flow (one Composite tick)

```
CPM publishes per-cell {Wnt@cell, y, volume, type}
   → SubcellularAdapter normalizes
      → SBMLSubcell integrates S over dt; sets state=S
         → BooleanSubcell reads S + neighbor fates; emits fate
            → CPMProcess.update applies set_cell_type(fate); runs MCS
               → new readouts published
```

Timescale: subcellular processes update every `K` CPM updates (`K = subcell
interval / mcs interval`), set so a cell traverses the crypt over many ODE
steps.

## Validation (demo exits nonzero on any fail)

1. **Basal stem retention**: mean COM-y of stem cells is below (more basal than)
   mean COM-y of differentiated cells by a margin.
2. **Progressive differentiation**: differentiated fraction increases from ~0 at
   start to a substantial fraction by the end (monotone up to noise).
3. **Both fates present**: Absorptive and Goblet both appear; secretory fraction
   within a plausible band (e.g. 0.1–0.5).
4. **Causality**: per-cell stemness `S` correlates with local Wnt / basal
   position above a threshold — proves the ODE drove fate, not chance.
5. **Composite integrity**: a smoke test asserts the run used
   `process_bigraph.Composite` with the expected number of subcell processes
   scheduled (not a hand-rolled loop).

## Testing

- Rust: unit + property test for `set_cell_type` (trackers invariant under
  retype); existing CPM tests stay green.
- Python unit: `BooleanSubcell` truth-table test; `SBMLSubcell` integrates a
  known 1-species decay and matches the analytic solution; `CPMProcess` coupling
  round-trip (wire a stub subcell that forces a fate, assert the cell's type
  changed after `update`).
- Composite integration: a 2-cell toy Composite runs end-to-end under the
  engine; one cell forced to differentiate, assert type + adhesion consequence.
- Demo validation: the five gates above.

## Viewer

Add a `subcell` model kind and a **color-by state** overlay: cells shaded by
their subcellular `state` (stemness) alongside the existing type/cell/volume
modes, so the differentiation wave is visible. Reuse the existing hover panel to
show a cell's `state` and current type.

## Dependencies

- `roadrunner` + `tellurium` (or `pbg-tellurium`) added to the repo venv.
- `pbg-tellurium` reused for `TelluriumProcess`; no fork.

## Risks / open points

- RoadRunner instance per stem cell is heavy but fine at crypt scale; if import
  or per-cell cost is prohibitive, fall back to a shared model instance stepped
  per cell (still one pbg process per cell). Flagged, not pre-optimized.
- Lateral-inhibition neighbor reads require the subcell to see neighbor fates;
  provided via a CPM output (per-cell neighbor type histogram) rather than
  cross-wiring cells directly, to keep wiring tree-shaped.
