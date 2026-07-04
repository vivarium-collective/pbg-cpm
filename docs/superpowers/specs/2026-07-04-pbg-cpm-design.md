# pbg-cpm — a native process-bigraph Cellular Potts framework

**Status:** design approved 2026-07-04 · **Scope of this doc:** umbrella design + spec for Sub-project A (CPM core engine). Sub-projects B–E get their own specs later.

## Goal

Remake CompuCell3D (CC3D) as a native [process-bigraph](https://github.com/vivarium-collective/process-bigraph) framework: a fast Cellular Potts Model (CPM) engine that runs 2D and 3D and scales to thousands of cells, a PDE solver for chemical fields, and a pluggable subcellular layer (ODE / Boolean / any model) attached per cell. Schemas align with CC3D's conceptual framework but are ontology-forward.

Two hard, explicit success criteria beyond "it works":

1. **Demo parity** — reproduce as many CC3D demos as possible (cell sorting, chemotaxis, bacterium-macrophage, diffusion, foam, …), but schema-driven. The union of what those demos need defines the schema surface. Demo parity is the north-star design driver.
2. **Beat CC3D in 3D.** CC3D's 3D is single-threaded and cache-hostile. Outperforming it in 3D is a first-class requirement, not a nice-to-have.

## Prior art in this ecosystem (reuse, don't reinvent)

- **`compucell-tissueforge`** — wraps CC3D/TissueForge as process-bigraph Processes (`CC3DProcess`, `TissueForgeProcess`, `tf_cc3d` composite). Reference for the *outside-wrapping* approach we are deliberately moving beyond.
- **`multicell-schema`** — already has a bigraph-style nested object model (`Universe ⊃ MaterialObjectSpace ⊃ CellCPM`, with `processes` referencing `participating_objects`), a `CellCPM` type, a `MotileForce` process, and a builder/validator/registry. Sub-project C **extends** this rather than inventing a new object model.
- **`OpenVTschema`** — schema for categorizing virtual-tissue simulators; source of the CBO/ontology alignment and the OpenVT community context.
- **`parsimony`** — Rust workspace (`parsimony-core / -spatial / -gpu / -cli / -bench`). We mirror its layout and can later share the GPU pattern (it already has a `parsimony-gpu` crate).
- **Existing pbg ODE/SBML wrappers** (RoadRunner/tellurium via biosimulators, `pbg-amici`, `pbg-simbio`) — reused as subcellular backends in Sub-project D.

## Key architectural decisions (approved)

- **CPM core: native compiled Rust**, exposed to Python via pyo3 + maturin. Rust owns the lattice; process-bigraph orchestrates.
- **CPM and PDE are two separate Rust-backed pbg Processes** sharing the field buffer (faithful to CC3D's Potts-vs-Steppable split; lets PDE solvers be swapped independently).
- **Repo name:** `pbg-cpm`.
- **Parallelization ordering:** build single-threaded and correct first, land + benchmark it, *then* add checkerboard/sublattice parallelism as an isolated, separately-validated step.
- **First milestone:** 2D cell sorting end-to-end, then 3D cell sorting.

## Decomposition into sub-projects

| # | Sub-project | Delivers |
|---|---|---|
| **A** | CPM core engine (Rust) | Fast 2D/3D Potts kernel: lattice, cell ownership, Metropolis sweep, energy plugins (Volume, Surface, Contact/Adhesion, Chemotaxis, ExternalPotential, Connectivity), incremental trackers (volume/surface/COM/neighbors). Python binding. **This spec.** |
| **B** | PDE field solver (Rust) | Diffusion–reaction–secretion–decay on lattice fields; per-cell-type secretion; boundary conditions. Separate pbg Process sharing field memory with A. |
| **C** | Schema + ontology layer | Extend multicell-schema into canonical typed schemas (Potts config, cell types, energy terms, fields, plugins), ontology-annotated. Round-trip a CC3DML/`.cc3d` demo ↔ our JSON. |
| **D** | Subcellular plugin layer | Per-cell ODE (SBML/RoadRunner), Boolean network, and generic pbg-Process attachment. Reuses existing pbg ODE wrappers. |
| **E** | Demo-parity suite | Port CC3D demos as schema files + regression tests. |
| **F** | Interactive 3D viewer | Dedicated web viewer for 3D CPM simulations (like parsimony's), rendering cells + fields with time playback, fed by an exported data-pack format. |

## Architecture — process/store topology

```
        ┌─────────────────── PBG STORE (single source of truth) ───────────────────┐
        │  potts_config   cells[ id,type,volume,target_volume,λ,surface,COM,… ]     │
        │  fields_meta    field_samples_at_cells   secretion_rates                  │
        └──────▲───────────────▲──────────────────────────▲────────────────▲────────┘
               │               │                          │                │
        ┌──────┴──────┐  ┌─────┴───────┐          ┌───────┴────────┐  ┌────┴─────────┐
        │ CPMProcess  │  │ Diffusion   │          │ per-cell ODE / │  │ Mitosis /    │
        │ (Rust core) │◄─┤ Process     │          │ Boolean Procs  │  │ secretion    │
        │ owns lattice│  │ (Rust PDE)  │          │ (Sub-proj D)   │  │ Steps        │
        └─────┬───────┘  └─────┬───────┘          └────────────────┘  └──────────────┘
              └── shared field buffer (Rust, zero-copy) ──┘
```

- **CPMProcess** owns the pixel lattice internally. Each `update()` runs N Monte Carlo sweeps, then syncs per-cell aggregates (volume, surface, COM, neighbor-contact table) into the store. Reads `target_volume` / type / adhesion / λ from the store so subcellular models can steer them.
- **DiffusionProcess** owns the field lattice; solves diffusion-reaction each step. CPM's Chemotaxis term needs the field, so the field buffer is **shared Rust memory** both Rust processes hold a handle to — the reason both are Rust. Only field *samples at cell COMs* + secretion rates cross into the store.
- Big arrays (lattice, fields) **never roundtrip** through Python each step; only cell-level scalars do. This is what keeps thousands of cells / large 3D lattices fast while staying genuinely composable.

## Repo structure

```
pbg-cpm/
  crates/
    cpm-core/      # lattice, cells, energy, Metropolis sweep (2D+3D, generic over dim)
    cpm-pde/       # field solver (Sub-project B)
    cpm-py/        # pyo3 bindings → `import cpm` (maturin build)
    cpm-bench/     # 3D scaling benchmarks vs CC3D
  cpm/             # thin Python: pbg Process wrappers + schema loader
    processes/     # CPMProcess, DiffusionProcess
  schemas/         # Sub-project C
  demos/           # Sub-project E (schema files + expected outputs)
  viewer/          # Sub-project F (static three.js web viewer + data/)
```

## Interactive 3D viewer (Sub-project F, sketch)

Mirrors parsimony's `viewer/`: a static three.js web app (importmap-loaded three.js, `OrbitControls`, `EffectComposer` postprocessing, a web worker for geometry building), deployable to R2/GitHub Pages, fed by an exported **`cpm.pack`** data format — no server required.

CPM-specific rendering (vs parsimony's meshes/spheres): the state is a 3D label lattice, so the viewer renders **cell surfaces extracted from boundary voxels** (per-cell surface mesh or marching-cubes, only boundary voxels — instancing every voxel of thousands of cells is too heavy), colored by cell type, with optional **field slices / iso-surfaces** overlaid and a **timeline scrubber** for playback across saved timesteps. The exporter (a seam added in Sub-project A: `cpm-py` already reads back lattice snapshots) writes `cpm.pack` frames; surface extraction can run in Rust (fast, reuses the boundary-voxel tracker) or in the viewer's worker.

## 3D performance strategy (the differentiator)

1. **Incremental everything** — volume, surface, and center-of-mass updated by ±deltas on each accepted flip, never rescanned. Made an invariant, property-tested against full recompute.
2. **Cache-friendly lattice** — flat `Vec<CellId>`; precomputed neighbor-offset tables per lattice type (2D 4/8-neighbor, 3D 6/18/26-neighbor); Morton/Z-order optional.
3. **Parallel sweeps via checkerboard / sublattice domain decomposition** (rayon) — non-adjacent blocks' copy-attempts run concurrently with halo locking so no two concurrent flips touch adjacent pixels. Main 3D win and the key research risk (parallelization perturbs detailed balance at block boundaries; mitigated with documented boundary handling). **Deferred to a validated follow-up after single-threaded lands.**
4. **GPU** — leave a `cpm-gpu` seam (parsimony has the pattern); not in Sub-project A.

## Schema + ontology alignment (sketch, detailed in C)

Build on multicell-schema's nested object model. Every schema type carries ontology annotations:

- **CBO** (Cell Behavior Ontology) — OpenVT/CC3D-native behavior ontology (chemotaxis, adhesion, mitosis). Primary alignment.
- **CL** (cell types), **GO** (subcellular processes), **SBO** (math/rate-law semantics), **KiSAO** (solver/algorithm identity — which diffusion solver, which sweep algorithm), **PATO** (phenotypic qualities — volume, surface).
- **Round-trip target:** `demos/cell_sorting/` schema JSON ↔ equivalent CC3DML, to pull CC3D demos in and validate a match.

Schema surface (shaped now so A's binding fits): `PottsConfig` (dims, steps, temperature, neighbor-order, boundary, lattice-type), `CellType`, `EnergyTerm` (Volume / Surface / Contact / Chemotaxis / … each a typed process with params), `Field`, `SubcellularModel`.

## Subcellular plugin mechanism (sketch, detailed in D)

Each cell is a place in the bigraph; a subcellular model is a pbg Process attached at that place. Standard interface: **read** {local field concentrations at cell, cell volume/type/age} → **write** {target_volume, target_surface, type, secretion_rate, death flag}. Backends: SBML/ODE (existing RoadRunner/tellurium/amici wrappers), Boolean (small logic evaluator), or any raw pbg Process.

## Testing & demo-parity strategy

- **Core:** Rust unit tests for energy deltas, trackers, boundary conditions; property test that incremental trackers == full recompute after random sweeps.
- **Determinism:** seeded RNG → reproducible runs → golden regression snapshots per demo.
- **Parity:** each ported demo asserts a qualitative + quantitative signature (cell-sorting: final heterotypic boundary length below threshold; chemotaxis: net COM displacement up-gradient).
- **Benchmark:** `cpm-bench` reports 3D sweeps/sec vs a CC3D baseline at matched lattice/cell counts — the "better in 3D" scorecard.

---

## Sub-project A — the vertical slice (this spec)

**Deliverable:** 2D cell sorting running end-to-end from a schema file, orchestrated as a process-bigraph Process, producing the classic sorted-blob result as a seeded regression test; then the same in 3D.

**In scope for A:**

- `cpm-core` crate, generic over dimension (2D/3D):
  - Lattice + cell-ownership storage; seeded RNG; boundary conditions (no-flux, periodic).
  - Metropolis copy-attempt sweep with Boltzmann acceptance at temperature `T`.
  - Energy plugins needed for sorting + near-term demos: **Volume**, **Surface**, **Contact/Adhesion** (type-pair `J` matrix). Stub seams for Chemotaxis / ExternalPotential / Connectivity (Chemotaxis wired in B).
  - Incremental trackers: volume, surface, center-of-mass, neighbor-contact table.
- `cpm-py` pyo3 bindings: construct a world from config, run N sweeps, read back per-cell aggregates + a lattice snapshot; export a minimal single-frame `cpm.pack` (the seam the Sub-project F viewer consumes).
- `cpm/processes/CPMProcess.py`: process-bigraph Process wrapping the core; a minimal schema loader that builds a world from a `PottsConfig` + `CellType` + `EnergyTerm` JSON (minimal slice of Sub-project C).
- Cell-sorting demo schema + regression tests (2D and 3D).
- `cpm-bench` skeleton with a first 3D sweeps/sec measurement (single-threaded baseline).

**Out of scope for A:** PDE/fields (B), full schema+ontology layer (C), subcellular models (D), broader demo suite (E), the full interactive viewer (F) — only the `cpm.pack` export seam is in A — and parallel/GPU sweeps.

**Acceptance:**

- 2D cell sorting from a schema file produces monotonic decrease in heterotypic boundary length; final value below a set threshold; result is deterministic under a fixed seed.
- 3D cell sorting runs and sorts.
- Property test: incremental trackers equal full recompute after random sweeps.
- `cpm-bench` prints a 3D single-threaded sweeps/sec baseline number.
