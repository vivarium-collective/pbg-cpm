# pbg-cpm

A [process-bigraph](https://github.com/vivarium-collective/process-bigraph) **Cellular Potts Model** framework — a fast Rust CPM engine (2D/3D, thousands of cells) with a Python layer for pluggable subcellular models, structural constraints, schema-driven world construction, and analysis metrics. A modern, composable remake of CompuCell3D built to do better in 3D.

## Install

```bash
pip install pbg-cpm                 # core: Rust CPM engine + process-bigraph layer
pip install "pbg-cpm[sbml]"        # + SBML/ODE subcellular models (pbg-tellurium)
pip install "pbg-cpm[ftu]"         # + Human Reference Atlas FTU -> CPM conversion
pip install "pbg-cpm[all]"         # everything (for the full demo suite)
```

From source (editable, requires a Rust toolchain + [maturin](https://www.maturin.rs)):

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop -m crates/cpm-py/Cargo.toml
pytest
```

## Use it from another project

```python
from cpm import load_world, cpm_core

spec = {
    "potts": {"dims": [50, 50, 1], "boundary": "periodic",
              "neighbor_order": 2, "temperature": 12.0, "seed": 0},
    "cells": [
        {"type": 1, "target_volume": 25, "lambda_volume": 1.0,
         "target_surface": 0, "lambda_surface": 0, "seed_block": [5, 5, 0, 13, 13, 1]},
    ],
    "contact": [{"a": 0, "b": 1, "j": 12.0}],
}
world = load_world(spec)
world.step(100)                     # run 100 Monte-Carlo sweeps
print(world.cell_volumes())
```

The engine itself is `cpm.cpm_core` (a compiled Rust extension). `load_world` builds a
world from a plain dict spec (cells or a seeded label array, contact energies, diffusion
fields, connectivity, basement membrane).

## Process-bigraph composites

Cells are wired as process-bigraph processes via import-path addresses, so any
process-bigraph `Composite` can embed them:

- `local:!cpm.processes.cpm_process.CPMProcess` — the CPM step as a process
- `local:!cpm.subcellular.sbml.SBMLSubcell` — a per-cell SBML/ODE model (needs `[sbml]`)
- `local:!cpm.subcellular.boolean.BooleanSubcell` — a per-cell Boolean fate network

See `cpm/composites/crypt.py` for a full crypt-differentiation composite (CPM + SBML
stemness ODE + Boolean fate switch), run with the process-bigraph `Composite` engine.

## Structural constraints

- **Connectivity** (E1): forbids copy attempts that would fragment a cell or pinch off
  interior medium (gaps). `spec["connectivity"] = {"types": [1, 2], "medium": true}`.
- **Basement membrane** (E3a): a basal anchor energy keeping epithelial cells in a thin
  band hugging a fixed membrane surface. `spec["membrane"] = {"anchors": [...], "k": ...,
  "band": ..., "types": [...]}`.

## Layout

```
crates/          Rust workspace: cpm-core (engine) + cpm-py (pyo3 bindings -> cpm.cpm_core)
cpm/             Python framework: schema, processes, subcellular, composites, metrics, ftu
demos/           runnable demos (each validates + exports a viewer model)
viewer/          browser 2D/3D viewer for the exported models
docs/            specs & implementation plans
tests/           Rust (cargo test) + Python (pytest) suites
```

## License

MIT
