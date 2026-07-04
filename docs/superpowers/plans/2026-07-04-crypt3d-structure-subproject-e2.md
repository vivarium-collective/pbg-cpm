# 3D Crypt Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-cell-thick 3D epithelial crypt (procedural test-tube shell) held together by the E1 connectivity constraint, with cells typed by axial position, and validate its structural consistency.

**Architecture:** A pure Python geometry generator tiles a capped-cylinder shell into cells and types them by height; a demo seeds a CPM from it, relaxes with connectivity ON (cells + medium), validates a monolayer that stays whole with an enclosed lumen, and exports 3D voxels for the viewer.

**Tech Stack:** Python (`cpm` pkg + `cpm_core`), the existing 3D viewer export.

## Global Constraints

- The generator is pure/deterministic (no RNG). Cells rasterising to `< 3` voxels are dropped and labels re-indexed consecutively from 1.
- The CPM relaxation uses the E1 connectivity constraint ON for all cell types AND the medium; a short, low-temperature run (structure-first — long-term stability is E3).
- Type ids: `1 = Epithelial Stem`, `2 = Absorptive`, `3 = Goblet`; `type_names = ["Epithelial Stem", "Absorptive", "Goblet"]` (index-aligned so `type_names[t-1]` names type `t`).
- Metrics assume a bounded `noflux` domain (matching E1).
- Build/test in the repo `.venv` (py3.12); pytest `pythonpath=["."]`.

---

## File Structure

- `cpm/crypt3d.py` (new) — `build_crypt3d(...) -> ((nx,ny,nz), labels, seg_to_type, type_names)`.
- `cpm/metrics.py` — add `radial_thickness(world, cx, cy, n_theta)`.
- `demos/run_crypt3d.py` (new) — build + relax + validate + export.
- Tests: `tests/test_crypt3d.py`.

---

### Task 1: Geometry generator + thickness metric

**Files:**
- Create: `cpm/crypt3d.py`
- Modify: `cpm/metrics.py` (add `radial_thickness`)
- Test: `tests/test_crypt3d.py`

**Interfaces:**
- Produces: `build_crypt3d(radius=8, cyl_height=28, wall=3, cell_pitch=6, margin=4) -> ((nx, ny, nz), labels, seg_to_type, type_names)` — `labels` row-major (`x + y*nx + z*nx*ny`), `seg_to_type` maps consecutive label → type id (1/2/3), `type_names` as in Global Constraints.
- Produces: `cpm.metrics.radial_thickness(world, cx=None, cy=None, n_theta=24) -> (mean, max)` — distinct cells crossed along radial rays.

- [ ] **Step 1: Write the failing test**

Create `tests/test_crypt3d.py`:

```python
import cpm_core
from cpm.crypt3d import build_crypt3d
from cpm.metrics import radial_thickness, interior_medium_pockets, connected_components


def test_build_crypt3d_is_a_thin_typed_shell():
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d()
    n_cells = len(seg_to_type)
    assert n_cells > 30                       # a real tiling, not a few blobs
    assert type_names == ["Epithelial Stem", "Absorptive", "Goblet"]
    # seed into a world so we can use the metrics
    w = cpm_core.World((nx, ny, nz), "noflux", 2, 5.0)
    m = w.seed_from_labels(labels, seg_to_type, 1, 20.0, 3.0)
    for t in range(1, 4):
        w.set_contact(0, t, 6.0)
        for u in range(t, 4):
            w.set_contact(t, u, 4.0)
    w.finalize(1)
    assert w.n_cells() == n_cells
    # thin shell: at most ~1 cell between lumen and exterior
    mean_t, max_t = radial_thickness(w, nx / 2.0, ny / 2.0)
    assert max_t <= 2 and mean_t < 1.6
    # lumen is an enclosed interior medium pocket
    assert interior_medium_pockets(w) >= 1
    # stem cells are basal (lower z) than goblet cells
    types = w.cell_types()
    coms = w.cell_coms()
    stem_z = [coms[c][2] for c in range(1, len(types)) if types[c] == 1]
    gob_z = [coms[c][2] for c in range(1, len(types)) if types[c] == 3]
    assert stem_z and gob_z and (sum(stem_z) / len(stem_z)) < (sum(gob_z) / len(gob_z))
```

- [ ] **Step 2: Run it, verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_crypt3d.py -v`
Expected: FAIL — `cpm.crypt3d` / `radial_thickness` not found.

- [ ] **Step 3: Implement `cpm/crypt3d.py`**

```python
"""Procedural 3D crypt geometry: a single-cell-thick epithelial shell shaped
like a test tube (hollow cylinder closed by a hemispherical base), tiled into
cells and typed by axial position (stem niche basal). Pure/deterministic."""
import math
from collections import Counter

TYPE_NAMES = ["Epithelial Stem", "Absorptive", "Goblet"]
STEM, ABS, GOB = 1, 2, 3


def build_crypt3d(radius=8, cyl_height=28, wall=3, cell_pitch=6, margin=4):
    R = float(radius)
    nx = ny = 2 * (radius + margin)
    nz = radius + cyl_height + 2 * margin
    cx = cy = nx / 2.0
    z_base = margin + radius              # cap pole at z=margin, equator at z=z_base
    half = wall / 2.0
    a_cap = R * (math.pi / 2.0)           # profile arc length of the cap
    a_max = a_cap + cyl_height

    labels = [0] * (nx * ny * nz)
    cellmap, seg_to_type = {}, {}
    next_seg = 1

    for z in range(nz):
        zc = z + 0.5
        for y in range(ny):
            dy = y + 0.5 - cy
            for x in range(nx):
                dx = x + 0.5 - cx
                r = math.hypot(dx, dy)
                if zc >= z_base:                    # cylinder
                    if abs(r - R) > half or zc > z_base + cyl_height:
                        continue
                    a = a_cap + (zc - z_base)
                    r_local = R
                else:                               # hemispherical cap
                    d = math.sqrt(dx * dx + dy * dy + (zc - z_base) ** 2)
                    if abs(d - R) > half:
                        continue
                    cphi = max(-1.0, min(1.0, (z_base - zc) / R))
                    phi = math.acos(cphi)           # 0 at pole -> pi/2 at equator
                    a = R * (math.pi / 2.0 - phi)   # 0 at pole -> a_cap at equator
                    r_local = max(1.0, R * math.sin(phi))
                if a < 0 or a > a_max:
                    continue
                axial_bin = int(a // cell_pitch)
                n_theta = max(1, round(2 * math.pi * r_local / cell_pitch))
                theta = math.atan2(dy, dx) + math.pi
                theta_bin = int(theta // (2 * math.pi / n_theta)) % n_theta
                key = (axial_bin, theta_bin)
                seg = cellmap.get(key)
                if seg is None:
                    seg = next_seg
                    next_seg += 1
                    cellmap[key] = seg
                    if a <= a_cap + cyl_height * 0.25:
                        seg_to_type[seg] = STEM
                    elif a <= a_cap + cyl_height * 0.60:
                        seg_to_type[seg] = ABS
                    else:
                        seg_to_type[seg] = GOB
                labels[x + y * nx + z * nx * ny] = seg

    # drop tiny cells and re-index consecutively from 1
    sizes = Counter(v for v in labels if v)
    keep = sorted(s for s, c in sizes.items() if c >= 3)
    remap = {s: i + 1 for i, s in enumerate(keep)}
    labels = [remap.get(v, 0) for v in labels]
    seg_to_type = {remap[s]: seg_to_type[s] for s in keep}
    return (nx, ny, nz), labels, seg_to_type, TYPE_NAMES
```

- [ ] **Step 4: Implement `radial_thickness`**

Append to `cpm/metrics.py`:

```python
def radial_thickness(world, cx=None, cy=None, n_theta=24):
    """Mean and max number of distinct cells crossed along radial rays from the
    axis (cx, cy) outward, over a (z, theta) sample grid. ~1 for a monolayer."""
    import math
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    if cx is None:
        cx = nx / 2.0
    if cy is None:
        cy = ny / 2.0
    rmax = int(math.hypot(nx, ny)) + 1
    counts = []
    for z in range(nz):
        base = z * nx * ny
        for ti in range(n_theta):
            theta = 2.0 * math.pi * ti / n_theta
            ct, st = math.cos(theta), math.sin(theta)
            crossed = set()
            for r in range(rmax):
                x = int(cx + r * ct)
                y = int(cy + r * st)
                if not (0 <= x < nx and 0 <= y < ny):
                    break
                o = labels[base + x + y * nx]
                if o != 0:
                    crossed.add(o)
            if crossed:
                counts.append(len(crossed))
    if not counts:
        return 0.0, 0
    return sum(counts) / len(counts), max(counts)
```

- [ ] **Step 5: Run the test, verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_crypt3d.py -v`
Expected: PASS. If `max_t > 2` at t=0 (the raster wall is thicker than one cell radially), reduce `wall` to `2` in the default or raise the `max_t` tolerance to `<= 2` only if the wall is genuinely one cell thick — the wall parameter controls raster thickness; a `wall` of 2–3 voxels is still ONE cell thick (cells span the wall radially), so counting DISTINCT cells (not voxels) along the ray is what matters and should be ~1. Do not loosen below a real single-cell wall.

- [ ] **Step 6: Commit**

```bash
git add cpm/crypt3d.py cpm/metrics.py tests/test_crypt3d.py
git commit -m "feat(crypt3d): procedural 3D crypt shell generator + radial-thickness metric"
```

---

### Task 2: 3D crypt demo + validation + export

**Files:**
- Create: `demos/run_crypt3d.py`

**Interfaces:**
- Consumes: `build_crypt3d`, `cpm.metrics.radial_thickness`/`interior_medium_pockets`/`connected_components`, the 3D `voxels` export + manifest conventions from `demos/run_cc3d_demos.py`/`run_extra_demos.py`.
- Produces: `viewer/data/crypt3d.json` (`kind="crypt3d"`, `is3d=True`) + manifest entry; exits nonzero on any failed gate.

- [ ] **Step 1: Write the demo**

Create `demos/run_crypt3d.py`:

```python
"""3D crypt structure: a procedural single-cell-thick epithelial shell held
together by the connectivity constraint. Builds it, relaxes briefly with
connectivity ON (cells + medium), validates that it stays a coherent monolayer
with an enclosed lumen and a basal stem niche, and exports it for the viewer.

Usage (repo root, venv active):  python demos/run_crypt3d.py
"""
import json
import os

import cpm_core
from cpm.crypt3d import build_crypt3d
from cpm.metrics import radial_thickness, interior_medium_pockets, connected_components

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))


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


def build_world(labels, seg_to_type, dims, median):
    nx, ny, nz = dims
    w = cpm_core.World((nx, ny, nz), "noflux", 2, 6.0)   # low T -> structure holds
    w.seed_from_labels(labels, seg_to_type, 1, float(median), 3.0)
    for t in range(1, 4):
        w.set_contact(0, t, 6.0)          # moderate medium contact
        for u in range(t, 4):
            w.set_contact(t, u, 4.0)      # cohesive cell-cell
        w.set_connectivity(t, True)       # E1: cells stay whole
    w.set_connectivity_medium(True)       # E1: lumen stays enclosed, no gaps
    w.finalize(1)
    return w


def main(n_frames=8, mcs_per_frame=3):
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d()
    from collections import Counter
    median = int(Counter(v for v in labels if v).most_common()[len(Counter(v for v in labels if v)) // 2][1])
    w = build_world(labels, seg_to_type, (nx, ny, nz), median)
    n0 = w.n_cells()

    frames, min_pockets = [], 10**9
    for f in range(n_frames + 1):
        frames.append({"mcs": f * mcs_per_frame, "voxels": surface_3d(w)})
        min_pockets = min(min_pockets, interior_medium_pockets(w))
        if f < n_frames:
            w.step(mcs_per_frame)

    types = w.cell_types()
    coms = w.cell_coms()
    vols = w.cell_volumes()
    mean_t, max_t = radial_thickness(w, nx / 2.0, ny / 2.0)
    frag = sum(1 for c in range(1, len(types)) if vols[c] > 0 and connected_components(w, c) != 1)
    alive = sum(1 for c in range(1, len(types)) if vols[c] > 0)
    stem_z = [coms[c][2] for c in range(1, len(types)) if types[c] == 1 and vols[c] > 0]
    gob_z = [coms[c][2] for c in range(1, len(types)) if types[c] == 3 and vols[c] > 0]

    checks = [
        (f"single-cell-thick monolayer (radial cells mean {mean_t:.2f} < 1.6, max {max_t} <= 2)",
         mean_t < 1.6 and max_t <= 2),
        (f"no cell fragmented ({frag} of {alive} cells split)", frag == 0),
        (f"lumen stays enclosed / no wall breach (min interior pockets {min_pockets} >= 1)",
         min_pockets >= 1),
        (f"stem niche is basal (mean stem z {sum(stem_z)/len(stem_z):.1f} < goblet z "
         f"{sum(gob_z)/len(gob_z):.1f})" if stem_z and gob_z else "stem + goblet present",
         bool(stem_z) and bool(gob_z) and sum(stem_z)/len(stem_z) < sum(gob_z)/len(gob_z)),
        (f"structure persists (all {n0} cells survive: {alive} alive)", alive == n0),
    ]

    data = {"name": "3D Crypt (structure)", "kind": "crypt3d", "dims": [nx, ny, nz],
            "is3d": True, "n_cells": w.n_cells(), "cell_types": list(w.cell_types()),
            "type_names": ["Medium"] + type_names, "frames": frames}
    os.makedirs(DATA, exist_ok=True)
    json.dump(data, open(os.path.join(DATA, "crypt3d.json"), "w"))
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    manifest = [m for m in manifest if m["file"] != "crypt3d.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "crypt3d.json", "name": data["name"], "is3d": True,
                     "n_cells": data["n_cells"], "dims": data["dims"], "kind": "crypt3d",
                     "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json", "hra_ftu.json", "crypt_differentiation.json",
             "connectivity_2d.json", "connectivity_3d.json", "connectivity_gap.json",
             "crypt3d.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (3D crypt structure) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the demo**

Run: `source .venv/bin/activate && python demos/run_crypt3d.py`
Expected: all 5 gates PASS, exit 0; writes `viewer/data/crypt3d.json` + manifest entry. If the shell collapses (a gate fails — thickness climbs, fragmentation appears, or the lumen breaches), REDUCE the stress: lower `n_frames`/`mcs_per_frame`, lower the temperature in `build_world` (e.g. `4.0`), or raise `lambda_volume` — do NOT weaken a gate. The connectivity constraint + low temperature + short run must keep the monolayer intact; tune until they do.

- [ ] **Step 3: Commit**

```bash
git add demos/run_crypt3d.py
git commit -m "feat(demo): 3D crypt structure demo (connectivity-held monolayer)"
```

---

## Self-Review

**Spec coverage:** geometry generator (capped cylinder, axial typing, tiny-cell drop) → Task 1; `radial_thickness` metric → Task 1; CPM assembly with connectivity ON (cells + medium), short low-T relaxation → Task 2; 5 validation gates (monolayer, no fragmentation, lumen enclosed, axial type order, structure persists) → Task 2; 3D voxel export + manifest → Task 2. Determinism: generator is pure; CPM seeded.

**Placeholder scan:** no TBD/TODO; complete code in every step. The one tuning point (shell must not collapse) is called out with concrete knobs (temperature/frames/lambda_volume), not vague.

**Type consistency:** `build_crypt3d(...)` return shape, `seg_to_type` type ids (1/2/3), `type_names`, `radial_thickness(world, cx, cy, n_theta) -> (mean, max)`, `interior_medium_pockets`/`connected_components` usage, and the manifest `kind="crypt3d"` + `order` entry are consistent across Tasks 1–2 and match the E1 metrics.
```
