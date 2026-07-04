"""Basement membrane (E3a): the E2 crypt monolayer, run under a relaxation hotter
than E2 uses, twice on the same seed -- once WITHOUT a membrane (control: the
epithelium DETACHES from its basal surface / curls off the lamina) and once WITH
the epithelium anchored to a basement membrane on its BASAL (outer) surface (it
stays a coherent monolayer sitting on the membrane). The anchor is what keeps the
tissue on its lamina. Validates + exports for the viewer.

The membrane is the BASAL face of the shell (the outer surface, adjacent to the
exterior medium -- NOT the enclosed lumen), and cells are held within `BAND` of it
by a soft quadratic anchor energy. band ~ wall means anchored cells relax freely
inside the wall shell but pay energy to detach outward or thicken past ~band into
the lumen. This is the spec's soft basal band, NOT a pin to the whole seeded
footprint -- so the effect is robust across a wide stiffness range (K ~ [8, 30]),
not a knife-edge.

SCOPE: the membrane resists DETACHMENT/curling (its real job). It does NOT seal a
thin-wall lumen perforation under high heat -- a basal anchor holds cells on the
surface but not the wall's integrity against a local breach. That is what
cell-cell junctions / wall mechanics (Sub-project E3b) add; the membrane run's
lumen state is reported here, not gated, so the demo does not over-claim.

Usage (repo root, venv active):  python demos/run_membrane_demo.py
"""
import json
import os
import sys
from collections import deque

from cpm import cpm_core

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cpm.crypt3d import build_crypt3d
from cpm.metrics import (radial_cell_counts, interior_medium_pockets,
                         connected_components, membrane_distance_field)

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
ANCHORED_TYPES = {1, 2, 3}   # all epithelial types feel the membrane
WALL = 3                     # crypt wall thickness (build_crypt3d(wall=WALL))
BAND = float(WALL)           # basal band ~ wall: cells stay within a wall of the surface
K = 12.0                     # anchor stiffness (robust across ~[8,30]; see SCOPE note)
TEMP = 10.0                  # hotter than E2 (which used 4.0) -> control detaches
N_FRAMES, MCS_PER_FRAME = 8, 4

_NB6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


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
                for dx, dy, dz in _NB6:
                    x2, y2, z2 = x+dx, y+dy, z+dz
                    if not (0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz) or owner(x2,y2,z2) != c:
                        out.append([x, y, z, c]); break
    return out


def basal_surface_anchors(labels, dims):
    """Basement membrane = the BASAL (outer) face of the epithelial shell: shell
    voxels 6-adjacent to EXTERIOR medium (medium reachable from the lattice
    border), as opposed to the enclosed lumen. A 1-voxel-thick surface on the
    basal side; the distance field grows from it across the wall (apical voxels
    ~wall-1) and outward into the exterior, so with band ~ wall the monolayer can
    relax inside the wall but not detach outward or thicken into the lumen."""
    nx, ny, nz = dims
    n = nx * ny * nz
    def idx(x, y, z):
        return x + y * nx + z * nx * ny
    # exterior medium: BFS through medium (label 0) from every border medium voxel
    exterior = bytearray(n)
    q = deque()
    for i in range(n):
        if labels[i] != 0:
            continue
        z, rem = divmod(i, nx * ny)
        y, x = divmod(rem, nx)
        if x in (0, nx - 1) or y in (0, ny - 1) or z in (0, nz - 1):
            if not exterior[i]:
                exterior[i] = 1
                q.append(i)
    while q:
        v = q.popleft()
        z, rem = divmod(v, nx * ny)
        y, x = divmod(rem, nx)
        for dx, dy, dz in _NB6:
            x2, y2, z2 = x + dx, y + dy, z + dz
            if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
                w = idx(x2, y2, z2)
                if labels[w] == 0 and not exterior[w]:
                    exterior[w] = 1
                    q.append(w)
    # basal surface = shell voxels with a 6-neighbour in exterior medium
    anchors = []
    for i in range(n):
        if labels[i] == 0:
            continue
        z, rem = divmod(i, nx * ny)
        y, x = divmod(rem, nx)
        for dx, dy, dz in _NB6:
            x2, y2, z2 = x + dx, y + dy, z + dz
            if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz and exterior[idx(x2, y2, z2)]:
                anchors.append(i)
                break
    return anchors


def anchored_distance_p95(w, dist, anchored_types):
    """95th-percentile membrane distance over the voxels of anchored-type cells.
    Robust to a single stray voxel; rises when cells detach from / thicken off the
    basal surface. (A mean is near-useless here -- the bulk of a wall-thick shell
    sits at small distance regardless of a few detached cells.)"""
    labels = w.snapshot()
    types = w.cell_types()
    ds = sorted(dist[i] for i, c in enumerate(labels) if c != 0 and types[c] in anchored_types)
    return ds[int(0.95 * len(ds))] if ds else 0.0


def build_world(dims, labels, seg_to_type, anchors, with_membrane):
    nx, ny, nz = dims
    w = cpm_core.World((nx, ny, nz), "noflux", 2, TEMP)
    w.seed_from_labels(labels, seg_to_type, 1, 20.0, 20.0)
    for t in range(1, 4):
        w.set_contact(0, t, 6.0)
        for u in range(t, 4):
            w.set_contact(t, u, 4.0)
        w.set_connectivity(t, True)
    w.set_connectivity_medium(True)
    if with_membrane:
        w.set_membrane(anchors, K, BAND)
        for t in ANCHORED_TYPES:
            w.set_membrane_anchored(t, True)
    w.finalize(1)
    vols0 = w.cell_volumes()
    for c in range(1, len(vols0)):
        if vols0[c] > 0:
            w.set_target_volume(c, float(vols0[c]))
    return w


def relax(w, dims, dist, capture):
    """Step the world, tracking worst-case structure over EVERY frame. Returns a
    dict of aggregates (+ frames if capture)."""
    nx, ny, nz = dims
    frames = []
    prev = None
    min_pockets = 10 ** 9
    worst_mean, worst_p90, worst_memp95 = 0.0, 0, 0.0
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
        worst_memp95 = max(worst_memp95, anchored_distance_p95(w, dist, ANCHORED_TYPES))
        min_pockets = min(min_pockets, interior_medium_pockets(w))
        if capture:
            frames.append({"mcs": f * MCS_PER_FRAME, "voxels": surface_3d(w)})
        if f < N_FRAMES:
            w.step(MCS_PER_FRAME)
    types = w.cell_types(); vols = w.cell_volumes()
    frag = sum(1 for c in range(1, len(types)) if vols[c] > 0 and connected_components(w, c) != 1)
    alive = sum(1 for c in range(1, len(types)) if vols[c] > 0)
    return {"worst_mean": worst_mean, "worst_p90": worst_p90, "worst_memp95": worst_memp95,
            "min_pockets": min_pockets, "min_step_churn": min_step_churn or 0,
            "frag": frag, "alive": alive, "n0": w.n_cells(),
            "cell_types": list(types), "frames": frames}


def main():
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d(wall=WALL)
    dims = (nx, ny, nz)
    anchors = basal_surface_anchors(labels, dims)
    dist = membrane_distance_field(dims, anchors)

    ctrl = relax(build_world(dims, labels, seg_to_type, anchors, False), dims, dist, capture=False)
    mem = relax(build_world(dims, labels, seg_to_type, anchors, True), dims, dist, capture=True)

    # The membrane's job is ANCHORING: keeping the epithelium on its basal
    # surface (resisting detachment/curling/thickening-off-the-lamina). The
    # headline is the contrast on the basal-distance p95: WITHOUT the membrane
    # the cells leave the basal band (p95 > band); WITH it they stay (p95 <= band).
    # This is robust across a wide K range (verified ~[8,30]).
    #
    # NOT claimed: sealing a thin-wall lumen perforation under this heat. A basal
    # anchor holds cells ON the surface but does not by itself stop a local wall
    # breach (a breach only makes medium more connected -- see the E2 crypt3d
    # finding). Sealing the wall is what cell-cell junctions / wall mechanics
    # (Sub-project E3b) add; here the membrane run's lumen state is REPORTED, not
    # gated, so the demo does not over-claim.
    control_detaches = ctrl["worst_memp95"] > BAND
    checks = [
        (f"CONTROL (no membrane) detaches from the basal surface under the stress "
         f"(p95 membrane distance {ctrl['worst_memp95']:.1f} > band {BAND}) -- proves the "
         f"anchor does the work", control_detaches),
        (f"WITH membrane: cells stay on the basal membrane (worst p95 membrane distance "
         f"{mem['worst_memp95']:.1f} <= band {BAND}; control was {ctrl['worst_memp95']:.1f}) "
         f"-- the same test the control FAILS", mem["worst_memp95"] <= BAND),
        (f"WITH membrane: single-cell monolayer holds (worst mean {mem['worst_mean']:.2f} < 1.5, "
         f"worst p90 {mem['worst_p90']} <= 2)", mem["worst_mean"] < 1.5 and mem["worst_p90"] <= 2),
        (f"WITH membrane: no cell fragments ({mem['frag']} split) and all survive "
         f"({mem['alive']}/{mem['n0']})", mem["frag"] == 0 and mem["alive"] == mem["n0"]),
        (f"WITH membrane: relaxation is non-trivial (min per-step churn "
         f"{mem['min_step_churn']} > 0 -- anchored, not frozen)", mem["min_step_churn"] > 0),
    ]

    data = {"name": "Basement Membrane (crypt)", "kind": "membrane", "dims": [nx, ny, nz],
            "is3d": True, "n_cells": mem["n0"], "cell_types": mem["cell_types"],
            "type_names": ["Medium"] + type_names, "frames": mem["frames"]}
    os.makedirs(DATA, exist_ok=True)
    json.dump(data, open(os.path.join(DATA, "membrane.json"), "w"))
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    manifest = [m for m in manifest if m["file"] != "membrane.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "membrane.json", "name": data["name"], "is3d": True,
                     "n_cells": data["n_cells"], "dims": data["dims"], "kind": "membrane",
                     "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json", "hra_ftu.json", "crypt_differentiation.json",
             "connectivity_2d.json", "connectivity_3d.json", "connectivity_gap.json",
             "crypt3d.json", "membrane.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (basement membrane) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print(f"   [info] lumen enclosure (NOT gated -- E3b junctions' job): "
          f"membrane min pockets {mem['min_pockets']}, control {ctrl['min_pockets']}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
