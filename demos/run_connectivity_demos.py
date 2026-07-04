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
import sys

from cpm import cpm_core

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
    # C-shape (2px-thick walls) around a medium bay in an 18x18 lattice; a
    # single-row mouth spanning the wall's full thickness sits at y=8 on the
    # +x side (leaving (13,8) and (14,8) as medium). Thick walls keep the
    # ring itself stable under thermal noise; the mouth remains the one
    # cheap single-flip energy win available, so it is what closes.
    x0, x1, y0, y1, thick = 3, 15, 3, 15, 2
    w.seed_block(c, x0, y0, 0, x0 + thick, y1, 1)          # left wall
    w.seed_block(c, x0, y0, 0, x1, y0 + thick, 1)          # bottom wall
    w.seed_block(c, x0, y1 - thick, 0, x1, y1, 1)          # top wall
    w.seed_block(c, x1 - thick, y0, 0, x1, 8, 1)           # right wall, below mouth
    w.seed_block(c, x1 - thick, 9, 0, x1, y1, 1)           # right wall, above mouth


def run_horseshoe(medium_connectivity):
    w = cpm_core.World((18, 18, 1), "noflux", 2, 10.0)
    c = w.add_cell(1, 78.0, 2.0, 0.0, 0.0)
    w.set_contact(0, 1, 8.0)   # positive medium adhesion -> cell closes the mouth
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
