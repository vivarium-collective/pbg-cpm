"""Basement membrane (E3a): the E2 crypt monolayer, run under a relaxation
STRONGER than E2 survives, twice on the same seed -- once WITHOUT a membrane
(control: it detaches/thickens or the lumen breaches) and once WITH the basal
shell anchored as a basement membrane (it stays a coherent monolayer hugging the
membrane). Shows the anchor is what holds the structure. Validates + exports.

Usage (repo root, venv active):  python demos/run_membrane_demo.py
"""
import json
import os
import sys

from cpm import cpm_core

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cpm.crypt3d import build_crypt3d
from cpm.metrics import (radial_cell_counts, interior_medium_pockets,
                         connected_components, membrane_distance_field,
                         mean_membrane_distance)

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
ANCHORED_TYPES = {1, 2, 3}   # all epithelial types feel the membrane
BAND = 0.5                   # tight band: anchored cells barely drift, but not zero
K = 20.0                     # stiff enough to seal the wall; higher freezes it (gate 6)
TEMP = 10.0                  # hotter than E2 (which used 4.0) -> control fails
N_FRAMES, MCS_PER_FRAME = 8, 4


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


def shell_anchors(labels, dims):
    """Membrane = the seeded shell's own footprint (the basal lamina the
    epithelium sits on). Anchoring to it with band ~ wall keeps cells in the
    membrane shell."""
    return [i for i, v in enumerate(labels) if v != 0]


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
    """Step the world, tracking worst-case structure over every frame. Returns a
    dict of aggregates (+ frames if capture)."""
    nx, ny, nz = dims
    frames = []
    prev = None
    min_pockets = 10 ** 9
    worst_mean, worst_p90, worst_mem = 0.0, 0, 0.0
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
        worst_mem = max(worst_mem, mean_membrane_distance(w, dist, ANCHORED_TYPES))
        min_pockets = min(min_pockets, interior_medium_pockets(w))
        if capture:
            frames.append({"mcs": f * MCS_PER_FRAME, "voxels": surface_3d(w)})
        if f < N_FRAMES:
            w.step(MCS_PER_FRAME)
    types = w.cell_types(); vols = w.cell_volumes()
    frag = sum(1 for c in range(1, len(types)) if vols[c] > 0 and connected_components(w, c) != 1)
    alive = sum(1 for c in range(1, len(types)) if vols[c] > 0)
    return {"worst_mean": worst_mean, "worst_p90": worst_p90, "worst_mem": worst_mem,
            "min_pockets": min_pockets, "min_step_churn": min_step_churn or 0,
            "frag": frag, "alive": alive, "n0": w.n_cells(),
            "cell_types": list(types), "frames": frames}


def main():
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d(wall=3)
    dims = (nx, ny, nz)
    anchors = shell_anchors(labels, dims)
    dist = membrane_distance_field(dims, anchors)

    ctrl = relax(build_world(dims, labels, seg_to_type, anchors, False), dims, dist, capture=False)
    mem = relax(build_world(dims, labels, seg_to_type, anchors, True), dims, dist, capture=True)

    control_degrades = ctrl["min_pockets"] == 0 or ctrl["worst_mean"] >= 1.5
    checks = [
        (f"CONTROL (no membrane) degrades under the stress -- proves the anchor "
         f"does the work (min pockets {ctrl['min_pockets']}, worst mean {ctrl['worst_mean']:.2f})",
         control_degrades),
        (f"WITH membrane: single-cell monolayer holds (worst mean {mem['worst_mean']:.2f} < 1.5, "
         f"worst p90 {mem['worst_p90']} <= 2)", mem["worst_mean"] < 1.5 and mem["worst_p90"] <= 2),
        (f"WITH membrane: lumen stays enclosed (min interior pockets {mem['min_pockets']} >= 1)",
         mem["min_pockets"] >= 1),
        (f"WITH membrane: cells stay anchored (worst mean membrane distance "
         f"{mem['worst_mem']:.2f} <= band {BAND})", mem["worst_mem"] <= BAND),
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
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
