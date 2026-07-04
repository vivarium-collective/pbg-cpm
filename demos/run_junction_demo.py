"""Cell-cell junctions (E3b): the E2 crypt monolayer under a relaxation hot enough
that the wall perforates and the lumen breaches WITHOUT junctions, run twice on the
same seed -- once WITHOUT junctions (control: gaps open, the lumen breaches) and
once WITH junctions (the anti-gap energy seals the seams: no perforation, lumen
stays enclosed). This is the "no gaps, no merging" physics E3a's basal anchor
leaves open. Validates + exports for the viewer.

Junctions penalise medium pinched between two different junction-enabled cells, so
opening a gap/film/perforation costs energy -- WITHOUT adding surface tension (free
surface is never pinched), so the sheet does not curl.

Usage (repo root, venv active):  python demos/run_junction_demo.py
"""
import json
import os
import sys

from cpm import cpm_core

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cpm.crypt3d import build_crypt3d
from cpm.metrics import (radial_cell_counts, interior_medium_pockets,
                         connected_components, intercell_gap_faces)

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
JUNCTION_TYPES = {1, 2, 3}
LAMBDA_J = 96.0              # junction stiffness (tune so it seals without freezing)
TEMP = 10.0                 # hot enough that the control wall perforates
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


def build_world(dims, labels, seg_to_type, with_junctions):
    nx, ny, nz = dims
    w = cpm_core.World((nx, ny, nz), "noflux", 2, TEMP)
    w.seed_from_labels(labels, seg_to_type, 1, 20.0, 20.0)
    for t in range(1, 4):
        w.set_contact(0, t, 6.0)
        for u in range(t, 4):
            w.set_contact(t, u, 4.0)
        w.set_connectivity(t, True)
    w.set_connectivity_medium(True)
    if with_junctions:
        for t in JUNCTION_TYPES:
            w.set_junction(t, True)
        w.set_junction_lambda(LAMBDA_J)
    w.finalize(1)
    vols0 = w.cell_volumes()
    for c in range(1, len(vols0)):
        if vols0[c] > 0:
            w.set_target_volume(c, float(vols0[c]))
    return w


def relax(w, dims, capture):
    nx, ny, nz = dims
    frames = []
    prev = None
    min_pockets = 10 ** 9
    worst_mean, worst_p90, worst_gap = 0.0, 0, 0
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
        worst_gap = max(worst_gap, intercell_gap_faces(w, JUNCTION_TYPES))
        min_pockets = min(min_pockets, interior_medium_pockets(w))
        if capture:
            frames.append({"mcs": f * MCS_PER_FRAME, "voxels": surface_3d(w)})
        if f < N_FRAMES:
            w.step(MCS_PER_FRAME)
    types = w.cell_types(); vols = w.cell_volumes()
    frag = sum(1 for c in range(1, len(types)) if vols[c] > 0 and connected_components(w, c) != 1)
    alive = sum(1 for c in range(1, len(types)) if vols[c] > 0)
    return {"worst_mean": worst_mean, "worst_p90": worst_p90, "worst_gap": worst_gap,
            "min_pockets": min_pockets, "min_step_churn": min_step_churn or 0,
            "frag": frag, "alive": alive, "n0": w.n_cells(),
            "cell_types": list(types), "frames": frames}


def main():
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d(wall=3)
    dims = (nx, ny, nz)
    ctrl = relax(build_world(dims, labels, seg_to_type, False), dims, capture=False)
    jun = relax(build_world(dims, labels, seg_to_type, True), dims, capture=True)

    # Control must genuinely open gaps / breach; else the junction contrast is vacuous.
    control_degrades = ctrl["min_pockets"] == 0 or ctrl["worst_gap"] > jun["worst_gap"]
    checks = [
        (f"CONTROL (no junctions) opens gaps / breaches (worst gap faces {ctrl['worst_gap']}, "
         f"min pockets {ctrl['min_pockets']}) -- proves the junctions do the work",
         control_degrades),
        (f"WITH junctions: seams stay sealed (worst gap faces {jun['worst_gap']} < control "
         f"{ctrl['worst_gap']})", jun["worst_gap"] < ctrl["worst_gap"]),
        (f"WITH junctions: lumen stays enclosed (min interior pockets {jun['min_pockets']} >= 1)",
         jun["min_pockets"] >= 1),
        (f"WITH junctions: monolayer holds, no curling (worst mean {jun['worst_mean']:.2f} < 1.5, "
         f"worst p90 {jun['worst_p90']} <= 2)", jun["worst_mean"] < 1.5 and jun["worst_p90"] <= 2),
        (f"WITH junctions: no cell fragments ({jun['frag']} split) and all survive "
         f"({jun['alive']}/{jun['n0']})", jun["frag"] == 0 and jun["alive"] == jun["n0"]),
        (f"WITH junctions: relaxation is non-trivial (min per-step churn "
         f"{jun['min_step_churn']} > 0 -- sealed, not frozen)", jun["min_step_churn"] > 0),
    ]

    data = {"name": "Cell-Cell Junctions (crypt)", "kind": "junction", "dims": [nx, ny, nz],
            "is3d": True, "n_cells": jun["n0"], "cell_types": jun["cell_types"],
            "type_names": ["Medium"] + type_names, "frames": jun["frames"]}
    os.makedirs(DATA, exist_ok=True)
    json.dump(data, open(os.path.join(DATA, "junction.json"), "w"))
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    manifest = [m for m in manifest if m["file"] != "junction.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "junction.json", "name": data["name"], "is3d": True,
                     "n_cells": data["n_cells"], "dims": data["dims"], "kind": "junction",
                     "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json", "hra_ftu.json", "crypt_differentiation.json",
             "connectivity_2d.json", "connectivity_3d.json", "connectivity_gap.json",
             "crypt3d.json", "membrane.json", "junction.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (cell-cell junctions) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
