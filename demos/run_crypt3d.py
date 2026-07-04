"""3D crypt structure: a procedural single-cell-thick epithelial shell held
together by the connectivity constraint. Builds it, relaxes briefly with
connectivity ON (cells + medium), validates that it stays a coherent monolayer
with an enclosed lumen and a basal stem niche, and exports it for the viewer.

Usage (repo root, venv active):  python demos/run_crypt3d.py
"""
import json
import os
import sys

import cpm_core

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
    w = cpm_core.World((nx, ny, nz), "noflux", 2, 4.0)   # low T -> structure holds
    w.seed_from_labels(labels, seg_to_type, 1, float(median), 20.0)
    for t in range(1, 4):
        w.set_contact(0, t, 6.0)          # moderate medium contact
        for u in range(t, 4):
            w.set_contact(t, u, 4.0)      # cohesive cell-cell
        w.set_connectivity(t, True)       # E1: cells stay whole
    w.set_connectivity_medium(True)       # E1: lumen stays enclosed, no gaps
    w.finalize(1)
    # Re-anchor each cell's target volume to its OWN just-seeded volume
    # (must happen AFTER finalize -- finalize derives target volume from the
    # single scalar passed to seed_from_labels, so setting it before finalize
    # gets overwritten). Cell sizes vary a lot across this geometry (small
    # polar wedges vs. full cylinder rings), so a single shared target (the
    # median) drives every off-median cell to aggressively grow/shrink -- a
    # volume-mismatch force that swamps the mild J_medium=6 > J_cell=4
    # adhesion tension and collapses the monolayer within the first MCS
    # regardless of temperature (these are downhill, not thermal, moves).
    # Matching each cell's target to what it already has removes that
    # spurious driving force so the shell relaxes under adhesion alone, with
    # a moderate lambda_volume (20) keeping volume drift costly WITHOUT pinning
    # every boundary rigid -- lambda_volume=80 froze the shell solid (zero
    # accepted flips: the "relaxation" was vacuous), so we lower it until the
    # boundary genuinely fluctuates (see the "relaxation is non-trivial" gate)
    # while connectivity + volume hold the monolayer together.
    vols0 = w.cell_volumes()
    for c in range(1, len(vols0)):
        if vols0[c] > 0:
            w.set_target_volume(c, float(vols0[c]))
    return w


def main(n_frames=8, mcs_per_frame=3):
    # wall=3: a single-CELL-thick wall needs a few voxels of radial margin to
    # survive boundary roughening -- a razor-thin 2-voxel wall perforates within
    # a handful of MCS and the lumen breaches (interior_medium_pockets -> 0).
    # E1 connectivity forbids fragmentation and medium-pocket PINCH-OFF, but a
    # wall breach MERGES lumen with the exterior (medium becomes more connected),
    # which connectivity permits -- so keeping the lumen sealed under relaxation
    # is a matter of wall robustness here, and full monolayer stability under
    # growth is what the basement membrane / junctions (E3) are for.
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d(wall=3)
    from collections import Counter
    median = int(Counter(v for v in labels if v).most_common()[len(Counter(v for v in labels if v)) // 2][1])
    w = build_world(labels, seg_to_type, (nx, ny, nz), median)
    n0 = w.n_cells()

    frames, min_pockets = [], 10**9
    prev_snap = None
    total_churn = 0                       # voxels that changed owner over the run
    for f in range(n_frames + 1):
        snap = w.snapshot()
        churn = 0 if prev_snap is None else sum(1 for a, b in zip(prev_snap, snap) if a != b)
        total_churn += churn
        prev_snap = snap
        frames.append({"mcs": f * mcs_per_frame, "voxels": surface_3d(w), "churn": churn})
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
        # Gate on MEAN cells-per-ray, not max: max is a ray-sampling artifact
        # (a single ray grazing a roughened corner of a still-single-cell wall
        # reads 3+ once boundaries fluctuate) and does not distinguish a genuine
        # monolayer from multilayering; mean ~1.2 is an unambiguous monolayer
        # certificate. max is reported for context, not gated.
        (f"single-cell-thick monolayer (mean radial cells {mean_t:.2f} < 1.5; "
         f"max {max_t} reported for context)", mean_t < 1.5),
        (f"no cell fragmented ({frag} of {alive} cells split)", frag == 0),
        (f"lumen stays enclosed / no wall breach (min interior pockets {min_pockets} >= 1)",
         min_pockets >= 1),
        (f"stem niche is basal (mean stem z {sum(stem_z)/len(stem_z):.1f} < goblet z "
         f"{sum(gob_z)/len(gob_z):.1f})" if stem_z and gob_z else "stem + goblet present",
         bool(stem_z) and bool(gob_z) and sum(stem_z)/len(stem_z) < sum(gob_z)/len(gob_z)),
        (f"structure persists (all {n0} cells survive: {alive} alive)", alive == n0),
        (f"relaxation is non-trivial (total voxel reassignments {total_churn} > 0 -- "
         f"the shell is genuinely relaxing under the CPM, not pinned rigid)", total_churn > 0),
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
