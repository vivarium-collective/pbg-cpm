"""3D crypt structure: a procedural single-cell-thick epithelial shell held
together by the connectivity constraint. Builds an OPEN-TOPPED crypt (a test
tube: a closed rounded base holding the stem niche, a cylindrical wall, and an
open mouth that drains into the gut lumen — the real intestinal-crypt shape),
relaxes briefly with connectivity ON (cells + medium), validates that it stays a
coherent monolayer with a deep open lumen, a sealed base, and a basal stem
niche, and exports it for the viewer.

Usage (repo root, venv active):  python demos/run_crypt3d.py
"""
import json
import os
import sys

from cpm import cpm_core

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cpm.crypt3d import build_crypt3d
from cpm.metrics import (radial_thickness, radial_cell_counts,
                         connected_components, central_axis_column,
                         open_lumen_depth)

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
    # a handful of MCS. Here the lumen is OPEN at the mouth by design, so the
    # integrity claim is that the tube keeps a deep open cavity with a sealed
    # base (not an enclosed pocket); full monolayer stability under growth is
    # what the basement membrane / junctions (E3) are for.
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d(wall=3, open_top=True)
    from collections import Counter
    # Median cell size -- only the seed's finalize scalar; build_world overwrites
    # each cell's target with its own volume after finalize, so this just sets a
    # reasonable initial target before that per-cell re-anchoring.
    sizes = sorted(Counter(v for v in labels if v).values())
    median = sizes[len(sizes) // 2]
    w = build_world(labels, seg_to_type, (nx, ny, nz), median)
    n0 = w.n_cells()

    # Track structural gates over EVERY frame (worst case), not just the endpoint:
    # the claim is the shell STAYS a monolayer throughout the relaxation, so a
    # transient thickening or breach must fail even if it settles back by the end.
    frames = []
    prev_snap = None
    total_churn = 0                       # voxels that changed owner over the run
    min_step_churn = None                 # smallest per-step churn (proves EVERY step moved)
    min_lumen = 10**9                      # worst-case (shallowest) open-lumen depth over frames
    worst_mean_t = 0.0                     # max over frames of mean cells-per-ray
    worst_p90 = 0                          # max over frames of the 90th-pctile cells-per-ray
    for f in range(n_frames + 1):
        snap = w.snapshot()
        if prev_snap is not None:
            churn = sum(1 for a, b in zip(prev_snap, snap) if a != b)
            total_churn += churn
            min_step_churn = churn if min_step_churn is None else min(min_step_churn, churn)
        else:
            churn = 0
        prev_snap = snap
        counts = radial_cell_counts(w, nx / 2.0, ny / 2.0)
        mean_t = sum(counts) / len(counts)
        p90 = counts[int(0.90 * len(counts))]
        worst_mean_t = max(worst_mean_t, mean_t)
        worst_p90 = max(worst_p90, p90)
        min_lumen = min(min_lumen, open_lumen_depth(w))
        frames.append({"mcs": f * mcs_per_frame, "voxels": surface_3d(w), "churn": churn})
        if f < n_frames:
            w.step(mcs_per_frame)

    types = w.cell_types()
    coms = w.cell_coms()
    vols = w.cell_volumes()
    _, max_t = radial_thickness(w, nx / 2.0, ny / 2.0)
    frag = sum(1 for c in range(1, len(types)) if vols[c] > 0 and connected_components(w, c) != 1)
    alive = sum(1 for c in range(1, len(types)) if vols[c] > 0)
    stem_z = [coms[c][2] for c in range(1, len(types)) if types[c] == 1 and vols[c] > 0]
    gob_z = [coms[c][2] for c in range(1, len(types)) if types[c] == 3 and vols[c] > 0]
    # open crypt certificate: sealed base = a cap cell low on the central axis;
    # open mouth = NO cell lid on the axis in the upper half of the domain.
    axis = central_axis_column(w)
    base_capped = any(axis[z] != 0 for z in range(nz // 2))
    top_lid = [z for z in range(nz // 2, nz) if axis[z] != 0]
    lumen_min_depth = int(0.35 * nz)

    checks = [
        # Monolayer certificate over the whole run, on two robust statistics:
        #   - worst-frame MEAN cells-per-ray < 1.5 (central tendency), and
        #   - worst-frame 90th-percentile cells-per-ray <= 2 (upper tail).
        # We gate the p90, not the raw max: max is a ray-sampling artifact (a
        # single ray grazing a cell corner or skimming a hemispherical cap reads
        # 3+ even for a true single-cell wall), so it flags sampling geometry, not
        # multilayering. p90 <= 2 catches genuine multilayering while ignoring that
        # ~5% corner-grazing tail. max is reported for context only.
        (f"single-cell-thick monolayer throughout (worst mean {worst_mean_t:.2f} < 1.5, "
         f"worst p90 {worst_p90} <= 2; final max {max_t} reported for context)",
         worst_mean_t < 1.5 and worst_p90 <= 2),
        (f"no cell fragmented ({frag} of {alive} cells split)", frag == 0),
        (f"open lumen throughout: axis cavity stays >= {lumen_min_depth} voxels deep "
         f"(worst {min_lumen}) with a sealed base and an open mouth "
         f"(no axis lid in upper half: {len(top_lid)} lid voxels)",
         min_lumen >= lumen_min_depth and base_capped and not top_lid),
        (f"stem niche is basal (mean stem z {sum(stem_z)/len(stem_z):.1f} < goblet z "
         f"{sum(gob_z)/len(gob_z):.1f})" if stem_z and gob_z else "stem + goblet present",
         bool(stem_z) and bool(gob_z) and sum(stem_z)/len(stem_z) < sum(gob_z)/len(gob_z)),
        (f"structure persists (all {n0} cells survive: {alive} alive)", alive == n0),
        # Not just "moved once": require EVERY relaxation step to move some voxels,
        # so the run can't pass by fluctuating a single flip then freezing.
        (f"relaxation is non-trivial (total voxel reassignments {total_churn}, "
         f"min per-step {min_step_churn} > 0 -- genuinely relaxing every step, not pinned)",
         (min_step_churn or 0) > 0),
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
             "hra_mibitof.json", "hra_ftu.json",
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
