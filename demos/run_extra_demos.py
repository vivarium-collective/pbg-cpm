"""Additional / heavier demos, merged into the existing viewer manifest:
  - cellsort_3d  : the 3D sort run LONGER (fuller sorting)
  - spheroid_3d  : advanced 3D — a proliferating tumor spheroid (growth + mitosis)
  - scale_2d     : how many cells can we run? a large packed 2D sort, timed

Usage (repo root, venv active):  python demos/run_extra_demos.py
Merges into viewer/data/index.json (keeps the CC3D + HRA demos).
"""
import json
import os
import time

from cpm import cpm_core
from run_cc3d_demos import run_cellsort, labels_2d, surface_3d, heterotypic_boundary

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))


def bbox_aspect_3d(world):
    nx, ny, nz = world.dims()
    lab = world.snapshot()
    mn = [nx, ny, nz]; mx = [-1, -1, -1]
    for z in range(nz):
        for y in range(ny):
            base = y * nx + z * nx * ny
            row = lab[base:base + nx]
            for x, c in enumerate(row):
                if c:
                    if x < mn[0]: mn[0] = x
                    if x > mx[0]: mx[0] = x
                    if y < mn[1]: mn[1] = y
                    if y > mx[1]: mx[1] = y
                    if z < mn[2]: mn[2] = z
                    if z > mx[2]: mx[2] = z
    ext = [mx[i] - mn[i] + 1 for i in range(3)]
    return max(ext) / max(1, min(ext)), ext


# ---------------- advanced 3D: proliferating tumor spheroid ----------------
def run_spheroid_3d(dims=(60, 60, 60), n_frames=30, mcs_per_frame=18, seed=5):
    nx, ny, nz = dims
    w = cpm_core.World(dims, "noflux", 2, 10.0)
    TUMOR = 1
    # seed a 2x2x2 cluster of 5^3 blocks at the center; healthy volume regime
    # (cells settle near ~48 vox), then grow and divide below that so waves fire
    c0 = (nx // 2 - 5, ny // 2 - 5, nz // 2 - 5)
    for dz in (0, 5):
        for dy in (0, 5):
            for dx in (0, 5):
                cid = w.add_cell(TUMOR, 30.0, 3.5, 0.0, 0.0)
                x0, y0, z0 = c0[0] + dx, c0[1] + dy, c0[2] + dz
                w.seed_block(cid, x0, y0, z0, x0 + 4, y0 + 4, z0 + 4)
    w.set_contact(0, TUMOR, 14.0)   # cohesive -> stays a compact spheroid
    w.set_contact(TUMOR, TUMOR, 3.0)
    w.finalize(seed)

    frames, counts = [], []
    for f in range(n_frames + 1):
        frames.append({"mcs": f * mcs_per_frame, "voxels": surface_3d(w), "n_cells": w.n_cells()})
        counts.append(w.n_cells())
        if f < n_frames:
            w.step(mcs_per_frame)
            w.grow(TUMOR, 3.5)              # push target volume up each frame
            w.divide_cells(46.0, 28.0)      # divide below the ~48 cells reach; daughters ~28
        print(f"    spheroid_3d: {f}/{n_frames} cells={w.n_cells()}", flush=True)
    aspect, ext = bbox_aspect_3d(w)
    max_vol = max(w.cell_volumes()[1:])
    checks = [
        (f"3D colony proliferates ({counts[0]} → {counts[-1]} cells)", counts[-1] >= 4 * counts[0]),
        (f"stays a compact spheroid (bbox aspect {aspect:.2f} < 1.6, extent {ext})", aspect < 1.6),
        (f"cell volumes bounded (max {max_vol} < 110)", max_vol < 110),
    ]
    data = {"name": "3D Tumor Spheroid (growth)", "kind": "growth", "dims": list(dims),
            "is3d": True, "n_cells": w.n_cells(), "cell_types": list(w.cell_types()),
            "frames": frames}
    return data, checks


# ---------------- scale: how many cells can we run? ----------------
def run_scale_2d(dim=500, cell_w=5, mcs_capture=200, n_frames=6, seed=1):
    nx = ny = dim
    w = cpm_core.World((nx, ny, 1), "periodic", 2, 10.0)
    ng = dim // cell_w
    import random
    rng = random.Random(seed)
    for iy in range(ng):
        for ix in range(ng):
            t = 1 if rng.random() < 0.5 else 2
            cid = w.add_cell(t, float(cell_w * cell_w), 2.0, 0.0, 0.0)
            w.seed_block(cid, ix * cell_w, iy * cell_w, 0, ix * cell_w + cell_w, iy * cell_w + cell_w, 1)
    w.set_contact(0, 1, 16.0); w.set_contact(0, 2, 16.0)
    w.set_contact(1, 1, 2.0); w.set_contact(2, 2, 16.0); w.set_contact(1, 2, 11.0)
    w.finalize(seed)
    n_cells = w.n_cells()

    # time the throughput
    t0 = time.time()
    w.step(20)
    rate = 20 / (time.time() - t0)  # MCS/s at this scale
    attempts_per_s = rate * nx * ny

    frames = []
    per = mcs_capture // n_frames
    for f in range(n_frames + 1):
        frames.append({"mcs": f * per, "labels": labels_2d(w)})
        if f < n_frames:
            w.step(per)
        print(f"    scale_2d: {f}/{n_frames}", flush=True)
    checks = [
        (f"ran {n_cells} cells on a {nx}×{ny} lattice", n_cells >= 8000),
        (f"throughput {rate:.1f} MCS/s ({attempts_per_s/1e6:.1f}M copy-attempts/s), single-thread",
         rate > 0),
    ]
    data = {"name": f"Scale Test — {n_cells:,} cells", "kind": "scale", "dims": [nx, ny, 1],
            "is3d": False, "n_cells": n_cells, "cell_types": list(w.cell_types()),
            "throughput_mcs_s": round(rate, 1),
            "attempts_per_s": round(attempts_per_s / 1e6, 1), "frames": frames}
    return data, checks


def emit(slug, data, checks, manifest, results):
    with open(os.path.join(DATA, slug + ".json"), "w") as fh:
        json.dump(data, fh)
    ok = all(p for _, p in checks)
    results.append((data["name"], checks, ok))
    entry = {"file": slug + ".json", "name": data["name"], "is3d": data["is3d"],
             "n_cells": data["n_cells"], "dims": data["dims"], "kind": data.get("kind"),
             "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]}
    manifest[:] = [m for m in manifest if m["file"] != slug + ".json"] + [entry]
    print(f"  {'✓' if ok else '✗'} {data['name']}")


def main():
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    results = []
    print("[1/3] cellsort_3D (longer)")
    emit("cellsort_3d", *run_cellsort("cellsort_3d", "3D Cell Sorting", (50, 50, 50), 5, 10,
                                      4200, 22, lam_vol=6.0), manifest, results)
    print("[2/3] spheroid_3D"); emit("spheroid_3d", *run_spheroid_3d(), manifest, results)
    print("[3/3] scale_2D"); emit("scale_2d", *run_scale_2d(), manifest, results)

    # keep a sensible display order
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    with open(idx, "w") as fh:
        json.dump({"models": manifest}, fh, indent=2)

    print("\n=============== VALIDATION ===============")
    allok = True
    for name, checks, ok in results:
        print(f"\n{name}: {'PASS' if ok else 'FAIL'}")
        for t, p in checks:
            print(f"   [{'PASS' if p else 'FAIL'}] {t}")
        allok = allok and ok
    print("\nALL EXTRA DEMOS VALIDATED" if allok else "\nSOME FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
