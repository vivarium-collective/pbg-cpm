"""Initialize a CPM simulation from REAL multiplexed-imaging tissue.

Pulls a MIBI-TOF human colon-carcinoma field of view (squidpy's `mibitof`,
scverse example data — the same modality as HuBMAP CODEX), takes its cell
SEGMENTATION MASK and per-cell NAMED cell types, and seeds a Cellular Potts
world with EXACT cell placement via `seed_from_labels`. Runs a short
structure-preserving relaxation, VALIDATES that placement + cell-type
composition are preserved, and exports the time-series for the viewer.

Usage (repo root, venv active):  python demos/run_hra_mibitof.py
"""
import json
import math
import os
from collections import Counter

import numpy as np
import squidpy as sq

import cpm_core

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
DOWN = 4          # downsample factor for the 1024^2 mask -> 256^2 lattice
MIN_PIX = 4       # drop cells smaller than this after downsampling


def load_fov(fov):
    ad = sq.datasets.mibitof()
    mask = np.asarray(ad.uns["spatial"][fov]["images"]["segmentation"])  # 1024x1024 int labels
    obs = ad.obs[ad.obs["library_id"] == fov]
    cellid_to_type = dict(zip(obs["cell_id"].astype(int), obs["Cluster"].astype(str)))
    return mask, cellid_to_type


def build_lattice_labels(mask, cellid_to_type):
    # nearest-neighbour decimation preserves label identity (no interpolation)
    small = mask[::DOWN, ::DOWN]
    ny, nx = small.shape
    # count pixels per raw label, keep those >= MIN_PIX; re-index to consecutive segment ids
    counts = Counter(int(v) for v in small.ravel() if v != 0)
    keep = {lab for lab, c in counts.items() if c >= MIN_PIX and lab in cellid_to_type}
    # stable type-name -> CPM type index (1..K)
    type_names = sorted({cellid_to_type[l] for l in keep})
    type_idx = {name: i + 1 for i, name in enumerate(type_names)}
    seg_of_raw = {}          # raw label -> new consecutive seg id
    seg_to_type = {}         # new seg id -> CPM type index
    next_seg = 1
    labels = [0] * (nx * ny)
    for j in range(ny):
        for i in range(nx):
            raw = int(small[j, i])
            if raw == 0 or raw not in keep:
                continue
            if raw not in seg_of_raw:
                seg_of_raw[raw] = next_seg
                seg_to_type[next_seg] = type_idx[cellid_to_type[raw]]
                next_seg += 1
            labels[j * nx + i] = seg_of_raw[raw]
    sizes = Counter(l for l in labels if l)
    median_size = int(np.median(list(sizes.values()))) if sizes else 20
    return (nx, ny), labels, seg_to_type, type_names, type_idx, median_size


def validate_and_run(fov="point16", mcs_total=120, n_frames=15):
    mask, cellid_to_type = load_fov(fov)
    (nx, ny), labels, seg_to_type, type_names, type_idx, median_size = \
        build_lattice_labels(mask, cellid_to_type)
    n_seed = len(seg_to_type)
    src_comp = Counter(seg_to_type.values())
    print(f"  FOV {fov}: {nx}x{ny} lattice, {n_seed} cells, {len(type_names)} types, "
          f"median cell size {median_size}px")

    w = cpm_core.World((nx, ny, 1), "noflux", 2, 8.0)
    # seed EXACTLY from the real segmentation; target = median cell size so cells keep scale
    seg_to_cell = w.seed_from_labels(labels, seg_to_type, 1, float(median_size), 2.0)
    # structure-preserving adhesion: medium costly (cells stay packed), uniform cell-cell
    n_types = len(type_names)
    for t in range(1, n_types + 1):
        w.set_contact(0, t, 16.0)
        for u in range(t, n_types + 1):
            w.set_contact(t, u, 6.0)
    w.finalize(1)

    # record initial COMs to measure drift (structure preservation)
    coms0 = {cid: w.cell_coms()[cid] for cid in seg_to_cell.values()}

    frames = []
    per = mcs_total // n_frames
    for f in range(n_frames + 1):
        frames.append({"mcs": f * per, "labels": list(w.snapshot())})
        if f < n_frames:
            w.step(per)
        print(f"    hra_mibitof: {f}/{n_frames}", flush=True)

    # ---- validation vs the source imaging ----
    cur_types = w.cell_types()
    cur_vols = w.cell_volumes()
    alive = [cid for cid in seg_to_cell.values() if cur_vols[cid] > 0]
    cur_comp = Counter(cur_types[cid] for cid in alive)

    def frac(comp):
        tot = sum(comp.values())
        return {t: comp[t] / tot for t in comp}
    f_src, f_cur = frac(src_comp), frac(cur_comp)
    max_drift = max(abs(f_src.get(t, 0) - f_cur.get(t, 0)) for t in set(f_src) | set(f_cur))
    coms1 = w.cell_coms()
    drifts = [math.hypot(coms1[c][0] - coms0[c][0], coms1[c][1] - coms0[c][1])
              for c in alive]
    mean_drift = sum(drifts) / len(drifts) if drifts else 0.0

    checks = [
        (f"seeded {n_seed} cells with exact placement from the real segmentation "
         f"({len(type_names)} cell types)", n_seed > 100 and w.n_cells() == n_seed),
        (f"cell-type composition preserved (max fraction drift {max_drift:.3f} < 0.03)",
         max_drift < 0.03),
        (f"cells stay near their imaged positions (mean COM drift {mean_drift:.1f}px < 4)",
         mean_drift < 4.0),
    ]

    data = {"name": "HRA / MIBI-TOF Colon (from imaging)", "kind": "imaging",
            "dims": [nx, ny, 1], "is3d": False, "n_cells": w.n_cells(),
            "cell_types": list(w.cell_types()),
            "type_names": ["Medium"] + type_names,   # index-aligned to CPM type ids
            "type_counts": {type_names[t-1]: src_comp[t] for t in sorted(src_comp)},
            "frames": frames}
    return data, checks


def main():
    os.makedirs(DATA, exist_ok=True)
    data, checks = validate_and_run()
    with open(os.path.join(DATA, "hra_mibitof.json"), "w") as fh:
        json.dump(data, fh)
    # merge into the manifest (keep existing CC3D demos)
    idx_path = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx_path))["models"] if os.path.exists(idx_path) else []
    manifest = [m for m in manifest if m["file"] != "hra_mibitof.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "hra_mibitof.json", "name": data["name"], "is3d": False,
                     "n_cells": data["n_cells"], "dims": data["dims"], "kind": "imaging",
                     "validated": ok,
                     "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    with open(idx_path, "w") as fh:
        json.dump({"models": manifest}, fh, indent=2)

    print("\n=========== VALIDATION (init from real MIBI-TOF imaging) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
