"""Convert an HRA 2D Functional Tissue Unit (FTU) illustration into a CPM.

The Human Reference Atlas publishes canonical 2D FTU illustrations in which
every cell is drawn as a polygon and grouped, in the SVG's ``Crosswalk``
layer, under its Cell-Ontology cell type. This is the *idealized reference*
counterpart to the raw MIBI-TOF imaging demo: instead of a segmentation mask
from one sample, we convert the atlas's prototypical tissue unit.

We take the colonic-crypt FTU (large intestine), rasterize each crosswalk
polygon into a labelled lattice with its cell type, seed a Cellular Potts
world with EXACT placement via ``seed_from_labels``, relax it, and VALIDATE
that the cell-type composition and cell positions are preserved.

Source (public, no auth): hubmapconsortium/ccf-2d-reference-object-library.

Usage (repo root, venv active):  python demos/run_hra_ftu.py
"""
import json
import math
import os
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
from matplotlib.path import Path as MplPath

import cpm_core

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "viewer", "data"))
SVG_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "ftu"))
SVG_URL = ("https://raw.githubusercontent.com/hubmapconsortium/"
           "ccf-2d-reference-object-library/main/v1.1/"
           "colonic_crypts_large_intestine.svg")
SVG_FILE = os.path.join(SVG_DIR, "colonic_crypts.svg")
TARGET_MAXDIM = 500     # longest lattice axis after scaling the FTU
MARGIN = 6              # lattice cells of medium padding around the tissue


def _tag(e):
    return e.tag.split("}")[-1]


def _poly_points(el):
    s = el.get("points") or ""
    nums = [float(x) for x in s.replace(",", " ").split()]
    return list(zip(nums[0::2], nums[1::2]))


def _clean_type(svg_id):
    # 'Absorptive_x5F_Cells' -> 'Absorptive Cells'
    return svg_id.replace("_x5F_", "_").replace("_", " ").strip()


def load_ftu_cells():
    """Return (cells, type_names): cells = list of (type_idx, polygon_pts)."""
    if not os.path.exists(SVG_FILE):
        os.makedirs(SVG_DIR, exist_ok=True)
        urllib.request.urlretrieve(SVG_URL, SVG_FILE)
    root = ET.parse(SVG_FILE).getroot()
    crosswalk = [g for g in root if _tag(g) == "g" and g.get("id") == "Crosswalk"][0]

    type_names = []
    cells = []
    for ct in crosswalk:
        name = _clean_type(ct.get("id", ""))
        if not name:
            continue
        type_names.append(name)
        tidx = len(type_names)          # CPM type ids are 1..K
        for el in ct.iter():            # each filled polygon is one cell
            if _tag(el) == "polygon":
                pts = _poly_points(el)
                if len(pts) >= 3:
                    cells.append((tidx, pts))
    return cells, type_names


def rasterize(cells):
    """Rasterize cell polygons to a labelled lattice.

    Returns (nx, ny), labels (row-major), seg_to_type, median_size.
    Each cell gets a unique consecutive segment id; overlapping pixels go to
    the first cell that claims them (illustrations barely overlap).
    """
    allpts = np.array([p for _, poly in cells for p in poly])
    x0, y0 = allpts[:, 0].min(), allpts[:, 1].min()
    x1, y1 = allpts[:, 0].max(), allpts[:, 1].max()
    scale = TARGET_MAXDIM / max(x1 - x0, y1 - y0)
    nx = int(math.ceil((x1 - x0) * scale)) + 2 * MARGIN
    ny = int(math.ceil((y1 - y0) * scale)) + 2 * MARGIN

    labels = np.zeros(ny * nx, dtype=np.int64)
    seg_to_type = {}
    seg = 0
    for tidx, poly in cells:
        pa = np.array(poly, dtype=float)
        px = (pa[:, 0] - x0) * scale + MARGIN
        py = (pa[:, 1] - y0) * scale + MARGIN
        # test pixel centres inside the polygon's bounding box
        cxmin, cxmax = int(px.min()), int(math.ceil(px.max()))
        cymin, cymax = int(py.min()), int(math.ceil(py.max()))
        xs = np.arange(cxmin, cxmax + 1)
        ys = np.arange(cymin, cymax + 1)
        gx, gy = np.meshgrid(xs + 0.5, ys + 0.5)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        mask = MplPath(np.column_stack([px, py])).contains_points(pts)
        if not mask.any():
            continue
        seg += 1
        seg_to_type[seg] = tidx
        inside = pts[mask]
        for (fx, fy) in inside:
            ix, iy = int(fx - 0.5), int(fy - 0.5)
            if 0 <= ix < nx and 0 <= iy < ny:
                idx = iy * nx + ix
                if labels[idx] == 0:
                    labels[idx] = seg
    sizes = Counter(int(v) for v in labels if v)
    # drop cells that rasterized to nothing (kept ids stay consecutive enough)
    seg_to_type = {s: t for s, t in seg_to_type.items() if sizes.get(s, 0) >= 3}
    median_size = int(np.median([c for c in sizes.values()])) if sizes else 12
    return (nx, ny), labels.tolist(), seg_to_type, median_size


def validate_and_run(mcs_total=60, n_frames=15):
    cells, type_names = load_ftu_cells()
    (nx, ny), labels, seg_to_type, median_size = rasterize(cells)
    n_seed = len(seg_to_type)
    src_comp = Counter(seg_to_type.values())
    print(f"  colonic-crypt FTU: {nx}x{ny} lattice, {n_seed} cells, "
          f"{len(type_names)} types, median cell size {median_size}px")

    w = cpm_core.World((nx, ny, 1), "noflux", 2, 8.0)
    seg_to_cell = w.seed_from_labels(labels, seg_to_type, 1, float(median_size), 2.0)
    # gentle structure-preserving adhesion: medium modestly costly, uniform cell-cell
    K = len(type_names)
    for t in range(1, K + 1):
        w.set_contact(0, t, 14.0)
        for u in range(t, K + 1):
            w.set_contact(t, u, 6.0)
    w.finalize(1)

    coms0 = {cid: w.cell_coms()[cid] for cid in seg_to_cell.values()}
    frames = []
    per = mcs_total // n_frames
    for f in range(n_frames + 1):
        frames.append({"mcs": f * per, "labels": list(w.snapshot())})
        if f < n_frames:
            w.step(per)
        print(f"    hra_ftu: {f}/{n_frames}", flush=True)

    cur_types, cur_vols = w.cell_types(), w.cell_volumes()
    alive = [cid for cid in seg_to_cell.values() if cur_vols[cid] > 0]
    cur_comp = Counter(cur_types[cid] for cid in alive)

    def frac(comp):
        tot = sum(comp.values()) or 1
        return {t: comp[t] / tot for t in comp}
    f_src, f_cur = frac(src_comp), frac(cur_comp)
    max_drift = max(abs(f_src.get(t, 0) - f_cur.get(t, 0)) for t in set(f_src) | set(f_cur))
    coms1 = w.cell_coms()
    drifts = [math.hypot(coms1[c][0] - coms0[c][0], coms1[c][1] - coms0[c][1]) for c in alive]
    mean_drift = sum(drifts) / len(drifts) if drifts else 0.0

    checks = [
        (f"converted {n_seed} cells with exact placement from the HRA FTU illustration "
         f"({len(type_names)} atlas cell types)", n_seed > 50 and w.n_cells() == n_seed),
        (f"cell-type composition preserved (max fraction drift {max_drift:.3f} < 0.05)",
         max_drift < 0.05),
        (f"cells stay near their illustrated positions (mean COM drift {mean_drift:.1f}px < 6)",
         mean_drift < 6.0),
    ]
    data = {"name": "HRA FTU — Colonic Crypt (converted)", "kind": "ftu",
            "dims": [nx, ny, 1], "is3d": False, "n_cells": w.n_cells(),
            "cell_types": list(w.cell_types()),
            "type_names": ["Medium"] + type_names,
            "type_counts": {type_names[t - 1]: src_comp[t] for t in sorted(src_comp)},
            "frames": frames}
    return data, checks


def main():
    os.makedirs(DATA, exist_ok=True)
    data, checks = validate_and_run()
    with open(os.path.join(DATA, "hra_ftu.json"), "w") as fh:
        json.dump(data, fh)
    idx_path = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx_path))["models"] if os.path.exists(idx_path) else []
    manifest = [m for m in manifest if m["file"] != "hra_ftu.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "hra_ftu.json", "name": data["name"], "is3d": False,
                     "n_cells": data["n_cells"], "dims": data["dims"], "kind": "ftu",
                     "validated": ok,
                     "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json", "hra_ftu.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    with open(idx_path, "w") as fh:
        json.dump({"models": manifest}, fh, indent=2)

    print("\n=========== VALIDATION (HRA FTU conversion) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
