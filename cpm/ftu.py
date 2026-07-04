"""Reusable HRA 2D Functional Tissue Unit (FTU) rasterizer.

The Human Reference Atlas publishes canonical 2D FTU illustrations in which
every cell is drawn as a polygon and grouped, in the SVG's ``Crosswalk``
layer, under its Cell-Ontology cell type. This module downloads the
colonic-crypt FTU SVG and rasterizes each crosswalk polygon into a labelled
lattice tagged with its cell type, ready to seed a Cellular Potts world via
``World.seed_from_labels``.

Source (public, no auth): hubmapconsortium/ccf-2d-reference-object-library.
"""
import math
import os
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
from matplotlib.path import Path as MplPath

HERE = os.path.dirname(__file__)
SVG_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "ftu"))
SVG_URL = ("https://raw.githubusercontent.com/hubmapconsortium/"
           "ccf-2d-reference-object-library/main/v1.1/"
           "colonic_crypts_large_intestine.svg")
SVG_FILE = os.path.join(SVG_DIR, "colonic_crypts.svg")


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


def rasterize(cells, target_maxdim=500, margin=6):
    """Rasterize cell polygons to a labelled lattice.

    Returns (nx, ny), labels (row-major), seg_to_type, median_size.
    Each cell gets a unique consecutive segment id; overlapping pixels go to
    the first cell that claims them (illustrations barely overlap).
    """
    allpts = np.array([p for _, poly in cells for p in poly])
    x0, y0 = allpts[:, 0].min(), allpts[:, 1].min()
    x1, y1 = allpts[:, 0].max(), allpts[:, 1].max()
    scale = target_maxdim / max(x1 - x0, y1 - y0)
    nx = int(math.ceil((x1 - x0) * scale)) + 2 * margin
    ny = int(math.ceil((y1 - y0) * scale)) + 2 * margin

    labels = np.zeros(ny * nx, dtype=np.int64)
    seg_to_type = {}
    seg = 0
    for tidx, poly in cells:
        pa = np.array(poly, dtype=float)
        px = (pa[:, 0] - x0) * scale + margin
        py = (pa[:, 1] - y0) * scale + margin
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


def load_crypt_labels(target_maxdim=500, margin=6):
    """Return ((nx, ny), labels, seg_to_type, type_names, median_size) for the
    HRA colonic-crypt FTU. Downloads the SVG on first use."""
    cells, type_names = load_ftu_cells()
    (nx, ny), labels, seg_to_type, median = rasterize(cells, target_maxdim, margin)
    return (nx, ny), labels, seg_to_type, type_names, median
