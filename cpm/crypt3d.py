"""Procedural 3D crypt geometry: a single-cell-thick epithelial shell tiled
into cells and typed by axial position (stem niche basal). Pure/deterministic.

Two shapes, selected by ``open_top``:

  * ``open_top=False`` (default) — a closed capsule (hollow cylinder capped by
    a hemisphere at BOTH ends). The lumen is a fully enclosed interior medium
    pocket, which the ``interior_medium_pockets`` integrity gate relies on.
    Used by the junction (E3b) and membrane (E3a) demos, and the geometry test.

  * ``open_top=True`` — a biologically faithful OPEN-TOPPED crypt (a test tube):
    a hemispherical closed BASE holding the stem niche, a cylindrical wall, and
    an OPEN mouth at the top that drains into the gut lumen. There is no top cap,
    so the lumen is an open cavity connected to the exterior through the mouth
    (validated by lumen depth + a sealed base, not by an enclosed pocket).

Both keep the same typing (basal stem niche -> absorptive -> goblet toward the
mouth), pitch, and wall-thickness handling.
"""
import math
from collections import Counter

TYPE_NAMES = ["Epithelial Stem", "Absorptive", "Goblet"]
STEM, ABS, GOB = 1, 2, 3


def _neighbors18(cx, cy, cz, nx, ny, nz):
    # Same adjacency as the CPM connectivity constraint / cpm.metrics
    # (18-neighbourhood: Manhattan offset <= 2, excluding the 8 cube corners).
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if abs(dx) + abs(dy) + abs(dz) == 3:
                    continue
                x2, y2, z2 = cx + dx, cy + dy, cz + dz
                if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
                    yield x2 + y2 * nx + z2 * nx * ny


def _split_disconnected(labels, seg_to_type, nx, ny, nz):
    """The (axial_bin, theta_bin) binning key can, near the cap/cylinder
    seam and the poles, be revisited by two spatially disjoint voxel blobs
    (float rounding of the arc-length parameter `a` folds a cylinder-side
    ring and a cap-side ring into the same bin). That silently merges two
    unconnected regions under one cell id -- a latent fragmented cell that
    breaks the CPM connectivity constraint's invariant before the CPM ever
    runs. Split every label into its 18-connected components (matching the
    constraint's own adjacency) so every returned cell id is one contiguous
    blob; extra components become new cell ids of the same type."""
    by_label = {}
    for i, v in enumerate(labels):
        if v:
            by_label.setdefault(v, []).append(i)

    next_id = (max(seg_to_type) if seg_to_type else 0) + 1
    for seg, sites in by_label.items():
        remaining = set(sites)
        first = True
        while remaining:
            start = next(iter(remaining))
            comp, stack = [], [start]
            remaining.discard(start)
            while stack:
                c = stack.pop()
                comp.append(c)
                cz, rem = divmod(c, nx * ny)
                cy, cx = divmod(rem, nx)
                for n in _neighbors18(cx, cy, cz, nx, ny, nz):
                    if n in remaining:
                        remaining.discard(n)
                        stack.append(n)
            if first:
                first = False
                continue  # keep the first component under the original id
            new_id = next_id
            next_id += 1
            seg_to_type[new_id] = seg_to_type[seg]
            for i in comp:
                labels[i] = new_id
    return labels, seg_to_type


def build_crypt3d(radius=8, cyl_height=28, wall=2, cell_pitch=6, margin=4, open_top=False):
    R = float(radius)
    nx = ny = 2 * (radius + margin)
    # a closed capsule needs room for a top cap (2*radius of z); an open crypt
    # only needs a margin of medium above the rim for the mouth.
    nz = (radius if open_top else 2 * radius) + cyl_height + 2 * margin
    cx = cy = nx / 2.0
    z_base = margin + radius              # bottom cap pole at z=margin, equator at z=z_base
    z_top = z_base + cyl_height            # cylinder top; closed cap equator OR open rim
    half = wall / 2.0
    a_cap = R * (math.pi / 2.0)           # profile arc length of one cap
    a_max = a_cap + cyl_height + (0.0 if open_top else a_cap)  # + top cap only when closed

    labels = [0] * (nx * ny * nz)
    cellmap, seg_to_type = {}, {}
    next_seg = 1

    for z in range(nz):
        zc = z + 0.5
        for y in range(ny):
            dy = y + 0.5 - cy
            for x in range(nx):
                dx = x + 0.5 - cx
                r = math.hypot(dx, dy)
                if zc >= z_top:                      # above the cylinder
                    if open_top:
                        continue                    # OPEN mouth: medium drains to the gut lumen
                    d = math.sqrt(dx * dx + dy * dy + (zc - z_top) ** 2)  # top hemispherical cap
                    if abs(d - R) > half:
                        continue
                    cphi = max(-1.0, min(1.0, (zc - z_top) / R))
                    phi = math.acos(cphi)           # 0 at equator -> pi/2 at pole
                    a = (a_cap + cyl_height) + R * (math.pi / 2.0 - phi)
                    r_local = max(1.0, R * math.sin(phi))
                elif zc >= z_base:                   # cylinder
                    if abs(r - R) > half:
                        continue
                    a = a_cap + (zc - z_base)
                    r_local = R
                else:                               # bottom hemispherical cap
                    d = math.sqrt(dx * dx + dy * dy + (zc - z_base) ** 2)
                    if abs(d - R) > half:
                        continue
                    cphi = max(-1.0, min(1.0, (z_base - zc) / R))
                    phi = math.acos(cphi)           # 0 at pole -> pi/2 at equator
                    a = R * phi                     # 0 at pole -> a_cap at equator (joins cylinder)
                    r_local = max(1.0, R * math.sin(phi))
                if a < 0 or a > a_max:
                    continue
                axial_bin = int(a // cell_pitch)
                n_theta = max(1, round(2 * math.pi * r_local / cell_pitch))
                theta = math.atan2(dy, dx) + math.pi
                theta_bin = int(theta // (2 * math.pi / n_theta)) % n_theta
                key = (axial_bin, theta_bin)
                seg = cellmap.get(key)
                if seg is None:
                    seg = next_seg
                    next_seg += 1
                    cellmap[key] = seg
                    # Axial typing: basal band -> stem niche, middle ->
                    # absorptive, upper -> goblet. The top cap (a beyond
                    # a_cap + cyl_height) exceeds the highest cutoff, so it
                    # falls through to GOB by construction — keep the cutoffs
                    # below a_cap + cyl_height if you retune, or the apical
                    # cap's type will change with them.
                    if a <= a_cap + cyl_height * 0.25:
                        seg_to_type[seg] = STEM
                    elif a <= a_cap + cyl_height * 0.60:
                        seg_to_type[seg] = ABS
                    else:
                        seg_to_type[seg] = GOB
                labels[x + y * nx + z * nx * ny] = seg

    # split any label whose voxels form >1 connected component (bin-key
    # collisions near the pole/cap-cylinder seam), then drop tiny cells and
    # re-index consecutively from 1
    labels, seg_to_type = _split_disconnected(labels, seg_to_type, nx, ny, nz)
    sizes = Counter(v for v in labels if v)
    keep = sorted(s for s, c in sizes.items() if c >= 3)
    remap = {s: i + 1 for i, s in enumerate(keep)}
    labels = [remap.get(v, 0) for v in labels]
    seg_to_type = {remap[s]: seg_to_type[s] for s in keep}
    return (nx, ny, nz), labels, seg_to_type, TYPE_NAMES
