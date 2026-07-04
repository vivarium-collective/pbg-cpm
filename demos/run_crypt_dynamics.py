"""Living 3D colonic crypt: the full epithelial life-cycle as a Cellular Potts
tissue. A bigger open-topped crypt (sized from Human Reference Atlas / literature
colonic-crypt proportions) runs the crypt conveyor:

  * stem cells at the BASE grow and DIVIDE (mitosis),
  * their progeny are pushed UP the crypt wall by proliferative pressure,
  * they DIFFERENTIATE by height (stem -> transit-amplifying -> goblet /
    colonocyte), and
  * they SLOUGH / die (apoptosis + extrusion) at the mouth,

so the population turns over while holding a homeostatic steady state. The tube
is kept a single-cell-thick monolayer by the E1 connectivity constraint plus a
basement membrane anchoring cells to the wall; the E1 medium-connectivity keeps
the lumen open at the mouth.

HRA / literature scaling (colonic crypt of Lieberkuhn, ~5 um/voxel): depth
~400 um, luminal ~100 um / basal ~58 um, ~23 cells around the ring, ~1700-2500
cells; zonation stem(base ~1%) -> transit-amplifying(lower 2/3) -> goblet ~20% /
colonocyte ~14% (upper); turnover ~3.4 days; shedding at the luminal mouth.

Usage (repo root, venv active):  python demos/run_crypt_dynamics.py
"""
import json
import os
import sys
from collections import Counter, deque

from cpm import cpm_core
from cpm.crypt3d import build_crypt3d
from cpm.metrics import connected_components, open_lumen_depth

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
STEM, TA, GOB, COL, DEAD = 1, 2, 3, 4, 5
TYPE_NAMES = ["Medium", "Stem", "Transit-amplifying", "Goblet", "Colonocyte", "Sloughing"]
_NB6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

# geometry + dynamics (tuned so the conveyor is a clean homeostatic steady state)
RADIUS, CYL_H, WALL, PITCH, MARGIN = 9, 54, 2, 5, 4
GROW_RATE, THRESH_MULT, LAM = 0.8, 1.6, 60.0
MCS, N_STEPS, GRACE = 6, 40, 2
CAP_MULT, TA_FRAC, DIFF_FRAC = 1.10, 0.18, 0.45


def basal_surface_anchors(labels, dims):
    """Basement membrane anchor voxels = shell voxels adjacent to medium reachable
    from the lattice border (both faces of the open tube), so anchored cells hug
    the wall and the monolayer can't bulge into the lumen or pile up."""
    nx, ny, nz = dims
    n = nx * ny * nz
    def idx(x, y, z): return x + y * nx + z * nx * ny
    exterior = bytearray(n); q = deque()
    for i in range(n):
        if labels[i] != 0:
            continue
        z, rem = divmod(i, nx * ny); y, x = divmod(rem, nx)
        if (x in (0, nx-1) or y in (0, ny-1) or z in (0, nz-1)) and not exterior[i]:
            exterior[i] = 1; q.append(i)
    while q:
        v = q.popleft(); z, rem = divmod(v, nx * ny); y, x = divmod(rem, nx)
        for dx, dy, dz in _NB6:
            x2, y2, z2 = x+dx, y+dy, z+dz
            if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
                w = idx(x2, y2, z2)
                if labels[w] == 0 and not exterior[w]:
                    exterior[w] = 1; q.append(w)
    anchors = []
    for i in range(n):
        if labels[i] == 0:
            continue
        z, rem = divmod(i, nx * ny); y, x = divmod(rem, nx)
        for dx, dy, dz in _NB6:
            x2, y2, z2 = x+dx, y+dy, z+dz
            if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz and exterior[idx(x2, y2, z2)]:
                anchors.append(i); break
    return anchors


def surface_voxels(w):
    """boundary voxels [x, y, z, cellId, cellType] for the viewer (per-frame type
    so the viewer can shade the differentiation zones)."""
    nx, ny, nz = w.dims()
    lab = w.snapshot()
    types = w.cell_types()
    def owner(x, y, z): return lab[x + y * nx + z * nx * ny]
    out = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                c = owner(x, y, z)
                if c == 0:
                    continue
                for dx, dy, dz in _NB6:
                    x2, y2, z2 = x+dx, y+dy, z+dz
                    if not (0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz) or owner(x2, y2, z2) != c:
                        out.append([x, y, z, c, int(types[c])]); break
    return out


def build():
    (nx, ny, nz), labels, seg_to_type, _ = build_crypt3d(
        radius=RADIUS, cyl_height=CYL_H, wall=WALL, cell_pitch=PITCH,
        margin=MARGIN, open_top=True)
    seg_to_type = {s: STEM for s in seg_to_type}   # retyped by height after finalize
    sizes = sorted(Counter(v for v in labels if v).values())
    V0 = sizes[len(sizes) // 2]
    w = cpm_core.World((nx, ny, nz), "noflux", 2, 4.0)
    w.seed_from_labels(labels, seg_to_type, 1, float(V0), LAM)
    for t in range(1, 6):
        w.set_contact(0, t, 6.0)
        for u in range(t, 6):
            w.set_contact(t, u, 4.0)
        w.set_connectivity(t, True)
    w.set_connectivity_medium(True)
    w.set_membrane(basal_surface_anchors(labels, (nx, ny, nz)), 14.0, float(WALL))
    for t in (STEM, TA, GOB, COL):
        w.set_membrane_anchored(t, True)
    w.finalize(1)
    vols0 = w.cell_volumes()
    for c in range(1, len(vols0)):
        if vols0[c] > 0:
            w.set_target_volume(c, float(vols0[c]))
    return w, (nx, ny, nz), V0


def main():
    w, (nx, ny, nz), V0 = build()
    z_base, z_top = MARGIN + RADIUS, MARGIN + RADIUS + CYL_H
    span = z_top - z_base
    def zfrac(z): return (z - z_base) / span

    thresh = THRESH_MULT * V0
    n0 = w.n_cells()
    target_pop = int(CAP_MULT * n0)
    dying = {}
    births = deaths = 0
    div_z, death_z = [], []
    live_hist, min_lumen = [], 10**9
    frames = [{"mcs": 0, "voxels": surface_voxels(w)}]

    for step in range(N_STEPS):
        w.grow(STEM, GROW_RATE); w.grow(TA, GROW_RATE)
        coms = w.cell_coms()
        for c in w.divide_cells(thresh, V0):
            births += 1
            div_z.append(zfrac(coms[c][2]) if c < len(coms) else 0.0)
        coms = w.cell_coms(); vols = w.cell_volumes()
        for c in range(1, len(vols)):
            if vols[c] <= 0 or c in dying:
                continue
            f = zfrac(coms[c][2])
            if f >= DIFF_FRAC:
                w.set_cell_type(c, GOB if (c * 2654435761) % 100 < 59 else COL)
                w.set_target_volume(c, float(vols[c]))
            elif f >= TA_FRAC:
                w.set_cell_type(c, TA)
            else:
                w.set_cell_type(c, STEM)
        live = sorted(((coms[c][2], c) for c in range(1, len(vols))
                       if vols[c] > 0 and c not in dying), reverse=True)
        i = 0
        while len(live) - i > target_pop and i < len(live):
            z, c = live[i]; i += 1
            w.set_cell_type(c, DEAD); w.set_target_volume(c, 0.0)
            dying[c] = 0; deaths += 1; death_z.append(zfrac(z))
        ready = [c for c, age in dying.items() if age >= GRACE]
        if ready:
            w.remove_cells(ready)
            for c in ready:
                del dying[c]
        for c in dying:
            dying[c] += 1
        w.step(MCS)
        vv = w.cell_volumes()
        live_hist.append(sum(1 for c in range(1, len(vv)) if vv[c] > 0 and c not in dying))
        min_lumen = min(min_lumen, open_lumen_depth(w))
        frames.append({"mcs": (step + 1) * MCS, "voxels": surface_voxels(w)})

    vols = w.cell_volumes(); types = w.cell_types(); coms = w.cell_coms()
    alive = [c for c in range(1, len(vols)) if vols[c] > 0]
    frag = sum(1 for c in alive if connected_components(w, c) != 1)
    def mean_z(*tps):
        zs = [zfrac(coms[c][2]) for c in alive if types[c] in tps]
        return sum(zs) / len(zs) if zs else float("nan")
    z_stem, z_ta, z_diff = mean_z(STEM), mean_z(TA), mean_z(GOB, COL)
    mdiv = sum(div_z) / len(div_z) if div_z else float("nan")
    mdeath = sum(death_z) / len(death_z) if death_z else float("nan")
    warm = live_hist[len(live_hist) // 2:]
    steady = min(warm) >= 0.8 * target_pop and max(warm) <= 1.25 * target_pop

    checks = [
        (f"the crypt renews: {births} births and {deaths} deaths (turnover), "
         f"living population held steady near {target_pop} "
         f"({min(warm)}-{max(warm)} over 2nd half)",
         births > 0.3 * n0 and deaths > 0.3 * n0 and steady),
        (f"conveyor: cells are BORN at the base (mean division height {mdiv:.2f}) and "
         f"SLOUGH at the mouth (mean death height {mdeath:.2f}); rise > 0.35",
         mdeath - mdiv > 0.35),
        (f"differentiation zonation by height: stem {z_stem:.2f} < transit-amplifying "
         f"{z_ta:.2f} < differentiated {z_diff:.2f}",
         z_stem < z_ta < z_diff and z_stem < 0.25 and z_diff > 0.4),
        (f"single-cell-thick monolayer stays intact: no cell fragments "
         f"({frag} of {len(alive)})", frag == 0),
        (f"lumen stays open throughout (min axial depth {min_lumen} >= {int(0.3*nz)})",
         min_lumen >= 0.3 * nz),
    ]

    data = {"name": "Living Crypt (life-cycle)", "kind": "cryptlife",
            "dims": [nx, ny, nz], "is3d": True, "n_cells": w.n_cells(),
            "cell_types": list(w.cell_types()), "type_names": TYPE_NAMES,
            "frames": frames}
    os.makedirs(DATA, exist_ok=True)
    json.dump(data, open(os.path.join(DATA, "crypt_dynamics.json"), "w"))
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    manifest = [m for m in manifest if m["file"] != "crypt_dynamics.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "crypt_dynamics.json", "name": data["name"], "is3d": True,
                     "n_cells": data["n_cells"], "dims": data["dims"], "kind": "cryptlife",
                     "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "connectivity_2d.json", "connectivity_3d.json", "connectivity_gap.json",
             "crypt3d.json", "crypt_dynamics.json", "crypt_differentiation.json",
             "hra_mibitof.json", "hra_ftu.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (living crypt life-cycle) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
