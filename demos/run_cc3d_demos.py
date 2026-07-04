"""Reimplement canonical CompuCell3D demos with CC3D's actual parameters, run
them on the pbg-cpm engine, VALIDATE the output against CC3D's expected
behavior, and export time-series for the viewer.

Demos (see docs/cc3d-reference/demo-parameters.md):
  1. cellsort_2D          differential-adhesion cell sorting (exact CC3D params)
  2. cellsort_3D          the same in 3D
  3. bacterium_macrophage macrophage chemotaxes up a secreted attractant
  4. growth_mitosis       proliferating colony (grow + divide)

Usage (repo root, venv active):  python demos/run_cc3d_demos.py
Writes viewer/data/<slug>.json + viewer/data/index.json, and prints PASS/FAIL
for each demo's expected-behavior validation.
"""
import json
import math
import os
import random

import cpm_core

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "viewer", "data"))
os.makedirs(DATA, exist_ok=True)

# CC3D cellsort types
MEDIUM, CONDENSING, NONCONDENSING = 0, 1, 2


# ---------- capture helpers ----------
def labels_2d(world):
    return list(world.snapshot())


def field_2d(world, fi):
    return [round(v, 3) for v in world.field_conc(fi)]


def surface_3d(world):
    nx, ny, nz = world.dims()
    lab = world.snapshot()

    def I(x, y, z):
        return x + y * nx + z * nx * ny

    out = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                c = lab[I(x, y, z)]
                if c == 0:
                    continue
                if (x == 0 or lab[I(x-1,y,z)] != c or x == nx-1 or lab[I(x+1,y,z)] != c or
                        y == 0 or lab[I(x,y-1,z)] != c or y == ny-1 or lab[I(x,y+1,z)] != c or
                        z == 0 or lab[I(x,y,z-1)] != c or z == nz-1 or lab[I(x,y,z+1)] != c):
                    out.append([x, y, z, c])
    return out


def heterotypic_boundary(world):
    """von Neumann faces between two non-medium cells of DIFFERENT types (each face once)."""
    nx, ny, nz = world.dims()
    lab = world.snapshot()
    typ = world.cell_types()

    def I(x, y, z):
        return x + y * nx + z * nx * ny

    n = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                a = lab[I(x, y, z)]
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    if dz and nz == 1:
                        continue
                    b = lab[I((x+dx) % nx, (y+dy) % ny, (z+dz) % nz)]
                    if a and b and a != b and typ[a] != typ[b]:
                        n += 1
    return n


def medium_contact_by_type(world):
    """Total cell-medium face contacts per cell type (how exposed each type is)."""
    nx, ny, nz = world.dims()
    lab = world.snapshot()
    typ = world.cell_types()

    def I(x, y, z):
        return x + y * nx + z * nx * ny

    contact = {}
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                c = lab[I(x, y, z)]
                if c == 0:
                    continue
                m = 0
                for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    if dz and nz == 1:
                        continue
                    if lab[I((x+dx) % nx, (y+dy) % ny, (z+dz) % nz)] == 0:
                        m += 1
                contact[typ[c]] = contact.get(typ[c], 0) + m
    return contact


def cell_count_by_type(world):
    typ = world.cell_types()
    out = {}
    for t in typ[1:]:
        out[t] = out.get(t, 0) + 1
    return out


# ============================ Demo 1/2: cellsort ============================
def build_cellsort(dims, cell_w, margin, lam_vol=2.0, seed=42):
    # Cells fill a CENTRAL region surrounded by Medium (as in CC3D cellsort): the
    # mass rounds up and Condensing sorts to the interior, engulfed by NonCondensing.
    # lam_vol: CC3D's 2D value is 2.0; 3D uses a stronger constraint (6.0) because at
    # 3D coordination the contact energies overwhelm lam=2 and squeeze the less-cohesive
    # NonCondensing cells to zero volume (a CPM volume-constraint artifact).
    nx, ny, nz = dims
    is3d = nz > 1
    w = cpm_core.World(dims, "noflux", 2, 10.0)            # Temperature 10, NeighborOrder 2
    rng = random.Random(seed)
    ngz = (nz - 2 * margin) // cell_w if is3d else 1
    ngy = (ny - 2 * margin) // cell_w
    ngx = (nx - 2 * margin) // cell_w
    for iz in range(ngz):
        for iy in range(ngy):
            for ix in range(ngx):
                t = CONDENSING if rng.random() < 0.5 else NONCONDENSING
                cid = w.add_cell(t, 25.0, lam_vol, 0.0, 0.0)  # TargetVolume 25
                x0, y0 = margin + ix * cell_w, margin + iy * cell_w
                z0 = margin + iz * cell_w if is3d else 0
                z1 = z0 + cell_w if is3d else 1
                w.seed_block(cid, x0, y0, z0, x0 + cell_w, y0 + cell_w, z1)
    # exact CC3D contact matrix: JMM0 JCC2 JCN11 JNN16 JCM16 JNM16
    w.set_contact(MEDIUM, MEDIUM, 0.0)
    w.set_contact(MEDIUM, CONDENSING, 16.0)
    w.set_contact(MEDIUM, NONCONDENSING, 16.0)
    w.set_contact(CONDENSING, CONDENSING, 2.0)
    w.set_contact(CONDENSING, NONCONDENSING, 11.0)
    w.set_contact(NONCONDENSING, NONCONDENSING, 16.0)
    w.finalize(seed)
    return w


def run_cellsort(slug, name, dims, cell_w, margin, mcs_total, n_frames, lam_vol=2.0, seed=42):
    is3d = dims[2] > 1
    w = build_cellsort(dims, cell_w, margin, lam_vol, seed)
    frames = []
    start_boundary = heterotypic_boundary(w)
    per = mcs_total // n_frames
    for f in range(n_frames + 1):
        frames.append({"mcs": f * per,
                       **({"voxels": surface_3d(w)} if is3d else {"labels": labels_2d(w)})})
        if f < n_frames:
            w.step(per)
        print(f"    {slug}: {f}/{n_frames}", flush=True)
    end_boundary = heterotypic_boundary(w)
    mc = medium_contact_by_type(w)
    # normalize medium contact per cell of each type
    counts = cell_count_by_type(w)
    cond_exposure = mc.get(CONDENSING, 0) / max(1, counts.get(CONDENSING, 1))
    noncond_exposure = mc.get(NONCONDENSING, 0) / max(1, counts.get(NONCONDENSING, 1))
    # VALIDATION (Steinberg): boundary shrinks + condensing is engulfed (less medium-exposed)
    checks = [
        (f"heterotypic boundary decreases ({start_boundary} → {end_boundary})",
         end_boundary < 0.7 * start_boundary),
        (f"Condensing engulfed: medium-exposure/cell C={cond_exposure:.2f} < NC={noncond_exposure:.2f}",
         cond_exposure < noncond_exposure),
    ]
    data = {"name": name, "kind": "cellsort", "dims": list(dims), "is3d": is3d,
            "n_cells": w.n_cells(), "cell_types": list(w.cell_types()), "frames": frames}
    return data, checks


# ==================== Demo 3: bacterium_macrophage ====================
def run_bacterium_macrophage(mcs_total, n_frames, seed=7):
    nx, ny = 100, 100
    w = cpm_core.World((nx, ny, 1), "noflux", 2, 10.0)
    BACT, MAC = 1, 2
    # bacterium: a compact cell near center; macrophage: away from it
    bact = w.add_cell(BACT, 25.0, 2.0, 0.0, 0.0)
    w.seed_block(bact, 48, 48, 0, 53, 53, 1)
    mac = w.add_cell(MAC, 36.0, 2.0, 0.0, 0.0)
    w.seed_block(mac, 15, 15, 0, 21, 21, 1)
    # ATTR field: CC3D-typical D=0.1, decay=5e-5; bacterium secretes at 100
    attr = w.add_field("ATTR", 0.10, 5e-5)
    w.set_secretion(attr, BACT, 100.0)
    w.set_chemotaxis(attr, MAC, 40.0)          # macrophage climbs the gradient (tuned)
    # keep cells compact
    w.set_contact(0, BACT, 16.0)
    w.set_contact(0, MAC, 16.0)
    w.set_contact(BACT, BACT, 2.0)
    w.set_contact(MAC, MAC, 2.0)
    w.set_contact(BACT, MAC, 16.0)
    w.finalize(seed)

    def dist_mac_to_bact():
        coms = w.cell_coms()
        bx, by, _ = coms[bact]
        mx, my, _ = coms[mac]
        return math.hypot(mx - bx, my - by)

    frames = []
    per = mcs_total // n_frames
    d0 = None
    attr_series = []
    dist_series = []
    fmax = 0.0
    for f in range(n_frames + 1):
        fc = field_2d(w, attr)
        fmax = max(fmax, max(fc) if fc else 0.0)
        frames.append({"mcs": f * per, "labels": labels_2d(w), "field": fc})
        d = dist_mac_to_bact()
        attr_series.append(w.field_mean_at_cell(attr, mac))
        dist_series.append(d)
        if d0 is None:
            d0 = d
        if f < n_frames:
            w.step(per)
        print(f"    bacterium_macrophage: {f}/{n_frames}  dist={d:.1f}", flush=True)
    dN = dist_series[-1]
    # VALIDATION: macrophage moves toward bacterium AND up the ATTR gradient
    checks = [
        (f"macrophage approaches bacterium (dist {d0:.1f} → {dN:.1f})", dN < d0 - 3),
        (f"ATTR at macrophage rises ({attr_series[0]:.2f} → {attr_series[-1]:.2f})",
         attr_series[-1] > attr_series[0]),
    ]
    data = {"name": "Bacterium–Macrophage (chemotaxis)", "kind": "chemotaxis",
            "dims": [nx, ny, 1], "is3d": False, "n_cells": w.n_cells(),
            "cell_types": list(w.cell_types()), "field_name": "ATTR",
            "field_max": round(fmax, 3), "frames": frames}
    return data, checks


# ==================== Demo 4: growth + mitosis ====================
def run_growth_mitosis(n_frames=30, mcs_per_frame=25, seed=3):
    nx, ny = 130, 130
    w = cpm_core.World((nx, ny, 1), "noflux", 2, 10.0)
    CELL = 1
    # a few seed cells in the center
    for (cx, cy) in [(60, 60), (66, 60), (60, 66), (66, 66)]:
        cid = w.add_cell(CELL, 25.0, 2.0, 0.0, 0.0)
        w.seed_block(cid, cx, cy, 0, cx + 5, cy + 5, 1)
    w.set_contact(0, CELL, 16.0)     # medium costly -> colony stays cohesive
    w.set_contact(CELL, CELL, 4.0)
    w.finalize(seed)

    frames = []
    counts = []
    for f in range(n_frames + 1):
        frames.append({"mcs": f * mcs_per_frame, "labels": labels_2d(w),
                       "n_cells": w.n_cells()})
        counts.append(w.n_cells())
        if f < n_frames:
            w.step(mcs_per_frame)
            w.grow(CELL, 2.0)                  # target_volume += 2 per frame
            w.divide_cells(48.0, 24.0)         # divide at ~2x, reset daughters to 24
        print(f"    growth_mitosis: {f}/{n_frames}  cells={w.n_cells()}", flush=True)
    max_vol = max(w.cell_volumes()[1:])
    # VALIDATION: colony grows geometrically, volumes bounded (no runaway)
    checks = [
        (f"cell count grows ({counts[0]} → {counts[-1]})", counts[-1] >= 3 * counts[0]),
        (f"cell volumes bounded (max {max_vol} < 96)", max_vol < 96),
    ]
    data = {"name": "Cell Growth & Division", "kind": "growth", "dims": [nx, ny, 1],
            "is3d": False, "n_cells": w.n_cells(), "cell_types": list(w.cell_types()),
            "frames": frames}
    return data, checks


# ================================ main ================================
def emit(slug, data, checks, manifest, results):
    path = os.path.join(DATA, slug + ".json")
    with open(path, "w") as fh:
        json.dump(data, fh)
    ok = all(p for _, p in checks)
    results.append((data["name"], checks, ok))
    manifest.append({"file": slug + ".json", "name": data["name"],
                     "is3d": data["is3d"], "n_cells": data["n_cells"],
                     "dims": data["dims"], "kind": data.get("kind"),
                     "validated": ok,
                     "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    tag = "✓ MB" if ok else "✗ MB"
    print(f"  {'✓' if ok else '✗'} {data['name']}: {os.path.getsize(path)/1e6:.1f} MB")


def main():
    manifest, results = [], []
    print("[1/4] cellsort_2D"); emit("cellsort_2d",
        *run_cellsort("cellsort_2d", "2D Cell Sorting", (100, 100, 1), 5, 20, 5000, 24), manifest, results)
    print("[2/4] cellsort_3D"); emit("cellsort_3d",
        *run_cellsort("cellsort_3d", "3D Cell Sorting", (50, 50, 50), 5, 10, 1600, 14, lam_vol=6.0),
        manifest, results)
    print("[3/4] bacterium_macrophage"); emit("bacterium_macrophage",
        *run_bacterium_macrophage(2400, 30), manifest, results)
    print("[4/4] growth_mitosis"); emit("growth_mitosis",
        *run_growth_mitosis(), manifest, results)

    with open(os.path.join(DATA, "index.json"), "w") as fh:
        json.dump({"models": manifest}, fh, indent=2)

    print("\n================ VALIDATION (vs CC3D expected behavior) ================")
    allok = True
    for name, checks, ok in results:
        print(f"\n{name}: {'PASS' if ok else 'FAIL'}")
        for t, p in checks:
            print(f"   [{'PASS' if p else 'FAIL'}] {t}")
        allok = allok and ok
    print(f"\n{'ALL DEMOS VALIDATED' if allok else 'SOME VALIDATIONS FAILED'}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
