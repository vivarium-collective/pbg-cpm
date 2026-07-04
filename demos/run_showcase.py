"""Run a set of large, visually-distinct CPM simulations and export them as
time-series data packs for the interactive viewer.

Usage (from repo root, venv active):
    python demos/run_showcase.py

Writes viewer/data/<slug>.json per model and viewer/data/index.json (the
manifest the viewer reads to populate its model selector).
"""
import json
import os
import time

from cpm import cpm_core

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "viewer", "data"))
os.makedirs(DATA, exist_ok=True)


def tile_cells(world, dims, cell_w, margin, type_of):
    """Tile the lattice with contiguous square/cube cells; return count.

    type_of(gx, gy, gz) -> (cell_type, target_volume, lambda_vol) OR None to skip.
    """
    nx, ny, nz = dims
    is3d = nz > 1
    gz_range = range((nz - 2 * margin) // cell_w) if is3d else [0]
    n = 0
    for gz in gz_range:
        for gy in range((ny - 2 * margin) // cell_w):
            for gx in range((nx - 2 * margin) // cell_w):
                spec = type_of(gx, gy, gz)
                if spec is None:
                    continue
                ctype, tvol, lvol = spec
                cid = world.add_cell(ctype, tvol, lvol, 0.0, 0.0)
                x0 = margin + gx * cell_w
                y0 = margin + gy * cell_w
                z0 = margin + gz * cell_w if is3d else 0
                z1 = z0 + cell_w if is3d else 1
                world.seed_block(cid, x0, y0, z0, x0 + cell_w, y0 + cell_w, z1)
                n += 1
    return n


def capture_labels_2d(world):
    return list(world.snapshot())


def capture_surface_3d(world):
    """Boundary voxels only: [x, y, z, cell_id] for voxels with a 6-neighbor
    of a different owner. Keeps 3D data small and is exactly what we render."""
    nx, ny, nz = world.dims()
    labels = world.snapshot()

    def idx(x, y, z):
        return x + y * nx + z * nx * ny

    out = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                c = labels[idx(x, y, z)]
                if c == 0:
                    continue
                surf = False
                if x == 0 or labels[idx(x - 1, y, z)] != c: surf = True
                elif x == nx - 1 or labels[idx(x + 1, y, z)] != c: surf = True
                elif y == 0 or labels[idx(x, y - 1, z)] != c: surf = True
                elif y == ny - 1 or labels[idx(x, y + 1, z)] != c: surf = True
                elif z == 0 or labels[idx(x, y, z - 1)] != c: surf = True
                elif z == nz - 1 or labels[idx(x, y, z + 1)] != c: surf = True
                if surf:
                    out.append([x, y, z, c])
    return out


def run_model(cfg):
    t0 = time.time()
    nx, ny, nz = cfg["dims"]
    is3d = nz > 1
    world = cpm_core.World((nx, ny, nz), cfg.get("boundary", "periodic"),
                           2, cfg["temperature"])
    n = tile_cells(world, cfg["dims"], cfg["cell_w"], cfg["margin"], cfg["type_of"])
    for a, b, j in cfg["contact"]:
        world.set_contact(a, b, j)
    world.finalize(cfg.get("seed", 17))

    frames = []
    n_frames = cfg["n_frames"]
    mcs_per_frame = cfg["mcs_total"] // n_frames
    for f in range(n_frames + 1):
        if is3d:
            frames.append({"mcs": f * mcs_per_frame, "voxels": capture_surface_3d(world)})
        else:
            frames.append({"mcs": f * mcs_per_frame, "labels": capture_labels_2d(world)})
        if f < n_frames:
            world.step(mcs_per_frame)
        print(f"    {cfg['slug']}: frame {f}/{n_frames} @ {f*mcs_per_frame} MCS", flush=True)

    data = {
        "name": cfg["name"],
        "description": cfg["description"],
        "dims": [nx, ny, nz],
        "is3d": is3d,
        "n_cells": n,
        "cell_types": list(world.cell_types()),
        "frames": frames,
    }
    path = os.path.join(DATA, cfg["slug"] + ".json")
    with open(path, "w") as fh:
        json.dump(data, fh)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  ✓ {cfg['name']}: {n} cells, {n_frames+1} frames, "
          f"{size_mb:.1f} MB, {time.time()-t0:.0f}s", flush=True)
    return {"file": cfg["slug"] + ".json", "name": cfg["name"],
            "description": cfg["description"], "is3d": is3d, "n_cells": n,
            "dims": [nx, ny, nz]}


# --- Contact-energy recipes (J = interfacial energy; Metropolis MINIMIZES it) ---
# Low J => that pair likes to touch. Cells minimize total boundary energy.

def checkerboard(gx, gy, gz):
    return (1 + (gx + gy + gz) % 2, 36, 2.0)

MODELS = [
    {
        "slug": "sort2d", "name": "2D Cell Sorting",
        "description": "Two adhesively-distinct cell types, mixed at random, "
                       "segregate into pure domains (Steinberg differential "
                       "adhesion). Like-type contact is cheap, unlike-type costly.",
        "dims": [200, 200, 1], "cell_w": 6, "margin": 4, "temperature": 10.0,
        "type_of": checkerboard, "n_frames": 24, "mcs_total": 4800, "seed": 7,
        # medium costly (16) -> cells stay packed; same-type cheap (2); cross costly (11)
        "contact": [(0, 1, 16.0), (0, 2, 16.0), (1, 1, 2.0), (2, 2, 2.0), (1, 2, 11.0)],
    },
    {
        "slug": "engulf2d", "name": "2D Engulfment",
        "description": "Asymmetric adhesion: the high-surface-tension type is "
                       "enveloped by the low-tension type, forming core–shell "
                       "islands instead of side-by-side domains.",
        "dims": [160, 160, 1], "cell_w": 6, "margin": 4, "temperature": 10.0,
        # ~40% type-2 (the engulfed core)
        "type_of": lambda gx, gy, gz: (2 if (gx * 7 + gy * 3) % 5 < 2 else 1, 36, 2.0),
        "n_frames": 22, "mcs_total": 4400, "seed": 11,
        # type-1 (shell) tolerates medium (8); type-2 (core) hates medium (18) so it
        # hides inside; cross (4) cheaper than type2-medium -> 1 engulfs 2.
        "contact": [(0, 1, 8.0), (0, 2, 18.0), (1, 1, 6.0), (2, 2, 2.0), (1, 2, 4.0)],
    },
    {
        "slug": "disperse2d", "name": "2D Dispersal",
        "description": "Weakly cohesive cells that prefer contact with medium "
                       "over each other: a packed sheet loosens and rounds up "
                       "into separated islands.",
        "dims": [160, 160, 1], "cell_w": 6, "margin": 4, "temperature": 12.0,
        "type_of": lambda gx, gy, gz: (1, 30, 2.0), "n_frames": 20,
        "mcs_total": 4000, "seed": 3,
        # medium cheap (2) -> cells expose surface; same-type costly (14) -> repel
        "contact": [(0, 1, 2.0), (1, 1, 14.0)],
    },
    {
        "slug": "sort3d", "name": "3D Cell Sorting",
        "description": "Differential-adhesion sorting in three dimensions — two "
                       "cell types demix into interpenetrating 3D domains. "
                       "Rotate to explore the sorted interface.",
        "dims": [56, 56, 56], "cell_w": 7, "margin": 2, "temperature": 10.0,
        "type_of": checkerboard, "n_frames": 14, "mcs_total": 1400, "seed": 5,
        "contact": [(0, 1, 16.0), (0, 2, 16.0), (1, 1, 2.0), (2, 2, 2.0), (1, 2, 11.0)],
    },
]


def main():
    print(f"Running {len(MODELS)} showcase simulations -> {DATA}")
    manifest = []
    for cfg in MODELS:
        print(f"\n[{cfg['name']}]")
        manifest.append(run_model(cfg))
    with open(os.path.join(DATA, "index.json"), "w") as fh:
        json.dump({"models": manifest}, fh, indent=2)
    print(f"\n✓ wrote manifest with {len(manifest)} models -> viewer/data/index.json")


if __name__ == "__main__":
    main()
