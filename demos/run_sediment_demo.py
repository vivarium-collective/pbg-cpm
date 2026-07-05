"""External-potential demo — sedimentation & density sorting (CC3D ExternalPotential).

Heavy and light cells start intermixed at the same height (a checkerboard). Both
feel a constant downward force, but the heavy type feels a stronger one, so it
sediments to the floor while the light type stays afloat above it — density
sorting / centrifugation, driven purely by the external-potential energy term
E = −f·com. The two populations start at the same mean height and end up sorted.

Usage (repo root, venv active):  python demos/run_sediment_demo.py
"""
import json
import os

from cpm import cpm_core

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
W, H, TEMP = 40, 48, 12.0
G_HEAVY, G_LIGHT = 3.5, 1.2           # downward force per type (heavy > light)
COLS, ROWS, SEED, PITCH = 6, 4, 5, 6  # checkerboard of 5x5 cells, 6px pitch
MCS, FRAMES = 6, 70
HEAVY, LIGHT = 1, 2


def build():
    w = cpm_core.World((W, H, 1), "noflux", 2, TEMP)
    w.set_contact(0, HEAVY, 3.0); w.set_contact(0, LIGHT, 3.0)
    w.set_contact(HEAVY, HEAVY, 2.0); w.set_contact(LIGHT, LIGHT, 2.0)
    w.set_contact(HEAVY, LIGHT, 3.0)
    heavy, light = [], []
    for gy in range(ROWS):
        for gx in range(COLS):
            t = HEAVY if (gx + gy) % 2 == 0 else LIGHT
            cid = w.add_cell(t, float(SEED * SEED), 2.0, 0.0, 0.0)
            (heavy if t == HEAVY else light).append(cid)
            ox, oy = 2 + gx * PITCH, 3 + gy * PITCH
            w.seed_block(cid, ox, oy, 0, ox + SEED, oy + SEED, 1)
    w.set_connectivity(HEAVY, True); w.set_connectivity(LIGHT, True)
    w.set_external_potential(HEAVY, 0.0, G_HEAVY, 0.0)
    w.set_external_potential(LIGHT, 0.0, G_LIGHT, 0.0)
    w.finalize(1)
    return w, heavy, light


def mean_y(w, ids):
    com = w.cell_coms()
    return sum(com[c][1] for c in ids) / len(ids)


def main():
    w, heavy, light = build()
    start_hy, start_ly = mean_y(w, heavy), mean_y(w, light)
    start_y = (start_hy + start_ly) / 2
    frames = []
    for f in range(FRAMES):
        frames.append({"mcs": f * MCS, "labels": list(w.snapshot())})
        if f < FRAMES - 1:
            w.step(MCS)
    hy, ly = mean_y(w, heavy), mean_y(w, light)

    checks = [
        (f"both populations start at the same height (heavy start {start_hy:.1f} "
         f"≈ light start {start_ly:.1f})", abs(start_hy - start_ly) < 1.5),
        (f"heavy cells sediment toward the floor (mean depth {hy:.1f} > start "
         f"{start_y:.1f} + 4)", hy > start_y + 4),
        (f"heavy settles BELOW light — density sorting (heavy {hy:.1f} > light "
         f"{ly:.1f} + 4)", hy > ly + 4),
    ]
    blurb = (
        "External potential — CC3D's ExternalPotential, added to the CPM core. A "
        "per-type constant force f gives every pixel the linear potential U = −f·r, so "
        "a cell drifts up the force. Here heavy (dark) and light (pale) cells start "
        "intermixed at the same height, both pulled DOWN, but the heavy type feels a "
        "stronger force — so it sediments to the floor while the light type stays afloat "
        "on top: density sorting / centrifugation from a single energy term. Because the "
        "force is constant, each flip costs only ΔH = (f_old − f_new)·r — O(1).")
    data = {"name": "External Potential — Sedimentation & Density Sorting",
            "kind": "cellsort", "dims": [W, H, 1], "is3d": False, "n_cells": len(heavy) + len(light),
            "cell_types": list(w.cell_types()),
            "type_names": ["Medium", "Heavy (strong gravity)", "Light (weak gravity)"],
            "blurb": blurb, "frames": frames}

    os.makedirs(DATA, exist_ok=True)
    json.dump(data, open(os.path.join(DATA, "sediment_sorting.json"), "w"))
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    ok = all(p for _, p in checks)
    manifest = [m for m in manifest if m["file"] != "sediment_sorting.json"]
    manifest.append({"file": "sediment_sorting.json", "name": data["name"], "is3d": False,
                     "n_cells": data["n_cells"], "dims": [W, H, 1], "kind": "cellsort",
                     "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "length_rods.json",
             "sediment_sorting.json", "scale_2d.json", "connectivity_2d.json",
             "connectivity_3d.json", "connectivity_gap.json", "crypt3d.json",
             "crypt_dynamics.json", "crypt_morphogen.json", "hra_mibitof.json", "hra_ftu.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (external potential) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
