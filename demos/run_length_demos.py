"""Length (elongation) constraint demo — CC3D's LengthConstraint in the CPM core.

The SAME lawn of identical square cells is relaxed twice, side by side. The LEFT
panel has no length constraint, so surface tension rounds every cell into an
isotropic blob (a cobblestone epithelium). The RIGHT panel adds a per-type length
spring E = λ·(ℓ − ℓ_target)² that pulls each cell's major-axis length toward a
target well above its round value, so the cells thin and elongate into rods (a
lawn of rod bacteria / a palisade of columnar cells) while volume is conserved.

We measure the mean major-axis length in each panel and require the constrained
rods to be much longer than the round controls and close to the target.

Usage (repo root, venv active):  python demos/run_length_demos.py
"""
import json
import os

from cpm import cpm_core

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
GAP = 3
GRID, SLOT, SEED = 3, 15, 5           # 3x3 cells, 15-wide slots, 5x5 square seeds
CONTACT, TEMP, LAM_VOL = 4.0, 2.0, 2.0  # medium adhesion + low T + stiff volume -> tight rounding
TARGET_LEN, LAM_LEN = 12.0, 8.0       # rod target length + spring stiffness (must beat contact)
MCS, FRAMES = 8, 30


def build(length_on):
    n = GRID * SLOT
    w = cpm_core.World((n, n, 1), "noflux", 2, TEMP)
    w.set_contact(0, 1, CONTACT)      # medium adhesion -> rounds cells when unconstrained
    ids = []
    for gy in range(GRID):
        for gx in range(GRID):
            cid = w.add_cell(1, float(SEED * SEED), LAM_VOL, 0.0, 0.0)
            ids.append(cid)
            ox, oy = gx * SLOT + (SLOT - SEED) // 2, gy * SLOT + (SLOT - SEED) // 2
            w.seed_block(cid, ox, oy, 0, ox + SEED, oy + SEED, 1)
    w.set_connectivity(1, True)        # keep every rod a single connected cell
    if length_on:
        w.set_length_constraint(1, TARGET_LEN, LAM_LEN)
    w.finalize(1)
    return w, ids, n


def run(length_on):
    w, ids, n = build(length_on)
    frames = []
    for f in range(FRAMES):
        frames.append({"mcs": f * MCS, "labels": list(w.snapshot())})
        if f < FRAMES - 1:
            w.step(MCS)
    lengths = w.cell_lengths()
    mean_len = sum(lengths[c] for c in ids) / len(ids)
    return w, frames, n, mean_len


def stitch(frames_off, frames_on, n):
    w_comb = 2 * n + GAP
    out = []
    for fo, fn in zip(frames_off, frames_on):
        grid = [0] * (w_comb * n)
        lo, ln = fo["labels"], fn["labels"]
        for y in range(n):
            row = y * w_comb
            for x in range(n):
                if lo[x + y * n]:
                    grid[row + x] = 1
                if ln[x + y * n]:
                    grid[row + x + n + GAP] = 2
        out.append({"mcs": fo["mcs"], "labels": grid})
    return out, (w_comb, n, 1)


def main():
    w_off, frames_off, n, len_off = run(False)
    w_on, frames_on, _, len_on = run(True)
    frames, dims = stitch(frames_off, frames_on, n)

    checks = [
        (f"unconstrained cells stay compact/round (mean length {len_off:.1f} "
         f"< {0.6 * TARGET_LEN:.1f})", len_off < 0.6 * TARGET_LEN),
        (f"length constraint elongates cells into rods (mean length {len_on:.1f} "
         f">= {0.8 * TARGET_LEN:.1f}, near target {TARGET_LEN:.0f})",
         len_on >= 0.8 * TARGET_LEN),
        (f"rods are far longer than the round controls ({len_on:.1f} > "
         f"1.7x {len_off:.1f})", len_on > 1.7 * len_off),
    ]
    blurb = (
        "Length (elongation) constraint — CC3D's LengthConstraint, added to the CPM "
        "core. The SAME lawn of identical square cells is relaxed twice. LEFT: no "
        "constraint, so surface tension rounds each cell into a compact blob "
        "(cobblestone epithelium). RIGHT: a per-cell spring E = λ·(ℓ − ℓ_target)² pulls "
        "each cell's major-axis length (largest eigenvalue of its gyration tensor) toward "
        "a target well above the round value, so cells thin and elongate into randomly "
        "oriented rods — a lawn of rod bacteria — while volume is conserved. Length is "
        "tracked incrementally, so the constraint costs O(1) per flip.")
    data = {"name": "Length Constraint — Round Cells vs Rods", "kind": "integrity",
            "dims": list(dims), "is3d": False, "n_cells": 2, "cell_types": [0, 1, 2],
            "type_names": ["Medium", "Unconstrained (round)", "Length constraint (rods)"],
            "blurb": blurb, "frames": frames}

    os.makedirs(DATA, exist_ok=True)
    json.dump(data, open(os.path.join(DATA, "length_rods.json"), "w"))
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    ok = all(p for _, p in checks)
    manifest = [m for m in manifest if m["file"] != "length_rods.json"]
    manifest.append({"file": "length_rods.json", "name": data["name"], "is3d": False,
                     "n_cells": 2, "dims": list(dims), "kind": "integrity",
                     "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "length_rods.json",
             "scale_2d.json", "connectivity_2d.json", "connectivity_3d.json",
             "connectivity_gap.json", "crypt3d.json", "crypt_dynamics.json",
             "crypt_morphogen.json", "hra_mibitof.json", "hra_ftu.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (length constraint) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
