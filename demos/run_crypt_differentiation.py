"""Crypt differentiation as a process-bigraph Composite.

Assembles the HRA colonic-crypt FTU as a Cellular-Potts ``CPMProcess`` plus,
PER Absorptive-progenitor cell, one ``SBMLSubcell`` (a stemness ODE driven by
the local Wnt concentration) and one ``BooleanSubcell`` (a fate switch). The
two base ``Epithelial Stem`` cells are the permanent Wnt niche and carry no
subcell, so the gradient never collapses. Progenitors near the niche see high
Wnt and stay undifferentiated; distal ones lose stemness and differentiate to
Goblet (secretory) / Absorptive via the ODE -> Boolean -> cell_type coupling.

The whole thing is advanced by ``Composite.run`` -- there is no bypass of the
engine. Gates are keyed on the stemness STATE (not the transient ``stem`` cell
type), validate the expected biology, and the run exits nonzero on any failure.

Usage (repo root, venv active):  python demos/run_crypt_differentiation.py
"""
import json
import os
import sys
from statistics import mean

import process_bigraph as pb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cpm.composites.crypt import build_crypt_composite

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))


def _corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs) or 1e-9
    syy = sum((y - my) ** 2 for y in ys) or 1e-9
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _cpm_world(comp):
    """The live CPM process's Rust world (for labels/types/COMs snapshots).
    process-bigraph stores the instantiated process under the node's
    ``instance`` key after composition."""
    return comp.state["cpm"]["instance"].world


def _state_list(comp, n):
    """Per-cell stemness aligned to cell id (0.0 for cells without a subcell);
    ``cell_state`` is a map keyed by str(cid)."""
    cs = comp.state.get("cell_state", {}) or {}
    arr = [0.0] * n
    for k, v in cs.items():
        i = int(k)
        if 0 <= i < n:
            arr[i] = float(v)
    return arr


def main(n_frames=30, mcs_per_update=80):
    core = pb.allocate_core()
    comp, meta = build_crypt_composite(core, downscale=1.0,
                                       mcs_per_update=mcs_per_update, subcell_every=1)
    stem = meta["stem_type"]
    goblet = meta["goblet_type"]
    absorp = meta["absorptive_type"]
    wired = list(meta["subcell_ids"])
    thr = meta["stemness_threshold"]
    nx, ny, _ = meta["dims"]

    world = _cpm_world(comp)

    def capture(mcs):
        types = list(world.cell_types())
        return {"mcs": mcs, "labels": list(world.snapshot()), "types": types,
                "state": _state_list(comp, len(types))}

    # One outer step advances the Composite by ONE time unit; the CPM runs
    # ``mcs_per_update`` MCS per time unit and each subcell (interval=1) fires
    # once. n_frames time units => n_frames*mcs_per_update MCS total.
    frames = [capture(0)]
    for f in range(n_frames):
        comp.run(1.0)
        frames.append(capture((f + 1) * mcs_per_update))
        print(f"    crypt: {f + 1}/{n_frames}", flush=True)

    # ---- final metrics (keyed on stemness STATE, not the ``stem`` cell type) --
    types = list(world.cell_types())
    coms = [list(c) for c in world.cell_coms()]
    field = comp.state.get("field_at_cell", {}) or {}
    cs = comp.state.get("cell_state", {}) or {}
    n = len(types)
    st = lambda c: float(cs.get(str(c), 0.0))
    fl = lambda c: float(field.get(str(c), 0.0))

    # base region = mean y of the (permanent) Wnt-secreting stem cells at t0.
    # They carry no subcell so they keep type==stem for the whole run.
    stem_ids = [c for c in range(1, n) if types[c] == stem]
    base_y = (mean([coms[c][1] for c in stem_ids]) if stem_ids
              else mean([coms[c][1] for c in wired]))
    dist = lambda c: abs(coms[c][1] - base_y)

    wired_now = [c for c in wired if c < n]
    high = [c for c in wired_now if st(c) >= thr]          # retained stemness
    low = [c for c in wired_now if st(c) < thr]            # differentiated
    n_high = len(high)
    dh = mean([dist(c) for c in high]) if high else float("inf")
    dl = mean([dist(c) for c in low]) if low else 0.0

    n_goblet = sum(1 for t in types[1:] if t == goblet)
    n_absorp = sum(1 for t in types[1:] if t == absorp)
    n0_goblet = meta["initial_counts"]["goblet"]

    # secretory fraction among DIFFERENTIATED wired cells:
    #   diff_gob = wired cells now Goblet; diff_abs = wired cells now Absorptive
    #   AND below threshold (differentiated-to-absorptive, not retained stem-like)
    diff_gob = sum(1 for c in wired_now if types[c] == goblet)
    diff_abs = sum(1 for c in wired_now if types[c] == absorp and st(c) < thr)
    sec_frac = diff_gob / max(1, diff_gob + diff_abs)

    # causality: stemness vs local Wnt across all wired cells
    r = _corr([fl(c) for c in wired_now], [st(c) for c in wired_now])

    checks = [
        (f"basal stemness niche retained: {n_high} wired cells stay stem "
         f"(state>={thr}) and sit nearer the Wnt source "
         f"(mean |y-base| {dh:.0f} < differentiated {dl:.0f})",
         n_high >= 3 and dh < dl),
        (f"differentiation progressed: Goblet {n0_goblet} -> {n_goblet} "
         f"(+{n_goblet - n0_goblet} via fate switch, no division)",
         n_goblet >= n0_goblet + 3),
        (f"both fates present (Goblet {n_goblet}, Absorptive {n_absorp}); "
         f"secretory fraction {sec_frac:.2f} in [0.1,0.6] "
         f"(diff_goblet {diff_gob} / (diff_goblet {diff_gob} + diff_absorptive {diff_abs}))",
         n_goblet > 0 and n_absorp > 0 and 0.1 <= sec_frac <= 0.6),
        (f"causality: corr(stemness, local Wnt) = {r:.2f} > 0.4 across "
         f"{len(wired_now)} wired cells (the ODE drove fate)",
         r > 0.4),
        (f"composite integrity: ran under process_bigraph.Composite with "
         f"{meta['n_subcells']} subcell processes",
         isinstance(comp, pb.Composite) and meta["n_subcells"] > 10),
    ]

    data = {"name": "Crypt Differentiation (subcellular)", "kind": "subcell",
            "dims": meta["dims"], "is3d": False, "n_cells": n - 1,
            "cell_types": types, "type_names": meta["type_names"],
            "subcell_ids": wired, "stemness_threshold": thr,
            "frames": frames}
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "crypt_differentiation.json"), "w") as fh:
        json.dump(data, fh)
    _merge_manifest(data, checks)

    print("\n=========== VALIDATION (crypt differentiation) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    ok = all(p for _, p in checks)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _merge_manifest(data, checks):
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    manifest = [m for m in manifest if m["file"] != "crypt_differentiation.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "crypt_differentiation.json", "name": data["name"],
                     "is3d": False, "n_cells": data["n_cells"], "dims": data["dims"],
                     "kind": "subcell", "validated": ok,
                     "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "hra_mibitof.json", "hra_ftu.json", "crypt_differentiation.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    with open(idx, "w") as fh:
        json.dump({"models": manifest}, fh, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
