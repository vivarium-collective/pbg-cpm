"""Morphogen-DRIVEN living crypt: differentiation emerges from a Wnt gradient,
not from a height rule. A fixed niche at the crypt base secretes Wnt (a real PDE:
diffusion + decay on the lattice); it forms a base-high gradient; every cell reads
its LOCAL Wnt concentration to choose fate — high Wnt keeps it stem, low Wnt makes
it differentiate. The stem→transit-amplifying→goblet/colonocyte axis therefore
*emerges* from the morphogen field (change the gradient and the zones move),
while the conveyor (divide at base → pushed up → slough at the mouth) and the
monolayer integrity (E1 connectivity + basement membrane) carry over.

Colour by TYPE to see the emergent zonation, or by STATE to see the Wnt gradient
itself painted on the cells.

Usage (repo root, venv active):  python demos/run_crypt_morphogen.py
"""
import json
import os
import sys
from collections import Counter, deque

from cpm import cpm_core
from cpm.crypt3d import build_crypt3d
from cpm.metrics import connected_components, open_lumen_depth

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "data"))
NICHE, STEM, TA, GOB, COL, DEAD = 1, 2, 3, 4, 5, 6
TYPE_NAMES = ["Medium", "Niche (Wnt source)", "Stem", "Transit-amplifying",
              "Goblet", "Colonocyte", "Sloughing"]
_NB6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

# geometry
RADIUS, CYL_H, WALL, PITCH, MARGIN = 9, 54, 2, 5, 4
# Wnt PDE (dt*D*2*ndim < 1 for stability; lambda = sqrt(D/decay) ~ 15 voxels)
WNT_D, WNT_DECAY, WNT_DT, WNT_SUBSTEPS, WNT_SECRETE, WNT_WARMUP = 0.6, 0.0027, 0.2, 30, 2.0, 80
NICHE_FRAC = 0.08              # bottom band that becomes the fixed niche source
STEM_WNT, TA_WNT = 0.45, 0.15  # fate thresholds on local Wnt / max Wnt
# dynamics
GROW, THRESH_MULT, LAM = 1.0, 1.6, 60.0
MCS, N_STEPS, GRACE, CAP_MULT = 6, 45, 2, 1.10


def basal_anchors(labels, dims):
    nx, ny, nz = dims
    n = nx * ny * nz
    def idx(x, y, z): return x + y * nx + z * nx * ny
    ext = bytearray(n); q = deque()
    for i in range(n):
        if labels[i]:
            continue
        z, r = divmod(i, nx * ny); y, x = divmod(r, nx)
        if (x in (0, nx-1) or y in (0, ny-1) or z in (0, nz-1)) and not ext[i]:
            ext[i] = 1; q.append(i)
    while q:
        v = q.popleft(); z, r = divmod(v, nx * ny); y, x = divmod(r, nx)
        for dx, dy, dz in _NB6:
            x2, y2, z2 = x+dx, y+dy, z+dz
            if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
                w = idx(x2, y2, z2)
                if labels[w] == 0 and not ext[w]:
                    ext[w] = 1; q.append(w)
    a = []
    for i in range(n):
        if labels[i] == 0:
            continue
        z, r = divmod(i, nx * ny); y, x = divmod(r, nx)
        for dx, dy, dz in _NB6:
            x2, y2, z2 = x+dx, y+dy, z+dz
            if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz and ext[idx(x2, y2, z2)]:
                a.append(i); break
    return a


def surface_voxels(w):
    """boundary voxels [x,y,z,cellId,cellType] for the viewer (per-frame type)."""
    nx, ny, nz = w.dims()
    lab = w.snapshot(); types = w.cell_types()
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


def wnt_state(w, wnt, ncells):
    """per-cell local Wnt (for the viewer's STATE colour mode = the gradient)."""
    s = [0.0] * (ncells + 1)
    vols = w.cell_volumes()
    for c in range(1, len(vols)):
        if c <= ncells and vols[c] > 0:
            s[c] = round(w.field_mean_at_cell(wnt, c), 4)
    return s


def build():
    (nx, ny, nz), labels, seg, _ = build_crypt3d(
        radius=RADIUS, cyl_height=CYL_H, wall=WALL, cell_pitch=PITCH,
        margin=MARGIN, open_top=True)
    seg = {s: STEM for s in seg}
    V0 = sorted(Counter(v for v in labels if v).values())[len(Counter(v for v in labels if v)) // 2]
    w = cpm_core.World((nx, ny, nz), "noflux", 2, 4.0)
    w.seed_from_labels(labels, seg, 1, float(V0), LAM)
    wnt = w.add_field("wnt", WNT_D, WNT_DECAY)      # BEFORE finalize
    for t in range(1, 7):
        w.set_contact(0, t, 6.0)
        for u in range(t, 7):
            w.set_contact(t, u, 4.0)
        w.set_connectivity(t, True)
    w.set_connectivity_medium(True)
    w.set_membrane(basal_anchors(labels, (nx, ny, nz)), 14.0, float(WALL))
    for t in (NICHE, STEM, TA, GOB, COL):
        w.set_membrane_anchored(t, True)
    w.finalize(1)
    z_base, z_top = MARGIN + RADIUS, MARGIN + RADIUS + CYL_H
    def zf(z): return (z - z_base) / (z_top - z_base)
    vols0 = w.cell_volumes()
    for c in range(1, len(vols0)):
        if vols0[c] > 0:
            w.set_target_volume(c, float(vols0[c]))
    coms = w.cell_coms(); niche = set()
    for c in range(1, len(vols0)):
        if vols0[c] > 0 and zf(coms[c][2]) < NICHE_FRAC:
            w.set_cell_type(c, NICHE); niche.add(c)
    w.set_secretion(wnt, NICHE, WNT_SECRETE)
    w.set_field_dynamics(wnt, WNT_DT, WNT_SUBSTEPS)
    w.advance_fields(WNT_WARMUP)                    # pre-equilibrate the gradient
    return w, wnt, niche, (nx, ny, nz), V0, zf


def main():
    w, wnt, niche, (nx, ny, nz), V0, zf = build()
    thresh = THRESH_MULT * V0
    n0 = w.n_cells()
    target = int(CAP_MULT * (n0 - len(niche)))
    dying = {}
    births = deaths = 0
    div_z, death_z = [], []
    live_hist, min_lumen = [], 10**9
    frames = [{"mcs": 0, "voxels": surface_voxels(w), "state": wnt_state(w, wnt, n0)}]

    for step in range(N_STEPS):
        w.grow(STEM, GROW); w.grow(TA, GROW)
        coms = w.cell_coms()
        for c in w.divide_cells(thresh, V0):
            births += 1
            div_z.append(zf(coms[c][2]) if c < len(coms) else 0.0)
        coms = w.cell_coms(); vols = w.cell_volumes()
        wmax = max((w.field_mean_at_cell(wnt, c) for c in range(1, len(vols)) if vols[c] > 0),
                   default=1.0) or 1.0
        for c in range(1, len(vols)):
            if vols[c] <= 0 or c in dying or c in niche:
                continue
            wr = w.field_mean_at_cell(wnt, c) / wmax    # LOCAL Wnt drives fate
            if wr >= STEM_WNT:
                w.set_cell_type(c, STEM)
            elif wr >= TA_WNT:
                w.set_cell_type(c, TA)
            else:
                w.set_cell_type(c, GOB if (c * 2654435761) % 100 < 59 else COL)
                w.set_target_volume(c, float(vols[c]))
        live = sorted(((coms[c][2], c) for c in range(1, len(vols))
                       if vols[c] > 0 and c not in dying and c not in niche), reverse=True)
        i = 0
        while len(live) - i > target and i < len(live):
            z, c = live[i]; i += 1
            w.set_cell_type(c, DEAD); w.set_target_volume(c, 0.0)
            dying[c] = 0; deaths += 1; death_z.append(zf(z))
        ready = [c for c, age in dying.items() if age >= GRACE]
        if ready:
            w.remove_cells(ready)
            for c in ready:
                del dying[c]
        for c in dying:
            dying[c] += 1
        w.step(MCS)
        vv = w.cell_volumes()
        live_hist.append(sum(1 for c in range(1, len(vv))
                             if vv[c] > 0 and c not in dying and c not in niche))
        min_lumen = min(min_lumen, open_lumen_depth(w))
        frames.append({"mcs": (step + 1) * MCS, "voxels": surface_voxels(w),
                       "state": wnt_state(w, wnt, n0)})

    vols = w.cell_volumes(); types = w.cell_types(); coms = w.cell_coms()
    alive = [c for c in range(1, len(vols)) if vols[c] > 0]
    frag = sum(1 for c in alive if connected_components(w, c) != 1)
    conc = w.field_conc(wnt); snap = w.snapshot()
    def band_wnt(lo, hi):
        tot = cnt = 0.0
        for i, cv in enumerate(conc):
            if snap[i] != 0 and lo <= zf(i // (nx * ny) + 0.5) < hi:
                tot += cv; cnt += 1
        return tot / cnt if cnt else 0.0
    w_base, w_top = band_wnt(0.0, 0.2), band_wnt(0.8, 1.0)
    def mwnt(*tp):
        xs = [w.field_mean_at_cell(wnt, c) for c in alive if types[c] in tp]
        return sum(xs) / len(xs) if xs else 0.0
    def mz(*tp):
        xs = [zf(coms[c][2]) for c in alive if types[c] in tp]
        return sum(xs) / len(xs) if xs else float("nan")
    ws, wt, wd = mwnt(STEM), mwnt(TA), mwnt(GOB, COL)
    zs, zt, zd = mz(STEM), mz(TA), mz(GOB, COL)
    mdiv = sum(div_z) / len(div_z) if div_z else 1.0
    mdeath = sum(death_z) / len(death_z) if death_z else 0.0
    warm = live_hist[len(live_hist) // 2:]
    steady = warm and min(warm) >= 0.7 * target and max(warm) <= 1.3 * target

    checks = [
        (f"a Wnt morphogen gradient forms from the basal niche: base {w_base:.2f} >> "
         f"mouth {w_top:.2f} (>= 3x)", w_base >= 3 * max(w_top, 1e-6)),
        (f"fate is DRIVEN by local Wnt (not height): mean Wnt at stem {ws:.2f} > "
         f"transit-amplifying {wt:.2f} > differentiated {wd:.2f}", ws > wt > wd),
        (f"the crypt axis EMERGES from the gradient: stem basal {zs:.2f} < "
         f"differentiated apical {zd:.2f}", zs < zd and zs < 0.3 and zd > 0.4),
        (f"conveyor + turnover: {births} births at the base (h {mdiv:.2f}) balanced by "
         f"{deaths} sloughs at the mouth (h {mdeath:.2f}); steady population",
         births > 0.2 * n0 and deaths > 0.2 * n0 and mdeath - mdiv > 0.35 and steady),
        (f"integrity: no cell fragments ({frag} of {len(alive)}); lumen stays open "
         f"(min depth {min_lumen} >= {int(0.3*nz)})", frag == 0 and min_lumen >= 0.3 * nz),
    ]

    blurb = (
        "Morphogen-DRIVEN crypt — the same colonic-crypt conveyor, but differentiation "
        "now EMERGES from a Wnt gradient instead of a height rule. A fixed niche at the "
        "base (a real reaction–diffusion PDE on the lattice: diffusion + decay) secretes "
        "Wnt, which forms a base-high gradient; every cell reads its LOCAL Wnt "
        "concentration and chooses fate — high Wnt stays stem, mid becomes "
        "transit-amplifying, low differentiates into goblet / colonocyte. The "
        "stem→differentiated axis is therefore a readout of the field, not imposed: move "
        "the gradient and the zones move. Colour by TYPE for the emergent zonation, or by "
        "WNT (state) to see the gradient itself painted on the cells. Divide at the base, "
        "pushed up, slough at the mouth; E1 connectivity + a basement membrane keep it a "
        "single-cell-thick monolayer with an open lumen.")
    data = {"name": "Living Crypt — Wnt-driven (morphogen)", "kind": "cryptlife",
            "dims": [nx, ny, nz], "is3d": True, "n_cells": w.n_cells(),
            "cell_types": list(w.cell_types()), "type_names": TYPE_NAMES,
            "state_label": "Wnt", "blurb": blurb, "frames": frames}
    os.makedirs(DATA, exist_ok=True)
    json.dump(data, open(os.path.join(DATA, "crypt_morphogen.json"), "w"))
    idx = os.path.join(DATA, "index.json")
    manifest = json.load(open(idx))["models"] if os.path.exists(idx) else []
    manifest = [m for m in manifest if m["file"] != "crypt_morphogen.json"]
    ok = all(p for _, p in checks)
    manifest.append({"file": "crypt_morphogen.json", "name": data["name"], "is3d": True,
                     "n_cells": data["n_cells"], "dims": data["dims"], "kind": "cryptlife",
                     "validated": ok, "checks": [{"text": t, "pass": bool(p)} for t, p in checks]})
    order = ["cellsort_2d.json", "cellsort_3d.json", "spheroid_3d.json",
             "bacterium_macrophage.json", "growth_mitosis.json", "scale_2d.json",
             "connectivity_2d.json", "connectivity_3d.json", "connectivity_gap.json",
             "crypt3d.json", "crypt_dynamics.json", "crypt_morphogen.json",
             "hra_mibitof.json", "hra_ftu.json"]
    manifest.sort(key=lambda m: order.index(m["file"]) if m["file"] in order else 99)
    json.dump({"models": manifest}, open(idx, "w"), indent=2)

    print("\n=========== VALIDATION (Wnt-driven crypt) ===========")
    for t, p in checks:
        print(f"   [{'PASS' if p else 'FAIL'}] {t}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
