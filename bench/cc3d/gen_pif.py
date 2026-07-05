"""Regenerate a CC3D PIF matching cpm-bench's seeding.

  python gen_pif.py                      -> blocks.piff (50^3, sparse 125 cells;
                                            original head-to-head, matches main.rs)
  python gen_pif.py 96 8 blocks_96.piff  -> dense DIM^3 tiled by CELL^3 cells,
                                            matching cpm-bench build(dim, cell)
"""
import sys

if len(sys.argv) >= 4:
    dim, cell, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    g = dim // cell
    lines, cid = [], 0
    for gz in range(g):
        for gy in range(g):
            for gx in range(g):
                cid += 1
                t = 1 + ((gx + gy + gz) % 2)
                x0, y0, z0 = gx * cell, gy * cell, gz * cell
                lines.append(f"{cid} Type{t} {x0} {x0+cell-1} {y0} {y0+cell-1} {z0} {z0+cell-1}")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{cid} cells, {dim}^3 dense -> {out}")
else:
    lines, cid = [], 0
    for gz in range(5):
        for gy in range(5):
            for gx in range(5):
                cid += 1
                t = 1 + ((gx + gy + gz) % 2)
                x0, y0, z0 = gx * 10, gy * 10, gz * 10
                lines.append(f"{cid} Type{t} {x0} {x0+7} {y0} {y0+7} {z0} {z0+7}")
    open("blocks.piff", "w").write("\n".join(lines) + "\n")
    print(f"{cid} cells, 50^3 sparse -> blocks.piff")
