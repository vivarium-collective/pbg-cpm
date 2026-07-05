"""Regenerate blocks.piff — the exact seeding of crates/cpm-bench/src/main.rs
(50^3, 125 cells, 8^3 blocks on a 10-pitch grid, type 1+((gx+gy+gz)%2))."""
lines = []
cid = 0
for gz in range(5):
    for gy in range(5):
        for gx in range(5):
            cid += 1
            t = 1 + ((gx + gy + gz) % 2)
            x0, y0, z0 = gx * 10, gy * 10, gz * 10
            lines.append(f"{cid} Type{t} {x0} {x0+7} {y0} {y0+7} {z0} {z0+7}")
open("blocks.piff", "w").write("\n".join(lines) + "\n")
print(f"{cid} cells")
