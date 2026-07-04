def heterotypic_boundary(world):
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    types = world.cell_types()

    def idx(x, y, z):
        return x + y * nx + z * nx * ny

    count = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                a = labels[idx(x, y, z)]
                # +x and +y faces only (each unordered face once); periodic wrap
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    if dz and nz == 1:
                        continue
                    nxp, nyp, nzp = (x + dx) % nx, (y + dy) % ny, (z + dz) % nz
                    b = labels[idx(nxp, nyp, nzp)]
                    if a == b:
                        continue
                    if a == 0 or b == 0:
                        continue
                    if types[a] != types[b]:
                        count += 1
    return count


def _neighbors_moore(cx, cy, cz, nx, ny, nz):
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                x2, y2, z2 = cx + dx, cy + dy, cz + dz
                if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
                    yield x2 + y2 * nx + z2 * nx * ny


def connected_components(world, cell_id):
    """Number of connected components of `cell_id`'s pixels (Moore adjacency,
    bounded/no-wrap domain)."""
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    sites = {i for i, v in enumerate(labels) if v == cell_id}
    seen, comps = set(), 0
    for s in sites:
        if s in seen:
            continue
        comps += 1
        stack = [s]
        seen.add(s)
        while stack:
            c = stack.pop()
            cz, rem = divmod(c, nx * ny)
            cy, cx = divmod(rem, nx)
            for n in _neighbors_moore(cx, cy, cz, nx, ny, nz):
                if n in sites and n not in seen:
                    seen.add(n)
                    stack.append(n)
    return comps


def interior_medium_pockets(world):
    """Count medium (cell 0) connected components that do NOT touch the lattice
    border — interior gap pockets. Assumes a bounded (noflux) domain."""
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    sites = {i for i, v in enumerate(labels) if v == 0}
    seen, pockets = set(), 0
    for s in sites:
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        touches_border = False
        while stack:
            c = stack.pop()
            cz, rem = divmod(c, nx * ny)
            cy, cx = divmod(rem, nx)
            if cx == 0 or cx == nx - 1 or cy == 0 or cy == ny - 1 or \
               (nz > 1 and (cz == 0 or cz == nz - 1)):
                touches_border = True
            for n in _neighbors_moore(cx, cy, cz, nx, ny, nz):
                if n in sites and n not in seen:
                    seen.add(n)
                    stack.append(n)
        if not touches_border:
            pockets += 1
    return pockets
