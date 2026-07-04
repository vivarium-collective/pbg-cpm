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
    # Adjacency matching the CPM lattice at neighbor_order 2: the 8-neighbourhood
    # in 2D (Moore) and the 18-neighbourhood in 3D (Manhattan distance <= 2, i.e.
    # excluding the 8 cube corners). Keeping this identical to the constraint's
    # notion of connectivity is what makes the integrity metrics measure exactly
    # what would_stay_connected enforces.
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if abs(dx) + abs(dy) + abs(dz) == 3:   # 3D cube corners: not adjacent
                    continue
                x2, y2, z2 = cx + dx, cy + dy, cz + dz
                if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
                    yield x2 + y2 * nx + z2 * nx * ny


def connected_components(world, cell_id):
    """Number of connected components of `cell_id`'s pixels, using the same
    adjacency as the CPM constraint (8-conn 2D / 18-conn 3D), bounded no-wrap
    domain."""
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


def radial_cell_counts(world, cx=None, cy=None, n_theta=24):
    """Sorted list of the number of distinct cells crossed along each radial ray
    from the axis (cx, cy) outward, over a (z, theta) sample grid. A single-cell
    wall yields mostly 1s (with a few 2s where a ray grazes a cell boundary or
    skims obliquely through a hemispherical cap). This is the raw distribution
    behind the monolayer thickness measures: the mean is the robust central
    certificate; an upper-tail percentile (e.g. p90 <= 2) catches genuine
    multilayering while ignoring the corner-grazing tail that inflates the max."""
    import math
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    if cx is None:
        cx = nx / 2.0
    if cy is None:
        cy = ny / 2.0
    rmax = int(math.hypot(nx, ny)) + 1
    counts = []
    for z in range(nz):
        base = z * nx * ny
        for ti in range(n_theta):
            theta = 2.0 * math.pi * ti / n_theta
            ct, st = math.cos(theta), math.sin(theta)
            crossed = set()
            for r in range(rmax):
                x = int(cx + r * ct)
                y = int(cy + r * st)
                if not (0 <= x < nx and 0 <= y < ny):
                    break
                o = labels[base + x + y * nx]
                if o != 0:
                    crossed.add(o)
            if crossed:
                counts.append(len(crossed))
    return sorted(counts)


def radial_thickness(world, cx=None, cy=None, n_theta=24):
    """Mean and max number of distinct cells crossed along radial rays from the
    axis (cx, cy) outward, over a (z, theta) sample grid. ~1 for a monolayer.
    (The max is sampling-sensitive; use radial_cell_counts for an upper-tail
    percentile when a robust multilayering guard is needed.)"""
    counts = radial_cell_counts(world, cx, cy, n_theta)
    if not counts:
        return 0.0, 0
    return sum(counts) / len(counts), max(counts)


def membrane_distance_field(dims, anchors):
    """Multi-source BFS (6-neighbour graph distance) from anchor voxels to every
    voxel. Mirrors cpm_core::membrane::build_distance_field for tests/demos."""
    from collections import deque
    nx, ny, nz = dims
    n = nx * ny * nz
    dist = [float("inf")] * n
    q = deque()
    for a in anchors:
        if 0 <= a < n and dist[a] != 0.0:
            dist[a] = 0.0
            q.append(a)
    while q:
        v = q.popleft()
        d = dist[v] + 1.0
        z, rem = divmod(v, nx * ny)
        y, x = divmod(rem, nx)
        nb = []
        if x + 1 < nx: nb.append(v + 1)
        if x >= 1: nb.append(v - 1)
        if y + 1 < ny: nb.append(v + nx)
        if y >= 1: nb.append(v - nx)
        if z + 1 < nz: nb.append(v + nx * ny)
        if z >= 1: nb.append(v - nx * ny)
        for w in nb:
            if d < dist[w]:
                dist[w] = d
                q.append(w)
    return dist


def mean_membrane_distance(world, dist_field, anchored_types):
    """Mean membrane distance over the voxels of cells whose type is in
    `anchored_types` (a set of type ids). ~<= band when the anchor holds."""
    nx, ny, nz = world.dims()
    labels = world.snapshot()
    types = world.cell_types()
    total, count = 0.0, 0
    for i, c in enumerate(labels):
        if c != 0 and types[c] in anchored_types:
            total += dist_field[i]
            count += 1
    return total / count if count else 0.0
