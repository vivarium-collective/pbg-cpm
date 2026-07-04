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
