from process_bigraph import Process

from cpm.schema import load_world


class CPMProcess(Process):
    """process-bigraph wrapper around the Rust CPM engine, with a per-cell
    coupling surface so subcellular processes can read the local environment
    and drive differentiation via a cell-type switch."""

    config_schema = {
        "spec": "tree",
        "mcs_per_update": {"_type": "integer", "_default": 10},
        "n_fields": {"_type": "integer", "_default": 0},
        "secretory_types": {"_type": "list", "_default": []},
    }

    def initialize(self, config):
        self.world = load_world(self.config["spec"])
        self.mcs = int(self.config["mcs_per_update"])
        self.n_fields = int(self.config["n_fields"])
        self.secretory = set(int(t) for t in self.config.get("secretory_types") or [])
        self.dims = self.world.dims()

    def inputs(self):
        return {"fates": "maybe[list]"}

    def outputs(self):
        return {
            "volumes": "overwrite[list]",
            "types": "overwrite[list]",
            "positions": "overwrite[list]",
            "field_at_cell": "overwrite[list]",
            "neighbor_secretory": "overwrite[list]",
        }

    def _neighbor_secretory_counts(self, types):
        """Count, per cell, face-adjacent cells whose type is secretory."""
        nx, ny, nz = self.dims
        lab = self.world.snapshot()
        n = len(types)
        counts = [0] * n
        seen = [set() for _ in range(n)]
        def owner(x, y, z):
            return lab[x + y * nx + z * nx * ny]
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    a = owner(x, y, z)
                    if a == 0:
                        continue
                    for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                        xx, yy, zz = x + dx, y + dy, z + dz
                        if xx >= nx or yy >= ny or zz >= nz:
                            continue
                        b = owner(xx, yy, zz)
                        if b == a or b == 0:
                            continue
                        if types[b] in self.secretory and b not in seen[a]:
                            counts[a] += 1; seen[a].add(b)
                        if types[a] in self.secretory and a not in seen[b]:
                            counts[b] += 1; seen[b].add(a)
        return counts

    def update(self, state, interval):
        fates = (state or {}).get("fates")
        if fates:
            for cid, t in enumerate(fates):
                if t and cid > 0:
                    self.world.set_cell_type(cid, int(t))
        self.world.step(self.mcs)

        types = list(self.world.cell_types())
        n = len(types)
        field_at = [0.0] * n
        if self.n_fields > 0:
            for cid in range(1, n):
                field_at[cid] = self.world.field_mean_at_cell(0, cid)
        return {
            "volumes": list(self.world.cell_volumes()),
            "types": types,
            "positions": [list(c) for c in self.world.cell_coms()],
            "field_at_cell": field_at,
            "neighbor_secretory": self._neighbor_secretory_counts(types),
        }
