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
        # fates is a map keyed by string cell id so subcellular processes can
        # wire a single cell's fate (``[fates, str(cid)]``); integer-index paths
        # into a plain list are rejected by the type system. The composite
        # pre-initialises the fates store with a key per wired cell so per-key
        # overwrite writes land (an absent map key would be dropped).
        return {"fates": "map[integer]"}

    def outputs(self):
        return {
            "volumes": "overwrite[list]",
            "types": "overwrite[list]",
            "positions": "overwrite[list]",
            # per-cell readouts are maps keyed by string cell id, so a subcell
            # can wire ``[<port>, str(cid)]`` to its own cell. ``overwrite`` wraps
            # the map so emitting the whole dict atomically replaces the store
            # (a bare ``map[..]`` apply ignores keys absent from the prior map).
            "field_at_cell": "overwrite[map[float]]",
            "neighbor_secretory": "overwrite[map[integer]]",
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
        fates = (state or {}).get("fates") or {}
        for cid_key, t in fates.items():
            cid = int(cid_key)
            if t and cid > 0:
                self.world.set_cell_type(cid, int(t))
        self.world.step(self.mcs)

        types = list(self.world.cell_types())
        n = len(types)
        field_at = {}
        if self.n_fields > 0:
            for cid in range(1, n):
                field_at[str(cid)] = self.world.field_mean_at_cell(0, cid)
        neigh = self._neighbor_secretory_counts(types)
        return {
            "volumes": list(self.world.cell_volumes()),
            "types": types,
            "positions": [list(c) for c in self.world.cell_coms()],
            "field_at_cell": field_at,
            "neighbor_secretory": {str(cid): neigh[cid] for cid in range(1, n)},
        }
