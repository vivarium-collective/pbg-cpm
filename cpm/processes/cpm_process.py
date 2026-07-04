from process_bigraph import Process

from cpm.schema import load_world


class CPMProcess(Process):
    """process-bigraph wrapper around the Rust CPM engine.

    Loads a world from a spec on construction and advances it by
    ``mcs_per_update`` Monte Carlo steps on each ``update``, returning a
    readback of per-cell volumes, surfaces, and centers of mass.
    """

    config_schema = {
        "spec": "tree",
        "mcs_per_update": {"_type": "integer", "_default": 10},
    }

    def initialize(self, config):
        self.world = load_world(self.config["spec"])
        self.mcs = int(self.config["mcs_per_update"])

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "cell_volumes": "list",
            "cell_surfaces": "list",
            "cell_coms": "list",
        }

    def update(self, state, interval):
        self.world.step(self.mcs)
        return {
            "cell_volumes": list(self.world.cell_volumes()),
            "cell_surfaces": list(self.world.cell_surfaces()),
            "cell_coms": [list(c) for c in self.world.cell_coms()],
        }
