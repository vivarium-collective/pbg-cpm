from process_bigraph import Process


class BooleanSubcell(Process):
    """A minimal synchronous Boolean fate switch with Notch-style lateral
    inhibition. When stemness falls below threshold the cell differentiates:
    it becomes secretory (goblet) unless a neighbour already committed
    secretory, in which case it becomes absorptive. Deterministic."""

    config_schema = {
        "stemness_threshold": {"_type": "float", "_default": 0.4},
        "goblet_type": {"_type": "integer", "_default": 3},
        "absorptive_type": {"_type": "integer", "_default": 2},
    }

    def initialize(self, config):
        self.thresh = float(config["stemness_threshold"])
        self.goblet = int(config["goblet_type"])
        self.absorptive = int(config["absorptive_type"])

    def inputs(self):
        return {"state": "float", "neighbor_secretory": "float"}

    def outputs(self):
        return {"fate": "overwrite[integer]"}

    def update(self, state, interval):
        s = float((state or {}).get("state", 1.0))
        if s >= self.thresh:
            return {"fate": 0}
        neigh = float((state or {}).get("neighbor_secretory", 0))
        return {"fate": self.goblet if neigh == 0 else self.absorptive}
