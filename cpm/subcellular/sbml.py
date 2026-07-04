from process_bigraph import Process
from pbg_tellurium.processes import TelluriumProcess


class SBMLSubcell(Process):
    """Per-cell SBML ODE via pbg-tellurium. The environmental ligand is pushed
    into a held floating species each step; a chosen species is published as
    the cell's scalar ``state`` (e.g. stemness)."""

    config_schema = {
        "model": {"_type": "string", "_default": ""},
        "ligand_species": {"_type": "string", "_default": "Wnt"},
        "state_species": {"_type": "string", "_default": "S"},
        "ligand_scale": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.ligand_species = config["ligand_species"]
        self.state_species = config["state_species"]
        self.scale = float(config["ligand_scale"])
        self._tp = TelluriumProcess({"model": config["model"]}, core=self.core)

    def inputs(self):
        return {"ligand": "float"}

    def outputs(self):
        return {"state": "overwrite[float]"}

    def update(self, state, interval):
        ligand = float((state or {}).get("ligand", 0.0)) * self.scale
        out = self._tp.update({"species": {self.ligand_species: ligand}}, interval)
        return {"state": float(out["species"][self.state_species])}
