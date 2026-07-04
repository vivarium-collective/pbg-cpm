import process_bigraph as pb
from cpm.subcellular.sbml import SBMLSubcell

# S is produced at a Wnt-gated rate and decays; Wnt is a held floating species
MODEL = """
J1: -> S; k_on*Wnt^4/(K^4 + Wnt^4);
J2: S -> ; k_off*S;
species S, Wnt;
S = 1.0; Wnt = 0.0;
k_on = 0.8; K = 0.3; k_off = 0.4;
"""

CFG = {"model": MODEL, "ligand_species": "Wnt", "state_species": "S", "ligand_scale": 1.0}


def _proc():
    return SBMLSubcell(CFG, core=pb.allocate_core())


def test_low_ligand_lets_stemness_decay():
    p = _proc()
    s_last = 1.0
    for _ in range(20):
        s_last = p.update({"ligand": 0.0}, 1.0)["state"]
    assert s_last < 0.2            # no Wnt -> S decays toward 0


def test_high_ligand_sustains_stemness():
    p = _proc()
    s_last = 1.0
    for _ in range(20):
        s_last = p.update({"ligand": 1.0}, 1.0)["state"]
    assert s_last > 0.8            # saturating Wnt -> S held high
