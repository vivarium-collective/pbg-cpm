import process_bigraph as pb
from cpm.subcellular.boolean import BooleanSubcell

CFG = {"stemness_threshold": 0.4, "goblet_type": 3, "absorptive_type": 2}


def _proc():
    return BooleanSubcell(CFG, core=pb.allocate_core())


def test_high_stemness_stays():
    assert _proc().update({"state": 0.9, "neighbor_secretory": 0}, 1.0)["fate"] == 0


def test_low_stemness_no_secretory_neighbor_becomes_goblet():
    assert _proc().update({"state": 0.1, "neighbor_secretory": 0}, 1.0)["fate"] == 3


def test_low_stemness_with_secretory_neighbor_becomes_absorptive():
    assert _proc().update({"state": 0.1, "neighbor_secretory": 2}, 1.0)["fate"] == 2
