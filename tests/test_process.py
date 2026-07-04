import process_bigraph as pb

from cpm.processes.cpm_process import CPMProcess


SPEC = {
    "potts": {"dims": [24, 24, 1], "boundary": "periodic",
              "neighbor_order": 2, "temperature": 12.0, "seed": 3},
    "cell_types": [{"name": "medium", "id": 0}, {"name": "a", "id": 1}],
    "contact": [{"a": 0, "b": 1, "j": 8.0}, {"a": 1, "b": 1, "j": 2.0}],
    "cells": [{"type": 1, "target_volume": 16, "lambda_volume": 2.0,
               "target_surface": 16, "lambda_surface": 0.0,
               "seed_block": [4, 4, 0, 10, 10, 1]}],
}


def test_process_update_returns_readback():
    core = pb.allocate_core()
    proc = CPMProcess({"spec": SPEC, "mcs_per_update": 5}, core=core)
    out = proc.update({}, 1.0)
    assert "cell_volumes" in out
    assert len(out["cell_volumes"]) == 2   # medium + 1 cell
    assert out["cell_volumes"][1] > 0
