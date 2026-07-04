import process_bigraph as pb
from cpm.processes.cpm_process import CPMProcess

SPEC = {
    "potts": {"dims": [24, 24, 1], "boundary": "periodic",
              "neighbor_order": 2, "temperature": 10.0, "seed": 3},
    "contact": [{"a": 0, "b": 1, "j": 8.0}, {"a": 0, "b": 2, "j": 8.0},
                {"a": 1, "b": 1, "j": 2.0}, {"a": 2, "b": 2, "j": 2.0}],
    "cells": [{"type": 1, "target_volume": 25, "lambda_volume": 2.0,
               "target_surface": 0, "lambda_surface": 0.0,
               "seed_block": [8, 8, 0, 14, 14, 1]}],
    "fields": [{"name": "L", "d": 0.1, "decay": 0.02,
                "secretion": [{"type": 1, "rate": 3.0}], "chemotaxis": []}],
}


def test_outputs_expose_per_cell_readouts():
    core = pb.allocate_core()
    proc = CPMProcess({"spec": SPEC, "mcs_per_update": 3, "n_fields": 1}, core=core)
    out = proc.update({}, 1.0)
    assert len(out["types"]) == 2 and out["types"][1] == 1
    assert out["field_at_cell"][1] >= 0.0
    assert len(out["positions"]) == 2 and len(out["positions"][1]) == 3


def test_fates_input_switches_cell_type():
    core = pb.allocate_core()
    proc = CPMProcess({"spec": SPEC, "mcs_per_update": 1, "n_fields": 1}, core=core)
    proc.update({}, 1.0)
    # cell 1 -> type 2
    out = proc.update({"fates": [0, 2]}, 1.0)
    assert out["types"][1] == 2
