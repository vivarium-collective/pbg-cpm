from cpm.schema import load_world


def test_load_world_builds_and_steps():
    spec = {
        "potts": {"dims": [24, 24, 1], "boundary": "periodic",
                  "neighbor_order": 2, "temperature": 12.0, "seed": 5},
        "cell_types": [{"name": "medium", "id": 0}, {"name": "a", "id": 1}],
        "contact": [{"a": 0, "b": 1, "j": 8.0}, {"a": 1, "b": 1, "j": 2.0}],
        "cells": [
            {"type": 1, "target_volume": 16, "lambda_volume": 2.0,
             "target_surface": 16, "lambda_surface": 0.0,
             "seed_block": [4, 4, 0, 10, 10, 1]},
        ],
    }
    w = load_world(spec)
    v0 = w.cell_volumes()[1]
    w.step(50)
    v1 = w.cell_volumes()[1]
    assert v0 == 36           # 6x6 seed block
    assert v1 != v0           # dynamics ran


def test_load_world_builds_fields():
    from cpm.schema import load_world
    spec = {
        "potts": {"dims": [20, 20, 1], "boundary": "noflux",
                  "neighbor_order": 2, "temperature": 10.0, "seed": 1},
        "contact": [{"a": 0, "b": 1, "j": 8.0}, {"a": 1, "b": 1, "j": 2.0}],
        "cells": [{"type": 1, "target_volume": 16, "lambda_volume": 2.0,
                   "target_surface": 0, "lambda_surface": 0.0,
                   "seed_block": [6, 6, 0, 12, 12, 1]}],
        "fields": [{"name": "Wnt", "d": 0.1, "decay": 0.05,
                    "secretion": [{"type": 1, "rate": 5.0}],
                    "chemotaxis": []}],
    }
    w = load_world(spec)
    w.step(3)
    conc = w.field_conc(0)
    assert max(conc) > 0.0            # the type-1 cell secreted Wnt
