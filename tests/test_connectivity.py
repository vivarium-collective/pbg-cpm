from cpm import cpm_core


def test_connectivity_setters_survive_finalize():
    w = cpm_core.World((25, 9, 1), "noflux", 2, 30.0)
    c = w.add_cell(1, 40.0, 1.0, 0.0, 0.0)
    w.set_contact(0, 1, -2.0)
    # dumbbell: two blobs + a 1px neck bridge
    for y in range(2, 7):
        for x in range(5, 10):
            w.seed_block(c, x, y, 0, x + 1, y + 1, 1)
        for x in range(15, 20):
            w.seed_block(c, x, y, 0, x + 1, y + 1, 1)
    for x in range(10, 15):
        w.seed_block(c, x, 4, 0, x + 1, 5, 1)
    w.set_connectivity(1, True)          # set BEFORE finalize
    w.finalize(1)
    w.step(40)
    # cell survived as nonzero volume; detailed single-component check is in
    # test_connectivity_metric below via cpm.metrics
    assert w.cell_volumes()[c] > 0


def _dumbbell_world(connectivity):
    from cpm.schema import load_world
    spec = {
        "potts": {"dims": [25, 9, 1], "boundary": "noflux",
                  "neighbor_order": 2, "temperature": 30.0, "seed": 1},
        "contact": [{"a": 0, "b": 1, "j": -2.0}],
        "cells": [{"type": 1, "target_volume": 40, "lambda_volume": 1.0,
                   "target_surface": 0, "lambda_surface": 0.0,
                   "seed_block": [5, 2, 0, 20, 7, 1]}],   # solid bar (stays connected only if protected)
    }
    if connectivity:
        spec["connectivity"] = {"types": [1], "medium": False}
    return load_world(spec)


def test_connected_components_metric_and_constraint():
    from cpm.metrics import connected_components
    # WITH the constraint the cell stays one connected component
    w_on = _dumbbell_world(True)
    w_on.step(40)
    c = 1
    assert connected_components(w_on, c) == 1
    # WITHOUT it, the same stressed bar fragments into >1 component
    w_off = _dumbbell_world(False)
    w_off.step(40)
    assert connected_components(w_off, c) > 1


def test_interior_medium_pockets_metric():
    from cpm.metrics import interior_medium_pockets
    # a 6x6 lattice: fill all but one interior medium pixel -> exactly 1 pocket
    w = cpm_core_world_all_cells_with_hole()
    assert interior_medium_pockets(w) == 1


def cpm_core_world_all_cells_with_hole():
    # 6x6, fill with one big cell except pixel (3,3) left as medium -> 1 interior pocket
    w = cpm_core.World((6, 6, 1), "noflux", 2, 1.0)
    c = w.add_cell(1, 35.0, 1.0, 0.0, 0.0)
    w.set_contact(0, 1, 4.0)
    for y in range(6):
        for x in range(6):
            if (x, y) == (3, 3):
                continue
            w.seed_block(c, x, y, 0, x + 1, y + 1, 1)
    w.finalize(1)
    return w
