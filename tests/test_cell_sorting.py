import json
import os
from cpm.schema import load_world
from cpm.metrics import heterotypic_boundary

DEMO = os.path.join(os.path.dirname(__file__), "..", "demos", "cell_sorting_2d.json")


def _build_spec():
    with open(DEMO) as f:
        spec = json.load(f)
    cells = []
    n = 5           # 5x5 grid
    w = 8           # cell width == stride: cells touch (no medium gap) so
                    # the checkerboard has real initial heterotypic contact
    for gy in range(n):
        for gx in range(n):
            t = 1 + ((gx + gy) % 2)
            x0, y0 = 2 + gx * w, 2 + gy * w
            cells.append({
                "type": t, "target_volume": 64, "lambda_volume": 2.0,
                "target_surface": 40, "lambda_surface": 0.0,
                "seed_block": [x0, y0, 0, x0 + 8, y0 + 8, 1],
            })
    spec["cells"] = cells
    return spec


def test_cell_sorting_reduces_heterotypic_boundary():
    spec = _build_spec()
    world = load_world(spec)
    start = heterotypic_boundary(world)
    world.step(400)
    end = heterotypic_boundary(world)
    assert end < start, f"sorting should reduce heterotypic boundary: {start} -> {end}"
    assert end < 0.85 * start


def test_cell_sorting_deterministic():
    spec = _build_spec()
    w1 = load_world(spec); w1.step(50)
    w2 = load_world(spec); w2.step(50)
    assert w1.snapshot() == w2.snapshot()
