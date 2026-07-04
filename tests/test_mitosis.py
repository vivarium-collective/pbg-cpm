import cpm_core


def _build_growing_world():
    w = cpm_core.World((30, 30, 1), "periodic", 2, 10.0)
    a = w.add_cell(1, 25.0, 2.0, 0.0, 0.0)
    w.set_contact(0, 1, 16.0)
    w.seed_block(a, 10, 10, 0, 15, 15, 1)  # 5x5 = 25 sites
    w.finalize(7)
    return w


def test_grow_and_divide_increases_cell_count_and_bounds_volume():
    w = _build_growing_world()
    threshold = 50.0
    reset_target = 25.0
    ceiling = 2 * threshold

    for _ in range(30):
        w.step(5)
        w.grow(1, 2.0)
        w.divide_cells(threshold, reset_target)

    assert w.n_cells() > 1, "growth + division should have increased the cell count"
    for v in w.cell_volumes()[1:]:  # skip medium (index 0)
        assert v < ceiling, f"volume {v} exceeded ceiling {ceiling}"
