import cpm_core


def test_single_cell_shrinks_to_target():
    w = cpm_core.World((30, 30, 1), "periodic", 2, 15.0)
    a = w.add_cell(1, 49.0, 5.0, 28.0, 0.0)
    w.set_contact(0, 1, 6.0)
    w.seed_block(a, 5, 5, 0, 15, 15, 1)   # 100 sites
    w.finalize(7)
    start = w.cell_volumes()[a]
    w.step(200)
    end = w.cell_volumes()[a]
    assert end < start
    assert abs(end - 49) < 30


def test_snapshot_shape_and_determinism():
    def run():
        w = cpm_core.World((20, 20, 1), "periodic", 2, 10.0)
        a = w.add_cell(1, 25.0, 2.0, 20.0, 0.0)
        w.set_contact(0, 1, 16.0)
        w.seed_block(a, 5, 5, 0, 10, 10, 1)
        w.finalize(42)
        w.step(5)
        return w.snapshot()
    s = run()
    assert len(s) == 400
    assert run() == s
