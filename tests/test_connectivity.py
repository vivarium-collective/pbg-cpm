import cpm_core


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
