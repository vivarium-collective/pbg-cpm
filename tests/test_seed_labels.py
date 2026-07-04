from cpm import cpm_core


def test_seed_from_labels_two_blocks():
    w = cpm_core.World((8, 8, 1), "noflux", 2, 10.0)
    labels = [0] * 64

    def idx(x, y):
        return x + y * 8

    # segment 1: 3x3 block at (0,0)-(2,2)
    for y in range(3):
        for x in range(3):
            labels[idx(x, y)] = 1
    # segment 2: 3x3 block at (5,5)-(7,7)
    for y in range(5, 8):
        for x in range(5, 8):
            labels[idx(x, y)] = 2

    w.set_contact(0, 1, 6.0)
    w.set_contact(0, 2, 6.0)
    w.set_contact(1, 2, 10.0)

    seg_map = w.seed_from_labels(labels, {1: 1, 2: 2}, 1, 25.0, 2.0)

    assert len(seg_map) == 2
    assert w.n_cells() == 2

    w.finalize(3)
    volumes = w.cell_volumes()
    assert volumes[seg_map[1]] == 9
    assert volumes[seg_map[2]] == 9

    w.step(5)  # runs without error
