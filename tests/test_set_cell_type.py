import cpm_core


def test_set_cell_type_changes_only_type():
    w = cpm_core.World((10, 10, 1), "periodic", 2, 10.0)
    c = w.add_cell(1, 16.0, 2.0, 0.0, 0.0)
    w.set_contact(0, 1, 8.0); w.set_contact(0, 2, 8.0); w.set_contact(1, 1, 2.0)
    w.seed_block(c, 2, 2, 0, 6, 6, 1)
    w.finalize(1)
    vol_before = w.cell_volumes()[c]
    w.set_cell_type(c, 2)
    assert w.cell_types()[c] == 2
    assert w.cell_volumes()[c] == vol_before  # relabel does not move mass
    w.set_target_volume(c, 30.0)              # smoke: no error, cell still alive
    w.step(1)
    assert w.cell_volumes()[c] > 0
