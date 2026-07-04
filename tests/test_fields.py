import cpm_core


def test_field_secretion_and_diffusion_through_bindings():
    w = cpm_core.World((10, 10, 1), "noflux", 2, 10.0)
    a = w.add_cell(1, 9.0, 1.0, 0.0, 0.0)
    w.set_contact(0, 1, 8.0)
    w.seed_block(a, 3, 3, 0, 6, 6, 1)  # 3x3 secreting cell

    fi = w.add_field("attr", 0.1, 1e-3)
    w.set_secretion(fi, 1, 100.0)
    w.set_chemotaxis(fi, 1, 0.0)

    w.finalize(11)
    w.step(3)

    conc = w.field_conc(fi)
    assert len(conc) == 100
    assert max(conc) > 0.0, "secretion+diffusion should raise concentration near the cell"

    mean_at_cell = w.field_mean_at_cell(fi, a)
    assert mean_at_cell > 0.0, "mean concentration over the secreting cell's pixels should be positive"
