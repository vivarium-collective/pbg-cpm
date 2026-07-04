from cpm import cpm_core
from cpm.crypt3d import build_crypt3d
from cpm.metrics import (radial_thickness, interior_medium_pockets, connected_components,
                         central_axis_column, open_lumen_depth)


def test_build_crypt3d_is_a_thin_typed_shell():
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d()
    n_cells = len(seg_to_type)
    assert n_cells > 30                       # a real tiling, not a few blobs
    assert type_names == ["Epithelial Stem", "Absorptive", "Goblet"]
    # seed into a world so we can use the metrics
    w = cpm_core.World((nx, ny, nz), "noflux", 2, 5.0)
    m = w.seed_from_labels(labels, seg_to_type, 1, 20.0, 3.0)
    for t in range(1, 4):
        w.set_contact(0, t, 6.0)
        for u in range(t, 4):
            w.set_contact(t, u, 4.0)
    w.finalize(1)
    assert w.n_cells() == n_cells
    # thin shell: at most ~1 cell between lumen and exterior
    mean_t, max_t = radial_thickness(w, nx / 2.0, ny / 2.0)
    assert max_t <= 2 and mean_t < 1.6
    # lumen is an enclosed interior medium pocket
    assert interior_medium_pockets(w) >= 1
    # stem cells are basal (lower z) than goblet cells
    types = w.cell_types()
    coms = w.cell_coms()
    stem_z = [coms[c][2] for c in range(1, len(types)) if types[c] == 1]
    gob_z = [coms[c][2] for c in range(1, len(types)) if types[c] == 3]
    assert stem_z and gob_z and (sum(stem_z) / len(stem_z)) < (sum(gob_z) / len(gob_z))


def test_open_top_crypt_has_an_open_lumen_and_sealed_base():
    # The open-topped crypt (test-tube shape) must be closed at the base and open
    # at the mouth: no cell lid on the central axis in the upper half, a cap cell
    # low on the axis, a deep open lumen, and NO enclosed medium pocket (it drains
    # through the mouth).
    (nx, ny, nz), labels, seg_to_type, type_names = build_crypt3d(open_top=True)
    assert len(seg_to_type) > 30
    assert type_names == ["Epithelial Stem", "Absorptive", "Goblet"]
    w = cpm_core.World((nx, ny, nz), "noflux", 2, 5.0)
    w.seed_from_labels(labels, seg_to_type, 1, 20.0, 3.0)
    w.finalize(1)
    axis = central_axis_column(w)
    assert any(axis[z] != 0 for z in range(nz // 2)), "base should be capped (cell low on axis)"
    assert not [z for z in range(nz // 2, nz) if axis[z] != 0], "mouth should be open (no lid)"
    assert open_lumen_depth(w) >= 0.35 * nz, "lumen should be a deep open cavity"
    assert interior_medium_pockets(w) == 0, "open lumen drains out; it is not an enclosed pocket"
    # still a thin monolayer wall
    mean_t, max_t = radial_thickness(w, nx / 2.0, ny / 2.0)
    assert mean_t < 1.6


def test_every_generated_cell_is_one_connected_component():
    # Regression guard for the (axial_bin, theta_bin) bin-key collision near the
    # cap/cylinder seam, which silently merged two spatially-disjoint voxel blobs
    # under one cell id -- a cell fragmented at MCS 0 that violates the
    # connectivity constraint's invariant before the CPM ever runs.
    # build_crypt3d must hand back only contiguous cells (same 18-adjacency the
    # constraint and cpm.metrics use). No relaxation: this is a pure geometry test.
    (nx, ny, nz), labels, seg_to_type, _ = build_crypt3d()
    w = cpm_core.World((nx, ny, nz), "noflux", 2, 5.0)
    w.seed_from_labels(labels, seg_to_type, 1, 20.0, 3.0)
    w.finalize(1)
    vols = w.cell_volumes()
    bad = [c for c in range(1, w.n_cells() + 1)
           if vols[c] > 0 and connected_components(w, c) != 1]
    assert bad == [], f"cells fragmented straight out of the generator: {bad}"
