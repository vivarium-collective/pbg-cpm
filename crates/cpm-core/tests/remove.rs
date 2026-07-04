use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::world::World;

#[test]
fn remove_cells_clears_voxels_and_rebuilds_trackers() {
    let lat = Lattice::new([6, 6, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 10.0);
    let a = w.add_cell(1, 9.0, 1.0, 0.0, 0.0);
    let b = w.add_cell(1, 9.0, 1.0, 0.0, 0.0);
    for y in 1..4 {
        w.paint(w.lattice.index(1, y, 0), a);
        w.paint(w.lattice.index(3, y, 0), b);
    }
    w.recompute_trackers();
    assert_eq!(w.cells[a as usize].volume, 3);
    assert_eq!(w.cells[b as usize].volume, 3);

    w.remove_cells(&[a]);

    // a's voxels are now medium; b is untouched; trackers rebuilt.
    assert_eq!(w.cells[a as usize].volume, 0, "sloughed cell must be gone");
    assert_eq!(w.cells[b as usize].volume, 3, "the other cell is untouched");
    for y in 1..4 {
        assert_eq!(w.lattice.owner(w.lattice.index(1, y, 0)), 0, "voxel became medium");
    }
}
