use cpm_core::energy::ContactMatrix;
use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::sweep::Cpm;
use cpm_core::world::World;

#[test]
fn incremental_equals_full_recompute_after_random_sweeps() {
    let lat = Lattice::new([25, 25, 1], [Boundary::Periodic; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 12.0);
    for k in 0..6 {
        let a = w.add_cell((1 + k % 2) as u16, 20.0, 2.0, 18.0, 0.5);
        let ox = 2 + (k % 3) * 7;
        let oy = 2 + (k / 3) * 10;
        for y in oy..oy + 4 {
            for x in ox..ox + 4 {
                let i = w.lattice.index(x, y, 0);
                w.paint(i, a);
            }
        }
    }
    let mut m = ContactMatrix::new(3);
    m.set(0, 1, 10.0);
    m.set(0, 2, 10.0);
    m.set(1, 2, 14.0);
    w.set_contact_matrix(m);
    w.recompute_trackers();

    let mut cpm = Cpm::new(w, 99);
    cpm.step(50);

    // snapshot incremental trackers
    let inc: Vec<(i64, i64, [f64; 3])> =
        cpm.world.cells.iter().map(|c| (c.volume, c.surface, c.com_sum)).collect();

    // full recompute and compare
    cpm.world.recompute_trackers();
    for (i, c) in cpm.world.cells.iter().enumerate() {
        assert_eq!(inc[i].0, c.volume, "volume drift cell {i}");
        assert_eq!(inc[i].1, c.surface, "surface drift cell {i}");
        for k in 0..3 {
            assert!((inc[i].2[k] - c.com_sum[k]).abs() < 1e-6, "com drift cell {i}");
        }
    }
}
