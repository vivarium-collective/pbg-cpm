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

#[test]
fn set_cell_type_relabels_without_disturbing_trackers() {
    use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
    use cpm_core::world::World;
    let lat = Lattice::new([6, 6, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 10.0);
    let a = w.add_cell(1, 9.0, 1.0, 0.0, 0.0);
    for y in 1..4 { for x in 1..4 { let i = w.lattice.index(x, y, 0); w.paint(i, a); } }
    w.recompute_trackers();
    let (v0, s0, com0) = (w.cells[a as usize].volume, w.cells[a as usize].surface, w.com(a));
    w.set_cell_type(a, 7);
    assert_eq!(w.cells[a as usize].cell_type, 7);
    // relabel must not touch volume/surface/COM trackers
    assert_eq!(w.cells[a as usize].volume, v0);
    assert_eq!(w.cells[a as usize].surface, s0);
    assert_eq!(w.com(a), com0);
    w.set_target_volume(a, 42.0);
    assert_eq!(w.cells[a as usize].target_volume, 42.0);
}

#[test]
fn would_stay_connected_flags_local_articulation() {
    use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
    use cpm_core::world::World;
    // 2D Moore neighbourhood
    let lat = Lattice::new([7, 7, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 10.0);
    let c = w.add_cell(1, 9.0, 1.0, 0.0, 0.0);
    // dumbbell: two 1x3 arms joined by a single neck pixel at (3,3)
    for x in 1..3 { let i = w.lattice.index(x, 3, 0); w.paint(i, c); }   // left arm x=1,2
    let neck = w.lattice.index(3, 3, 0); w.paint(neck, c);              // neck x=3
    for x in 4..6 { let i = w.lattice.index(x, 3, 0); w.paint(i, c); }   // right arm x=4,5
    w.recompute_trackers();
    // removing the neck would split the cell -> not safe
    assert!(!w.would_stay_connected(neck, c));
    // removing a tip is safe (its only same-owner neighbour is one pixel)
    let tip = w.lattice.index(1, 3, 0);
    assert!(w.would_stay_connected(tip, c));

    // connectivity flags
    assert!(!w.any_connectivity());
    w.set_connectivity(1, true);
    assert!(w.any_connectivity());
    assert!(w.type_is_constrained(1));
    assert!(!w.type_is_constrained(2));
    w.set_connectivity(1, false);
    w.set_connectivity_medium(true);
    assert!(w.any_connectivity());
}
