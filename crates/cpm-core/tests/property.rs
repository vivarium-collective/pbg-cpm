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

#[test]
fn connectivity_keeps_cell_in_one_piece_under_stress() {
    use cpm_core::energy::ContactMatrix;
    use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
    use cpm_core::sweep::Cpm;
    use cpm_core::world::World;

    // Count connected components of a cell's pixels via a global flood-fill
    // (Moore adjacency), for the test only.
    fn components(w: &World, cid: u32) -> usize {
        let [nx, ny, nz] = [w.lattice.dims_x(), w.lattice.dims_y(), w.lattice.dims_z()];
        let sites: Vec<usize> = (0..w.lattice.n_sites())
            .filter(|&i| w.lattice.owner(i) == cid)
            .collect();
        let inset: std::collections::HashSet<usize> = sites.iter().copied().collect();
        let mut seen = std::collections::HashSet::new();
        let mut comps = 0;
        for &s in &sites {
            if seen.contains(&s) { continue; }
            comps += 1;
            let mut stack = vec![s];
            seen.insert(s);
            while let Some(c) = stack.pop() {
                let cz = c / (nx * ny);
                let cy = (c % (nx * ny)) / nx;
                let cx = c % nx;
                for dz in -1i64..=1 { for dy in -1i64..=1 { for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    let (x2, y2, z2) = (cx as i64 + dx, cy as i64 + dy, cz as i64 + dz);
                    if x2 < 0 || y2 < 0 || z2 < 0 || x2 >= nx as i64 || y2 >= ny as i64 || z2 >= nz as i64 { continue; }
                    let n = x2 as usize + y2 as usize * nx + z2 as usize * nx * ny;
                    if inset.contains(&n) && !seen.contains(&n) { seen.insert(n); stack.push(n); }
                }}}
            }
        }
        comps
    }

    fn dumbbell(connectivity: bool) -> usize {
        let lat = Lattice::new([25, 9, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 30.0); // high temperature -> stress
        let c = w.add_cell(1, 40.0, 1.0, 0.0, 0.0);
        // two 5x5 blobs joined by a single 1px neck at x=12,y=4
        for y in 2..7 { for x in 5..10 { let i = w.lattice.index(x, y, 0); w.paint(i, c); } }
        for y in 2..7 { for x in 15..20 { let i = w.lattice.index(x, y, 0); w.paint(i, c); } }
        let neck = w.lattice.index(12, 4, 0); w.paint(neck, c);
        // bridge the neck to both blobs so it starts connected
        for x in 10..12 { let i = w.lattice.index(x, 4, 0); w.paint(i, c); }
        for x in 13..15 { let i = w.lattice.index(x, 4, 0); w.paint(i, c); }
        let mut m = ContactMatrix::new(2);
        m.set(0, 1, -2.0); // negative medium adhesion -> cell wants boundary -> shreds
        w.set_contact_matrix(m);
        w.recompute_trackers();
        if connectivity { w.set_connectivity(1, true); }
        let mut cpm = Cpm::new(w, 1);
        cpm.step(40);
        components(&cpm.world, c)
    }

    // Without the constraint the stressed dumbbell fragments; with it, it stays whole.
    assert!(dumbbell(false) > 1, "stress must actually fragment (guard against vacuous test)");
    assert_eq!(dumbbell(true), 1, "connectivity must keep the cell in one piece");
}
