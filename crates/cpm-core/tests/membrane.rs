use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::membrane::build_distance_field;
use cpm_core::sweep::Cpm;
use cpm_core::world::World;

const DIMS: [usize; 3] = [12, 12, 8];

fn anchors_z0() -> Vec<usize> {
    // the whole z=0 plane is the membrane surface
    let (nx, ny) = (DIMS[0], DIMS[1]);
    (0..nx * ny).collect()
}

fn mean_membrane_dist(cpm: &Cpm, cell: u32, dist: &[f32]) -> f64 {
    let n = cpm.world.lattice.n_sites();
    let (mut sum, mut cnt) = (0.0f64, 0u64);
    for i in 0..n {
        if cpm.world.lattice.owner(i) == cell {
            sum += dist[i] as f64;
            cnt += 1;
        }
    }
    if cnt == 0 { 0.0 } else { sum / cnt as f64 }
}

fn run(anchored: bool) -> f64 {
    let anchors = anchors_z0();
    let lat = Lattice::new(DIMS, [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    let mut w = World::new(lat, 20.0); // hot
    // a 64-voxel type-1 cell seeded as a 4x4x4 block sitting a bit above z=0
    let a = w.add_cell(1, 64.0, 1.0, 0.0, 0.0);
    for z in 2..6 {
        for y in 4..8 {
            for x in 4..8 {
                let i = w.lattice.index(x, y, z);
                w.paint(i, a);
            }
        }
    }
    w.set_membrane(&anchors, 4.0, 2.0);
    if anchored {
        w.set_membrane_anchored(1, true);
    }
    w.recompute_trackers();
    let mut cpm = Cpm::new(w, 7);
    cpm.step(40);
    let dist = build_distance_field(DIMS, &anchors);
    mean_membrane_dist(&cpm, a, &dist)
}

#[test]
fn membrane_anchor_holds_cell_near_surface() {
    let anchored = run(true);
    let free = run(false);
    // z=0-plane anchors -> membrane distance == z. band=2, so an anchored cell
    // settles into z<=~2-3; a free hot cell wanders further up.
    assert!(anchored <= 3.5, "anchored cell drifted off the membrane: {anchored}");
    assert!(free > anchored + 0.5, "membrane made no difference: free {free} vs anchored {anchored}");
}

#[test]
fn membrane_run_is_deterministic() {
    assert_eq!(run(true), run(true));
}
