//! Basement-membrane anchor for the CPM sweep. `build_distance_field` is a pure
//! multi-source BFS (6-neighbour graph distance) from a fixed set of anchor
//! voxels; `cost` is the per-voxel anchor penalty (0 within `band`, quadratic
//! beyond). `World::delta_membrane` uses them to bias the Metropolis accept test
//! so anchored cells stay in a thin band hugging the membrane. RNG-free.
use std::collections::VecDeque;

/// Distance (6-neighbour graph steps) from every voxel to the nearest anchor.
/// Voxels unreachable from any anchor (only when `anchors` is empty) are +inf.
pub fn build_distance_field(dims: [usize; 3], anchors: &[usize]) -> Vec<f32> {
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let n = nx * ny * nz;
    let mut dist = vec![f32::INFINITY; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &a in anchors {
        if a < n && dist[a] != 0.0 {
            dist[a] = 0.0;
            queue.push_back(a);
        }
    }
    while let Some(v) = queue.pop_front() {
        let d = dist[v] + 1.0;
        let z = v / (nx * ny);
        let rem = v % (nx * ny);
        let y = rem / nx;
        let x = rem % nx;
        let mut nb: Vec<usize> = Vec::with_capacity(6);
        if x + 1 < nx { nb.push(v + 1); }
        if x >= 1 { nb.push(v - 1); }
        if y + 1 < ny { nb.push(v + nx); }
        if y >= 1 { nb.push(v - nx); }
        if z + 1 < nz { nb.push(v + nx * ny); }
        if z >= 1 { nb.push(v - nx * ny); }
        for w in nb {
            if d < dist[w] {
                dist[w] = d;
                queue.push_back(w);
            }
        }
    }
    dist
}

/// Anchor cost for a voxel at membrane distance `d`: 0 within `band`, quadratic
/// beyond, scaled by stiffness `k`.
#[inline]
pub fn cost(d: f32, k: f64, band: f64) -> f64 {
    let over = d as f64 - band;
    if over > 0.0 { k * over * over } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};
    use crate::world::World;

    #[test]
    fn bfs_line_distance() {
        assert_eq!(build_distance_field([5, 1, 1], &[0]), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bfs_two_anchors_take_min() {
        assert_eq!(build_distance_field([5, 1, 1], &[0, 4]), vec![0.0, 1.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn bfs_empty_anchors_all_infinite() {
        assert!(build_distance_field([3, 1, 1], &[]).iter().all(|x| x.is_infinite()));
    }

    #[test]
    fn cost_zero_within_band_quadratic_beyond() {
        assert_eq!(cost(1.0, 5.0, 2.0), 0.0);
        assert_eq!(cost(2.0, 5.0, 2.0), 0.0);
        assert_eq!(cost(4.0, 5.0, 2.0), 20.0); // 5 * (4-2)^2
    }

    #[test]
    fn delta_membrane_sign_and_gating() {
        // 5x1x1 line; membrane anchored at index 0 -> dist = [0,1,2,3,4].
        // band=1, k=1 so cost(d) = max(0, d-1)^2.
        let lat = Lattice::new([5, 1, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 5.0, 1.0, 0.0, 0.0);   // type 1
        let b = w.add_cell(2, 5.0, 1.0, 0.0, 0.0);   // type 2 (left unanchored)
        w.paint(0, a);
        w.set_membrane(&[0], 1.0, 1.0);
        w.set_membrane_anchored(1, true);            // only type 1 feels the anchor
        w.recompute_trackers();

        // Site 4 (dist 4) currently medium. Assigning it to `a` (anchored) costs
        // cost(4)-cost(medium) = (4-1)^2 - 0 = 9 > 0.
        assert!((w.delta_membrane(4, a) - 9.0).abs() < 1e-9);
        // Assigning site 4 to `b` (unanchored) costs 0.
        assert_eq!(w.delta_membrane(4, b), 0.0);
        // A site within band (dist 1) assigned to `a` costs 0.
        assert_eq!(w.delta_membrane(1, a), 0.0);
        // No type anchored -> any_membrane false.
        let mut w2 = w;
        w2.set_membrane_anchored(1, false);
        assert!(!w2.any_membrane());
    }
}
