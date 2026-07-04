//! Cell-cell junction (anti-gap) predicate for the CPM sweep. A "pinch" is a
//! medium voxel whose two opposite axis-neighbours are both junction-enabled
//! cells of DIFFERENT ids -- a thin medium film/gap between two bonded cells.
//! `axis_is_pinch` is the pure core; `World` supplies the per-site geometry.

/// One axis is pinched iff both sides are non-medium (`id != 0`) junction-enabled
/// (`true`) cells with different ids. Each side is (owner_id, is_junction_type);
/// medium is (0, false).
#[inline]
pub fn axis_is_pinch(a: (u32, bool), b: (u32, bool)) -> bool {
    let (ia, ja) = a;
    let (ib, jb) = b;
    ia != 0 && ib != 0 && ja && jb && ia != ib
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};
    use crate::world::World;

    #[test]
    fn axis_pinch_only_between_two_different_junction_cells() {
        assert!(axis_is_pinch((1, true), (2, true)));      // two diff junction cells
        assert!(!axis_is_pinch((1, true), (1, true)));     // same cell (a hole; E1's job)
        assert!(!axis_is_pinch((1, true), (0, false)));    // one side medium
        assert!(!axis_is_pinch((1, false), (2, true)));    // one side not junction-enabled
        assert!(!axis_is_pinch((0, false), (0, false)));   // free medium
    }

    #[test]
    fn delta_junction_opens_and_closes_gaps() {
        // 3x1x1: cell A | medium | cell B, both type 1 (junction-enabled), lambda 2.
        let lat = Lattice::new([3, 1, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 1.0, 1.0, 0.0, 0.0);
        let b = w.add_cell(1, 1.0, 1.0, 0.0, 0.0);
        w.paint(0, a);
        w.paint(2, b);                       // site 1 = medium -> a pinch exists
        w.set_junction(1, true);
        w.set_junction_lambda(2.0);
        w.recompute_trackers();

        // closing the gap (fill site 1 with A) removes the pinch -> favourable (-lambda)
        assert!((w.delta_junction(1, a) + 2.0).abs() < 1e-9);

        // fill it, then opening it back (A -> medium) creates the pinch -> +lambda
        w.paint(1, a);                       // A | A | B, no gap
        assert!((w.delta_junction(1, crate::MEDIUM) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn delta_junction_zero_on_free_surface() {
        // 3x1x1: cell A | medium | medium. Filling site 1 touches no pinch.
        let lat = Lattice::new([3, 1, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 1.0, 1.0, 0.0, 0.0);
        w.paint(0, a);                       // sites 1,2 medium
        w.set_junction(1, true);
        w.set_junction_lambda(2.0);
        w.recompute_trackers();
        assert_eq!(w.delta_junction(1, a), 0.0);   // no cell on the far side -> no pinch
    }

    #[test]
    fn any_junction_requires_lambda_and_a_type() {
        let lat = Lattice::new([3, 1, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        w.set_junction(1, true);
        assert!(!w.any_junction(), "lambda 0 -> off");
        w.set_junction_lambda(2.0);
        assert!(w.any_junction());
        w.set_junction(1, false);
        assert!(!w.any_junction(), "no type enabled -> off");
    }
}
