use crate::world::World;
use crate::CellId;
use std::collections::HashMap;

impl World {
    /// Seed the lattice from a per-pixel segmentation label array (len == n_sites(), same
    /// index order as the lattice: idx = x + y*nx + z*nx*ny). Label 0 == background (stays
    /// Medium). Each distinct nonzero label becomes ONE CPM cell painted over exactly its
    /// pixels, with cell_type = types[label] (defaulting to `default_type` if absent) and the
    /// given volume params. Returns a map label -> assigned CellId. Call before finalize.
    pub fn seed_from_labels(
        &mut self,
        labels: &[u32],
        types: &HashMap<u32, u16>,
        default_type: u16,
        target_volume: f64,
        lambda_volume: f64,
    ) -> HashMap<u32, CellId> {
        assert_eq!(labels.len(), self.lattice.n_sites(), "labels length must equal n_sites");
        let mut seg_to_cell: HashMap<u32, CellId> = HashMap::new();
        for (idx, &seg) in labels.iter().enumerate() {
            if seg == 0 { continue; } // background = Medium
            let cid = *seg_to_cell.entry(seg).or_insert_with(|| {
                let t = types.get(&seg).copied().unwrap_or(default_type);
                self.add_cell(t, target_volume, lambda_volume, 0.0, 0.0)
            });
            self.paint(idx, cid);
        }
        seg_to_cell
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};
    use crate::world::World;
    use crate::MEDIUM;

    fn small_world() -> World {
        let lat = Lattice::new([6, 6, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        World::new(lat, 10.0)
    }

    #[test]
    fn seed_two_segments() {
        let mut w = small_world();
        let n = w.lattice.n_sites();
        let mut labels = vec![0u32; n];
        // segment 5 over a 2x2 block at (0,0)-(1,1)
        for y in 0..2 {
            for x in 0..2 {
                let idx = w.lattice.index(x, y, 0);
                labels[idx] = 5;
            }
        }
        // segment 9 over a distinct 3x1 block at (3..6, 3)
        for x in 3..6 {
            let idx = w.lattice.index(x, 3, 0);
            labels[idx] = 9;
        }
        let mut types = HashMap::new();
        types.insert(5u32, 1u16);
        types.insert(9u32, 2u16);

        let seg_map = w.seed_from_labels(&labels, &types, 1, 25.0, 2.0);
        w.recompute_trackers();

        assert_eq!(seg_map.len(), 2);
        assert_eq!(w.cells.len(), 3); // medium + 2 cells

        let id5 = seg_map[&5];
        let id9 = seg_map[&9];
        let cell5 = &w.cells[id5 as usize];
        let cell9 = &w.cells[id9 as usize];
        assert_eq!(cell5.volume, 4);
        assert_eq!(cell5.cell_type, 1);
        assert_eq!(cell9.volume, 3);
        assert_eq!(cell9.cell_type, 2);

        // background pixels stay MEDIUM
        let bg_idx = w.lattice.index(5, 5, 0);
        assert_eq!(w.lattice.owner(bg_idx), MEDIUM);
    }

    #[test]
    fn default_type_when_missing() {
        let mut w = small_world();
        let n = w.lattice.n_sites();
        let mut labels = vec![0u32; n];
        let idx = w.lattice.index(0, 0, 0);
        labels[idx] = 7; // no entry in types map
        let types = HashMap::new();

        let seg_map = w.seed_from_labels(&labels, &types, 3, 25.0, 2.0);
        w.recompute_trackers();

        let id7 = seg_map[&7];
        assert_eq!(w.cells[id7 as usize].cell_type, 3);
    }

    #[test]
    #[should_panic]
    fn length_mismatch_panics() {
        let mut w = small_world();
        let labels = vec![0u32; 3]; // wrong length
        let types = HashMap::new();
        w.seed_from_labels(&labels, &types, 1, 25.0, 2.0);
    }
}
