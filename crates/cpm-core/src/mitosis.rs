use crate::world::World;
use crate::{CellId, MEDIUM};

impl World {
    /// Increase target_volume of every non-medium cell of `cell_type` by `rate`.
    pub fn grow(&mut self, cell_type: u16, rate: f64) {
        for c in self.cells.iter_mut() {
            if c.id != MEDIUM && c.cell_type == cell_type {
                c.target_volume += rate;
            }
        }
    }

    /// Divide every non-medium cell whose volume >= threshold into two daughters, split by a
    /// plane perpendicular to the cell's longest bounding-box axis through the box midpoint.
    /// The original cell keeps its id (one daughter); a NEW cell (same type/lambdas) is the other.
    /// Both daughters' target_volume is set to `reset_target`. Returns the new daughter ids.
    pub fn divide_cells(&mut self, threshold: f64, reset_target: f64) -> Vec<CellId> {
        // 1. which cells divide (snapshot ids now; volume read from trackers)
        let dividing: Vec<CellId> = self
            .cells
            .iter()
            .filter(|c| c.id != MEDIUM && (c.volume as f64) >= threshold)
            .map(|c| c.id)
            .collect();
        if dividing.is_empty() {
            return Vec::new();
        }

        // 2. pass 1: bounding box per dividing cell -> choose split axis + midpoint.
        // bbox[i] corresponds to dividing[i]; (min[3], max[3]).
        let mut bbox: Vec<([usize; 3], [usize; 3])> =
            vec![([usize::MAX; 3], [0usize; 3]); dividing.len()];
        // map CellId -> index into `dividing` (Vec-indexed by CellId; sparse but bounded by
        // total cell count, avoids HashMap iteration).
        let mut idx_of: Vec<Option<usize>> = vec![None; self.cells.len()];
        for (i, &id) in dividing.iter().enumerate() {
            idx_of[id as usize] = Some(i);
        }

        let n = self.lattice.n_sites();
        for site in 0..n {
            let owner = self.lattice.owner(site);
            if let Some(i) = idx_of[owner as usize] {
                let [x, y, z] = self.lattice.coords(site);
                let coord = [x, y, z];
                let (min, max) = &mut bbox[i];
                for k in 0..3 {
                    if coord[k] < min[k] {
                        min[k] = coord[k];
                    }
                    if coord[k] > max[k] {
                        max[k] = coord[k];
                    }
                }
            }
        }

        // choose axis (largest extent) + midpoint per dividing cell.
        let mut axis_of: Vec<usize> = Vec::with_capacity(dividing.len());
        let mut mid_of: Vec<usize> = Vec::with_capacity(dividing.len());
        for (min, max) in &bbox {
            let mut best_axis = 0usize;
            let mut best_extent = 0usize;
            for k in 0..3 {
                let extent = max[k].saturating_sub(min[k]);
                if extent > best_extent {
                    best_extent = extent;
                    best_axis = k;
                }
            }
            let mid = (min[best_axis] + max[best_axis]) / 2;
            axis_of.push(best_axis);
            mid_of.push(mid);
        }

        // 3. create a new daughter cell per dividing cell (same type + lambdas),
        //    remember parent_id -> new_id (indexed in the same order as `dividing`).
        let mut new_ids: Vec<CellId> = Vec::with_capacity(dividing.len());
        for &parent_id in &dividing {
            let parent = &self.cells[parent_id as usize];
            let (cell_type, lambda_volume, target_surface, lambda_surface) =
                (parent.cell_type, parent.lambda_volume, parent.target_surface, parent.lambda_surface);
            let new_id = self.add_cell(cell_type, reset_target, lambda_volume, target_surface, lambda_surface);
            new_ids.push(new_id);
        }

        // 4. pass 2: iterate all sites; if owner is a dividing cell and coords[axis] > mid,
        //    set_owner(site, new_id_for_that_parent).
        for site in 0..n {
            let owner = self.lattice.owner(site);
            if let Some(i) = idx_of[owner as usize] {
                let [x, y, z] = self.lattice.coords(site);
                let coord = [x, y, z];
                if coord[axis_of[i]] > mid_of[i] {
                    self.lattice.set_owner(site, new_ids[i]);
                }
            }
        }

        // 5. set BOTH daughters' target_volume = reset_target (parent + each new cell).
        for &parent_id in &dividing {
            self.cells[parent_id as usize].target_volume = reset_target;
        }
        // (new daughters already created with target_volume = reset_target above)

        // 6. recompute_trackers()
        self.recompute_trackers();

        // 7. return the list of new ids (in the order created)
        new_ids
    }
}

#[cfg(test)]
mod tests {
    use crate::lattice::{Boundary, Lattice, Neighborhood};
    use crate::world::World;

    fn small_world(dims: [usize; 3]) -> World {
        let lat = Lattice::new(dims, [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        World::new(lat, 10.0)
    }

    #[test]
    fn grow_only_affects_matching_type() {
        let mut w = small_world([5, 5, 1]);
        let a = w.add_cell(1, 25.0, 2.0, 20.0, 0.0);
        let b = w.add_cell(2, 25.0, 2.0, 20.0, 0.0);
        w.grow(1, 3.0);
        assert_eq!(w.cells[a as usize].target_volume, 28.0);
        assert_eq!(w.cells[b as usize].target_volume, 25.0);
    }

    #[test]
    fn divide_one_cell_conserves_mass_and_resets_target() {
        let mut w = small_world([12, 10, 1]);
        let a = w.add_cell(1, 48.0, 2.0, 20.0, 0.0);
        // paint an 8x6 block -> volume 48
        for y in 1..7 {
            for x in 1..9 {
                let idx = w.lattice.index(x, y, 0);
                w.paint(idx, a);
            }
        }
        w.recompute_trackers();
        assert_eq!(w.cells[a as usize].volume, 48);

        let n_cells_before = w.cells.len();
        let new_ids = w.divide_cells(40.0, 24.0);
        assert_eq!(new_ids.len(), 1);
        assert_eq!(w.cells.len(), n_cells_before + 1);

        let vol_a = w.cells[a as usize].volume;
        let vol_new = w.cells[new_ids[0] as usize].volume;
        assert_eq!(vol_a + vol_new, 48, "mass must be conserved on division");
        assert!(vol_a > 0 && vol_a >= 16 && vol_a <= 32, "vol_a = {vol_a}");
        assert!(vol_new > 0 && vol_new >= 16 && vol_new <= 32, "vol_new = {vol_new}");
        assert_eq!(w.cells[a as usize].target_volume, 24.0);
        assert_eq!(w.cells[new_ids[0] as usize].target_volume, 24.0);
    }

    #[test]
    fn split_is_perpendicular_to_long_axis() {
        let mut w = small_world([16, 8, 1]);
        let a = w.add_cell(1, 48.0, 2.0, 20.0, 0.0);
        // 12 wide (x) x 4 tall (y): long axis is x.
        for y in 1..5 {
            for x in 1..13 {
                let idx = w.lattice.index(x, y, 0);
                w.paint(idx, a);
            }
        }
        w.recompute_trackers();

        let new_ids = w.divide_cells(40.0, 24.0);
        assert_eq!(new_ids.len(), 1);

        let com_a = w.com(a);
        let com_b = w.com(new_ids[0]);
        let dx = (com_a[0] - com_b[0]).abs();
        let dy = (com_a[1] - com_b[1]).abs();
        assert!(dx > dy, "expected split along long (x) axis: dx={dx} dy={dy}");
    }

    #[test]
    fn below_threshold_does_not_divide() {
        let mut w = small_world([8, 8, 1]);
        let a = w.add_cell(1, 20.0, 2.0, 20.0, 0.0);
        for y in 1..5 {
            for x in 1..6 {
                let idx = w.lattice.index(x, y, 0);
                w.paint(idx, a);
            }
        }
        w.recompute_trackers();
        assert_eq!(w.cells[a as usize].volume, 20);

        let n_cells_before = w.cells.len();
        let new_ids = w.divide_cells(40.0, 12.0);
        assert!(new_ids.is_empty());
        assert_eq!(w.cells.len(), n_cells_before);
    }
}
