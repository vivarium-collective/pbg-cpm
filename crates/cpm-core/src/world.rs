use crate::lattice::Lattice;
use crate::{CellId, MEDIUM};

#[derive(Clone, Debug)]
pub struct Cell {
    pub id: CellId,
    pub cell_type: u16,
    pub volume: i64,
    pub surface: i64,
    pub com_sum: [f64; 3],
    pub target_volume: f64,
    pub lambda_volume: f64,
    pub target_surface: f64,
    pub lambda_surface: f64,
}

pub struct World {
    pub lattice: Lattice,
    pub cells: Vec<Cell>,
    pub temperature: f64,
    pub contact: crate::energy::ContactMatrix,
    pub fields: Vec<crate::field::Field>,
    pub connectivity_types: Vec<bool>,
    pub connectivity_medium: bool,
}

impl World {
    pub fn new(lattice: Lattice, temperature: f64) -> World {
        let medium = Cell {
            id: MEDIUM,
            cell_type: 0,
            volume: 0,
            surface: 0,
            com_sum: [0.0; 3],
            target_volume: 0.0,
            lambda_volume: 0.0,
            target_surface: 0.0,
            lambda_surface: 0.0,
        };
        World {
            lattice,
            cells: vec![medium],
            temperature,
            contact: crate::energy::ContactMatrix::new(1),
            fields: Vec::new(),
            connectivity_types: Vec::new(),
            connectivity_medium: false,
        }
    }

    pub fn set_contact_matrix(&mut self, m: crate::energy::ContactMatrix) {
        self.contact = m;
    }

    pub fn set_connectivity(&mut self, cell_type: u16, on: bool) {
        let t = cell_type as usize;
        if t >= self.connectivity_types.len() {
            self.connectivity_types.resize(t + 1, false);
        }
        self.connectivity_types[t] = on;
    }

    pub fn set_connectivity_medium(&mut self, on: bool) {
        self.connectivity_medium = on;
    }

    pub fn any_connectivity(&self) -> bool {
        self.connectivity_medium || self.connectivity_types.iter().any(|&b| b)
    }

    pub fn type_is_constrained(&self, cell_type: u16) -> bool {
        self.connectivity_types
            .get(cell_type as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Would removing pixel `site` keep cell `target` locally connected?
    /// Local test over `site`'s same-owner neighbours; O(neighbourhood^2).
    pub fn would_stay_connected(&self, site: usize, target: CellId) -> bool {
        let members: Vec<usize> = self
            .lattice
            .neighbors(site)
            .into_iter()
            .filter(|&n| self.lattice.owner(n) == target)
            .collect();
        if members.len() <= 1 {
            return true;
        }
        let adj = |a: usize, b: usize| self.lattice.neighbors(a).contains(&b);
        crate::connectivity::count_components(&members, &adj) == 1
    }

    pub fn add_cell(
        &mut self,
        cell_type: u16,
        target_volume: f64,
        lambda_volume: f64,
        target_surface: f64,
        lambda_surface: f64,
    ) -> CellId {
        let id = self.cells.len() as CellId;
        self.cells.push(Cell {
            id,
            cell_type,
            volume: 0,
            surface: 0,
            com_sum: [0.0; 3],
            target_volume,
            lambda_volume,
            target_surface,
            lambda_surface,
        });
        id
    }

    pub fn paint(&mut self, idx: usize, c: CellId) {
        self.lattice.set_owner(idx, c);
    }

    /// Relabel a live cell's type. Type affects only contact energy, never the
    /// volume/surface/COM trackers, so this is a pure O(1) label write.
    pub fn set_cell_type(&mut self, cell_id: CellId, new_type: u16) {
        self.cells[cell_id as usize].cell_type = new_type;
    }

    /// Set a cell's target volume (per-cell growth control).
    pub fn set_target_volume(&mut self, cell_id: CellId, v: f64) {
        self.cells[cell_id as usize].target_volume = v;
    }

    pub fn recompute_trackers(&mut self) {
        for cell in self.cells.iter_mut() {
            cell.volume = 0;
            cell.surface = 0;
            cell.com_sum = [0.0; 3];
        }
        let n = self.lattice.n_sites();
        for idx in 0..n {
            let owner = self.lattice.owner(idx);
            let [x, y, z] = self.lattice.coords(idx);
            {
                let cell = &mut self.cells[owner as usize];
                cell.volume += 1;
                cell.com_sum[0] += x as f64;
                cell.com_sum[1] += y as f64;
                cell.com_sum[2] += z as f64;
            }
            let mut unlike = 0i64;
            for nidx in self.lattice.neighbors(idx) {
                if self.lattice.owner(nidx) != owner {
                    unlike += 1;
                }
            }
            self.cells[owner as usize].surface += unlike;
        }
    }

    pub fn com(&self, c: CellId) -> [f64; 3] {
        let cell = &self.cells[c as usize];
        if cell.volume == 0 {
            return [0.0; 3];
        }
        let v = cell.volume as f64;
        [cell.com_sum[0] / v, cell.com_sum[1] / v, cell.com_sum[2] / v]
    }

    pub fn surface_deltas(&self, site: usize, new_owner: CellId) -> Vec<(CellId, i64)> {
        let a = self.lattice.owner(site);
        let b = new_owner;
        if a == b {
            return Vec::new();
        }
        let mut acc: Vec<(CellId, i64)> = Vec::new();
        fn entry(acc: &mut Vec<(CellId, i64)>, c: CellId) -> &mut i64 {
            if let Some(pos) = acc.iter().position(|&(k, _)| k == c) {
                &mut acc[pos].1
            } else {
                acc.push((c, 0));
                let last = acc.len() - 1;
                &mut acc[last].1
            }
        }
        let neighbors = self.lattice.neighbors(site);
        // Site term
        let mut unlike_a = 0i64;
        let mut unlike_b = 0i64;
        for &q in &neighbors {
            let c = self.lattice.owner(q);
            if c != a { unlike_a += 1; }
            if c != b { unlike_b += 1; }
        }
        *entry(&mut acc, a) -= unlike_a;
        *entry(&mut acc, b) += unlike_b;
        // Neighbor term
        for &q in &neighbors {
            let c = self.lattice.owner(q);
            let delta = (if b != c { 1 } else { 0 }) - (if a != c { 1 } else { 0 });
            *entry(&mut acc, c) += delta;
        }
        acc.sort_by_key(|&(c, _)| c);
        acc
    }

    pub fn apply_flip(&mut self, site: usize, new_owner: CellId) {
        let a = self.lattice.owner(site);
        let b = new_owner;
        if a == b {
            return;
        }
        let deltas = self.surface_deltas(site, b);
        for (c, d) in deltas {
            self.cells[c as usize].surface += d;
        }
        let [x, y, z] = self.lattice.coords(site);
        // volume + com
        {
            let ca = &mut self.cells[a as usize];
            ca.volume -= 1;
            ca.com_sum[0] -= x as f64;
            ca.com_sum[1] -= y as f64;
            ca.com_sum[2] -= z as f64;
        }
        {
            let cb = &mut self.cells[b as usize];
            cb.volume += 1;
            cb.com_sum[0] += x as f64;
            cb.com_sum[1] += y as f64;
            cb.com_sum[2] += z as f64;
        }
        self.lattice.set_owner(site, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};

    fn small_world() -> World {
        let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        World::new(lat, 10.0)
    }

    #[test]
    fn add_cell_and_paint_volume() {
        let mut w = small_world();
        let a = w.add_cell(1, 9.0, 1.0, 12.0, 1.0);
        // paint a 3x3 block -> volume 9
        for y in 1..4 {
            for x in 1..4 {
                let idx = w.lattice.index(x, y, 0);
                w.paint(idx, a);
            }
        }
        w.recompute_trackers();
        assert_eq!(w.cells[a as usize].volume, 9);
        assert_eq!(w.com(a), [2.0, 2.0, 0.0]);
    }

    #[test]
    fn surface_of_isolated_3x3_moore_is_correct() {
        // 3x3 block, Moore neighborhood, center of a 5x5 NoFlux lattice (no
        // lattice-edge clipping). Hand-verified unlike-neighbor counts per
        // site: the 4 corners each have 5 unlike Moore-neighbors, the 4
        // edge-midpoints each have 3 unlike neighbors, the center has 0.
        // Total unlike ordered pairs = 4*5 + 4*3 + 0 = 32.
        let mut w = small_world();
        let a = w.add_cell(1, 9.0, 1.0, 12.0, 1.0);
        for y in 1..4 {
            for x in 1..4 {
                let idx = w.lattice.index(x, y, 0);
                w.paint(idx, a);
            }
        }
        w.recompute_trackers();
        assert_eq!(w.cells[a as usize].surface, 32);
    }

    #[test]
    fn apply_flip_matches_full_recompute() {
        use crate::lattice::{Boundary, Lattice, Neighborhood};
        let lat = Lattice::new([6, 6, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 9.0, 1.0, 12.0, 1.0);
        let b = w.add_cell(2, 9.0, 1.0, 12.0, 1.0);
        for y in 1..4 { for x in 1..4 { let i = w.lattice.index(x, y, 0); w.paint(i, a); } }
        for y in 1..4 { for x in 3..5 { let i = w.lattice.index(x, y, 0); w.paint(i, b); } }
        // fix overlap: column x=3 belongs to b above; repaint cleanly
        for y in 1..4 { let i = w.lattice.index(3, y, 0); w.paint(i, b); }
        w.recompute_trackers();

        // flip site (2,2) from a to b, then compare to a fresh full recompute
        let site = w.lattice.index(2, 2, 0);
        w.apply_flip(site, b);

        let mut ref_w = World::new(
            Lattice::new([6, 6, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2)),
            10.0,
        );
        ref_w.add_cell(1, 9.0, 1.0, 12.0, 1.0);
        ref_w.add_cell(2, 9.0, 1.0, 12.0, 1.0);
        for idx in 0..w.lattice.n_sites() {
            ref_w.paint(idx, w.lattice.owner(idx));
        }
        ref_w.recompute_trackers();

        for c in 0..w.cells.len() {
            assert_eq!(w.cells[c].volume, ref_w.cells[c].volume, "volume cell {c}");
            assert_eq!(w.cells[c].surface, ref_w.cells[c].surface, "surface cell {c}");
            for k in 0..3 {
                assert!((w.cells[c].com_sum[k] - ref_w.cells[c].com_sum[k]).abs() < 1e-9);
            }
        }
    }
}
