use crate::world::World;
use crate::CellId;

pub struct ContactMatrix {
    n_types: usize,
    j: Vec<f64>,
}

impl ContactMatrix {
    pub fn new(n_types: usize) -> ContactMatrix {
        ContactMatrix { n_types: n_types.max(1), j: vec![0.0; n_types.max(1) * n_types.max(1)] }
    }
    pub fn set(&mut self, a: u16, b: u16, val: f64) {
        let (a, b) = (a as usize, b as usize);
        self.j[a * self.n_types + b] = val;
        self.j[b * self.n_types + a] = val;
    }
    pub fn get(&self, a: u16, b: u16) -> f64 {
        let (a, b) = (a as usize, b as usize);
        if a >= self.n_types || b >= self.n_types {
            // Untracked type pair (matrix not sized/populated for it): no
            // contact energy contribution, rather than an out-of-bounds panic.
            return 0.0;
        }
        self.j[a * self.n_types + b]
    }
}

impl World {
    pub fn delta_hamiltonian(&self, site: usize, new_owner: CellId) -> f64 {
        let a = self.lattice.owner(site);
        let b = new_owner;
        if a == b {
            return 0.0;
        }
        // Volume
        let mut d = 0.0;
        for (c, dv) in [(a, -1i64), (b, 1i64)] {
            if c == crate::MEDIUM {
                continue;
            }
            let cell = &self.cells[c as usize];
            let before = cell.volume as f64;
            let after = (cell.volume + dv) as f64;
            d += cell.lambda_volume
                * ((after - cell.target_volume).powi(2) - (before - cell.target_volume).powi(2));
        }
        // Surface
        for (c, ds) in self.surface_deltas(site, b) {
            if c == crate::MEDIUM {
                continue;
            }
            let cell = &self.cells[c as usize];
            let before = cell.surface as f64;
            let after = (cell.surface + ds) as f64;
            d += cell.lambda_surface
                * ((after - cell.target_surface).powi(2) - (before - cell.target_surface).powi(2));
        }
        // Contact
        let ta = self.cells[a as usize].cell_type;
        let tb = self.cells[b as usize].cell_type;
        for q in self.lattice.neighbors(site) {
            let c = self.lattice.owner(q);
            let tc = self.cells[c as usize].cell_type;
            let after = if c != b { self.contact.get(tb, tc) } else { 0.0 };
            let before = if c != a { self.contact.get(ta, tc) } else { 0.0 };
            d += after - before;
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};

    #[test]
    fn contact_matrix_symmetric() {
        let mut m = ContactMatrix::new(3);
        m.set(1, 2, 5.0);
        assert_eq!(m.get(1, 2), 5.0);
        assert_eq!(m.get(2, 1), 5.0);
    }

    #[test]
    fn delta_h_volume_penalizes_growth_away_from_target() {
        // one cell at target volume; growing it by 1 raises volume energy.
        let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 9.0, 1.0, 0.0, 0.0); // target vol 9, no surface term
        for y in 1..4 { for x in 1..4 { let i = w.lattice.index(x,y,0); w.paint(i, a); } }
        w.recompute_trackers();
        assert_eq!(w.cells[a as usize].volume, 9);
        // medium site adjacent to the cell tries to become the cell -> volume 9->10
        let site = w.lattice.index(4, 2, 0); // medium, neighbor of (3,2)
        let dh = w.delta_hamiltonian(site, a);
        // ΔVolume = 1*((10-9)^2 - (9-9)^2) = 1 ; contact from medium term = 0 (J all zero)
        assert!((dh - 1.0).abs() < 1e-9, "dh was {dh}");
    }
}
