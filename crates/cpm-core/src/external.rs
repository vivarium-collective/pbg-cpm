//! External potential — CC3D's ExternalPotential energy term.
//!
//! A per-type constant force `f` gives every pixel of a cell the linear potential
//! U(r) = −f·r, so a cell's energy is −f·com_sum. Cells therefore drift up the
//! force direction: gravity/sedimentation (`f` along +z), directional taxis, or a
//! positioning bias. The force is constant, so a flip only moves one pixel between
//! owners and ΔH = (f_old − f_new)·r — O(1), no per-cell state needed.

use crate::world::World;
use crate::CellId;

impl World {
    /// Set a per-type constant force vector (0,0,0 disables it for that type).
    pub fn set_external_potential(&mut self, cell_type: u16, fx: f64, fy: f64, fz: f64) {
        let t = cell_type as usize;
        if t >= self.ext_potential.len() {
            self.ext_potential.resize(t + 1, [0.0; 3]);
        }
        self.ext_potential[t] = [fx, fy, fz];
    }

    pub fn any_external(&self) -> bool {
        self.ext_potential.iter().any(|f| f != &[0.0; 3])
    }

    #[inline]
    fn force_of(&self, owner: CellId) -> [f64; 3] {
        let t = self.cells[owner as usize].cell_type as usize;
        self.ext_potential.get(t).copied().unwrap_or([0.0; 3])
    }

    /// External-potential energy change for reassigning `site` to `new_owner`.
    /// Total energy is Σ_pixels −f_{type(owner)}·r, and only `site` changes owner,
    /// so ΔH = (f_old − f_new)·r. Medium force is (0,0,0).
    pub fn delta_external(&self, site: usize, new_owner: CellId) -> f64 {
        let a = self.lattice.owner(site);
        let b = new_owner;
        if a == b {
            return 0.0;
        }
        let fa = self.force_of(a);
        let fb = self.force_of(b);
        let [x, y, z] = self.lattice.coords(site);
        let r = [x as f64, y as f64, z as f64];
        (fa[0] - fb[0]) * r[0] + (fa[1] - fb[1]) * r[1] + (fa[2] - fb[2]) * r[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::ContactMatrix;
    use crate::lattice::{Boundary, Lattice, Neighborhood};
    use crate::sweep::Cpm;

    #[test]
    fn cell_sediments_along_the_force() {
        // A single cell with a downward (+y) force should drift toward larger y.
        let lat = Lattice::new([20, 30, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 8.0);
        let a = w.add_cell(1, 25.0, 2.0, 0.0, 0.0);
        for y in 3..8 { for x in 8..13 { let i = w.lattice.index(x, y, 0); w.paint(i, a); } }
        let mut m = ContactMatrix::new(2);
        m.set(0, 1, 4.0);
        w.set_contact_matrix(m);
        w.set_external_potential(1, 0.0, 2.0, 0.0);
        w.recompute_trackers();
        let y0 = w.com(a)[1];
        let mut cpm = Cpm::new(w, 5);
        cpm.step(60);
        let y1 = cpm.world.com(a)[1];
        assert!(y1 > y0 + 3.0, "cell should sediment: com-y {y0} -> {y1}");
    }

    #[test]
    fn delta_external_signs() {
        let lat = Lattice::new([6, 6, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 8.0);
        let a = w.add_cell(1, 4.0, 1.0, 0.0, 0.0);
        let i = w.lattice.index(2, 2, 0);
        w.paint(i, a);
        w.recompute_trackers();
        w.set_external_potential(1, 0.0, 1.0, 0.0);
        // growing cell into medium site (2,3): a gains it => (f_a - f_b)·r with
        // a=medium(0) losing, b=cell(1) gaining -> (0 - f_b)·r = -1*3 = -3 (favoured)
        let grow = w.lattice.index(2, 3, 0);
        assert!((w.delta_external(grow, a) + 3.0).abs() < 1e-9);
    }
}
