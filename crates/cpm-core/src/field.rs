use crate::lattice::Lattice;
use crate::world::World;
use crate::CellId;

pub struct Field {
    pub name: String,
    pub conc: Vec<f32>,        // len == lattice.n_sites()
    pub d: f64,                // diffusion constant
    pub decay: f64,            // decay constant
    pub dt: f64,               // time step (default 1.0)
    pub substeps: u32,         // diffusion sub-steps per MCS (default 1)
    pub secretion: Vec<f64>,   // per cell_type secretion rate; index = cell_type; len = n_types
    pub chemotaxis: Vec<f64>,  // per cell_type chemotaxis lambda; index = cell_type; len = n_types
}

impl Field {
    pub fn new(name: String, n_sites: usize, d: f64, decay: f64, n_types: usize) -> Field {
        Field {
            name,
            conc: vec![0.0; n_sites],
            d,
            decay,
            dt: 1.0,
            substeps: 1,
            secretion: vec![0.0; n_types],
            chemotaxis: vec![0.0; n_types],
        }
    }

    /// One explicit forward-Euler diffusion+decay sub-step over the whole lattice.
    pub fn diffuse_step(&mut self, lattice: &Lattice) {
        let n = self.conc.len();
        let mut next = vec![0.0f32; n];
        for i in 0..n {
            let ci = self.conc[i];
            let mut lap = 0.0f32;
            for j in lattice.face_neighbors(i) {
                lap += self.conc[j] - ci;
            }
            let updated = ci as f64 + self.dt * (self.d * lap as f64 - self.decay * ci as f64);
            next[i] = updated.max(0.0) as f32; // concentrations stay non-negative
        }
        self.conc = next;
    }
}

impl World {
    pub fn add_field(&mut self, name: &str, d: f64, decay: f64) -> usize {
        let max_type = self.cells.iter().map(|c| c.cell_type).max().unwrap_or(0);
        let n_types = (max_type as usize + 1).max(1);
        let f = Field::new(name.to_string(), self.lattice.n_sites(), d, decay, n_types);
        self.fields.push(f);
        self.fields.len() - 1
    }

    /// Control the PDE solver for a field: `dt` (forward-Euler step; must keep
    /// `dt*d*2*ndim < 1` for stability) and `substeps` diffusion sub-steps per MCS
    /// (more sub-steps = faster equilibration / larger effective diffusion length).
    pub fn set_field_dynamics(&mut self, field_idx: usize, dt: f64, substeps: u32) {
        let f = &mut self.fields[field_idx];
        f.dt = dt;
        f.substeps = substeps.max(1);
    }

    pub fn set_secretion(&mut self, field_idx: usize, cell_type: u16, rate: f64) {
        let t = cell_type as usize;
        let field = &mut self.fields[field_idx];
        if t >= field.secretion.len() {
            field.secretion.resize(t + 1, 0.0);
        }
        field.secretion[t] = rate;
    }

    pub fn set_chemotaxis(&mut self, field_idx: usize, cell_type: u16, lambda: f64) {
        let t = cell_type as usize;
        let field = &mut self.fields[field_idx];
        if t >= field.chemotaxis.len() {
            field.chemotaxis.resize(t + 1, 0.0);
        }
        field.chemotaxis[t] = lambda;
    }

    /// Advance every field one MCS: `substeps` diffusion sub-steps, then
    /// secretion at secreting cells' pixels.
    pub fn advance_fields(&mut self) {
        for fi in 0..self.fields.len() {
            let substeps = self.fields[fi].substeps;
            for _ in 0..substeps {
                let lattice = &self.lattice;
                self.fields[fi].diffuse_step(lattice);
            }
            for idx in 0..self.lattice.n_sites() {
                let owner = self.lattice.owner(idx) as usize;
                let t = self.cells[owner].cell_type as usize;
                let rate = self.fields[fi].secretion.get(t).copied().unwrap_or(0.0);
                if rate != 0.0 {
                    let dt = self.fields[fi].dt;
                    self.fields[fi].conc[idx] += (rate * dt) as f32;
                }
            }
        }
    }

    pub fn field_conc(&self, field_idx: usize) -> Vec<f32> {
        self.fields[field_idx].conc.clone()
    }

    pub fn field_mean_at_cell(&self, field_idx: usize, cell_id: CellId) -> f64 {
        let cell = &self.cells[cell_id as usize];
        if cell.volume == 0 {
            return 0.0;
        }
        let field = &self.fields[field_idx];
        let mut sum = 0.0f64;
        let mut count = 0i64;
        for idx in 0..self.lattice.n_sites() {
            if self.lattice.owner(idx) == cell_id {
                sum += field.conc[idx] as f64;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// CC3D chemotaxis: ΔH = -λ(c(destination) - c(source)); destination=site s,
    /// source=source_pixel n, λ = chemotaxis lambda of the moving (new_owner /
    /// source) cell's type, summed over fields.
    pub fn delta_chemotaxis(&self, site: usize, source_pixel: usize, new_owner: CellId) -> f64 {
        let t = self.cells[new_owner as usize].cell_type as usize;
        let mut d = 0.0;
        for f in &self.fields {
            let lambda = f.chemotaxis.get(t).copied().unwrap_or(0.0);
            if lambda != 0.0 {
                d += -lambda * (f.conc[site] as f64 - f.conc[source_pixel] as f64);
            }
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};
    use crate::sweep::Cpm;

    #[test]
    fn diffusion_spreads_and_decays() {
        let lat = Lattice::new([7, 7, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let fi = w.add_field("attr", 0.1, 1e-3);
        let center = w.lattice.index(3, 3, 0);
        w.fields[fi].conc[center] = 100.0;
        let total_before: f64 = w.fields[fi].conc.iter().map(|&c| c as f64).sum();
        let neighbor = w.lattice.face_neighbors(center)[0];
        assert_eq!(w.fields[fi].conc[neighbor], 0.0);

        for _ in 0..5 {
            let lattice = &w.lattice;
            w.fields[fi].diffuse_step(lattice);
        }

        let after_center = w.fields[fi].conc[center];
        let after_neighbor = w.fields[fi].conc[neighbor];
        let total_after: f64 = w.fields[fi].conc.iter().map(|&c| c as f64).sum();
        assert!(after_center < 100.0, "center should decrease, got {after_center}");
        assert!(after_neighbor > 0.0, "neighbor should increase from 0, got {after_neighbor}");
        assert!(total_after < total_before, "decay should shrink total mass: {total_before} -> {total_after}");
    }

    #[test]
    fn secretion_increases_field_at_cell_pixels() {
        let lat = Lattice::new([6, 6, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 9.0, 1.0, 0.0, 0.0);
        for y in 1..4 {
            for x in 1..4 {
                let idx = w.lattice.index(x, y, 0);
                w.paint(idx, a);
            }
        }
        w.recompute_trackers();
        let fi = w.add_field("attr", 0.0, 0.0);
        w.set_secretion(fi, 1, 100.0);

        w.advance_fields();

        let idx = w.lattice.index(2, 2, 0);
        let conc = w.fields[fi].conc[idx];
        assert!((conc as f64 - 100.0).abs() < 1e-6, "expected ~100 secretion, got {conc}");
    }

    #[test]
    fn field_dynamics_substeps_accelerate_spread() {
        // set_field_dynamics(dt, substeps): one advance_fields() runs `substeps`
        // diffusion sub-steps, so more sub-steps spread a point source further in
        // the same MCS. Compare 1 sub-step vs 8 sub-steps from an identical source.
        fn spread(substeps: u32) -> f32 {
            let lat = Lattice::new([11, 11, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
            let mut w = World::new(lat, 10.0);
            let fi = w.add_field("m", 0.15, 0.0);
            w.set_field_dynamics(fi, 0.2, substeps);
            let c = w.lattice.index(5, 5, 0);
            w.fields[fi].conc[c] = 100.0;
            w.advance_fields();
            let edge = w.lattice.index(5, 8, 0);
            w.fields[fi].conc[edge]
        }
        assert!(spread(8) > spread(1), "more sub-steps should spread further");
    }

    #[test]
    fn chemotaxis_climbs_the_gradient() {
        // 2D world with a fixed linear gradient (D=0, decay=0, no secretion so
        // the imposed gradient persists), one motile cell starting at the
        // LOW-concentration end, large positive lambda -> should move up-gradient (+x).
        let dims = [20usize, 10usize, 1usize];
        let lat = Lattice::new(dims, [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 9.0, 2.0, 0.0, 0.0);
        // seed a 3x3 block near the low-x end
        for y in 4..7 {
            for x in 2..5 {
                let idx = w.lattice.index(x, y, 0);
                w.paint(idx, a);
            }
        }
        let mut m = crate::energy::ContactMatrix::new(2);
        m.set(0, 1, 8.0);
        w.set_contact_matrix(m);
        w.recompute_trackers();

        let fi = w.add_field("attr", 0.0, 0.0);
        for idx in 0..w.lattice.n_sites() {
            let [x, _, _] = w.lattice.coords(idx);
            w.fields[fi].conc[idx] = x as f32; // linear gradient in +x
        }
        w.set_chemotaxis(fi, 1, 1e3);

        let com_before = w.com(a);
        let mut cpm = Cpm::new(w, 123);
        cpm.step(30);
        let com_after = cpm.world.com(a);

        assert!(
            com_after[0] > com_before[0],
            "cell should climb the gradient (com-x): {} -> {}",
            com_before[0],
            com_after[0]
        );
    }
}
