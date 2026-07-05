//! CPM length (elongation) constraint — CC3D's LengthConstraint.
//!
//! Each cell carries the gyration tensor (second central moments) of its pixels,
//! maintained incrementally in `World::apply_flip`. The variance along the cell's
//! major principal axis is the largest eigenvalue λ_max of the covariance matrix;
//! for a uniform body of extent ℓ along that axis, variance = ℓ²/12, so we report
//! the **major-axis length** ℓ = √(12·λ_max). A per-type spring
//! E = λ_L·(ℓ − ℓ_target)² then pulls cells toward a target length, letting them
//! elongate (rods, columnar epithelium) or stay compact.

use crate::world::World;
use crate::CellId;

/// Largest eigenvalue of the symmetric 3×3 matrix
/// [[a11,a12,a13],[a12,a22,a23],[a13,a23,a33]] (Smith's analytic method).
pub fn max_eig_sym3(a11: f64, a22: f64, a33: f64, a12: f64, a13: f64, a23: f64) -> f64 {
    let p1 = a12 * a12 + a13 * a13 + a23 * a23;
    if p1 == 0.0 {
        // already diagonal
        return a11.max(a22).max(a33);
    }
    let q = (a11 + a22 + a33) / 3.0;
    let p2 = (a11 - q).powi(2) + (a22 - q).powi(2) + (a33 - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if p == 0.0 {
        return q;
    }
    // B = (A - qI)/p ; r = det(B)/2, clamped to [-1,1] for acos stability
    let b11 = (a11 - q) / p;
    let b22 = (a22 - q) / p;
    let b33 = (a33 - q) / p;
    let b12 = a12 / p;
    let b13 = a13 / p;
    let b23 = a23 / p;
    let det = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13)
        + b13 * (b12 * b23 - b22 * b13);
    let r = (det / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    // largest eigenvalue corresponds to phi (cos is largest there)
    q + 2.0 * p * phi.cos()
}

/// Major-axis length ℓ = √(12·λ_max) from a cell's volume, com_sum and moment_sum.
/// Returns 0 for cells with fewer than 2 pixels (length undefined / degenerate).
pub fn major_axis_length(volume: i64, com_sum: [f64; 3], moment_sum: [f64; 6]) -> f64 {
    if volume < 2 {
        return 0.0;
    }
    let v = volume as f64;
    let (cx, cy, cz) = (com_sum[0] / v, com_sum[1] / v, com_sum[2] / v);
    let cxx = moment_sum[0] / v - cx * cx;
    let cyy = moment_sum[1] / v - cy * cy;
    let czz = moment_sum[2] / v - cz * cz;
    let cxy = moment_sum[3] / v - cx * cy;
    let cxz = moment_sum[4] / v - cx * cz;
    let cyz = moment_sum[5] / v - cy * cz;
    let lam = max_eig_sym3(cxx, cyy, czz, cxy, cxz, cyz).max(0.0);
    (12.0 * lam).sqrt()
}

impl World {
    /// Give cells of `cell_type` a target major-axis length and spring stiffness.
    /// λ = 0 (default) disables the constraint for that type.
    pub fn set_length_constraint(&mut self, cell_type: u16, target_length: f64, lambda: f64) {
        let t = cell_type as usize;
        if t >= self.length_lambda.len() {
            self.length_lambda.resize(t + 1, 0.0);
            self.length_target.resize(t + 1, 0.0);
        }
        self.length_lambda[t] = lambda;
        self.length_target[t] = target_length;
    }

    pub fn any_length(&self) -> bool {
        self.length_lambda.iter().any(|&l| l > 0.0)
    }

    /// Current major-axis length of a cell.
    pub fn cell_length(&self, c: CellId) -> f64 {
        let cell = &self.cells[c as usize];
        major_axis_length(cell.volume, cell.com_sum, cell.moment_sum)
    }

    // Length energy of a hypothetical cell state (type, volume, moments).
    #[inline]
    fn length_energy(&self, cell_type: u16, volume: i64, com_sum: [f64; 3], moment_sum: [f64; 6]) -> f64 {
        let lambda = self.length_lambda.get(cell_type as usize).copied().unwrap_or(0.0);
        if lambda == 0.0 || volume < 2 {
            return 0.0;
        }
        let target = self.length_target.get(cell_type as usize).copied().unwrap_or(0.0);
        let ell = major_axis_length(volume, com_sum, moment_sum);
        lambda * (ell - target).powi(2)
    }

    /// Length-constraint energy change for reassigning `site` to `new_owner`:
    /// the losing cell drops the pixel, the gaining cell adds it, and each cell's
    /// (ℓ − ℓ_target)² spring is re-evaluated. MEDIUM contributes 0 (λ default 0).
    pub fn delta_length(&self, site: usize, new_owner: CellId) -> f64 {
        let a = self.lattice.owner(site);
        let b = new_owner;
        if a == b {
            return 0.0;
        }
        let [x, y, z] = self.lattice.coords(site);
        let (xf, yf, zf) = (x as f64, y as f64, z as f64);
        let m = [xf * xf, yf * yf, zf * zf, xf * yf, xf * zf, yf * zf];
        let sub = |s: [f64; 6]| { let mut o = s; for k in 0..6 { o[k] -= m[k]; } o };
        let add = |s: [f64; 6]| { let mut o = s; for k in 0..6 { o[k] += m[k]; } o };
        let ca = &self.cells[a as usize];
        let cb = &self.cells[b as usize];
        let e_a_old = self.length_energy(ca.cell_type, ca.volume, ca.com_sum, ca.moment_sum);
        let e_b_old = self.length_energy(cb.cell_type, cb.volume, cb.com_sum, cb.moment_sum);
        let a_com = [ca.com_sum[0] - xf, ca.com_sum[1] - yf, ca.com_sum[2] - zf];
        let b_com = [cb.com_sum[0] + xf, cb.com_sum[1] + yf, cb.com_sum[2] + zf];
        let e_a_new = self.length_energy(ca.cell_type, ca.volume - 1, a_com, sub(ca.moment_sum));
        let e_b_new = self.length_energy(cb.cell_type, cb.volume + 1, b_com, add(cb.moment_sum));
        (e_a_new + e_b_new) - (e_a_old + e_b_old)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Boundary, Lattice, Neighborhood};

    #[test]
    fn horizontal_bar_length_matches_extent() {
        // a 1x7 horizontal bar has major-axis length ~ sqrt(7^2 - 1) = 6.93
        let lat = Lattice::new([10, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 7.0, 1.0, 0.0, 0.0);
        for x in 1..8 { let i = w.lattice.index(x, 2, 0); w.paint(i, a); }
        w.recompute_trackers();
        let ell = w.cell_length(a);
        assert!((ell - (48.0f64).sqrt()).abs() < 1e-6, "bar length {ell}");
    }

    #[test]
    fn square_is_shorter_than_a_bar_of_equal_area() {
        // 4x4 square vs 1x16 bar: same volume, but the bar is far longer.
        let lat = Lattice::new([20, 20, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let sq = w.add_cell(1, 16.0, 1.0, 0.0, 0.0);
        let bar = w.add_cell(1, 16.0, 1.0, 0.0, 0.0);
        for y in 2..6 { for x in 2..6 { let i = w.lattice.index(x, y, 0); w.paint(i, sq); } }
        for x in 2..18 { let i = w.lattice.index(x, 12, 0); w.paint(i, bar); }
        w.recompute_trackers();
        assert!(w.cell_length(bar) > 2.0 * w.cell_length(sq),
            "bar {} should dwarf square {}", w.cell_length(bar), w.cell_length(sq));
    }

    #[test]
    fn delta_length_matches_recompute() {
        // delta_length for a flip must equal the exact before/after energy diff.
        let lat = Lattice::new([12, 12, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 10.0);
        let a = w.add_cell(1, 9.0, 1.0, 0.0, 0.0);
        let b = w.add_cell(1, 9.0, 1.0, 0.0, 0.0);
        for y in 2..5 { for x in 2..5 { let i = w.lattice.index(x, y, 0); w.paint(i, a); } }
        for y in 2..5 { for x in 5..8 { let i = w.lattice.index(x, y, 0); w.paint(i, b); } }
        w.recompute_trackers();
        w.set_length_constraint(1, 6.0, 2.5);
        let site = w.lattice.index(4, 3, 0); // owned by a, adjacent to b
        let predicted = w.delta_length(site, b);
        let before = w.cell_length(a); let _ = before;
        // exact: energies before, apply, energies after
        let e_before = {
            let ea = 2.5 * (w.cell_length(a) - 6.0).powi(2);
            let eb = 2.5 * (w.cell_length(b) - 6.0).powi(2);
            ea + eb
        };
        w.apply_flip(site, b);
        let e_after = {
            let ea = 2.5 * (w.cell_length(a) - 6.0).powi(2);
            let eb = 2.5 * (w.cell_length(b) - 6.0).powi(2);
            ea + eb
        };
        assert!((predicted - (e_after - e_before)).abs() < 1e-9,
            "predicted {predicted} vs actual {}", e_after - e_before);
    }
}
