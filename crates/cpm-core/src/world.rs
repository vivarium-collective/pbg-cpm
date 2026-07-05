use crate::lattice::Lattice;
use crate::{CellId, MEDIUM};

#[derive(Clone, Debug)]
pub struct Cell {
    pub id: CellId,
    pub cell_type: u16,
    pub volume: i64,
    pub surface: i64,
    pub com_sum: [f64; 3],
    // second moments Σ[x², y², z², xy, xz, yz] — the gyration tensor, for the
    // length (elongation) constraint. Maintained incrementally like com_sum.
    pub moment_sum: [f64; 6],
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
    pub membrane_dist: Vec<f32>,
    pub membrane_k: f64,
    pub membrane_band: f64,
    pub membrane_types: Vec<bool>,
    pub junction_types: Vec<bool>,
    pub lambda_junction: f64,
    pub length_target: Vec<f64>,
    pub length_lambda: Vec<f64>,
}

impl World {
    pub fn new(lattice: Lattice, temperature: f64) -> World {
        let medium = Cell {
            id: MEDIUM,
            cell_type: 0,
            volume: 0,
            surface: 0,
            com_sum: [0.0; 3],
            moment_sum: [0.0; 6],
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
            membrane_dist: Vec::new(),
            membrane_k: 0.0,
            membrane_band: 0.0,
            membrane_types: Vec::new(),
            junction_types: Vec::new(),
            lambda_junction: 0.0,
            length_target: Vec::new(),
            length_lambda: Vec::new(),
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

    pub fn set_membrane(&mut self, anchors: &[usize], k: f64, band: f64) {
        // Empty anchors => no membrane. Otherwise build_distance_field would
        // return an all-INFINITY field, and delta_membrane on an anchored cell
        // would then see cost(inf) = +inf, freezing anchored cells solid — a
        // silent foot-gun. Treat "no anchors" as "membrane unset" instead.
        if anchors.is_empty() {
            self.membrane_dist = Vec::new();
            return;
        }
        let dims = [self.lattice.dims_x(), self.lattice.dims_y(), self.lattice.dims_z()];
        self.membrane_dist = crate::membrane::build_distance_field(dims, anchors);
        self.membrane_k = k;
        self.membrane_band = band;
    }

    pub fn set_membrane_anchored(&mut self, cell_type: u16, on: bool) {
        let t = cell_type as usize;
        if t >= self.membrane_types.len() {
            self.membrane_types.resize(t + 1, false);
        }
        self.membrane_types[t] = on;
    }

    pub fn any_membrane(&self) -> bool {
        !self.membrane_dist.is_empty() && self.membrane_types.iter().any(|&b| b)
    }

    fn membrane_type_anchored(&self, cell_type: u16) -> bool {
        self.membrane_types.get(cell_type as usize).copied().unwrap_or(false)
    }

    /// Membrane anchor energy change for reassigning `site` to `new_owner`.
    /// Only `site` changes membership, so this is cost(site,new) - cost(site,old).
    pub fn delta_membrane(&self, site: usize, new_owner: CellId) -> f64 {
        if self.membrane_dist.is_empty() {
            return 0.0;
        }
        let d = self.membrane_dist[site];
        let target = self.lattice.owner(site);
        let cost_for = |c: CellId| -> f64 {
            if c == crate::MEDIUM {
                return 0.0;
            }
            if self.membrane_type_anchored(self.cells[c as usize].cell_type) {
                crate::membrane::cost(d, self.membrane_k, self.membrane_band)
            } else {
                0.0
            }
        };
        cost_for(new_owner) - cost_for(target)
    }

    pub fn set_junction(&mut self, cell_type: u16, on: bool) {
        let t = cell_type as usize;
        if t >= self.junction_types.len() {
            self.junction_types.resize(t + 1, false);
        }
        self.junction_types[t] = on;
    }

    pub fn set_junction_lambda(&mut self, lambda: f64) {
        self.lambda_junction = lambda;
    }

    pub fn any_junction(&self) -> bool {
        self.lambda_junction > 0.0 && self.junction_types.iter().any(|&b| b)
    }

    fn junction_type_enabled(&self, cell_type: u16) -> bool {
        self.junction_types.get(cell_type as usize).copied().unwrap_or(false)
    }

    // owner of `idx`, but treating site `s` as if it were `new`
    #[inline]
    fn owner_ov(&self, idx: usize, s: usize, new: CellId) -> CellId {
        if idx == s { new } else { self.lattice.owner(idx) }
    }

    // (owner_id, is_junction_type) for `idx` under the s->new override; medium = (0,false)
    #[inline]
    fn junction_tag(&self, idx: usize, s: usize, new: CellId) -> (u32, bool) {
        let o = self.owner_ov(idx, s, new);
        if o == crate::MEDIUM {
            (0, false)
        } else {
            (o, self.junction_type_enabled(self.cells[o as usize].cell_type))
        }
    }

    // number of pinched axes at medium centre `c`, under the s->new override
    fn pinch_at(&self, c: usize, s: usize, new: CellId) -> u32 {
        if self.owner_ov(c, s, new) != crate::MEDIUM {
            return 0; // only a medium voxel can be a pinch centre
        }
        let (nx, ny, nz) = (self.lattice.dims_x(), self.lattice.dims_y(), self.lattice.dims_z());
        let z = c / (nx * ny);
        let rem = c % (nx * ny);
        let y = rem / nx;
        let x = rem % nx;
        let mut n = 0u32;
        if x >= 1 && x + 1 < nx
            && crate::junction::axis_is_pinch(self.junction_tag(c - 1, s, new), self.junction_tag(c + 1, s, new))
        {
            n += 1;
        }
        if y >= 1 && y + 1 < ny
            && crate::junction::axis_is_pinch(self.junction_tag(c - nx, s, new), self.junction_tag(c + nx, s, new))
        {
            n += 1;
        }
        if nz > 1 && z >= 1 && z + 1 < nz
            && crate::junction::axis_is_pinch(
                self.junction_tag(c - nx * ny, s, new),
                self.junction_tag(c + nx * ny, s, new),
            )
        {
            n += 1;
        }
        n
    }

    /// Junction (anti-gap) energy change for reassigning `site` to `new_owner`.
    /// Only `site` changes owner, so pinch counts change only at `site` (a pinch
    /// centre while medium) and its medium 6-neighbours (for which `site` is one
    /// axis side). E = lambda_junction * total_pinches.
    pub fn delta_junction(&self, site: usize, new_owner: CellId) -> f64 {
        let old = self.lattice.owner(site);
        let mut d: i64 = self.pinch_at(site, site, new_owner) as i64
            - self.pinch_at(site, site, old) as i64;
        let (nx, ny, nz) = (self.lattice.dims_x(), self.lattice.dims_y(), self.lattice.dims_z());
        let z = site / (nx * ny);
        let rem = site % (nx * ny);
        let y = rem / nx;
        let x = rem % nx;
        let mut nb: Vec<usize> = Vec::with_capacity(6);
        if x + 1 < nx { nb.push(site + 1); }
        if x >= 1 { nb.push(site - 1); }
        if y + 1 < ny { nb.push(site + nx); }
        if y >= 1 { nb.push(site - nx); }
        if nz > 1 && z + 1 < nz { nb.push(site + nx * ny); }
        if nz > 1 && z >= 1 { nb.push(site - nx * ny); }
        for m in nb {
            if self.lattice.owner(m) == crate::MEDIUM {
                d += self.pinch_at(m, site, new_owner) as i64 - self.pinch_at(m, site, old) as i64;
            }
        }
        self.lambda_junction * d as f64
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
            moment_sum: [0.0; 6],
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

    /// Instantly extrude (slough) cells: set every voxel they own to medium and
    /// rebuild trackers. Models apoptotic shedding at the crypt mouth so the
    /// neighbours flow in and the monolayer advances. O(n_sites); batch ids for a
    /// single tracker rebuild. Ids that are already gone / medium are ignored.
    pub fn remove_cells(&mut self, ids: &[CellId]) {
        if ids.is_empty() {
            return;
        }
        let mut kill = vec![false; self.cells.len()];
        for &id in ids {
            if id != MEDIUM && (id as usize) < kill.len() {
                kill[id as usize] = true;
            }
        }
        let n = self.lattice.n_sites();
        for i in 0..n {
            let o = self.lattice.owner(i);
            if o != MEDIUM && kill[o as usize] {
                self.lattice.set_owner(i, MEDIUM);
            }
        }
        self.recompute_trackers();
    }

    pub fn recompute_trackers(&mut self) {
        for cell in self.cells.iter_mut() {
            cell.volume = 0;
            cell.surface = 0;
            cell.com_sum = [0.0; 3];
            cell.moment_sum = [0.0; 6];
        }
        let n = self.lattice.n_sites();
        for idx in 0..n {
            let owner = self.lattice.owner(idx);
            let [x, y, z] = self.lattice.coords(idx);
            let (xf, yf, zf) = (x as f64, y as f64, z as f64);
            {
                let cell = &mut self.cells[owner as usize];
                cell.volume += 1;
                cell.com_sum[0] += xf;
                cell.com_sum[1] += yf;
                cell.com_sum[2] += zf;
                cell.moment_sum[0] += xf * xf;
                cell.moment_sum[1] += yf * yf;
                cell.moment_sum[2] += zf * zf;
                cell.moment_sum[3] += xf * yf;
                cell.moment_sum[4] += xf * zf;
                cell.moment_sum[5] += yf * zf;
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
        let (xf, yf, zf) = (x as f64, y as f64, z as f64);
        let m = [xf * xf, yf * yf, zf * zf, xf * yf, xf * zf, yf * zf];
        // volume + com + moments
        {
            let ca = &mut self.cells[a as usize];
            ca.volume -= 1;
            ca.com_sum[0] -= xf;
            ca.com_sum[1] -= yf;
            ca.com_sum[2] -= zf;
            for k in 0..6 { ca.moment_sum[k] -= m[k]; }
        }
        {
            let cb = &mut self.cells[b as usize];
            cb.volume += 1;
            cb.com_sum[0] += xf;
            cb.com_sum[1] += yf;
            cb.com_sum[2] += zf;
            for k in 0..6 { cb.moment_sum[k] += m[k]; }
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
            for k in 0..6 {
                assert!((w.cells[c].moment_sum[k] - ref_w.cells[c].moment_sum[k]).abs() < 1e-9,
                    "moment {k} cell {c}");
            }
        }
    }
}
