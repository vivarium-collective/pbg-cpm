use crate::CellId;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Boundary {
    NoFlux,
    Periodic,
}

pub struct Neighborhood {
    offsets: Vec<[i64; 3]>,
}

impl Neighborhood {
    /// `order` = maximum Manhattan distance of an offset.
    /// 2D order 1 -> 4 (von Neumann), order 2 -> 8 (Moore).
    /// 3D order 1 -> 6, order 2 -> 18, order 3 -> 26.
    pub fn new(is_3d: bool, order: u8) -> Neighborhood {
        let zr: i64 = if is_3d { 1 } else { 0 };
        let mut offsets = Vec::new();
        for dz in -zr..=zr {
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let manhattan = dx.abs() + dy.abs() + dz.abs();
                    if manhattan <= order as i64 {
                        offsets.push([dx, dy, dz]);
                    }
                }
            }
        }
        Neighborhood { offsets }
    }

    pub fn offsets(&self) -> &[[i64; 3]] {
        &self.offsets
    }
}

pub struct Lattice {
    pub dims: [usize; 3],
    pub boundary: [Boundary; 3],
    // Owner of each site. AtomicU32 gives interior mutability so the parallel
    // checkerboard sweep can write owners from `&Lattice` across threads; relaxed
    // load/store compile to plain moves, so the sequential path pays nothing.
    site: Vec<AtomicU32>,
    pub nbr: Neighborhood,
}

impl Lattice {
    pub fn new(dims: [usize; 3], boundary: [Boundary; 3], nbr: Neighborhood) -> Lattice {
        let n = dims[0] * dims[1] * dims[2];
        let site = (0..n).map(|_| AtomicU32::new(crate::MEDIUM)).collect();
        Lattice { dims, boundary, site, nbr }
    }

    pub fn n_sites(&self) -> usize {
        self.site.len()
    }

    pub fn dims_x(&self) -> usize { self.dims[0] }
    pub fn dims_y(&self) -> usize { self.dims[1] }
    pub fn dims_z(&self) -> usize { self.dims[2] }

    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }

    pub fn coords(&self, idx: usize) -> [usize; 3] {
        let nx = self.dims[0];
        let ny = self.dims[1];
        let x = idx % nx;
        let y = (idx / nx) % ny;
        let z = idx / (nx * ny);
        [x, y, z]
    }

    #[inline]
    pub fn owner(&self, idx: usize) -> CellId {
        self.site[idx].load(Ordering::Relaxed)
    }

    // `&self` (not `&mut`): interior mutability via the atomic. The parallel sweep
    // writes owners across threads through a shared `&Lattice`; colour-phasing
    // guarantees no two concurrently-processed sites are neighbours, so these
    // relaxed stores never conflict on the same or adjacent cells' owner reads.
    #[inline]
    pub fn set_owner(&self, idx: usize, c: CellId) {
        self.site[idx].store(c, Ordering::Relaxed);
    }

    /// Resolve one axis coordinate + offset under this axis' boundary.
    /// Returns None if NoFlux and out of range.
    #[inline]
    fn wrap(&self, coord: usize, off: i64, dim: usize, axis: usize) -> Option<usize> {
        let v = coord as i64 + off;
        match self.boundary[axis] {
            Boundary::NoFlux => {
                if v < 0 || v >= dim as i64 {
                    None
                } else {
                    Some(v as usize)
                }
            }
            Boundary::Periodic => Some(((v % dim as i64 + dim as i64) % dim as i64) as usize),
        }
    }

    // Neighbour lists are tiny (<=18 for order-2 3D) and fetched on every flip
    // attempt, so return an inline SmallVec: no heap allocation in the hot loop.
    pub fn neighbors(&self, idx: usize) -> SmallVec<[usize; 18]> {
        let [x, y, z] = self.coords(idx);
        let mut out = SmallVec::new();
        for off in self.nbr.offsets() {
            let nx = match self.wrap(x, off[0], self.dims[0], 0) { Some(v) => v, None => continue };
            let ny = match self.wrap(y, off[1], self.dims[1], 1) { Some(v) => v, None => continue };
            let nz = match self.wrap(z, off[2], self.dims[2], 2) { Some(v) => v, None => continue };
            out.push(self.index(nx, ny, nz));
        }
        out
    }

    /// Axis-aligned (face/von-Neumann) neighbors only: up to 2 per dimension
    /// in use (2D: up to 4; 3D: up to 6), honoring per-axis boundary (NoFlux
    /// drops out-of-range, Periodic wraps). Independent of the CPM
    /// neighborhood order used for `neighbors` — diffusion always uses faces.
    pub fn face_neighbors(&self, idx: usize) -> SmallVec<[usize; 6]> {
        let [x, y, z] = self.coords(idx);
        let ndim = if self.dims[2] > 1 { 3 } else { 2 };
        let mut out = SmallVec::new();
        let coords = [x, y, z];
        for axis in 0..ndim {
            for off in [-1i64, 1i64] {
                if let Some(v) = self.wrap(coords[axis], off, self.dims[axis], axis) {
                    let mut nc = coords;
                    nc[axis] = v;
                    out.push(self.index(nc[0], nc[1], nc[2]));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_roundtrip() {
        let lat = Lattice::new([4, 3, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let idx = lat.index(2, 1, 0);
        assert_eq!(lat.coords(idx), [2, 1, 0]);
        assert_eq!(lat.n_sites(), 12);
    }

    #[test]
    fn moore_2d_interior_has_8_neighbors() {
        let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let center = lat.index(2, 2, 0);
        assert_eq!(lat.neighbors(center).len(), 8);
    }

    #[test]
    fn noflux_corner_drops_neighbors() {
        let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let corner = lat.index(0, 0, 0);
        assert_eq!(lat.neighbors(corner).len(), 3); // E, N, NE
    }

    #[test]
    fn periodic_corner_wraps_to_full() {
        let lat = Lattice::new([5, 5, 1], [Boundary::Periodic; 3], Neighborhood::new(false, 2));
        let corner = lat.index(0, 0, 0);
        assert_eq!(lat.neighbors(corner).len(), 8);
    }

    #[test]
    fn von_neumann_3d_has_6() {
        let lat = Lattice::new([5, 5, 5], [Boundary::NoFlux; 3], Neighborhood::new(true, 1));
        let c = lat.index(2, 2, 2);
        assert_eq!(lat.neighbors(c).len(), 6);
    }

    #[test]
    fn face_neighbors_3d_interior_has_6() {
        let lat = Lattice::new([5, 5, 5], [Boundary::NoFlux; 3], Neighborhood::new(true, 2));
        let c = lat.index(2, 2, 2);
        assert_eq!(lat.face_neighbors(c).len(), 6);
    }

    #[test]
    fn face_neighbors_2d_interior_has_4() {
        let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let c = lat.index(2, 2, 0);
        assert_eq!(lat.face_neighbors(c).len(), 4);
    }

    #[test]
    fn face_neighbors_noflux_corner_2d_has_2() {
        let lat = Lattice::new([5, 5, 1], [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
        let corner = lat.index(0, 0, 0);
        assert_eq!(lat.face_neighbors(corner).len(), 2);
    }

    #[test]
    fn face_neighbors_noflux_corner_3d_has_3() {
        let lat = Lattice::new([5, 5, 5], [Boundary::NoFlux; 3], Neighborhood::new(true, 2));
        let corner = lat.index(0, 0, 0);
        assert_eq!(lat.face_neighbors(corner).len(), 3);
    }
}
