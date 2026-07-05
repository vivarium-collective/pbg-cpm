//! Parallel checkerboard Metropolis sweep (opt-in; the default `Cpm::step` stays
//! sequential and bit-exact).
//!
//! The lattice is tiled into cubic blocks of side `B`. Blocks are 8-coloured in
//! 3D / 4-coloured in 2D by `(bx%2, by%2, bz%2)`; each colour phase runs its
//! blocks in parallel (rayon). Because same-colour blocks are ≥2 blocks apart,
//! no two concurrently-processed sites are neighbours, so **owner reads/writes
//! and the contact/surface neighbour scans are exact**. Per-cell tracker deltas
//! (volume/surface/com/moments) are accumulated per block and summed at a barrier
//! after each phase, so **the trackers themselves stay exact**.
//!
//! The single, well-bounded approximation: a flip's acceptance test may use a
//! cell volume that does not yet include a *concurrent* block's change to the
//! same (spread-out) cell within the same phase. This is resynced every phase
//! (8×/MCS), so the error is small — validated against the sequential engine.
//!
//! Not supported in parallel: the length (elongation) constraint (its energy
//! needs live com/moments); models using it must use the sequential `step`.

use crate::world::World;
use crate::{CellId, MEDIUM};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Default)]
struct CellDelta {
    dvol: i64,
    dsurf: i64,
    dcom: [f64; 3],
    dmom: [f64; 6],
}

/// ΔH for reassigning `site` to `new_owner` under a block's thread-local volume/
/// surface deltas. Volume/surface use frozen cell trackers + this block's own
/// deltas; contact/chemotaxis/membrane/junction/external read only owners (atomic)
/// and read-only fields, so they reuse the sequential methods unchanged.
fn par_delta_h(
    world: &World,
    site: usize,
    pick: usize,
    new_owner: CellId,
    local: &HashMap<CellId, CellDelta>,
) -> f64 {
    let a = world.lattice.owner(site);
    let b = new_owner;
    let vol_of = |c: CellId| world.cells[c as usize].volume + local.get(&c).map_or(0, |d| d.dvol);
    let surf_of = |c: CellId| world.cells[c as usize].surface + local.get(&c).map_or(0, |d| d.dsurf);
    let mut d = 0.0;
    for (c, dv) in [(a, -1i64), (b, 1i64)] {
        if c == MEDIUM {
            continue;
        }
        let cell = &world.cells[c as usize];
        let before = vol_of(c) as f64;
        let after = (vol_of(c) + dv) as f64;
        d += cell.lambda_volume
            * ((after - cell.target_volume).powi(2) - (before - cell.target_volume).powi(2));
    }
    if world.any_surface() {
        for (c, ds) in world.surface_deltas(site, b) {
            if c == MEDIUM {
                continue;
            }
            let cell = &world.cells[c as usize];
            let before = surf_of(c) as f64;
            let after = (surf_of(c) + ds) as f64;
            d += cell.lambda_surface
                * ((after - cell.target_surface).powi(2) - (before - cell.target_surface).powi(2));
        }
    }
    d += world.delta_contact(site, b);
    d += world.delta_chemotaxis(site, pick, b);
    if world.any_membrane() {
        d += world.delta_membrane(site, b);
    }
    if world.any_junction() {
        d += world.delta_junction(site, b);
    }
    if world.any_external() {
        d += world.delta_external(site, b);
    }
    d
}

#[cfg(test)]
mod tests {
    use crate::energy::ContactMatrix;
    use crate::lattice::{Boundary, Lattice, Neighborhood};
    use crate::sweep::Cpm;
    use crate::world::World;

    // A packed 2-type checkerboard of cells; returns (mean|vol-target|, min volume).
    fn build() -> World {
        let dim = 40usize;
        let cell = 8usize;
        let lat = Lattice::new([dim, dim, dim], [Boundary::Periodic; 3], Neighborhood::new(true, 2));
        let mut w = World::new(lat, 10.0);
        let g = dim / cell;
        for gz in 0..g { for gy in 0..g { for gx in 0..g {
            let t = 1 + ((gx + gy + gz) % 2) as u16;
            let a = w.add_cell(t, (cell*cell*cell) as f64, 1.0, 0.0, 0.0);
            for z in gz*cell..gz*cell+cell { for y in gy*cell..gy*cell+cell { for x in gx*cell..gx*cell+cell {
                let i = w.lattice.index(x,y,z); w.paint(i, a);
            }}}
        }}}
        let mut m = ContactMatrix::new(3);
        m.set(0,1,16.0); m.set(0,2,16.0); m.set(1,2,11.0);
        w.set_contact_matrix(m);
        w.recompute_trackers();
        w
    }

    fn stats(w: &World) -> (f64, i64) {
        let mut s = 0.0; let mut n = 0.0; let mut min = i64::MAX;
        for c in w.cells.iter().skip(1) {
            s += (c.volume as f64 - c.target_volume).abs(); n += 1.0;
            min = min.min(c.volume);
        }
        (s / n, min)
    }

    #[test]
    fn parallel_matches_sequential_statistics() {
        // Same initial state, same MCS count. Parallel is approximate + not
        // bit-identical, but its aggregate statistics (mean deviation from target
        // volume, no cell death) must track the sequential engine closely.
        let mut seq = Cpm::new(build(), 1);
        seq.step(40);
        let (dev_s, min_s) = stats(&seq.world);

        let mut par = Cpm::new(build(), 1);
        par.step_parallel(40, 8);
        let (dev_p, min_p) = stats(&par.world);

        assert!(min_s > 0 && min_p > 0, "no cell should vanish (seq {min_s}, par {min_p})");
        // parallel mean deviation within 2x of sequential (both small vs target 512)
        assert!(dev_p < dev_s.max(5.0) * 2.0,
            "parallel deviation {dev_p:.1} should track sequential {dev_s:.1}");
    }

    #[test]
    fn parallel_conserves_trackers_exactly() {
        // After a parallel sweep, cell.volume must equal a fresh recount from the
        // lattice — the per-phase delta barriers keep trackers exact even though
        // acceptance used slightly-stale cross-block volumes.
        let mut par = Cpm::new(build(), 2);
        par.step_parallel(15, 8);
        let recount_vol: Vec<i64> = {
            let w = &par.world;
            let mut v = vec![0i64; w.cells.len()];
            for i in 0..w.lattice.n_sites() { v[w.lattice.owner(i) as usize] += 1; }
            v
        };
        for c in 1..par.world.cells.len() {
            assert_eq!(par.world.cells[c].volume, recount_vol[c],
                "cell {c} tracked volume drifted from the true lattice count");
        }
    }
}

impl crate::sweep::Cpm {
    /// Parallel checkerboard sweep. `block` is the tile side (>=2; e.g. 8). Uses
    /// the current rayon thread pool (set `RAYON_NUM_THREADS` to cap it).
    pub fn step_parallel(&mut self, mcs: u64, block: usize) {
        assert!(block >= 2, "block side must be >= 2");
        assert!(!self.world.any_length(),
            "parallel sweep does not support the length constraint; use step()");
        let (nx, ny, nz) = (
            self.world.lattice.dims_x(),
            self.world.lattice.dims_y(),
            self.world.lattice.dims_z(),
        );
        let nbx = nx.div_ceil(block);
        let nby = ny.div_ceil(block);
        let nbz = nz.div_ceil(block);
        let n_blocks = nbx * nby * nbz;
        let n_colors = if nz > 1 { 8 } else { 4 };
        let temp = self.world.temperature;
        let seed_base = self.next_seed();

        for step in 0..mcs {
            for color in 0..n_colors {
                let (cbx, cby, cbz) = (color & 1, (color >> 1) & 1, (color >> 2) & 1);
                let block_ids: Vec<usize> = (0..n_blocks)
                    .filter(|&bid| {
                        let bx = bid % nbx;
                        let by = (bid / nbx) % nby;
                        let bz = bid / (nbx * nby);
                        bx % 2 == cbx && by % 2 == cby && bz % 2 == cbz
                    })
                    .collect();

                let world: &World = &self.world;
                // Each block sweeps ALL its sites once (a full n_sites/MCS sweep, like
                // CC3D). Interior no-ops skip cheaply; no boundary set is maintained,
                // so there is no sequential refresh barrier — only the delta merge.
                let outcomes: Vec<HashMap<CellId, CellDelta>> = block_ids
                    .par_iter()
                    .map(|&bid| {
                        let bx = bid % nbx;
                        let by = (bid / nbx) % nby;
                        let bz = bid / (nbx * nby);
                        let (x0, x1) = (bx * block, ((bx + 1) * block).min(nx));
                        let (y0, y1) = (by * block, ((by + 1) * block).min(ny));
                        let (z0, z1) = (bz * block, ((bz + 1) * block).min(nz));
                        let mut rng = SmallRng::seed_from_u64(
                            seed_base
                                ^ (step.wrapping_mul(0x9E3779B9))
                                ^ ((color as u64).wrapping_mul(0x85EBCA77))
                                ^ (bid as u64).wrapping_mul(0xC2B2AE3D),
                        );
                        let mut local: HashMap<CellId, CellDelta> = HashMap::new();
                        for z in z0..z1 {
                            for y in y0..y1 {
                                for x in x0..x1 {
                                    let s = world.lattice.index(x, y, z);
                                    let target = world.lattice.owner(s);
                                    let neighbors = world.lattice.neighbors(s);
                                    if neighbors.is_empty() {
                                        continue;
                                    }
                                    let pick = neighbors[rng.gen_range(0..neighbors.len())];
                                    let source = world.lattice.owner(pick);
                                    if source == target {
                                        continue;
                                    }
                                    let dh = par_delta_h(world, s, pick, source, &local);
                                    let accept =
                                        dh <= 0.0 || rng.gen::<f64>() < (-dh / temp).exp();
                                    if !accept {
                                        continue;
                                    }
                                    if world.any_connectivity() {
                                        let constrained = if target == MEDIUM {
                                            world.connectivity_medium
                                        } else {
                                            world.type_is_constrained(
                                                world.cells[target as usize].cell_type,
                                            )
                                        };
                                        if constrained && !world.would_stay_connected(s, target) {
                                            continue;
                                        }
                                    }
                                    let sd = world.surface_deltas(s, source);
                                    world.lattice.set_owner(s, source);
                                    let (xf, yf, zf) = (x as f64, y as f64, z as f64);
                                    let m =
                                        [xf * xf, yf * yf, zf * zf, xf * yf, xf * zf, yf * zf];
                                    let da = local.entry(target).or_default();
                                    da.dvol -= 1;
                                    da.dcom[0] -= xf; da.dcom[1] -= yf; da.dcom[2] -= zf;
                                    for k in 0..6 { da.dmom[k] -= m[k]; }
                                    let db = local.entry(source).or_default();
                                    db.dvol += 1;
                                    db.dcom[0] += xf; db.dcom[1] += yf; db.dcom[2] += zf;
                                    for k in 0..6 { db.dmom[k] += m[k]; }
                                    for (c, ds) in sd {
                                        local.entry(c).or_default().dsurf += ds;
                                    }
                                }
                            }
                        }
                        local
                    })
                    .collect();

                // Barrier: sum every block's per-cell deltas into the trackers (exact).
                for oc in outcomes {
                    for (c, d) in oc {
                        let cell = &mut self.world.cells[c as usize];
                        cell.volume += d.dvol;
                        cell.surface += d.dsurf;
                        for k in 0..3 { cell.com_sum[k] += d.dcom[k]; }
                        for k in 0..6 { cell.moment_sum[k] += d.dmom[k]; }
                    }
                }
            }

            self.world.advance_fields();
        }
    }
}
