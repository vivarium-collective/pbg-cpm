use cpm_core::energy::ContactMatrix;
use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::sweep::Cpm;
use cpm_core::world::World;
use std::time::Instant;

// dim^3 lattice tiled with contiguous cells of side `cell` (periodic), a 2-type
// checkerboard. Same physics as before; `dim` scales the problem size.
fn build(dim: usize, cell: usize) -> (World, usize, usize) {
    let lat = Lattice::new([dim, dim, dim], [Boundary::Periodic; 3], Neighborhood::new(true, 2));
    let mut w = World::new(lat, 10.0);
    let g = dim / cell;
    let tv = (cell * cell * cell) as f64;
    for gz in 0..g {
        for gy in 0..g {
            for gx in 0..g {
                let t = 1 + ((gx + gy + gz) % 2) as u16;
                let a = w.add_cell(t, tv, 1.0, 0.0, 0.0);
                let (x0, y0, z0) = (gx * cell, gy * cell, gz * cell);
                for z in z0..z0 + cell {
                    for y in y0..y0 + cell {
                        for x in x0..x0 + cell {
                            let i = w.lattice.index(x, y, z);
                            w.paint(i, a);
                        }
                    }
                }
            }
        }
    }
    let mut m = ContactMatrix::new(3);
    m.set(0, 1, 16.0);
    m.set(0, 2, 16.0);
    m.set(1, 2, 11.0);
    w.set_contact_matrix(m);
    w.recompute_trackers();
    (w, dim * dim * dim, g * g * g)
}

// mean over cells of |volume - target|, a stability/agreement statistic
fn mean_abs_dev(w: &World) -> f64 {
    let mut s = 0.0;
    let mut n = 0.0;
    for c in w.cells.iter().skip(1) {
        s += (c.volume as f64 - c.target_volume).abs();
        n += 1.0;
    }
    s / n
}

fn main() {
    let mcs = 20u64;
    let mode = std::env::args().nth(1).unwrap_or_default();
    let dim: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let block: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let cell = 8usize;

    if mode == "par" {
        let (w, n_sites, n_cells) = build(dim, cell);
        let mut cpm = Cpm::new(w, 1);
        let t0 = Instant::now();
        cpm.step_parallel(mcs, block);
        let secs = t0.elapsed().as_secs_f64();
        let threads = rayon::current_num_threads();
        println!(
            "PARALLEL {dim}^3 ({n_cells} cells), {mcs} MCS, block {block}, {threads} thr: {:.1} MCS/s | mean|vol-tgt| {:.1}",
            mcs as f64 / secs, mean_abs_dev(&cpm.world)
        );
    } else {
        let (w, n_sites, n_cells) = build(dim, cell);
        let _ = n_sites;
        let mut cpm = Cpm::new(w, 1);
        let t0 = Instant::now();
        cpm.step(mcs);
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "SEQUENTIAL {dim}^3 ({n_cells} cells), {mcs} MCS: {:.1} MCS/s | mean|vol-tgt| {:.1}",
            mcs as f64 / secs, mean_abs_dev(&cpm.world)
        );
    }
}
