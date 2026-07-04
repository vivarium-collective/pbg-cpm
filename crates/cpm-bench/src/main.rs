use cpm_core::energy::ContactMatrix;
use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::sweep::Cpm;
use cpm_core::world::World;
use std::time::Instant;

fn main() {
    let dim = 50usize;
    let mcs = 20u64;
    let lat = Lattice::new([dim, dim, dim], [Boundary::Periodic; 3], Neighborhood::new(true, 2));
    let mut w = World::new(lat, 10.0);
    // ~125 cells in a 5x5x5 grid of 8^3 blocks
    let step = dim / 5;
    for gz in 0..5 {
        for gy in 0..5 {
            for gx in 0..5 {
                let t = 1 + ((gx + gy + gz) % 2) as u16;
                let a = w.add_cell(t, 512.0, 1.0, 0.0, 0.0);
                let (x0, y0, z0) = (gx * step, gy * step, gz * step);
                for z in z0..z0 + 8 {
                    for y in y0..y0 + 8 {
                        for x in x0..x0 + 8 {
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

    let n_sites = w.lattice.n_sites();
    let mut cpm = Cpm::new(w, 1);
    let t0 = Instant::now();
    cpm.step(mcs);
    let secs = t0.elapsed().as_secs_f64();
    let attempts = mcs as f64 * n_sites as f64;
    println!("3D bench {dim}^3, {mcs} MCS: {:.2} MCS/s, {:.2e} copy-attempts/s",
             mcs as f64 / secs, attempts / secs);
}
