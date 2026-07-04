use cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use cpm_core::sweep::Cpm;
use cpm_core::world::World;

const DIMS: [usize; 3] = [16, 16, 1];

// count pinched medium faces between two different junction-enabled cells (the P quantity)
fn gap_faces(w: &World, jt: &[bool]) -> u32 {
    let (nx, ny) = (DIMS[0], DIMS[1]);
    let enabled = |o: u32| o != 0 && *jt.get(w.cells[o as usize].cell_type as usize).unwrap_or(&false);
    let mut p = 0u32;
    for y in 0..ny {
        for x in 0..nx {
            let i = w.lattice.index(x, y, 0);
            if w.lattice.owner(i) != 0 {
                continue;
            }
            if x >= 1 && x + 1 < nx {
                let a = w.lattice.owner(i - 1);
                let b = w.lattice.owner(i + 1);
                if enabled(a) && enabled(b) && a != b { p += 1; }
            }
            if y >= 1 && y + 1 < ny {
                let a = w.lattice.owner(i - nx);
                let b = w.lattice.owner(i + nx);
                if enabled(a) && enabled(b) && a != b { p += 1; }
            }
        }
    }
    p
}

// two 3-wide cells pressed together as a 6x8 block, split down the middle
fn run(junctions: bool) -> (u32, u64) {
    let lat = Lattice::new(DIMS, [Boundary::NoFlux; 3], Neighborhood::new(false, 2));
    // temp=10 / lambda=80 tuned up from the brief's 22/12 starting point: at 22/12 the
    // `with` run still transiently opened 1-4 gap faces over 30 MCS across many seeds
    // (verified by an ad-hoc 40-seed sweep), so both knobs were raised together per the
    // brief's guidance until `with` was 0 across all 40 seeds with margin to spare.
    let mut w = World::new(lat, 10.0); // hot -> boundary churns, gaps want to open
    let a = w.add_cell(1, 24.0, 1.0, 0.0, 0.0);
    let b = w.add_cell(1, 24.0, 1.0, 0.0, 0.0);
    for y in 4..12 {
        for x in 5..8 {
            w.paint(w.lattice.index(x, y, 0), a);
        }
        for x in 8..11 {
            w.paint(w.lattice.index(x, y, 0), b);
        }
    }
    // weak adhesion so the two cells don't strongly stick on their own
    let mut m = cpm_core::energy::ContactMatrix::new(2);
    m.set(0, 1, 4.0);
    m.set(1, 1, 4.0);
    w.set_contact_matrix(m);
    if junctions {
        w.set_junction(1, true);
        w.set_junction_lambda(80.0);
    }
    w.recompute_trackers();
    // Both cells are cell_type 1; track gaps between them regardless of whether the
    // junction force is active in this run. (A `jt` keyed off `junctions` itself would
    // make the control's gap-face count tautologically 0, since then no cell_type would
    // ever be marked "enabled" -- that would make `without > 0` unfalsifiable.)
    let jt = vec![false, true];
    let n = w.lattice.n_sites();
    let snap = |cpm: &Cpm| -> Vec<u32> { (0..n).map(|i| cpm.world.lattice.owner(i)).collect() };
    let mut cpm = Cpm::new(w, 5);
    let mut worst = 0u32;
    let mut churn = 0u64; // total owner changes -> proves the with-run isn't frozen
    for _ in 0..30 {
        let before = snap(&cpm);
        cpm.step(1);
        let after = snap(&cpm);
        churn += before.iter().zip(&after).filter(|(a, b)| a != b).count() as u64;
        worst = worst.max(gap_faces(&cpm.world, &jt));
    }
    (worst, churn)
}

#[test]
fn junctions_prevent_gaps_between_cells() {
    let (with, with_churn) = run(true);
    let (without, _) = run(false);
    assert_eq!(with, 0, "junctions should keep the two cells gap-free, saw {with}");
    assert!(without > 0, "control must open a gap or the test is vacuous, saw {without}");
    // ...and the with-run seals gaps while STILL relaxing (the A-B interface churns
    // freely -- the junction term is 0 for cell-cell moves), not by freezing.
    assert!(with_churn > 0, "with-junctions run froze (churn 0) -> 'gap-free' is vacuous");
}

#[test]
fn junction_run_is_deterministic() {
    assert_eq!(run(true), run(true));
}
