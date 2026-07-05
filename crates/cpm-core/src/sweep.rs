use crate::world::World;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub struct Cpm {
    pub world: World,
    rng: SmallRng,
}

impl Cpm {
    pub fn new(world: World, seed: u64) -> Cpm {
        Cpm { world, rng: SmallRng::seed_from_u64(seed) }
    }

    pub fn step(&mut self, mcs: u64) {
        let t = self.world.temperature;
        for _ in 0..mcs {
            // One MCS = |B| attempts, where |B| is the boundary-site count at the
            // start of the sweep. Sampling only boundary sites is exactly the full
            // sweep minus its guaranteed-no-op interior attempts (see world.rs).
            let m = self.world.boundary_len();
            for _ in 0..m {
                let blen = self.world.boundary_len();
                if blen == 0 {
                    break;
                }
                let s = self.world.boundary_site(self.rng.gen_range(0..blen));
                let neighbors = self.world.lattice.neighbors(s);
                if neighbors.is_empty() {
                    continue;
                }
                let pick = neighbors[self.rng.gen_range(0..neighbors.len())];
                let source_owner = self.world.lattice.owner(pick);
                if source_owner == self.world.lattice.owner(s) {
                    continue;
                }
                let dh = self.world.delta_hamiltonian(s, source_owner)
                    + self.world.delta_chemotaxis(s, pick, source_owner)
                    + if self.world.any_membrane() {
                        self.world.delta_membrane(s, source_owner)
                    } else {
                        0.0
                    }
                    + if self.world.any_junction() {
                        self.world.delta_junction(s, source_owner)
                    } else {
                        0.0
                    }
                    + if self.world.any_length() {
                        self.world.delta_length(s, source_owner)
                    } else {
                        0.0
                    }
                    + if self.world.any_external() {
                        self.world.delta_external(s, source_owner)
                    } else {
                        0.0
                    };
                let accept = dh <= 0.0 || self.rng.gen::<f64>() < (-dh / t).exp();
                if accept {
                    if self.world.any_connectivity() {
                        let target = self.world.lattice.owner(s);
                        let constrained = if target == 0 {
                            self.world.connectivity_medium
                        } else {
                            self.world
                                .type_is_constrained(self.world.cells[target as usize].cell_type)
                        };
                        if constrained && !self.world.would_stay_connected(s, target) {
                            continue; // reject: would fragment `target`
                        }
                    }
                    self.world.apply_flip(s, source_owner);
                }
            }
            self.world.advance_fields();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::ContactMatrix;
    use crate::lattice::{Boundary, Lattice, Neighborhood};

    #[test]
    fn deterministic_under_seed() {
        fn run(seed: u64) -> Vec<i64> {
            let lat = Lattice::new([20, 20, 1], [Boundary::Periodic; 3], Neighborhood::new(false, 2));
            let mut w = World::new(lat, 10.0);
            let a = w.add_cell(1, 25.0, 2.0, 20.0, 0.0);
            for y in 5..10 { for x in 5..10 { let i = w.lattice.index(x,y,0); w.paint(i, a); } }
            let mut m = ContactMatrix::new(2);
            m.set(0, 1, 16.0);
            w.set_contact_matrix(m);
            w.recompute_trackers();
            let mut cpm = Cpm::new(w, seed);
            cpm.step(5);
            cpm.world.cells.iter().map(|c| c.volume).collect()
        }
        assert_eq!(run(42), run(42));
    }

    #[test]
    fn single_cell_relaxes_toward_target_volume() {
        let lat = Lattice::new([30, 30, 1], [Boundary::Periodic; 3], Neighborhood::new(false, 2));
        let mut w = World::new(lat, 15.0);
        // start far above target: 100 sites, target 49
        let a = w.add_cell(1, 49.0, 5.0, 28.0, 0.0);
        for y in 5..15 { for x in 5..15 { let i = w.lattice.index(x,y,0); w.paint(i, a); } }
        let mut m = ContactMatrix::new(2);
        m.set(0, 1, 6.0);
        w.set_contact_matrix(m);
        w.recompute_trackers();
        let start = w.cells[a as usize].volume;
        let mut cpm = Cpm::new(w, 7);
        cpm.step(200);
        let end = cpm.world.cells[a as usize].volume;
        assert!(end < start, "volume should shrink toward target: {start} -> {end}");
        assert!((end - 49).abs() < 30, "should be near target 49, got {end}");
    }
}
