# pbg-cpm vs CompuCell3D — head-to-head benchmark

A like-for-like single-core throughput comparison between the pbg-cpm Rust engine
and CompuCell3D (CC3D) 4.10 on an **identical** model.

## Model (identical in both engines)

- 50×50×50 lattice, **periodic** boundaries
- 125 cells: a 5×5×5 grid of 8³ blocks (`blocks.piff` == `crates/cpm-bench/src/main.rs` seeding)
- Energy terms: **Volume** (target 512, λ=1) + **Contact** (J: medium–cell 16, cell–cell 11), nothing else
- Neighbor order 2 (18-neighbour), temperature 10
- One MCS = **125 000 pixel-copy attempts** (`n_sites`) — verified equal in both:
  CC3D logs `total number of pixel copy attempts=125000` and uses its full-lattice
  `Metropolis Fast` sweep, the same as our `sweep.rs` inner loop.

Both engines short-circuit same-cell picks (~80% of attempts), so the ~25k
energy evaluations/MCS are also matched. This is a fair, single-threaded,
same-machine comparison.

## Results (Apple M-series, single-threaded, release)

| Engine | MCS/s | copy-attempts/s |
|--------|-------|-----------------|
| pbg-cpm (before surface guard) | 67.7 | 8.47 M |
| **pbg-cpm (current)** | **~83** | **~10.4 M** |
| CompuCell3D 4.10 | ~85 | ~10.7 M |

The head-to-head exposed a real inefficiency: our `delta_hamiltonian` always ran
the surface-energy term (a neighbour scan + heap allocation per attempt) even
when λ_surface = 0. CC3D loads no surface plugin here, so it never paid it.
Guarding the term (`World::any_surface`) + dropping the allocation (`SmallVec`)
closed the 1.24× gap — pbg-cpm is now at **parity** with CC3D's mature C++ core.

## Reproduce

pbg-cpm side:

    cargo run --release -p cpm-bench

CC3D side (needs a conda/mamba env with `compucell3d=4.10`):

    micromamba create -n cc3d -c compucell3d -c conda-forge compucell3d=4.10.0 python=3.12
    micromamba run -n cc3d python bench/cc3d/cc3d_bench.py 30

`gen_pif.py` regenerates `blocks.piff` if the Rust seeding changes.
