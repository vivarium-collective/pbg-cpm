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

## Parallel: beating CC3D

`step_parallel` runs a checkerboard-coloured sweep across CPU cores (opt-in;
default `step` stays sequential + bit-exact). Same physics (validated: cell
trackers stay exact, aggregate statistics track the sequential engine).

Direct head-to-head on an identical **dense 96³ / 1728-cell** model
(`gen_pif.py 96 8 blocks_96.piff`), Apple M4 Max:

| Engine | MCS/s | vs CC3D |
|--------|-------|---------|
| CompuCell3D 4.10 (single-thread) | 9.1 | 1× |
| pbg-cpm `step_parallel`, 12 cores | 52.0 | **5.7×** |
| pbg-cpm `step_parallel`, 16 cores | 57.3 | **6.3×** |

Pure parallel efficiency (128³, self-relative): 1→4→8→12 threads = 1× → 3.5× →
6.3× → **7.9×** (≈66% efficiency on the 12 performance cores).

Reproduce:

    CC3D_DIM=96 CC3D_PIF=blocks_96.piff micromamba run -n cc3d python bench/cc3d/cc3d_bench.py 10
    RAYON_NUM_THREADS=12 cargo run --release -p cpm-bench -- par 96 16

## Reproduce

pbg-cpm side:

    cargo run --release -p cpm-bench

CC3D side (needs a conda/mamba env with `compucell3d=4.10`):

    micromamba create -n cc3d -c compucell3d -c conda-forge compucell3d=4.10.0 python=3.12
    micromamba run -n cc3d python bench/cc3d/cc3d_bench.py 30

`gen_pif.py` regenerates `blocks.piff` if the Rust seeding changes.
