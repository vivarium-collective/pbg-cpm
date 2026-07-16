# Glazier & Graner (1993) — Simulation of the differential-adhesion driven rearrangement of biological cells

*Phys. Rev. E **47**(3), 2128–2154. DOI: [10.1103/PhysRevE.47.2128](https://doi.org/10.1103/PhysRevE.47.2128).*
BibTeX key: `glazier1993` · PDF: `workspace/references/papers/glazier-graner-1993.pdf`

The source paper this investigation reproduces. An extended large-Q Cellular
Potts model shows that **differential adhesion + fluctuations alone** reproduce
the full range of cell-rearrangement phenomena.

## Model (Sec. II)
- Hamiltonian: type-dependent contact energies `J(τ,τ')` + area constraint
  `λ(a-A)²`. Types: light `l`, dark `d`, medium `M`.
- Surface tensions (Eq. 5): `γ_ld = J_ld-(J_dd+J_ll)/2`, `γ_lM = J_lM-J_ll/2`,
  `γ_dM = J_dM-J_dd/2`.
- Second-nearest-neighbour (Moore-8) square lattice. **1 MCS = 16 site sweeps.**
- **T=0 annealing (Sec. II D 2):** statistics/displays are taken after **2 MCS
  of T=0 annealing applied to a *copy*** of the pattern (the running spin array
  is untouched) — removes lattice crumpling/defects without over-relaxing.
- **Initial condition (Sec. II D 3):** a rectangular brick tiling of one cell
  type equilibrated at T=5 for 400 MCS → a rounded ~1000-cell aggregate (Fig 4b);
  cell types are then assigned (random for sorting; clean top/bottom split for
  engulfment).

## Quantitative anchors for the reproduced studies
- **Cell sorting (Fig 12-13):** random mix → dark cluster wrapped by a light
  monolayer; monolayer by ~600 MCS, dark cluster rounds by ~13 500 MCS.
  Light-dark heterotypic contact collapses; light comes to own the medium
  surface. Logarithmically slow.
- **Engulfment (Fig 18-19):** energies as sorting; **clean top-light/bottom-dark
  split (Fig 18a).** Light slowly engulfs dark; **still incomplete at 10 000 MCS**
  — a linear fit (R²=0.987) extrapolates complete engulfment at **~11 000 MCS.**
  The two blocks stay coherent (heterotypic contact stays low — it does *not*
  balloon like mixing).
- **Position reversal (Fig 20-21):** raise `γ_lM=23` (`J_lM=30`) → **dark forms
  the outer monolayer.** Dark monolayer complete by ~40 MCS; **Fig 21(b): the
  light-Medium correlation falls to ~0** (dark owns essentially the whole surface).
- **Partial sorting (Fig 22-24):** `γ_ld=7.5` violates the Young condition →
  sorting **stalls**: clusters coarsen and trap heterotypic inclusions, **no
  light monolayer forms,** logarithmic at all times.
- **Checkerboard (Fig 7-9):** negative `γ_ld=-3` → heterotypic contact dominates
  (mixing), homotypic contact collapses.

## Why this note exists
The 2026-07-16 investigation found that the reproduction's measurement path
scrambled cell types (see the engine fix + `tests/test_gg1993_anneal_fidelity.py`).
The paper's Fig 18a clean interface and Fig 21b light-Medium→0 were the
ground-truth used to rebuild the acceptance tests.
