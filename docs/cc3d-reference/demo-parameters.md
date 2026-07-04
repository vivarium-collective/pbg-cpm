# CC3D reference parameters (for faithful reimplementation)

Exact CompuCell3D demo parameters and formulas we reproduce, with sources. Where a
per-version constant could not be retrieved verbatim, the CC3D-typical value is used and
marked; in those cases the *validated behavior* (not the exact constant) is the fidelity
criterion.

## Chemotaxis energy (CC3D Chemotaxis plugin)

For a pixel-copy attempt, the effective-energy change from chemotaxis is:

    ΔH_chem = -λ · ( c(x_destination) - c(x_source) )

- `x_destination` = the pixel being overwritten (the target site `s` in our sweep).
- `x_source` = the neighbor pixel whose spin is copied (`n`).
- `λ` = chemotaxis coefficient of the **cell doing the moving** (the source cell / new owner), per field, set by cell type (`ChemotaxisByType Lambda`).
- Sign: **λ > 0 ⇒ move up-gradient (toward higher concentration)**; λ < 0 ⇒ away.
- Medium does not chemotax (λ = 0).

Source: CC3D Reference Manual, Chemotaxis plugin.

## Diffusion (CC3D DiffusionSolverFE)

Field evolves by  ∂c/∂t = D∇²c − k·c + secretion, solved explicitly (forward-Euler) each MCS:

    c_new[i] = c[i] + Δt·D·Σ_neighbors(c[j] - c[i]) − Δt·k·c[i]   (+ secretion at secreting cells)

- D = GlobalDiffusionConstant, k = GlobalDecayConstant.
- Secretion: cells of a secreting type add their rate to `c` at each of their pixels per step.
- Explicit-scheme stability requires Δt·D·(2·ndim) ≤ 1; sub-step if needed.

Source: CC3D Reference Manual, DiffusionSolverFE.

---

## Demo 1 — cellsort (cellsort_2D / cellsort_3D)  [EXACT]

Canonical differential-adhesion cell sorting (Steinberg differential adhesion hypothesis).

- Lattice: 100×100×1 (2D); 3D uses a comparable cube. Temperature **10.0**. NeighborOrder **2**.
- Cell types: **Medium, Condensing, NonCondensing**.
- Volume plugin: TargetVolume **25**, LambdaVolume **2.0** (both types). **3D note:** CC3D's
  LambdaVolume 2.0 is a 2D value; in 3D's higher coordination the contact energies (up to 16)
  overwhelm λ=2 and squeeze the less-cohesive NonCondensing cells to zero volume (a CPM
  volume-constraint artifact). The 3D demo uses **LambdaVolume 6.0** so both types survive and
  sort correctly (Condensing engulfed) — verified: with λ=2 all NonCondensing cells vanish; with
  λ=6 both types keep volume ~20–23.
- Contact energy matrix (J):

  | pair | J |
  |------|---|
  | Medium–Medium (JMM) | 0 |
  | Condensing–Condensing (JCC) | 2 |
  | Condensing–NonCondensing (JCN) | 11 |
  | NonCondensing–NonCondensing (JNN) | 16 |
  | Condensing–Medium (JCM) | 16 |
  | NonCondensing–Medium (JNM) | 16 |

- **Expected behavior (validation):** Condensing cells (cohesive, JCC=2) sort into a single
  central cluster **engulfed by** NonCondensing; heterotypic (Condensing–NonCondensing)
  boundary length decreases monotonically over time; total effective energy decreases.

Source: CC3D docs, "Building CC3DML-Based Simulations" (values JCC=2, JCN=11, JNN=JCM=JNM=16, JMM=0).

## Demo 2 — bacterium_macrophage  [structure EXACT; field constants CC3D-typical]

Bacterium secretes a diffusive chemoattractant; macrophage detects the gradient and hunts it.

- Cell types: **Medium, Bacterium, Macrophage**.
- Volume: TargetVolume 25, LambdaVolume 2.0.
- Field **ATTR** (DiffusionSolverFE): GlobalDiffusionConstant **0.10**, GlobalDecayConstant **5e-5**.
- Secretion: **Bacterium secretes ATTR at rate 100** (per pixel per MCS).
- Chemotaxis: **Macrophage λ = +ve toward ATTR** (up-gradient); Bacterium λ = 0.
  (Exact λ magnitude varies by CC3D version; tune to CC3D-typical ~1e3 scale so the macrophage
  visibly climbs the gradient, then validate the behavior.)
- Contact: adhesion values in the CC3D-typical range (Medium–cell ~16, cell–cell ~4–8) so cells
  stay compact; exact values secondary to the chemotaxis behavior.
- **Expected behavior (validation):** the Macrophage's net displacement is **up the ATTR
  gradient / toward the Bacterium** — its distance to the bacterium decreases over the run, and
  its center-of-mass moves toward higher mean ATTR concentration. (This is the fidelity check,
  per CC3D's "the macrophage hunts the bacterium.")

Sources: CC3D bacterium_macrophage demo description (nanoHUB, CC3D manual); DiffusionSolverFE
example constants (D≈0.1, decay≈5e-5, Bacterium secretes ATTR at 100).

## Demo 3 — cell growth & division (mitosis)  [standard CC3D pattern]

Cells grow and divide — proliferating tissue.

- Cell type: **Cell** (single proliferating type). TargetVolume start **25**, LambdaVolume **2.0**.
- Growth: each MCS, `targetVolume += growthRate` (growthRate ≈ 1.0) for growing cells.
- Mitosis: when a cell's **volume ≥ 2× (≈50)**, divide it into two along a plane through its
  center of mass; the daughter is a new cell of the same type; **reset both daughters'
  TargetVolume to 25**. (CC3D MitosisSteppable, divide-along-plane / random orientation.)
- Contact: Medium–Cell ~16, Cell–Cell ~4 so the colony stays cohesive.
- **Expected behavior (validation):** cell **count grows ~geometrically** (doubling per growth
  cycle); individual cell volumes stay **bounded** in ~[25, 50] (no runaway growth, no
  vanishing cells); total mass increases.

Source: CC3D standard cell-growth + MitosisSteppable pattern.
