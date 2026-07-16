"""The 11 Glazier & Graner (1993) simulation examples, one per parameter set.

Each entry carries the paper's exact energies (J_ll, J_dd, J_ld, J_lM, J_dM),
temperature T, area-constraint lambda, per-type target areas, initial condition,
the pattern-panel timepoints (paper MCS), the time-series run length, and the
temperature / lambda sweep points where the paper has them.

Types: light=2, dark=1, medium=0.  All MCS are *paper* MCS (engine = x16).
"""

from .types import LIGHT, DARK, MEDIUM
from .engine import surface_tensions


def _J(J_ll, J_dd, J_ld, J_lM, J_dM):
    return dict(J_ll=J_ll, J_dd=J_dd, J_ld=J_ld, J_lM=J_lM, J_dM=J_dM)


# Shared energies for the cell-sorting family (sorting, engulfment, cavity)
_SORT_J = _J(14, 2, 11, 16, 16)  # gamma_ld=3, gamma_lM=9, gamma_dM=15

STUDIES = {}


def _add(**kw):
    kw.setdefault("lam", 1.0)
    kw.setdefault("target", {LIGHT: 40.0, DARK: 40.0})
    kw.setdefault("display_anneal_mcs", 2)
    kw.setdefault("unannealed_display", False)
    kw.setdefault("frac_light", 0.5)
    kw.setdefault("temps", None)
    kw.setdefault("lambdas", None)
    kw.setdefault("series_linear", False)
    kw["gamma"] = surface_tensions(**kw["J"])
    STUDIES[kw["slug"]] = kw


# 1. Annealing (Figs 2, 3) -- all-light equilibrated pattern relaxed at T=0
_add(
    slug="annealing", title="Annealing", section="II — model characterization",
    J=_J(2, 2, 2, 8, 8), temperature=0.0, ic="equilibrated_light",
    image_mcs=[0, 2, 20], series_max_mcs=40, series_linear=True,
    unannealed_display=True,  # the frames ARE the T=0 annealing steps
    figures=["fig2", "fig3"],
    notes="J_ll=2, J_lM=8 (gamma_lM=7), T=0. Fig2: <n> bulk/total & moments vs "
          "MCS 0-40. Fig3: cell-wall detail at 0 (unannealed), 2, 20 MCS.",
)

# 2. Global pattern equilibration (Figs 4, 5) -- builds the canonical IC
_add(
    slug="global_equilibration", title="Global pattern equilibration",
    section="II — model characterization",
    J=_J(2, 2, 2, 8, 8), temperature=5.0, ic="brick_equilibrate",
    image_mcs=[0, 400], series_max_mcs=600, series_linear=True,
    display_anneal_mcs=10,
    figures=["fig4", "fig5"],
    notes="Rectangular brick tiling -> rounded disk after 400 MCS at T=5. "
          "Stats after 10 MCS T=0. Fig5: total boundary length, moments, "
          "light-Medium fractional length vs MCS 0-600.",
)

# 3. Checkerboard -- negative surface tension (Figs 7, 8, 9, Table I)
_add(
    slug="checkerboard", title="Checkerboard (negative surface tension)",
    section="III A", J=_J(10, 8, 6, 12, 12), temperature=10.0,
    ic="equilibrated_random",
    image_mcs=[10, 100, 1000, 2000], series_max_mcs=2000,
    temps=[0, 2, 5, 10, 15, 40],
    figures=["fig7", "fig8", "fig9", "table1"],
    notes="gamma_ld=-3, gamma_lM=7, gamma_dM=8. Random IC. Fig7 patterns "
          "10/100/1000/2000 MCS. Fig8 total/frac/<n>/moments. Fig9 temp sweep "
          "(ll, ld, lM interfaces). Table I moments vs T.",
)

# 4. Cell sorting (Figs 12-16, Tables II, III)
_add(
    slug="cell_sorting", title="Cell sorting", section="III B",
    J=dict(_SORT_J), temperature=10.0, ic="equilibrated_random",
    image_mcs=[0, 10, 100, 1000, 3000, 4000, 5000, 13500],
    series_max_mcs=13500,
    temps=[2, 5, 10, 15, 20, 40, 80],
    lambdas=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    figures=["fig12", "fig13", "fig14", "fig15", "fig16", "table2", "table3"],
    notes="gamma_ld=3, gamma_lM=9, gamma_dM=15. Random IC. Fig12 8 patterns to "
          "13500 MCS. Fig13 lengths+correlation. Fig14 <n>/moments. Fig15 temp "
          "sweep. Fig16 lambda sweep. Tables II (T), III (lambda).",
)

# 5. Engulfment (Figs 18, 19)
_add(
    slug="engulfment", title="Engulfment", section="III C",
    J=dict(_SORT_J), temperature=10.0, ic="half_split",
    image_mcs=[0, 1000, 5000, 10000, 20000], series_max_mcs=20000,
    figures=["fig18", "fig19"],
    notes="Same energies as cell sorting. IC: upper half light, lower half "
          "dark (clean interface, Fig 18a). Paper runs to 10000 MCS (still "
          "incomplete; linear fit R^2=0.987 extrapolates complete engulfment at "
          "~11000 MCS). We extend to 20000 MCS to observe the light monolayer "
          "actually close over the dark mass. Fig18 patterns 0/1000/5000/10000/"
          "20000. Fig19 homotypic (ll,dd) & heterotypic (lM,dM,ld) frac lengths.",
)

# 6. Position reversal (Figs 20, 21)
_add(
    slug="position_reversal", title="Position reversal", section="III D",
    J=_J(14, 2, 11, 30, 16), temperature=10.0, ic="equilibrated_random",
    image_mcs=[0, 50, 5000], series_max_mcs=10000,
    figures=["fig20", "fig21"],
    notes="J_lM=30 (gamma_lM=23) >> others -> dark sorts OUTWARD (reversed). "
          "Random IC. Fig20 patterns 0/50/5000. Fig21 frac (ll,dd,ld) & "
          "medium correlation (lM, dM).",
)

# 7. Partial cell sorting (Figs 22, 23, 24)
_add(
    slug="partial_sorting", title="Partial cell sorting", section="III E",
    J=_J(11, 2, 14, 16, 16), temperature=5.0, ic="equilibrated_random",
    image_mcs=[10, 100, 1000, 2000], series_max_mcs=2000,
    figures=["fig22", "fig23", "fig24"],
    notes="J_ll and J_ld swapped vs sorting: gamma_ld=7.5 (Young condition "
          "fails) -> sorting stalls partial. Fig22 patterns. Fig23 cell-cell & "
          "medium fractional lengths. Fig24 partial-vs-normal comparison.",
)

# 8. Dispersal -- light-light / sloughing (Fig 25)
_add(
    slug="dispersal_sloughing", title="Dispersal — light-cell sloughing",
    section="III F", J=_J(14, 4, 11, 2, 16), temperature=5.0,
    ic="equilibrated_random", image_mcs=[480], series_max_mcs=480,
    unannealed_display=True,
    figures=["fig25"],
    notes="gamma_lM=-5 (negative light-Medium tension) -> light cells disperse "
          "into medium; dark stays compact. Fig25 single pattern 480 MCS "
          "(unannealed).",
)

# 9. Dispersal -- light-dark, clusters separate (Fig 26)
_add(
    slug="dispersal_separate", title="Dispersal — clusters separate",
    section="III F", J=_J(14, 2, 35, 16, 16), temperature=5.0,
    ic="equilibrated_random", image_mcs=[2000], series_max_mcs=2000,
    figures=["fig26"],
    notes="gamma_ld=27 (very high) -> light & dark clusters fully separate; "
          "a few isolated dark cells. Fig26 single pattern 2000 MCS.",
)

# 10. Dispersal -- light-dark, clusters do NOT separate (Fig 27)
_add(
    slug="dispersal_no_separate", title="Dispersal — clusters do not separate",
    section="III F", J=_J(14, 2, 29, 16, 16), temperature=5.0,
    ic="equilibrated_random", image_mcs=[2000], series_max_mcs=2000,
    figures=["fig27"],
    notes="gamma_ld=21 -> extreme partial sorting, clusters stay attached. "
          "Fig27 single pattern 2000 MCS.",
)

# 11. Vacancy nucleation / cavity (Fig 28)
_add(
    slug="vacancy_cavity", title="Vacancy nucleation (cavity)", section="III F",
    J=dict(_SORT_J), temperature=5.0, ic="equilibrated_random",
    target={LIGHT: 20.0, DARK: 40.0},
    image_mcs=[200], series_max_mcs=200,
    figures=["fig28"],
    notes="Sorting energies but unequal target areas A_l=20, A_d=40; medium "
          "vacancy nucleation allowed -> light-cell-lined fluid cavity. Fig28 "
          "single pattern 200 MCS.",
)


ORDER = [
    "annealing", "global_equilibration", "checkerboard", "cell_sorting",
    "engulfment", "position_reversal", "partial_sorting", "dispersal_sloughing",
    "dispersal_separate", "dispersal_no_separate", "vacancy_cavity",
]


if __name__ == "__main__":
    for slug in ORDER:
        s = STUDIES[slug]
        g = s["gamma"]
        print(f"{slug:22} T={s['temperature']:<4} "
              f"gamma_ld={g['gamma_ld']:+.1f} gamma_lM={g['gamma_lM']:+.1f} "
              f"gamma_dM={g['gamma_dM']:+.1f}  ic={s['ic']}")
