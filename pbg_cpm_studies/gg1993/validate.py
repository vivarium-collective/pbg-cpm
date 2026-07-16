"""Quantitative validation of each Glazier & Graner (1993) study against the
behaviour the paper reports.

Each study gets ONE acceptance test: a named, paper-grounded assertion evaluated
against the saved metric time series (``workspace/gg1993_data/results/<slug>/
<slug>_series.json``). A test returns the measured quantity, the expected
condition, and PASS/FAIL — no hand-set verdicts, so a regression in the engine
flips the card to FAIL.

Run: ``python -m pbg_cpm_studies.gg1993.validate`` — evaluates every study,
prints a table, writes ``workspace/gg1993_data/validation.json``, AND calls
``apply_conclusions()`` to write the derived tests / run outcomes / gate verdict
back into each ``study.yaml``. So the report's conclusions are never hand-set:
re-run the sims (``driver.py``) then re-run this (or the full pipeline,
``python -m pbg_cpm_studies.gg1993.gallery``, which calls it) and every study's
"Ran · Tests · Verdict" strip updates automatically from the current data.
"""

from __future__ import annotations

import json
import os

# Path-only — deliberately does NOT import .run (which pulls the Rust engine),
# so conclusions can be re-derived from the committed series JSON in a light CI
# step (no build) as well as locally.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pbg_cpm_studies/
WS_ROOT = os.path.dirname(_PKG_DIR)
DATA_DIR = os.path.join(WS_ROOT, "workspace", "gg1993_data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
OUT_PATH = os.path.join(DATA_DIR, "validation.json")
STU_DIR = os.path.join(WS_ROOT, "workspace", "studies")

# readout key each study's acceptance test keys off (the `measure` field)
MEASURE = {
    "annealing": "n_bulk", "global_equilibration": "bl_total", "checkerboard": "frac_ld",
    "cell_sorting": "bl_total", "engulfment": "frac_dM", "position_reversal": "corr_dM",
    "partial_sorting": "frac_ld", "dispersal_sloughing": "frac_lM",
    "dispersal_separate": "bl_total", "dispersal_no_separate": "bl_total",
    "vacancy_cavity": "frac_dd",
}


def _series(slug):
    p = os.path.join(RESULTS_DIR, slug, f"{slug}_series.json")
    with open(p) as f:
        d = json.load(f)
    return d["series_mcs"], d["series"]


def _first_last(series, key):
    vals = [s[key] for s in series if s.get(key) is not None]
    return vals[0], vals[-1]


def _minmax(series, key):
    vals = [s[key] for s in series if s.get(key) is not None]
    return min(vals), max(vals)


# ---------------------------------------------------------------------------
# One acceptance test per study. Each returns a dict:
#   name, paper_ref, expected (human), measured (human), value (float), passed
# ---------------------------------------------------------------------------

def _t_annealing(mcs, S):
    n0, n1 = _first_last(S, "n_bulk")
    m0, m1 = _first_last(S, "mu2")
    passed = abs(n1 - 6.0) < 0.3 and m1 < m0
    return dict(
        name="Topological equilibrium under T=0 annealing",
        paper_ref="Fig 2 / Sec. II D 2 — bulk ⟨n⟩→6, moments settle",
        expected="final ⟨n⟩_bulk ≈ 6 (within 0.3) and μ₂ decreases while annealing",
        measured=f"⟨n⟩_bulk {n0:.2f}→{n1:.2f}, μ₂ {m0:.3f}→{m1:.3f}",
        value=round(n1, 3), passed=passed)


def _t_global_equilibration(mcs, S):
    b0, b1 = _first_last(S, "bl_total")
    n0, n1 = _first_last(S, "n_bulk")
    passed = b1 < b0 and abs(n1 - 6.0) < 0.3
    return dict(
        name="Brick tiling rounds to a disk at topological equilibrium",
        paper_ref="Fig 4-5 — rectangular tiling rounds; bulk ⟨n⟩≈6",
        expected="total boundary length decreases (rounding) and ⟨n⟩_bulk ≈ 6",
        measured=f"boundary {b0:.0f}→{b1:.0f}, ⟨n⟩_bulk {n0:.2f}→{n1:.2f}",
        value=round(n1, 3), passed=passed)


def _t_checkerboard(mcs, S):
    ll1 = _first_last(S, "frac_ll")[1]
    dd1 = _first_last(S, "frac_dd")[1]
    ld1 = _first_last(S, "frac_ld")[1]
    passed = ld1 > 0.5 and ld1 > ll1 and ld1 > dd1
    return dict(
        name="Heterotypic contacts dominate (checkerboard mixing)",
        paper_ref="Fig 7-8 — negative γ_ld drives a light/dark checkerboard",
        expected="light-dark fraction > 0.5 and larger than light-light & dark-dark",
        measured=f"frac_ld={ld1:.3f} vs frac_ll={ll1:.3f}, frac_dd={dd1:.3f}",
        value=round(ld1, 3), passed=passed)


def _t_cell_sorting(mcs, S):
    # Paper Fig 12-13: a random mix demixes into a dark cluster wrapped by a
    # light monolayer. Signatures: heterotypic light-dark contact collapses,
    # AND light comes to own essentially the whole medium surface (monolayer).
    ld0, ld1 = _first_last(S, "frac_ld")
    clM0, clM1 = _first_last(S, "corr_lM")
    demix = ld1 < 0.55 * ld0          # heterotypic contact roughly halves+
    monolayer = clM1 > 0.80           # light holds >80% of the medium contact
    passed = demix and monolayer
    return dict(
        name="Cells sort: dark cluster wrapped by a light monolayer",
        paper_ref="Fig 12-13 — light-dark contact collapses; light monolayer (light-Medium correlation→1)",
        expected="light-dark fraction falls ≥45% AND light holds >80% of the medium surface (monolayer)",
        measured=f"frac_ld {ld0:.3f}→{ld1:.3f} ({100*(1-ld1/max(ld0,1e-9)):.0f}% drop), corr_lM {clM0:.3f}→{clM1:.3f}",
        value=round(clM1, 3), passed=passed)


def _t_engulfment(mcs, S):
    # Paper Fig 18-19: a CLEAN top-light/bottom-dark split; light slowly
    # engulfs the dark mass. Two things distinguish true engulfment from mere
    # mixing: (1) the two blocks stay coherent — heterotypic light-dark contact
    # stays LOW throughout (it does NOT balloon the way a checkerboard/partial
    # sort does); (2) the dark mass loses its medium surface as the light
    # monolayer closes over it. The paper's run is still incomplete at 10000
    # MCS (extrapolated ~11000); we run to 20000 to see it close.
    dM0, dM1 = _first_last(S, "frac_dM")
    clM1 = _first_last(S, "corr_lM")[1]
    ld_min, ld_max = _minmax(S, "frac_ld")
    coherent = ld_max < 0.20          # blocks slide past each other, never mix
    engulfed = dM1 < 0.20 * dM0       # dark loses ≥80% of its medium contact
    monolayer = clM1 > 0.85           # light owns >85% of the medium surface
    passed = coherent and engulfed and monolayer
    return dict(
        name="Light engulfs the coherent dark mass (no mixing)",
        paper_ref="Fig 18-19 — clean split; dark-Medium→~0, light monolayer closes; ld stays low",
        expected="light-dark stays low (<0.20, no mixing) AND dark-medium falls ≥80% AND light owns >85% of the surface",
        measured=f"frac_dM {dM0:.4f}→{dM1:.4f} ({100*(1-dM1/max(dM0,1e-9)):.0f}% drop), corr_lM_final={clM1:.3f}, max frac_ld={ld_max:.3f}",
        value=round(clM1, 3), passed=passed)


def _t_position_reversal(mcs, S):
    # Paper Fig 20-21: raising gamma_lM (=23) reverses the layering — DARK forms
    # the outer monolayer. Fig 21(b): the light-Medium correlation falls to ~0,
    # i.e. dark comes to own essentially the entire medium surface. A bare
    # corr_dM > corr_lM (the old test) passed at a trivial 0.51 vs 0.49; the
    # paper demands near-total reversal.
    clM0, clM1 = _first_last(S, "corr_lM")
    cdM1 = _first_last(S, "corr_dM")[1]
    ld0, ld1 = _first_last(S, "frac_ld")
    reversed_surface = cdM1 > 0.80    # dark owns >80% of the medium surface
    demix = ld1 < 0.80 * ld0          # genuine sorting occurred (not frozen mix)
    passed = reversed_surface and demix
    return dict(
        name="Position reversal: dark forms the outer monolayer",
        paper_ref="Fig 20-21 — light-Medium correlation→0; dark owns the surface",
        expected="dark holds >80% of the medium surface (light-Medium→0) AND light-dark contact falls (sorting)",
        measured=f"corr_dM_final={cdM1:.3f} (corr_lM {clM0:.3f}→{clM1:.3f}), frac_ld {ld0:.3f}→{ld1:.3f}",
        value=round(cdM1, 3), passed=passed)


def _t_partial_sorting(mcs, S):
    # Paper Fig 22-24: the Young condition is violated (gamma_ld=7.5), so
    # sorting starts — clusters coarsen and heterotypic contact drops — but
    # STALLS: no light monolayer ever closes, clusters trap heterotypic
    # inclusions. The discriminating signature vs complete sorting is that
    # light never comes to own the medium surface. So we require BOTH: real
    # coarsening happened, AND the light monolayer did NOT form.
    ld0, ld1 = _first_last(S, "frac_ld")
    clM1 = _first_last(S, "corr_lM")[1]
    # Partial sorting DOES coarsen (heterotypic contact can fall a lot via
    # clustering); what makes it *partial* is that no light monolayer ever
    # closes — light does not come to own the aggregate surface the way
    # complete sorting does (corr_lM stays well short of the ~0.85 monolayer
    # value; measured ~0.62 here vs >0.85 for complete cell_sorting).
    coarsened = ld1 < 0.8 * ld0       # sorting genuinely progressed
    no_monolayer = clM1 < 0.75        # light never wrapped the aggregate
    passed = coarsened and no_monolayer
    return dict(
        name="Partial sorting stalls: coarsening but no monolayer",
        paper_ref="Fig 22-24 — Young condition fails; clusters trap inclusions, no monolayer",
        expected="light-dark contact drops (coarsening) BUT light never owns the surface — no monolayer (corr_lM<0.75, vs >0.85 for complete sorting)",
        measured=f"frac_ld {ld0:.3f}→{ld1:.3f}, corr_lM_final={clM1:.3f} (no monolayer)",
        value=round(clM1, 3), passed=passed)


def _t_dispersal_sloughing(mcs, S):
    lM0, lM1 = _first_last(S, "frac_lM")
    b0, b1 = _first_last(S, "bl_total")
    passed = lM1 > 3.0 * lM0 and b1 > b0
    return dict(
        name="Light cells slough off and disperse into the medium",
        paper_ref="Fig 25 — light dissociates; light-medium interface surges",
        expected="light-medium fraction rises ≥3× and total boundary length increases (fragmentation)",
        measured=f"frac_lM {lM0:.3f}→{lM1:.3f} ({lM1/max(lM0,1e-9):.1f}×), boundary {b0:.0f}→{b1:.0f}",
        value=round(lM1 / max(lM0, 1e-9), 2), passed=passed)


def _t_dispersal_separate(mcs, S):
    b0, b1 = _first_last(S, "bl_total")
    lM0, lM1 = _first_last(S, "frac_lM")
    dM0, dM1 = _first_last(S, "frac_dM")
    passed = b1 > b0 and lM1 > lM0 and dM1 > dM0
    return dict(
        name="Clusters separate (both types expose new medium surface)",
        paper_ref="Fig 26 — clusters break apart and separate",
        expected="total boundary length increases and both medium-contact fractions rise",
        measured=f"boundary {b0:.0f}→{b1:.0f}, frac_lM {lM0:.3f}→{lM1:.3f}, frac_dM {dM0:.3f}→{dM1:.3f}",
        value=round(b1 / b0, 4), passed=passed)


def _t_dispersal_no_separate(mcs, S):
    b0, b1 = _first_last(S, "bl_total")
    # Contrast with dispersal_separate: the aggregate stays compact (boundary
    # does NOT grow) because γ keeps the clusters together.
    passed = b1 <= b0
    return dict(
        name="Clusters stay aggregated (no separation)",
        paper_ref="Fig 27 — same setup but clusters do NOT separate",
        expected="total boundary length does not increase (aggregate stays compact)",
        measured=f"boundary {b0:.0f}→{b1:.0f} (ratio {b1/b0:.3f})",
        value=round(b1 / b0, 4), passed=passed)


def _t_vacancy_cavity(mcs, S):
    b0, b1 = _first_last(S, "bl_total")
    dd0, dd1 = _first_last(S, "frac_dd")
    passed = b1 < b0 and dd1 > dd0
    return dict(
        name="Cavity nucleates and rounds; dark accumulates",
        paper_ref="Fig 28 — vacancy/cavity nucleation with dark accumulation",
        expected="total boundary length decreases (cavity rounds) and dark-dark contact rises",
        measured=f"boundary {b0:.0f}→{b1:.0f}, frac_dd {dd0:.3f}→{dd1:.3f}",
        value=round(dd1, 3), passed=passed)


TESTS = {
    "annealing": _t_annealing,
    "global_equilibration": _t_global_equilibration,
    "checkerboard": _t_checkerboard,
    "cell_sorting": _t_cell_sorting,
    "engulfment": _t_engulfment,
    "position_reversal": _t_position_reversal,
    "partial_sorting": _t_partial_sorting,
    "dispersal_sloughing": _t_dispersal_sloughing,
    "dispersal_separate": _t_dispersal_separate,
    "dispersal_no_separate": _t_dispersal_no_separate,
    "vacancy_cavity": _t_vacancy_cavity,
}


def validate_all():
    out = {}
    for slug, fn in TESTS.items():
        mcs, S = _series(slug)
        res = fn(mcs, S)
        res["slug"] = slug
        out[slug] = res
    return out


def apply_conclusions(results=None):
    """Re-derive and write every study's conclusions from the CURRENT run data,
    so the report is never stale: the acceptance ``behavior_tests``, a completed
    ``runs`` entry whose ``outcomes`` carry the PASS/FAIL result, and the
    ``gate_status`` verdict. Idempotent — safe to run on every regeneration.

    This is the single source of truth for the studies' verdicts: change the
    engine or re-run the sims, re-run this, and the report's tests/verdicts (and
    the "Ran · Tests · Verdict" strip derived from them) update automatically.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
    yaml = YAML(); yaml.preserve_quotes = True; yaml.width = 4096
    if results is None:
        results = validate_all()
    for slug, r in results.items():
        p = os.path.join(STU_DIR, slug, "study.yaml")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            doc = yaml.load(f)
        passed = r["passed"]
        tname = r["name"]
        doc["behavior_tests"] = [CommentedMap([
            ("name", tname),
            ("classification", "primary"),
            ("description", f"Paper: {r['paper_ref']}. Pass criterion: {r['expected']}. "
                            f"Measured from the reproduced run: {r['measured']}."),
            ("measure", MEASURE.get(slug, "")),
            ("pass_if", r["expected"]),
            ("status", "passed" if passed else "failed"),
            ("result", "PASS" if passed else "FAIL"),
            ("paper_reference", r["paper_ref"]),
        ])]
        outcome = CommentedMap()
        outcome[tname] = CommentedMap([
            ("result", "PASS" if passed else "FAIL"),
            ("measured_value", r["value"]),
            ("notes", r["measured"]),
            ("evaluated_by", "pbg_cpm_studies.gg1993.validate"),
        ])
        doc["runs"] = [CommentedMap([
            ("run_id", f"{slug}-reproduction"),
            ("name", f"{slug} reproduction (seed 17)"),
            ("status", "completed"),
            ("canonical", True),
            ("timestamp", "2026-07-14"),
            ("seeds", [17]),
            ("emitter", "gg1993_data/results series-json"),
            ("outcomes", outcome),
        ])]
        doc["gate_status"] = "passed" if passed else "failed"
        with open(p, "w") as f:
            yaml.dump(doc, f)
    return results


def main():
    results = validate_all()
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    apply_conclusions(results)  # keep study.yaml verdicts in sync with the data
    npass = sum(1 for r in results.values() if r["passed"])
    print(f"Glazier & Graner (1993) reproduction validation — "
          f"{npass}/{len(results)} studies PASS\n")
    for slug, r in results.items():
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {slug}")
        print(f"       test: {r['name']}")
        print(f"       expect: {r['expected']}")
        print(f"       measured: {r['measured']}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
