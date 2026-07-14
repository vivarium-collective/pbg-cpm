"""Enumerate and execute every Glazier & Graner (1993) run (study mains plus
temperature / lambda sweeps), in parallel, saving results for figure assembly.

Each run is executed with the *serial* engine sweep (faithful sequential
dynamics); throughput comes from running many single-core runs concurrently via
a process pool.  Sweep runs are series-only (no pattern frames) and are bounded
to `SWEEP_MAX_MCS` where the paper's curves have plateaued.
"""

from __future__ import annotations

import os
import time
import argparse
import multiprocessing as mp

import numpy as np

from . import run as runmod
from . import engine
from .params import STUDIES, ORDER
from .types import LIGHT, DARK

RESULTS_DIR = os.path.join(runmod.DATA_DIR, "results")
SWEEP_MAX_MCS = 3000   # sweep curves plateau well before this (Figs 9,15,16)
SEED = 17


def _base_spec(s, run_slug, temperature=None, lam=None, series_max=None,
               with_images=True):
    return {
        "slug": run_slug,
        "temperature": s["temperature"] if temperature is None else temperature,
        "J": s["J"],
        "lam": s["lam"] if lam is None else lam,
        "target": s["target"],
        "image_mcs": list(s["image_mcs"]) if with_images else [],
        "series_max_mcs": s["series_max_mcs"] if series_max is None else series_max,
        "series_linear": s.get("series_linear", False),
        "display_anneal_mcs": s["display_anneal_mcs"],
        "unannealed_display": s["unannealed_display"],
    }


def enumerate_runs(only=None):
    """Return list of run descriptors: dict(run_slug, spec, ic, frac_light)."""
    runs = []
    for slug in ORDER:
        if only and slug not in only:
            continue
        s = STUDIES[slug]
        # main run (with pattern frames + full-length series)
        runs.append(dict(run_slug=slug, spec=_base_spec(s, slug),
                         ic=s["ic"], frac_light=s["frac_light"]))
        # temperature sweep (series only)
        for t in (s.get("temps") or []):
            rs = f"{slug}__T{t}"
            runs.append(dict(run_slug=rs,
                             spec=_base_spec(s, rs, temperature=float(t),
                                             series_max=min(s["series_max_mcs"],
                                                            SWEEP_MAX_MCS),
                                             with_images=False),
                             ic=s["ic"], frac_light=s["frac_light"]))
        # lambda sweep (series only)
        for lam in (s.get("lambdas") or []):
            rs = f"{slug}__lam{lam}"
            runs.append(dict(run_slug=rs,
                             spec=_base_spec(s, rs, lam=float(lam),
                                             series_max=min(s["series_max_mcs"],
                                                            SWEEP_MAX_MCS),
                                             with_images=False),
                             ic=s["ic"], frac_light=s["frac_light"]))
    return runs


def _make_ic(ic_kind, frac_light, seed):
    """Return (owner_ic_2d, types_map) for a run."""
    if ic_kind == "brick_equilibrate":
        # Global equilibration: the IC *is* the raw brick tiling (Fig 4a);
        # run_study then equilibrates it at T=5 (-> Fig 4b).
        nx = ny = 300
        labels = engine.brick_tiling_labels(nx, ny, (50, 50, 250, 250), 5, 8)
        owner = labels.reshape(ny, nx)
        types_map = runmod.assign_all_light(owner)
        return owner, types_map
    owner = runmod.load_canonical_ic()
    if ic_kind == "equilibrated_light":
        return owner, runmod.assign_all_light(owner)
    if ic_kind == "equilibrated_random":
        return owner, runmod.assign_random(owner, frac_light, seed=seed)
    if ic_kind == "half_split":
        return owner, runmod.assign_half_split(owner, light_on_top=True)
    raise ValueError(f"unknown ic kind {ic_kind}")


def execute_run(desc):
    run_slug = desc["run_slug"]
    out = os.path.join(RESULTS_DIR, run_slug.split("__")[0])
    done_marker = os.path.join(out, f"{run_slug}_series.json")
    if os.path.exists(done_marker) and os.environ.get("GG_FORCE") != "1":
        return f"skip {run_slug} (exists)"
    t0 = time.time()
    owner_ic, types_map = _make_ic(desc["ic"], desc["frac_light"], SEED)
    res = runmod.run_study(desc["spec"], owner_ic, types_map, seed=SEED,
                           parallel=False, verbose=False)
    runmod.save_result(res, out)
    return f"done {run_slug} in {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these study slugs")
    args = ap.parse_args()

    runmod.load_canonical_ic()  # ensure cached before forking
    runs = enumerate_runs(only=args.only)
    n_studies = len({r['run_slug'].split('__')[0] for r in runs})
    print(f"{len(runs)} runs across {n_studies} studies on {args.procs} procs",
          flush=True)
    for r in runs:
        print("  -", r["run_slug"], flush=True)
    t0 = time.time()
    with mp.Pool(args.procs) as pool:
        for msg in pool.imap_unordered(execute_run, runs):
            print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
