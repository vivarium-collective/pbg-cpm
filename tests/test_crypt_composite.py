import process_bigraph as pb
from cpm.composites.crypt import build_crypt_composite


def test_crypt_composite_runs_and_differentiates():
    core = pb.allocate_core()
    comp, meta = build_crypt_composite(core, downscale=0.5, mcs_per_update=6,
                                       subcell_every=1)
    assert meta["n_subcells"] > 10           # per-stem-cell processes exist
    types0 = list(comp.state["types"]) if "types" in comp.state else None
    comp.run(40.0)
    types1 = list(comp.state["types"])
    stem = meta["stem_type"]
    goblet, absorp = meta["goblet_type"], meta["absorptive_type"]
    # at least one stem cell differentiated to a non-stem epithelial fate
    differentiated = sum(1 for t in types1 if t in (goblet, absorp))
    assert differentiated >= 1
