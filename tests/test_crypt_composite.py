import process_bigraph as pb
from cpm.composites.crypt import build_crypt_composite


def test_crypt_composite_runs_and_differentiates():
    core = pb.allocate_core()
    comp, meta = build_crypt_composite(core, downscale=0.5, mcs_per_update=6,
                                       subcell_every=1)
    assert meta["n_subcells"] > 10           # per-stem-cell processes exist
    comp.run(40.0)
    types1 = list(comp.state["types"])
    goblet = meta["goblet_type"]
    n_goblet_after = sum(1 for t in types1 if t == goblet)
    # new goblet cells can ONLY arise from a fate switch (no division) ->
    # a strict increase proves the SBML->Boolean->CPM loop actually fired
    assert n_goblet_after > meta["initial_counts"]["goblet"]
    assert meta["n_subcells"] > 10           # per-cell subcell processes exist
