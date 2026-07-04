"""Crypt differentiation as a process-bigraph ``Composite``.

A single ``CPMProcess`` runs the colonic-crypt Cellular Potts world seeded
from the HRA FTU illustration. Every Epithelial-Stem cell additionally gets
two subcellular processes wired to its local environment:

  * an ``SBMLSubcell`` integrating a stemness ODE driven by the local Wnt
    concentration (``field_at_cell`` -> ``ligand``), publishing stemness ``S``;
  * a ``BooleanSubcell`` that reads stemness plus the count of secretory
    neighbours (``neighbor_secretory``) and emits a fate: stay stem, become
    goblet (secretory), or become absorptive (Notch-style lateral inhibition).

The CPM consumes the per-cell fates on the next update and relabels cells,
so differentiation feeds back into the tissue. The whole thing is advanced by
``Composite.run`` -- there is no bypass of the engine.
"""
import process_bigraph as pb

from cpm.ftu import load_crypt_labels
from cpm.schema import load_world

# Stemness ODE (Task 5): Wnt drives synthesis of stemness factor S via a Hill
# term; S decays. High local Wnt (crypt base) keeps S high (stem); as cells
# move up and Wnt falls, S drops below threshold and the cell differentiates.
STEMNESS_MODEL = """
J1: -> S; k_on*Wnt^4/(K^4 + Wnt^4);
J2: S -> ; k_off*S;
species S, Wnt;
S = 1.0; Wnt = 0.0;
k_on = 0.8; K = 0.3; k_off = 0.4;
"""

CPM_ADDR = "local:!cpm.processes.cpm_process.CPMProcess"
SBML_ADDR = "local:!cpm.subcellular.sbml.SBMLSubcell"
BOOL_ADDR = "local:!cpm.subcellular.boolean.BooleanSubcell"


def build_crypt_composite(core, *, downscale=1.0, mcs_per_update=8, subcell_every=1):
    (nx, ny), labels, seg_to_type, type_names, median = load_crypt_labels(
        target_maxdim=int(500 * downscale))
    # type ids from the FTU order (1-based); index 0 == Medium
    names = ["Medium"] + type_names
    stem = names.index("Epithelial Stem Cells")
    goblet = names.index("Goblet Cells")
    absorp = names.index("Absorptive Cells")

    spec = {
        "potts": {"dims": [nx, ny, 1], "boundary": "noflux",
                  "neighbor_order": 2, "temperature": 8.0, "seed": 1},
        "seed_labels": {"labels": labels, "types": seg_to_type,
                        "default_type": stem, "target_volume": float(median),
                        "lambda_volume": 2.0},
        "contact": _crypt_contacts(len(names) - 1, stem, goblet, absorp),
        "fields": [{"name": "Wnt", "d": 0.12, "decay": 0.15,
                    "secretion": [{"type": stem, "rate": 6.0}], "chemotaxis": []}],
    }

    # Which cell ids receive stemness + fate subcells? seed_from_labels assigns
    # ids in the order segments FIRST APPEAR row-major -- NOT sorted-seg order --
    # so derive them from a deterministic probe world built from the same spec.
    #
    # The HRA colonic-crypt FTU labels only TWO explicit "Epithelial Stem Cells"
    # (the crypt base); the rest are already-committed Absorptive (71) / Goblet
    # (39) / rare cells. Two stem cells cannot support a differentiation model
    # (n_subcells > 10) nor a demonstrable gradient. So the subcellular
    # machinery is attached to the crypt's *progenitor pool* = Stem + Absorptive
    # (the transit-amplifying column that differentiates as it leaves the base).
    # Wnt is still secreted ONLY by the base stem type, giving a base-localized
    # gradient: base progenitors stay stem, upper ones lose stemness and adopt a
    # goblet/absorptive fate.
    progenitor_types = {stem, absorp}
    probe = load_world(spec)
    probe_types = list(probe.cell_types())          # index == cell id, 0 == medium
    stem_cell_ids = [cid for cid in range(1, len(probe_types))
                     if probe_types[cid] in progenitor_types]

    state = {
        # pre-seed the fates map with a 0 entry per wired cell so the per-cell
        # Boolean writes (overwrite[integer] into ``[fates, str(cid)]``) update an
        # existing key instead of being dropped by the map's apply.
        "fates": {str(cid): 0 for cid in stem_cell_ids},
        "cpm": {
            "_type": "process", "address": CPM_ADDR,
            "config": {"spec": spec, "mcs_per_update": mcs_per_update,
                       "n_fields": 1, "secretory_types": [goblet]},
            "inputs": {"fates": ["fates"]},
            "outputs": {"volumes": ["volumes"], "types": ["types"],
                        "positions": ["positions"], "field_at_cell": ["field_at_cell"],
                        "neighbor_secretory": ["neighbor_secretory"]},
        },
    }
    for cid in stem_cell_ids:
        state[f"sbml_{cid}"] = {
            "_type": "process", "address": SBML_ADDR,
            "config": {"model": STEMNESS_MODEL, "ligand_species": "Wnt",
                       "state_species": "S", "ligand_scale": 1.0},
            "interval": float(subcell_every),
            "inputs": {"ligand": ["field_at_cell", str(cid)]},
            "outputs": {"state": ["cell_state", str(cid)]},
        }
        state[f"bool_{cid}"] = {
            "_type": "process", "address": BOOL_ADDR,
            "config": {"stemness_threshold": 0.4, "goblet_type": goblet,
                       "absorptive_type": absorp},
            "interval": float(subcell_every),
            "inputs": {"state": ["cell_state", str(cid)],
                       "neighbor_secretory": ["neighbor_secretory", str(cid)]},
            "outputs": {"fate": ["fates", str(cid)]},
        }

    comp = pb.Composite({"state": state}, core=core)
    meta = {"dims": [nx, ny, 1], "type_names": names, "stem_type": stem,
            "goblet_type": goblet, "absorptive_type": absorp, "wnt_field": 0,
            "n_subcells": len(stem_cell_ids)}
    return comp, meta


def _crypt_contacts(k, stem, goblet, absorp):
    pairs = []
    for t in range(1, k + 1):
        pairs.append({"a": 0, "b": t, "j": 14.0})       # medium costly -> packed
        for u in range(t, k + 1):
            pairs.append({"a": t, "b": u, "j": 6.0})     # uniform cell-cell
    return pairs
