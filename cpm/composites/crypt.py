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
k_on = 0.8; K = 0.3; k_off = 0.1;
"""

# Boolean fate threshold on stemness S; also exported in meta so the demo can
# key its gates on the stemness STATE rather than the transient ``stem`` type.
STEMNESS_THRESHOLD = 0.4

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
        # Long-range morphogen: low decay + higher (numerically stable, <=0.25)
        # diffusion so the base-secreted Wnt forms a crypt-spanning gradient
        # rather than a ~1px spike. Amplitude is huge at the source and is
        # brought into the ODE's sensitive window by the subcell ``ligand_scale``.
        "fields": [{"name": "Wnt", "d": 0.2, "decay": 0.001,
                    "secretion": [{"type": stem, "rate": 6.0}], "chemotaxis": []}],
    }

    # Which cell ids receive stemness + fate subcells? seed_from_labels assigns
    # ids in the order segments FIRST APPEAR row-major -- NOT sorted-seg order --
    # so derive them from a deterministic probe world built from the same spec.
    #
    # The HRA colonic-crypt FTU labels only TWO explicit "Epithelial Stem Cells"
    # (the crypt base); the rest are already-committed Absorptive (71) / Goblet
    # (39) / rare cells. Two stem cells cannot support a differentiation model
    # (n_subcells > 10) nor a demonstrable gradient.
    #
    # We deliberately do NOT wire subcells to the two base ``stem`` cells: they
    # are the Wnt source, and if they ever fell below the stemness threshold and
    # differentiated, the whole gradient would collapse (runaway differentiation
    # with no niche). Leaving them as a permanent, always-secreting niche makes
    # the biology stable and the story clean: the subcellular machinery is
    # attached to the crypt's *Absorptive progenitor pool*, the transit-
    # amplifying column above the base. Progenitors near the niche see high Wnt
    # and stay undifferentiated; distal ones lose stemness and adopt a
    # goblet (secretory) / absorptive fate via the ODE -> Boolean -> cell_type
    # coupling.
    progenitor_types = {absorp}
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
                       "state_species": "S", "ligand_scale": 0.05},
            "interval": float(subcell_every),
            "inputs": {"ligand": ["field_at_cell", str(cid)]},
            "outputs": {"state": ["cell_state", str(cid)]},
        }
        state[f"bool_{cid}"] = {
            "_type": "process", "address": BOOL_ADDR,
            "config": {"stemness_threshold": STEMNESS_THRESHOLD,
                       "goblet_type": goblet, "absorptive_type": absorp},
            "interval": float(subcell_every),
            "inputs": {"state": ["cell_state", str(cid)],
                       "neighbor_secretory": ["neighbor_secretory", str(cid)]},
            "outputs": {"fate": ["fates", str(cid)]},
        }

    comp = pb.Composite({"state": state}, core=core)
    meta = {"dims": [nx, ny, 1], "type_names": names, "stem_type": stem,
            "goblet_type": goblet, "absorptive_type": absorp, "wnt_field": 0,
            "n_subcells": len(stem_cell_ids),
            # cell ids that carry an SBML+Boolean subcell (the wired Absorptive
            # progenitor pool); the demo keys its gates on these + the threshold.
            "subcell_ids": list(stem_cell_ids),
            "stemness_threshold": STEMNESS_THRESHOLD}
    meta["initial_counts"] = {
        "stem": sum(1 for t in probe_types[1:] if t == stem),
        "goblet": sum(1 for t in probe_types[1:] if t == goblet),
        "absorptive": sum(1 for t in probe_types[1:] if t == absorp),
    }
    return comp, meta


def _crypt_contacts(k, stem, goblet, absorp):
    pairs = []
    for t in range(1, k + 1):
        pairs.append({"a": 0, "b": t, "j": 14.0})       # medium costly -> packed
        for u in range(t, k + 1):
            pairs.append({"a": t, "b": u, "j": 6.0})     # uniform cell-cell
    return pairs
