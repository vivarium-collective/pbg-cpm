from cpm import cpm_core
from cpm.schema import load_world
from cpm.metrics import membrane_distance_field, mean_membrane_distance, interior_medium_pockets


def _flat_sheet_labels(nx, ny, nz, z_sheet):
    # one 1-cell-thick type-1 sheet at height z_sheet, tiled into a few cells
    labels = [0] * (nx * ny * nz)
    seg_to_type = {}
    sid = 1
    for y in range(ny):
        for x in range(nx):
            # 2x2 patches -> distinct cells
            seg = 1 + (x // 2) + (y // 2) * (nx // 2)
            seg_to_type[seg] = 1
            labels[x + y * nx + z_sheet * nx * ny] = seg
    return labels, seg_to_type


def test_membrane_schema_wires_and_holds_sheet():
    nx = ny = 12
    nz = 8
    z_sheet = 4
    labels, seg_to_type = _flat_sheet_labels(nx, ny, nz, z_sheet)
    anchors = [x + y * nx + z_sheet * nx * ny for y in range(ny) for x in range(nx)]
    # NOTE: this is the ACTUAL cpm/schema.py shape — a "potts" wrapper (dims,
    # boundary, neighbor_order, temperature, seed); seed_labels with keys
    # labels/types/default_type/target_volume/lambda_volume; and contact as a
    # list of {a,b,j} dicts. The new membrane block is {anchors,k,band,types}.
    # NOTE on parameters: the brief's original draft values (temperature=18,
    # lambda_volume=1.0) made a 1-voxel-thick sheet dissolve to nothing within
    # ~6 MCS regardless of the membrane (verified empirically) — the weak
    # volume constraint let the strong cell/medium surface-tension differential
    # (j=6 vs j=2) erode cells to extinction before any anchored-vs-free drift
    # could be observed, so both sides read 0.0/0.0 (division-by-zero guard).
    # Tuned to temperature=10.0 / lambda_volume=25.0 / j=(4.0, 2.0), which keeps
    # the sheet population alive over 30 MCS and lets it buckle away from the
    # anchor plane without a membrane while holding flat with one; verified
    # stable (d_anchored < d_free, d_anchored <= 1.5) across seeds 1-42.
    spec = {
        "potts": {"dims": [nx, ny, nz], "boundary": "noflux", "neighbor_order": 2,
                  "temperature": 10.0, "seed": 3},
        "seed_labels": {"labels": labels, "types": seg_to_type, "default_type": 1,
                        "target_volume": 4.0, "lambda_volume": 25.0},
        "contact": [{"a": 0, "b": 1, "j": 4.0}, {"a": 1, "b": 1, "j": 2.0}],
        "membrane": {"anchors": anchors, "k": 6.0, "band": 1.0, "types": [1]},
    }
    w = load_world(spec)
    dist = membrane_distance_field((nx, ny, nz), anchors)
    w.step(30)
    d_anchored = mean_membrane_distance(w, dist, {1})

    # same seed/stress WITHOUT the membrane
    import copy
    spec2 = copy.deepcopy(spec); spec2.pop("membrane")
    w2 = load_world(spec2)
    w2.step(30)
    d_free = mean_membrane_distance(w2, dist, {1})

    assert d_anchored < d_free, f"membrane did not hold the sheet: {d_anchored} vs {d_free}"
    assert d_anchored <= 1.5, f"anchored sheet drifted off the membrane: {d_anchored}"
