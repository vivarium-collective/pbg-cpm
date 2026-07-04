from cpm import cpm_core
from cpm.schema import load_world
from cpm.metrics import intercell_gap_faces


def _two_cell_block(nx, ny):
    # two type-1 cells pressed together (left half / right half of a block)
    labels = [0] * (nx * ny)
    seg_to_type = {1: 1, 2: 1}
    for y in range(4, ny - 4):
        for x in range(nx // 2 - 3, nx // 2):
            labels[x + y * nx] = 1
        for x in range(nx // 2, nx // 2 + 3):
            labels[x + y * nx] = 2
    return labels, seg_to_type


def _spec(nx, ny, labels, seg_to_type, junctions):
    spec = {
        # temp=10 / lambda=80 (not the brief's starting 22/12): the cpm-core
        # tests/junction.rs integration test already discovered, on this exact
        # 16x16 two-3-wide-cells-pressed-together geometry, that 22/12 lets the
        # `with`-junctions run transiently open 1-4 gap faces over 30 MCS across
        # many seeds (see its comment) -- so both knobs were raised together
        # until `with` was 0 with margin. Reusing that already-validated tuning
        # here instead of rediscovering it.
        "potts": {"dims": [nx, ny, 1], "boundary": "noflux", "neighbor_order": 2,
                  "temperature": 10.0, "seed": 5},
        "seed_labels": {"labels": labels, "types": seg_to_type, "default_type": 1,
                        "target_volume": 24.0, "lambda_volume": 1.0},
        "contact": [{"a": 0, "b": 1, "j": 4.0}, {"a": 1, "b": 1, "j": 4.0}],
    }
    if junctions:
        spec["junctions"] = {"types": [1], "lambda": 80.0}
    return spec


def test_junctions_wire_and_prevent_gaps():
    nx = ny = 16
    labels, seg_to_type = _two_cell_block(nx, ny)

    w = load_world(_spec(nx, ny, labels, seg_to_type, True))
    worst_with = 0
    for _ in range(30):
        w.step(1)
        worst_with = max(worst_with, intercell_gap_faces(w, {1}))

    w2 = load_world(_spec(nx, ny, labels, seg_to_type, False))
    worst_without = 0
    for _ in range(30):
        w2.step(1)
        worst_without = max(worst_without, intercell_gap_faces(w2, {1}))

    assert worst_with == 0, f"junctions should keep the cells gap-free, saw {worst_with}"
    assert worst_without > 0, f"control must open a gap (else vacuous), saw {worst_without}"
