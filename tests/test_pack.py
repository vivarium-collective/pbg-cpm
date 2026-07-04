import json
import cpm_core
from cpm.pack import write_pack


def test_write_pack_roundtrip(tmp_path):
    w = cpm_core.World((8, 8, 1), "noflux", 2, 10.0)
    a = w.add_cell(1, 9.0, 1.0, 12.0, 1.0)
    w.seed_block(a, 2, 2, 0, 5, 5, 1)
    w.finalize(1)
    p = tmp_path / "frame.cpm.json"
    d = write_pack(w, str(p))
    assert d["format"] == "cpm.pack.v1"
    assert d["dims"] == [8, 8, 1]
    assert len(d["labels"]) == 64
    on_disk = json.loads(p.read_text())
    assert on_disk == d
