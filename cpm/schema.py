from cpm import cpm_core


def load_world(spec):
    p = spec["potts"]
    dims = tuple(p["dims"])
    world = cpm_core.World(dims, p["boundary"], int(p["neighbor_order"]), float(p["temperature"]))
    sl = spec.get("seed_labels")
    ids = []
    if sl is None:
        for c in spec.get("cells", []):
            cid = world.add_cell(
                int(c["type"]),
                float(c["target_volume"]),
                float(c["lambda_volume"]),
                float(c["target_surface"]),
                float(c["lambda_surface"]),
            )
            ids.append(cid)  # remember assigned id, locally (do not mutate spec)
    for pair in spec.get("contact", []):
        world.set_contact(int(pair["a"]), int(pair["b"]), float(pair["j"]))
    if sl is not None:
        # seed exact cell placement from a segmentation label array; skip the
        # per-cell `cells` loop (the two seeding paths are mutually exclusive).
        world.seed_from_labels(
            list(sl["labels"]),
            {int(k): int(v) for k, v in sl["types"].items()},
            int(sl["default_type"]),
            float(sl["target_volume"]),
            float(sl["lambda_volume"]),
        )
    else:
        for c, cid in zip(spec.get("cells", []), ids):
            x0, y0, z0, x1, y1, z1 = c["seed_block"]
            world.seed_block(cid, x0, y0, z0, x1, y1, z1)
    for fi, f in enumerate(spec.get("fields", [])):
        idx = world.add_field(f["name"], float(f["d"]), float(f["decay"]))
        # idx equals fi by construction; keep them in sync
        for s in f.get("secretion", []):
            world.set_secretion(idx, int(s["type"]), float(s["rate"]))
        for c in f.get("chemotaxis", []):
            world.set_chemotaxis(idx, int(c["type"]), float(c["lambda"]))
    conn = spec.get("connectivity")
    if conn is not None:
        for t in conn.get("types", []):
            world.set_connectivity(int(t), True)
        if conn.get("medium", False):
            world.set_connectivity_medium(True)
    world.finalize(int(p["seed"]))
    return world
