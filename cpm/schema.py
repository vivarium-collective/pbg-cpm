import cpm_core


def load_world(spec):
    p = spec["potts"]
    dims = tuple(p["dims"])
    world = cpm_core.World(dims, p["boundary"], int(p["neighbor_order"]), float(p["temperature"]))
    ids = []
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
    for c, cid in zip(spec.get("cells", []), ids):
        x0, y0, z0, x1, y1, z1 = c["seed_block"]
        world.seed_block(cid, x0, y0, z0, x1, y1, z1)
    world.finalize(int(p["seed"]))
    return world
