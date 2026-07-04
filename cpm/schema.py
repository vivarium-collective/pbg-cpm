import cpm_core


def load_world(spec):
    p = spec["potts"]
    dims = tuple(p["dims"])
    world = cpm_core.World(dims, p["boundary"], int(p["neighbor_order"]), float(p["temperature"]))
    for c in spec.get("cells", []):
        cid = world.add_cell(
            int(c["type"]),
            float(c["target_volume"]),
            float(c["lambda_volume"]),
            float(c["target_surface"]),
            float(c["lambda_surface"]),
        )
        c["_id"] = cid  # remember assigned id
    for pair in spec.get("contact", []):
        world.set_contact(int(pair["a"]), int(pair["b"]), float(pair["j"]))
    for c in spec.get("cells", []):
        x0, y0, z0, x1, y1, z1 = c["seed_block"]
        world.seed_block(c["_id"], x0, y0, z0, x1, y1, z1)
    world.finalize(int(p["seed"]))
    return world
