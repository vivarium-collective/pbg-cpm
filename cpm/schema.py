"""Declarative CPM spec → Rust `World`.

A spec is a plain dict describing a Cellular Potts model. `load_world` builds and
finalizes a `World` from it. Sections (all optional unless noted):

  potts (required): {dims:[x,y,z], boundary:"noflux"|"periodic",
                     neighbor_order:int, temperature:float, seed:int}
  cells:    [{type, target_volume, lambda_volume, target_surface,
              lambda_surface, seed_block:[x0,y0,z0,x1,y1,z1]}]  # explicit placement
  seed_labels: {labels:[...], types:{id:type}, default_type, target_volume,
              lambda_volume}                                     # from a segmentation
              (cells and seed_labels are mutually exclusive)
  contact:  [{a, b, j}]                                          # adhesion matrix
  fields:   [{name, d, decay, secretion:[{type,rate}],
              chemotaxis:[{type,lambda}],
              dynamics:{dt, substeps}}]                          # reaction–diffusion PDE
  connectivity: {types:[...], medium:bool}                       # anti-fragmentation (E1)
  membrane: {anchors:[...], k, band, types:[...]}                # basement membrane (E3a)
  junctions: {types:[...], lambda}                               # anti-gap (E3b)
  length:   [{type, target_length, lambda}]                      # elongation constraint
  external: [{type, fx?, fy?, fz?}]                              # constant force (gravity/taxis)

Energy terms compose additively in the Metropolis Hamiltonian; each `*_type`
list opts specific cell types into a term, so tissues can mix behaviours.
"""
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
        # optional PDE solver settings (forward-Euler dt + diffusion sub-steps per
        # MCS); keep dt*d*2*ndim < 1 for stability. Absent => engine defaults 1/1.
        dyn = f.get("dynamics")
        if dyn is not None:
            world.set_field_dynamics(idx, float(dyn["dt"]), int(dyn["substeps"]))
    conn = spec.get("connectivity")
    if conn is not None:
        for t in conn.get("types", []):
            world.set_connectivity(int(t), True)
        if conn.get("medium", False):
            world.set_connectivity_medium(True)
    mem = spec.get("membrane")
    if mem:
        world.set_membrane(list(mem["anchors"]), float(mem["k"]), float(mem["band"]))
        for t in mem.get("types", []):
            world.set_membrane_anchored(int(t), True)
    jn = spec.get("junctions")
    if jn:
        for t in jn.get("types", []):
            world.set_junction(int(t), True)
        world.set_junction_lambda(float(jn.get("lambda", 0.0)))
    # length (elongation) constraint: per-type target major-axis length + spring
    for lc in spec.get("length", []):
        world.set_length_constraint(int(lc["type"]), float(lc["target_length"]), float(lc["lambda"]))
    # external potential: per-type constant force vector (gravity / taxis / bias)
    for ep in spec.get("external", []):
        world.set_external_potential(
            int(ep["type"]), float(ep.get("fx", 0.0)),
            float(ep.get("fy", 0.0)), float(ep.get("fz", 0.0)))
    world.finalize(int(p["seed"]))
    return world
