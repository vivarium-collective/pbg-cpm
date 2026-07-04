import json


def write_pack(world, path, frame=0):
    nx, ny, nz = world.dims()
    data = {
        "format": "cpm.pack.v1",
        "dims": [nx, ny, nz],
        "frame": frame,
        "cell_types": list(world.cell_types()),
        "labels": list(world.snapshot()),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return data
