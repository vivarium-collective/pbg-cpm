"""CC3D head-to-head benchmark — mirrors crates/cpm-bench/src/main.rs exactly.

50^3 periodic lattice, 125 cells (5x5x5 grid of 8^3 blocks), Volume + Contact,
neighbor order 2, temperature 10. Times N MCS (n_sites flip attempts each, the
same MCS definition as our Rust engine). Prints MCS/s + copy-attempts/s.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

PIF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocks.piff")
DIM = 50
TEMP = 10.0
MCS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

from cc3d.core.PyCoreSpecs import (
    PottsCore, CellTypePlugin, VolumePlugin, ContactPlugin, PIFInitializer,
)
from cc3d.core.simservice.CC3DSimService import CC3DSimService


def build_specs():
    potts = PottsCore(
        dim_x=DIM, dim_y=DIM, dim_z=DIM,
        steps=MCS + 1, neighbor_order=2, random_seed=1,
        boundary_x="Periodic", boundary_y="Periodic", boundary_z="Periodic",
        fluctuation_amplitude=TEMP,
    )
    cell_type = CellTypePlugin("Type1", "Type2")
    volume = VolumePlugin()
    volume.param_new("Type1", target_volume=512, lambda_volume=1.0)
    volume.param_new("Type2", target_volume=512, lambda_volume=1.0)
    contact = ContactPlugin(neighbor_order=2)
    contact.param_new("Medium", "Medium", 0.0)
    contact.param_new("Medium", "Type1", 16.0)
    contact.param_new("Medium", "Type2", 16.0)
    contact.param_new("Type1", "Type1", 0.0)
    contact.param_new("Type2", "Type2", 0.0)
    contact.param_new("Type1", "Type2", 11.0)
    pif = PIFInitializer(pif_name=PIF)
    return [potts, cell_type, volume, contact, pif]


def main():
    sim = CC3DSimService()
    sim.register_specs(build_specs())
    sim.run()
    sim.init()
    sim.start()
    n_sites = DIM * DIM * DIM

    t0 = time.perf_counter()
    for _ in range(MCS):
        sim.step()
    secs = time.perf_counter() - t0
    sim.finish()

    attempts = MCS * n_sites
    print(f"CC3D 4.10  {DIM}^3, {MCS} MCS: "
          f"{MCS / secs:.2f} MCS/s, {attempts / secs:.2e} copy-attempts/s "
          f"({secs:.2f}s wall)")


if __name__ == "__main__":
    main()
