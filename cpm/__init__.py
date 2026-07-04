"""pbg-cpm — a process-bigraph Cellular Potts Model framework.

A fast Rust CPM engine (``cpm.cpm_core``) with a process-bigraph Python layer:
CPM and subcellular processes, schema-driven world construction, connectivity /
basement-membrane structural constraints, and analysis metrics.

Typical use from another project::

    from cpm import load_world, cpm_core            # engine + schema builder
    world = load_world(spec)                        # build a CPM world from a dict
    world.step(100)

Cells are wired as process-bigraph composites via import-path addresses, e.g.
``local:!cpm.processes.cpm_process.CPMProcess`` and
``local:!cpm.subcellular.sbml.SBMLSubcell`` — see ``cpm.composites`` for examples.
"""
from cpm import cpm_core
from cpm.schema import load_world

__all__ = ["cpm_core", "load_world"]
__version__ = "0.1.0"
