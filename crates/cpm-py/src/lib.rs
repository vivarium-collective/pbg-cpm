use ::cpm_core::energy::ContactMatrix;
use ::cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use ::cpm_core::sweep::Cpm;
use ::cpm_core::world::World as CoreWorld;
use pyo3::prelude::*;

#[pyclass]
struct World {
    // Before finalize(): hold the core world. After: hold the Cpm driver.
    core: Option<CoreWorld>,
    cpm: Option<Cpm>,
    dims: [usize; 3],
    max_type: u16,
    contacts: Vec<(u16, u16, f64)>,
}

impl World {
    fn world_ref(&self) -> &CoreWorld {
        if let Some(c) = &self.cpm { &c.world } else { self.core.as_ref().unwrap() }
    }
}

#[pymethods]
impl World {
    #[new]
    fn new(dims: (usize, usize, usize), boundary: &str, neighbor_order: u8, temperature: f64) -> PyResult<Self> {
        let b = match boundary {
            "noflux" => Boundary::NoFlux,
            "periodic" => Boundary::Periodic,
            other => return Err(pyo3::exceptions::PyValueError::new_err(format!("bad boundary {other}"))),
        };
        let dims = [dims.0, dims.1, dims.2];
        let is_3d = dims[2] > 1;
        let lat = Lattice::new(dims, [b; 3], Neighborhood::new(is_3d, neighbor_order));
        let core = CoreWorld::new(lat, temperature);
        Ok(World { core: Some(core), cpm: None, dims, max_type: 0, contacts: Vec::new() })
    }

    fn add_cell(&mut self, cell_type: u16, target_volume: f64, lambda_volume: f64, target_surface: f64, lambda_surface: f64) -> PyResult<u32> {
        let core = self.core.as_mut().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("add_cell after finalize"))?;
        self.max_type = self.max_type.max(cell_type);
        Ok(core.add_cell(cell_type, target_volume, lambda_volume, target_surface, lambda_surface))
    }

    fn set_contact(&mut self, type_a: u16, type_b: u16, j: f64) {
        self.max_type = self.max_type.max(type_a).max(type_b);
        self.contacts.push((type_a, type_b, j));
    }

    fn seed_block(&mut self, cell_id: u32, x0: usize, y0: usize, z0: usize, x1: usize, y1: usize, z1: usize) -> PyResult<()> {
        let core = self.core.as_mut().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("seed after finalize"))?;
        for z in z0..z1 {
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = core.lattice.index(x, y, z);
                    core.paint(idx, cell_id);
                }
            }
        }
        Ok(())
    }

    fn finalize(&mut self, seed: u64) -> PyResult<()> {
        let mut core = self.core.take().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("already finalized"))?;
        let mut m = ContactMatrix::new((self.max_type as usize) + 1);
        for (a, b, j) in &self.contacts {
            m.set(*a, *b, *j);
        }
        core.set_contact_matrix(m);
        core.recompute_trackers();
        self.cpm = Some(Cpm::new(core, seed));
        Ok(())
    }

    fn step(&mut self, mcs: u64) -> PyResult<()> {
        let cpm = self.cpm.as_mut().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("call finalize() first"))?;
        cpm.step(mcs);
        Ok(())
    }

    fn cell_volumes(&self) -> Vec<i64> {
        self.world_ref().cells.iter().map(|c| c.volume).collect()
    }
    fn cell_surfaces(&self) -> Vec<i64> {
        self.world_ref().cells.iter().map(|c| c.surface).collect()
    }
    fn cell_types(&self) -> Vec<u16> {
        self.world_ref().cells.iter().map(|c| c.cell_type).collect()
    }
    fn cell_coms(&self) -> Vec<(f64, f64, f64)> {
        let w = self.world_ref();
        w.cells.iter().map(|c| {
            let com = w.com(c.id);
            (com[0], com[1], com[2])
        }).collect()
    }
    fn snapshot(&self) -> Vec<u32> {
        let w = self.world_ref();
        (0..w.lattice.n_sites()).map(|i| w.lattice.owner(i)).collect()
    }
    fn dims(&self) -> (usize, usize, usize) {
        (self.dims[0], self.dims[1], self.dims[2])
    }
}

#[pymodule]
fn cpm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    Ok(())
}
