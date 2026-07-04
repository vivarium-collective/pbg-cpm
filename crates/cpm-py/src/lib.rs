use ::cpm_core::energy::ContactMatrix;
use ::cpm_core::lattice::{Boundary, Lattice, Neighborhood};
use ::cpm_core::sweep::Cpm;
use ::cpm_core::world::World as CoreWorld;
use pyo3::prelude::*;
use std::collections::HashMap;

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
    fn world_mut(&mut self) -> &mut CoreWorld {
        if let Some(c) = &mut self.cpm { &mut c.world } else { self.core.as_mut().unwrap() }
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

    fn seed_from_labels(
        &mut self,
        labels: Vec<u32>,
        types: HashMap<u32, u16>,
        default_type: u16,
        target_volume: f64,
        lambda_volume: f64,
    ) -> PyResult<HashMap<u32, u32>> {
        self.max_type = self.max_type.max(default_type);
        for t in types.values() {
            self.max_type = self.max_type.max(*t);
        }
        let core = self.core.as_mut().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("seed after finalize"))?;
        Ok(core.seed_from_labels(&labels, &types, default_type, target_volume, lambda_volume))
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

    fn add_field(&mut self, name: &str, d: f64, decay: f64) -> PyResult<usize> {
        let core = self.core.as_mut().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("add_field after finalize"))?;
        Ok(core.add_field(name, d, decay))
    }

    fn set_secretion(&mut self, field_idx: usize, cell_type: u16, rate: f64) {
        self.world_mut().set_secretion(field_idx, cell_type, rate);
    }

    fn set_chemotaxis(&mut self, field_idx: usize, cell_type: u16, lambda_val: f64) {
        self.world_mut().set_chemotaxis(field_idx, cell_type, lambda_val);
    }

    fn field_conc(&self, field_idx: usize) -> Vec<f32> {
        self.world_ref().field_conc(field_idx)
    }

    fn field_mean_at_cell(&self, field_idx: usize, cell_id: u32) -> f64 {
        self.world_ref().field_mean_at_cell(field_idx, cell_id)
    }

    fn grow(&mut self, cell_type: u16, rate: f64) {
        self.world_mut().grow(cell_type, rate);
    }

    fn set_cell_type(&mut self, cell_id: u32, new_type: u16) {
        self.world_mut().set_cell_type(cell_id, new_type);
    }

    fn set_connectivity(&mut self, cell_type: u16, on: bool) {
        self.max_type = self.max_type.max(cell_type);
        self.world_mut().set_connectivity(cell_type, on);
    }

    fn set_connectivity_medium(&mut self, on: bool) {
        self.world_mut().set_connectivity_medium(on);
    }

    fn set_membrane(&mut self, anchors: Vec<usize>, k: f64, band: f64) {
        self.world_mut().set_membrane(&anchors, k, band);
    }

    fn set_membrane_anchored(&mut self, cell_type: u16, on: bool) {
        self.max_type = self.max_type.max(cell_type);
        self.world_mut().set_membrane_anchored(cell_type, on);
    }

    fn set_junction(&mut self, cell_type: u16, on: bool) {
        self.max_type = self.max_type.max(cell_type);
        self.world_mut().set_junction(cell_type, on);
    }

    fn set_junction_lambda(&mut self, lambda: f64) {
        self.world_mut().set_junction_lambda(lambda);
    }

    fn set_target_volume(&mut self, cell_id: u32, v: f64) {
        self.world_mut().set_target_volume(cell_id, v);
    }

    fn remove_cells(&mut self, ids: Vec<u32>) {
        self.world_mut().remove_cells(&ids);
    }

    fn divide_cells(&mut self, threshold: f64, reset_target: f64) -> Vec<u32> {
        self.world_mut().divide_cells(threshold, reset_target)
    }

    fn n_cells(&self) -> usize {
        self.world_ref().cells.len() - 1
    }
}

#[pymodule]
fn cpm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    Ok(())
}
