// N-Body Gravitational Simulation - Rust Implementation
// Using PyO3 for Python bindings
// Adam Field - Computational Physics ISP

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};

/// Compute gravitational accelerations on all particles
/// O(N²) pairwise force calculation
fn compute_acceleration_internal(
    positions: &[f64],
    masses: &[f64],
    n_particles: usize,
    g: f64,
    softening: f64,
) -> Vec<f64> {
    let mut accelerations = vec![0.0; n_particles * 3];
    
    for i in 0..n_particles {
        for j in 0..n_particles {
            if i != j {
                // Displacement vector
                let dx = positions[j * 3 + 0] - positions[i * 3 + 0];
                let dy = positions[j * 3 + 1] - positions[i * 3 + 1];
                let dz = positions[j * 3 + 2] - positions[i * 3 + 2];
                
                // Distance with softening
                let dist_sq = dx * dx + dy * dy + dz * dz + softening * softening;
                let dist = dist_sq.sqrt();
                
                // 1/r³ term
                let inv_dist_cubed = 1.0 / (dist * dist_sq);
                
                // Accumulate acceleration
                let force_factor = g * masses[j] * inv_dist_cubed;
                accelerations[i * 3 + 0] += force_factor * dx;
                accelerations[i * 3 + 1] += force_factor * dy;
                accelerations[i * 3 + 2] += force_factor * dz;
            }
        }
    }
    
    accelerations
}

/// Single timestep using Velocity Verlet integration
fn velocity_verlet_step(
    positions: &mut [f64],
    velocities: &mut [f64],
    masses: &[f64],
    n_particles: usize,
    g: f64,
    softening: f64,
    dt: f64,
) {
    let dt_sq = dt * dt;
    
    // Compute current acceleration
    let acc_current = compute_acceleration_internal(positions, masses, n_particles, g, softening);
    
    // Update positions
    for i in 0..n_particles * 3 {
        positions[i] += velocities[i] * dt + 0.5 * acc_current[i] * dt_sq;
    }
    
    // Compute new acceleration
    let acc_new = compute_acceleration_internal(positions, masses, n_particles, g, softening);
    
    // Update velocities
    for i in 0..n_particles * 3 {
        velocities[i] += 0.5 * (acc_current[i] + acc_new[i]) * dt;
    }
}

/// Run N-body simulation for n_steps
#[pyfunction]
fn simulate(
    py: Python,
    positions_init: PyReadonlyArray2<f64>,
    velocities_init: PyReadonlyArray2<f64>,
    masses: PyReadonlyArray1<f64>,
    n_steps: usize,
    g: f64,
    softening: f64,
    dt: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    // Get array dimensions
    let pos_array = positions_init.as_array();
    let vel_array = velocities_init.as_array();
    let mass_array = masses.as_array();
    
    let n_particles = pos_array.shape()[0];
    
    // Flatten arrays (row-major for Python compatibility)
    let mut pos: Vec<f64> = pos_array.iter().copied().collect();
    let mut vel: Vec<f64> = vel_array.iter().copied().collect();
    let mass_vec: Vec<f64> = mass_array.iter().copied().collect();
    
    // Time integration loop
    for _ in 0..n_steps {
        velocity_verlet_step(&mut pos, &mut vel, &mass_vec, n_particles, g, softening, dt);
    }
    
    // Convert back to 2D arrays
    let pos_2d = PyArray2::from_vec2(
        py,
        &pos.chunks(3)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>(),
    )?;
    
    let vel_2d = PyArray2::from_vec2(
        py,
        &vel.chunks(3)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>(),
    )?;
    
    Ok((pos_2d.into(), vel_2d.into()))
}

/// Compute total energy of the system
#[pyfunction]
fn compute_energy(
    positions: PyReadonlyArray2<f64>,
    velocities: PyReadonlyArray2<f64>,
    masses: PyReadonlyArray1<f64>,
    g: f64,
    softening: f64,
) -> PyResult<(f64, f64, f64)> {
    let pos_array = positions.as_array();
    let vel_array = velocities.as_array();
    let mass_array = masses.as_array();
    
    let n_particles = pos_array.shape()[0];
    
    // Kinetic energy
    let mut ke = 0.0;
    for i in 0..n_particles {
        let vx = vel_array[[i, 0]];
        let vy = vel_array[[i, 1]];
        let vz = vel_array[[i, 2]];
        let v_sq = vx * vx + vy * vy + vz * vz;
        ke += 0.5 * mass_array[i] * v_sq;
    }
    
    // Potential energy (sum over pairs)
    let mut pe = 0.0;
    for i in 0..n_particles {
        for j in (i + 1)..n_particles {
            let dx = pos_array[[j, 0]] - pos_array[[i, 0]];
            let dy = pos_array[[j, 1]] - pos_array[[i, 1]];
            let dz = pos_array[[j, 2]] - pos_array[[i, 2]];
            
            let dist = (dx * dx + dy * dy + dz * dz + softening * softening).sqrt();
            
            pe -= g * mass_array[i] * mass_array[j] / dist;
        }
    }
    
    Ok((ke, pe, ke + pe))
}

/// Python module
#[pymodule]
fn nbody_rust_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(compute_energy, m)?)?;
    Ok(())
}