"""
N-Body Gravitational Simulation - Pure Python/NumPy Implementation
(Baseline for performance comparison)

Adam Field - Computational Physics ISP
"""

import numpy as np
import time


def compute_acceleration(positions, masses, G=1.0, softening=0.1):
    """
    Compute gravitational accelerations on all particles
    Pure NumPy implementation - O(N²)
    
    Args:
        positions: (N, 3) array of particle positions
        masses: (N,) array of particle masses
        G: Gravitational constant
        softening: Softening parameter
    
    Returns:
        (N, 3) array of accelerations
    """
    N = positions.shape[0]
    accelerations = np.zeros_like(positions)
    
    # Pairwise force calculation (vectorized)
    # displacements[i, j] = positions[j] - positions[i]
    displacements = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    
    # distances[i, j] = |r_j - r_i|
    distances_sq = np.sum(displacements**2, axis=2) + softening**2
    distances = np.sqrt(distances_sq)
    
    # Force per unit mass
    inv_dist_cubed = 1.0 / (distances * distances_sq)
    
    # accelerations = sum_j(G * m_j * (r_j - r_i) / |r_j - r_i|^3)
    force_directions = displacements * inv_dist_cubed[:, :, np.newaxis]
    accelerations = G * np.sum(force_directions * masses[np.newaxis, :, np.newaxis], axis=1)
    
    return accelerations


def velocity_verlet_step(positions, velocities, masses, G=1.0, softening=0.1, dt=0.01):
    """
    Single timestep using Velocity Verlet integration
    
    Args:
        positions: (N, 3) current positions
        velocities: (N, 3) current velocities
        masses: (N,) particle masses
        G, softening, dt: physics parameters
    
    Returns:
        new_positions, new_velocities
    """
    # Current acceleration
    acc_current = compute_acceleration(positions, masses, G, softening)
    
    # Update positions
    new_positions = positions + velocities * dt + 0.5 * acc_current * dt**2
    
    # Compute new acceleration
    acc_new = compute_acceleration(new_positions, masses, G, softening)
    
    # Update velocities
    new_velocities = velocities + 0.5 * (acc_current + acc_new) * dt
    
    return new_positions, new_velocities


def simulate(positions_init, velocities_init, masses, n_steps, G=1.0, softening=0.1, dt=0.01):
    """
    Run N-body simulation for n_steps
    
    Args:
        positions_init: (N, 3) initial positions
        velocities_init: (N, 3) initial velocities
        masses: (N,) particle masses
        n_steps: number of timesteps
        G, softening, dt: physics parameters
    
    Returns:
        final_positions, final_velocities
    """
    positions = positions_init.copy()
    velocities = velocities_init.copy()
    
    for step in range(n_steps):
        positions, velocities = velocity_verlet_step(
            positions, velocities, masses, G, softening, dt
        )
    
    return positions, velocities


def compute_energy(positions, velocities, masses, G=1.0, softening=0.1):
    """
    Compute total energy of the system
    
    Returns:
        (kinetic_energy, potential_energy, total_energy)
    """
    # Kinetic energy
    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    
    # Potential energy (sum over pairs)
    N = positions.shape[0]
    displacements = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    distances = np.sqrt(np.sum(displacements**2, axis=2) + softening**2)
    
    mass_products = masses[np.newaxis, :] * masses[:, np.newaxis]
    mask = np.tril(np.ones((N, N)), k=-1)
    
    pe = -G * np.sum(mask * mass_products / distances)
    
    return ke, pe, ke + pe


if __name__ == "__main__":
    print("Pure Python N-Body Simulation")
    print("=" * 50)
    
    # Create test system
    N = 100
    np.random.seed(42)
    
    positions = np.random.uniform(-10, 10, (N, 3))
    velocities = 0.5 * np.random.randn(N, 3)
    masses = np.random.uniform(0.1, 1.0, N)
    
    # Initial energy
    ke0, pe0, E0 = compute_energy(positions, velocities, masses)
    print(f"Initial energy: KE={ke0:.4f}, PE={pe0:.4f}, Total={E0:.4f}")
    
    # Run simulation
    n_steps = 1000
    print(f"\nRunning {n_steps} steps...")
    
    start = time.time()
    pos_final, vel_final = simulate(positions, velocities, masses, n_steps)
    elapsed = time.time() - start
    
    print(f"Runtime: {elapsed:.4f} seconds")
    print(f"Time per step: {elapsed/n_steps*1000:.4f} ms")
    
    # Final energy
    ke_f, pe_f, E_f = compute_energy(pos_final, vel_final, masses)
    print(f"\nFinal energy: KE={ke_f:.4f}, PE={pe_f:.4f}, Total={E_f:.4f}")
    print(f"Energy drift: {abs(E_f - E0)/abs(E0)*100:.6f}%")