"""
N-Body Gravitational Simulation - JAX Implementation
GPU-accelerated using JAX's JIT compilation and automatic vectorization

Adam Field - Computational Physics ISP
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple, NamedTuple
import time


class NBodyState(NamedTuple):
    """State of the N-body system at a given time"""
    positions: jnp.ndarray  # Shape: (N, 3) - x, y, z coordinates
    velocities: jnp.ndarray # Shape: (N, 3) - vx, vy, vz
    masses: jnp.ndarray     # Shape: (N,) - particle masses
    time: float


class NBodyConfig(NamedTuple):
    """Configuration parameters for the simulation"""
    G: float = 1.0          # Gravitational constant
    softening: float = 0.1  # Softening parameter to avoid singularities
    dt: float = 0.01        # Timestep


@jit
def compute_acceleration(positions: jnp.ndarray, 
                         masses: jnp.ndarray,
                         G: float,
                         softening: float) -> jnp.ndarray:
    """
    Compute gravitational acceleration on all particles
    
    Uses pairwise force calculation: O(N²) complexity
    
    Args:
        positions: (N, 3) array of particle positions
        masses: (N,) array of particle masses
        G: Gravitational constant
        softening: Softening parameter (prevents singularities)
    
    Returns:
        (N, 3) array of accelerations
    """
    N = positions.shape[0]
    
    # Compute all pairwise displacement vectors
    # positions[None, :, :] has shape (1, N, 3)
    # positions[:, None, :] has shape (N, 1, 3)
    # Broadcasting gives (N, N, 3) - all pairwise displacements
    displacements = positions[None, :, :] - positions[:, None, :]
    
    # Compute distances with softening
    # Shape: (N, N)
    distances_sq = jnp.sum(displacements**2, axis=2) + softening**2
    distances = jnp.sqrt(distances_sq)
    
    # Compute force magnitudes: F = G * m_i * m_j / r²
    # But we need F/m_i = G * m_j / r² for acceleration
    # Shape: (N, N)
    inv_dist_cubed = 1.0 / (distances * distances_sq)
    
    # Force contributions (before multiplying by masses)
    # Shape: (N, N, 3)
    force_directions = displacements * inv_dist_cubed[:, :, None]
    
    # Multiply by masses of attracting particles
    # masses[None, :, None] has shape (1, N, 1)
    # Broadcasting gives (N, N, 3)
    accelerations = G * jnp.sum(force_directions * masses[None, :, None], axis=1)
    
    return accelerations


@jit
def velocity_verlet_step(state: NBodyState, config: NBodyConfig) -> NBodyState:
    """
    Advance the system by one timestep using Velocity Verlet integration
    
    Velocity Verlet is symplectic and conserves energy better than Euler
    
    Algorithm:
        1. r(t + dt) = r(t) + v(t)*dt + 0.5*a(t)*dt²
        2. a(t + dt) = compute_acceleration(r(t + dt))
        3. v(t + dt) = v(t) + 0.5*(a(t) + a(t + dt))*dt
    
    Args:
        state: Current state of the system
        config: Simulation configuration
    
    Returns:
        New state after one timestep
    """
    # Current acceleration
    acc_current = compute_acceleration(
        state.positions, state.masses, config.G, config.softening
    )
    
    # Update positions
    new_positions = (state.positions + 
                     state.velocities * config.dt + 
                     0.5 * acc_current * config.dt**2)
    
    # Compute new acceleration at updated positions
    acc_new = compute_acceleration(
        new_positions, state.masses, config.G, config.softening
    )
    
    # Update velocities using average of old and new acceleration
    new_velocities = state.velocities + 0.5 * (acc_current + acc_new) * config.dt
    
    return NBodyState(
        positions=new_positions,
        velocities=new_velocities,
        masses=state.masses,
        time=state.time + config.dt
    )


def simulate(initial_state: NBodyState,
             config: NBodyConfig,
             n_steps: int,
             save_every: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run the N-body simulation for n_steps
    
    Args:
        initial_state: Initial conditions
        config: Simulation parameters
        n_steps: Number of timesteps to simulate
        save_every: Save state every N steps (for memory efficiency)
    
    Returns:
        Tuple of (times, positions, velocities) arrays
    """
    state = initial_state
    
    # Pre-allocate storage
    n_save = n_steps // save_every + 1
    N = initial_state.positions.shape[0]
    
    times = jnp.zeros(n_save)
    positions_history = jnp.zeros((n_save, N, 3))
    velocities_history = jnp.zeros((n_save, N, 3))
    
    # Store initial state
    positions_history = positions_history.at[0].set(state.positions)
    velocities_history = velocities_history.at[0].set(state.velocities)
    times = times.at[0].set(state.time)
    
    save_idx = 1
    for step in range(n_steps):
        state = velocity_verlet_step(state, config)
        
        if (step + 1) % save_every == 0:
            positions_history = positions_history.at[save_idx].set(state.positions)
            velocities_history = velocities_history.at[save_idx].set(state.velocities)
            times = times.at[save_idx].set(state.time)
            save_idx += 1
    
    return times, positions_history, velocities_history


def create_random_system(N: int, 
                         key: jax.random.PRNGKey,
                         position_scale: float = 10.0,
                         velocity_scale: float = 1.0,
                         mass_range: Tuple[float, float] = (0.1, 1.0)) -> NBodyState:
    """
    Create random initial conditions for N-body system
    
    Args:
        N: Number of particles
        key: JAX random key
        position_scale: Spatial extent of initial distribution
        velocity_scale: Magnitude of initial velocities
        mass_range: (min, max) mass range
    
    Returns:
        Initial NBodyState
    """
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Random positions (uniform in cube)
    positions = jax.random.uniform(k1, (N, 3), minval=-position_scale, maxval=position_scale)
    
    # Random velocities (Gaussian)
    velocities = velocity_scale * jax.random.normal(k2, (N, 3))
    
    # Random masses (uniform in range)
    masses = jax.random.uniform(k3, (N,), minval=mass_range[0], maxval=mass_range[1])
    
    return NBodyState(positions=positions, velocities=velocities, masses=masses, time=0.0)


def create_solar_system() -> NBodyState:
    """
    Create a simple solar system-like initial condition
    (Sun + a few planets in circular orbits)
    """
    # Sun at center
    sun_mass = 1000.0
    
    # 3 planets in circular orbits
    planet_masses = jnp.array([1.0, 2.0, 0.5])
    radii = jnp.array([5.0, 10.0, 15.0])
    
    # Circular orbit velocity: v = sqrt(G*M/r)
    G = 1.0
    velocities_mag = jnp.sqrt(G * sun_mass / radii)
    
    # Positions (planets on x-axis for simplicity)
    planet_positions = jnp.column_stack([radii, jnp.zeros(3), jnp.zeros(3)])
    
    # Velocities (in y-direction for circular orbits)
    planet_velocities = jnp.column_stack([jnp.zeros(3), velocities_mag, jnp.zeros(3)])
    
    # Combine sun + planets
    positions = jnp.vstack([jnp.zeros(3), planet_positions])
    velocities = jnp.vstack([jnp.zeros(3), planet_velocities])
    masses = jnp.concatenate([jnp.array([sun_mass]), planet_masses])
    
    return NBodyState(positions=positions, velocities=velocities, masses=masses, time=0.0)


def compute_energy(state: NBodyState, config: NBodyConfig) -> Tuple[float, float, float]:
    """
    Compute total energy of the system (for verification/debugging)
    
    Returns:
        (kinetic_energy, potential_energy, total_energy)
    """
    # Kinetic energy: 0.5 * m * v²
    ke = 0.5 * jnp.sum(state.masses[:, None] * state.velocities**2)
    
    # Potential energy: -G * m_i * m_j / r_ij (summed over all pairs)
    N = state.positions.shape[0]
    displacements = state.positions[None, :, :] - state.positions[:, None, :]
    distances = jnp.sqrt(jnp.sum(displacements**2, axis=2) + config.softening**2)
    
    # Avoid double counting and self-interaction
    mass_products = state.masses[None, :] * state.masses[:, None]
    # Zero out diagonal and upper triangle
    mask = jnp.tril(jnp.ones((N, N)), k=-1)
    
    pe = -config.G * jnp.sum(mask * mass_products / distances)
    
    return float(ke), float(pe), float(ke + pe)


if __name__ == "__main__":
    print("JAX N-Body Simulation")
    print("=" * 50)
    
    # Create initial conditions
    key = jax.random.PRNGKey(42)
    N = 100
    print(f"Creating system with {N} particles...")
    
    # Random system
    initial_state = create_random_system(N, key, position_scale=10.0, velocity_scale=0.5)
    
    # Or solar system
    # initial_state = create_solar_system()
    
    config = NBodyConfig(G=1.0, softening=0.1, dt=0.01)
    
    # Compute initial energy
    ke0, pe0, E0 = compute_energy(initial_state, config)
    print(f"Initial energy: KE={ke0:.4f}, PE={pe0:.4f}, Total={E0:.4f}")
    
    # Run simulation
    n_steps = 1000
    save_every = 10
    
    print(f"\nRunning {n_steps} steps...")
    start_time = time.time()
    
    # First run compiles (JIT)
    times, positions, velocities = simulate(initial_state, config, n_steps, save_every)
    
    elapsed = time.time() - start_time
    print(f"First run (with compilation): {elapsed:.4f} seconds")
    
    # Second run (compiled, should be much faster)
    start_time = time.time()
    times, positions, velocities = simulate(initial_state, config, n_steps, save_every)
    elapsed = time.time() - start_time
    
    print(f"Second run (compiled): {elapsed:.4f} seconds")
    print(f"Time per step: {elapsed/n_steps*1000:.4f} ms")
    
    # Compute final energy
    final_state = NBodyState(
        positions=positions[-1], 
        velocities=velocities[-1],
        masses=initial_state.masses,
        time=times[-1]
    )
    ke_f, pe_f, E_f = compute_energy(final_state, config)
    
    print(f"\nFinal energy: KE={ke_f:.4f}, PE={pe_f:.4f}, Total={E_f:.4f}")
    print(f"Energy drift: {abs(E_f - E0)/abs(E0)*100:.6f}%")
    
    print(f"\nOutput shapes:")
    print(f"  Times: {times.shape}")
    print(f"  Positions: {positions.shape}")
    print(f"  Velocities: {velocities.shape}")