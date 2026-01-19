"""
Python wrapper for Rust N-body implementation
Provides a consistent interface matching other implementations

To build the Rust module:
    cd nbody/rust
    pip install maturin
    maturin develop --release

Adam Field - Computational Physics ISP
"""

import numpy as np

try:
    import nbody_rust_module
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("Warning: Rust module not compiled. Run 'maturin develop --release' in nbody/rust/")


def simulate(positions_init, velocities_init, masses, n_steps, G=1.0, softening=0.1, dt=0.01):
    """
    Run N-body simulation using Rust backend
    
    Args:
        positions_init: (N, 3) array
        velocities_init: (N, 3) array
        masses: (N,) array
        n_steps: number of timesteps
        G, softening, dt: physics parameters
    
    Returns:
        final_positions, final_velocities (both (N, 3) arrays)
    """
    if not HAS_RUST:
        raise ImportError("Rust module not available")
    
    # Ensure correct types and shapes
    positions = np.ascontiguousarray(positions_init, dtype=np.float64)
    velocities = np.ascontiguousarray(velocities_init, dtype=np.float64)
    masses_arr = np.ascontiguousarray(masses, dtype=np.float64)
    
    return nbody_rust_module.simulate(
        positions, velocities, masses_arr, n_steps, G, softening, dt
    )


def compute_energy(positions, velocities, masses, G=1.0, softening=0.1):
    """
    Compute energy using Rust backend
    
    Returns:
        (kinetic_energy, potential_energy, total_energy)
    """
    if not HAS_RUST:
        raise ImportError("Rust module not available")
    
    # Ensure correct types
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    velocities = np.ascontiguousarray(velocities, dtype=np.float64)
    masses_arr = np.ascontiguousarray(masses, dtype=np.float64)
    
    return nbody_rust_module.compute_energy(positions, velocities, masses_arr, G, softening)


if __name__ == "__main__":
    if not HAS_RUST:
        print("ERROR: Rust module not compiled!")
        print("\nTo compile:")
        print("  cd nbody/rust")
        print("  pip install maturin")
        print("  maturin develop --release")
        exit(1)
    
    print("Testing Rust wrapper...")
    
    # Quick test
    N = 10
    np.random.seed(42)
    positions = np.random.uniform(-10, 10, (N, 3))
    velocities = np.random.randn(N, 3) * 0.5
    masses = np.random.uniform(0.1, 1.0, N)
    
    print(f"Testing with N={N} particles...")
    
    ke, pe, E = compute_energy(positions, velocities, masses)
    print(f"Initial energy: KE={ke:.4f}, PE={pe:.4f}, Total={E:.4f}")
    
    pos_final, vel_final = simulate(positions, velocities, masses, 100)
    
    ke_f, pe_f, E_f = compute_energy(pos_final, vel_final, masses)
    print(f"Final energy: KE={ke_f:.4f}, PE={pe_f:.4f}, Total={E_f:.4f}")
    print(f"Energy drift: {abs(E_f - E)/abs(E)*100:.6f}%")
    print("\nRust wrapper works!")