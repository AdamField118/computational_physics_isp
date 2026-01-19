"""
Python wrapper for Julia N-body implementation using PyJulia
Direct Julia calling without subprocess overhead

Adam Field - Computational Physics ISP
"""

import numpy as np
from pathlib import Path

# Import PyJulia
try:
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    HAS_PYJULIA = True
except ImportError:
    HAS_PYJULIA = False
    print("PyJulia not available. Install with: pip install julia")

# Get path to Julia implementation
JULIA_DIR = Path(__file__).parent
JULIA_SCRIPT = JULIA_DIR / "nbody.jl"

# Flag to track if Julia code is loaded
_JULIA_LOADED = False


def _load_julia_code():
    """Load Julia code into Python session (only once)"""
    global _JULIA_LOADED
    
    if not _JULIA_LOADED and HAS_PYJULIA:
        # Load the Julia implementation
        Main.include(str(JULIA_SCRIPT))
        _JULIA_LOADED = True


def simulate(positions_init, velocities_init, masses, n_steps, 
            G=1.0, softening=0.1, dt=0.01):
    """
    Run N-body simulation using Julia backend via PyJulia
    
    Args:
        positions_init: (N, 3) array
        velocities_init: (N, 3) array  
        masses: (N,) array
        n_steps: number of timesteps
        G, softening, dt: physics parameters
    
    Returns:
        final_positions, final_velocities (both (N, 3) arrays)
    """
    if not HAS_PYJULIA:
        raise ImportError("PyJulia not available. Install with: pip install julia")
    
    # Load Julia code if not already loaded
    _load_julia_code()
    
    # Convert to correct types (Julia expects Float64)
    positions = positions_init.astype(np.float64)
    velocities = velocities_init.astype(np.float64)
    masses_arr = masses.astype(np.float64)
    
    # Call Julia function directly
    # Julia will automatically convert NumPy arrays
    pos_final, vel_final = Main.simulate(
        positions, velocities, masses_arr, 
        n_steps, G, softening, dt, True  # True = use vectorized version
    )
    
    # Convert back to NumPy arrays
    return np.array(pos_final), np.array(vel_final)


def compute_energy(positions, velocities, masses, G=1.0, softening=0.1):
    """
    Compute energy using Julia backend via PyJulia
    
    Returns:
        (kinetic_energy, potential_energy, total_energy)
    """
    if not HAS_PYJULIA:
        raise ImportError("PyJulia not available")
    
    # Load Julia code if not already loaded
    _load_julia_code()
    
    # Convert to correct types
    positions = positions.astype(np.float64)
    velocities = velocities.astype(np.float64)
    masses_arr = masses.astype(np.float64)
    
    # Call Julia function directly
    ke, pe, total = Main.compute_energy(
        positions, velocities, masses_arr, G, softening
    )
    
    return float(ke), float(pe), float(total)


def check_pyjulia_available():
    """Check if PyJulia is available and working"""
    if not HAS_PYJULIA:
        return False
    
    try:
        # Try to execute simple Julia code
        Main.eval("1 + 1")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    print("Testing PyJulia wrapper...")
    print("="*50)
    
    if not HAS_PYJULIA:
        print("ERROR: PyJulia not available!")
        print("\nTo install:")
        print("  pip install julia")
        print("  python -c 'import julia; julia.install()'")
        exit(1)
    
    # Load Julia code
    _load_julia_code()
    
    # Quick test
    N = 10
    np.random.seed(42)
    positions = np.random.uniform(-10, 10, (N, 3))
    velocities = np.random.randn(N, 3) * 0.5
    masses = np.random.uniform(0.1, 1.0, N)
    
    print(f"\nTesting with N={N} particles...")
    
    import time
    
    # Initial energy
    t0 = time.perf_counter()
    ke, pe, E = compute_energy(positions, velocities, masses)
    print(f"Initial energy: KE={ke:.4f}, PE={pe:.4f}, Total={E:.4f}")
    
    # Run simulation
    t0 = time.perf_counter()
    pos_final, vel_final = simulate(positions, velocities, masses, 100)
    elapsed = time.perf_counter() - t0
    
    # Final energy
    ke_f, pe_f, E_f = compute_energy(pos_final, vel_final, masses)
    print(f"Final energy: KE={ke_f:.4f}, PE={pe_f:.4f}, Total={E_f:.4f}")
    print(f"Energy drift: {abs(E_f - E)/abs(E)*100:.6f}%")
    print(f"Runtime: {elapsed:.4f}s ({elapsed/100*1000:.4f} ms/step)")
    
    print("\n" + "="*50)
    print("✓ PyJulia wrapper works!")
    print("="*50)