"""
Python wrapper for C N-body implementation using ctypes
"""

import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import os

# Load shared library
lib_path = os.path.join(os.path.dirname(__file__), 'nbody_c.so')
if not os.path.exists(lib_path):
    raise ImportError(f"C library not found at {lib_path}")

_nbody_c = ctypes.CDLL(lib_path)

# Define function signatures
_nbody_c.simulate.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions_init
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities_init
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
    ctypes.c_int,                                       # n_particles
    ctypes.c_int,                                       # n_steps
    ctypes.c_double,                                    # G
    ctypes.c_double,                                    # softening
    ctypes.c_double,                                    # dt
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions_out
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities_out
]

_nbody_c.compute_energy.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
    ctypes.c_int,                                       # n_particles
    ctypes.c_double,                                    # G
    ctypes.c_double,                                    # softening
    ctypes.POINTER(ctypes.c_double),                   # kinetic
    ctypes.POINTER(ctypes.c_double),                   # potential
    ctypes.POINTER(ctypes.c_double),                   # total
]


def simulate(positions_init, velocities_init, masses, n_steps, G=1.0, softening=0.1, dt=0.01):
    """
    Run N-body simulation (Python wrapper for C code)
    
    Args:
        positions_init: (N, 3) array
        velocities_init: (N, 3) array
        masses: (N,) array
        n_steps: number of timesteps
        G, softening, dt: physics parameters
    
    Returns:
        final_positions, final_velocities (both (N, 3) arrays)
    """
    n_particles = len(masses)
    
    # Flatten and ensure C-contiguous
    pos_flat = np.ascontiguousarray(positions_init.flatten(), dtype=np.float64)
    vel_flat = np.ascontiguousarray(velocities_init.flatten(), dtype=np.float64)
    masses_c = np.ascontiguousarray(masses, dtype=np.float64)
    
    # Allocate output arrays
    pos_out = np.zeros_like(pos_flat)
    vel_out = np.zeros_like(vel_flat)
    
    # Call C function
    _nbody_c.simulate(
        pos_flat, vel_flat, masses_c, 
        n_particles, n_steps,
        G, softening, dt, 
        pos_out, vel_out
    )
    
    # Reshape back to (N, 3)
    return pos_out.reshape(-1, 3), vel_out.reshape(-1, 3)


def compute_energy(positions, velocities, masses, G=1.0, softening=0.1):
    """
    Compute energy (Python wrapper for C code)
    
    Returns:
        (kinetic_energy, potential_energy, total_energy)
    """
    n_particles = len(masses)
    
    pos_flat = np.ascontiguousarray(positions.flatten(), dtype=np.float64)
    vel_flat = np.ascontiguousarray(velocities.flatten(), dtype=np.float64)
    masses_c = np.ascontiguousarray(masses, dtype=np.float64)
    
    ke = ctypes.c_double()
    pe = ctypes.c_double()
    total = ctypes.c_double()
    
    _nbody_c.compute_energy(
        pos_flat, vel_flat, masses_c, n_particles,
        G, softening, 
        ctypes.byref(ke), ctypes.byref(pe), ctypes.byref(total)
    )
    
    return ke.value, pe.value, total.value


if __name__ == "__main__":
    print("Testing C wrapper...")
    
    # Quick test
    N = 10
    positions = np.random.uniform(-10, 10, (N, 3))
    velocities = np.random.randn(N, 3) * 0.5
    masses = np.random.uniform(0.1, 1.0, N)
    
    ke, pe, E = compute_energy(positions, velocities, masses)
    print(f"Initial energy: KE={ke:.4f}, PE={pe:.4f}, Total={E:.4f}")
    
    pos_final, vel_final = simulate(positions, velocities, masses, 100)
    
    ke_f, pe_f, E_f = compute_energy(pos_final, vel_final, masses)
    print(f"Final energy: KE={ke_f:.4f}, PE={pe_f:.4f}, Total={E_f:.4f}")
    print(f"Energy drift: {abs(E_f - E)/abs(E)*100:.6f}%")
    print("C wrapper works!")