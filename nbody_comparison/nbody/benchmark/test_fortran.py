"""Quick test of Fortran module"""
import numpy as np
import time
import nbody_fortran_module as fortran_nbody

print("Testing Fortran N-body module")
print("=" * 50)

# Create test system
N = 100
np.random.seed(42)

positions = np.random.uniform(-10, 10, (N, 3)).astype(np.float64)
velocities = 0.5 * np.random.randn(N, 3).astype(np.float64)
masses = np.random.uniform(0.1, 1.0, N).astype(np.float64)

# Fortran uses column-major order
positions_f = np.asfortranarray(positions)
velocities_f = np.asfortranarray(velocities)
masses_f = np.asfortranarray(masses)

# Physics parameters
G = 1.0
softening = 0.1
dt = 0.01
n_steps = 1000

print(f"N particles: {N}")
print(f"Steps: {n_steps}")

# Compute initial energy
ke0, pe0, E0 = fortran_nbody.nbody_fortran.compute_energy(
    positions_f, velocities_f, masses_f, G, softening
)
print(f"\nInitial energy: KE={ke0:.4f}, PE={pe0:.4f}, Total={E0:.4f}")

# Run simulation
print(f"\nRunning simulation...")
start = time.time()
pos_final, vel_final = fortran_nbody.nbody_fortran.simulate(
    positions_f, velocities_f, masses_f, n_steps, G, softening, dt
)
elapsed = time.time() - start

print(f"Runtime: {elapsed:.4f} seconds")
print(f"Time per step: {elapsed/n_steps*1000:.4f} ms")

# Compute final energy
ke_f, pe_f, E_f = fortran_nbody.nbody_fortran.compute_energy(
    pos_final, vel_final, masses_f, G, softening
)
print(f"\nFinal energy: KE={ke_f:.4f}, PE={pe_f:.4f}, Total={E_f:.4f}")
print(f"Energy drift: {abs(E_f - E0)/abs(E0)*100:.6f}%")

print("\n" + "=" * 50)
print("Fortran test complete!")