"""Comprehensive P3 diagnostic"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from p3_mesh_generator import generate_p3_structured_mesh
from p3_assembly import solve_poisson_p3

# Simple case: constant source on unit square
# ∇²ψ = 2κ where κ = constant
# For κ = -1, on [0,1]×[0,1] with ψ=0 on boundary:
# Analytic solution exists but complex, so just check basic properties

print("Diagnostic 1: Constant source")
print("="*70)
mesh = generate_p3_structured_mesh(4, 4, 0, 1, 0, 1)
nodes = np.array(mesh.nodes)

kappa = -1.0 * np.ones(len(nodes))  # Constant
psi = solve_poisson_p3(mesh, kappa)

print(f"Solution range: [{psi.min():.6f}, {psi.max():.6f}]")
print(f"Boundary values: min={psi[mesh.boundary].min():.2e}, max={psi[mesh.boundary].max():.2e}")
print(f"Interior max: {psi.max():.6f}")

# Check: for ∇²ψ = -2, max should be around 1/8 at center (from analytic)
if psi.max() > 0.2:
    print("❌ Solution too large - check source term sign/factor")
elif psi.max() < 0.05:
    print("❌ Solution too small - check source term sign/factor")
else:
    print("✅ Solution magnitude reasonable")

if np.max(np.abs(psi[mesh.boundary])) > 1e-8:
    print("❌ Boundary conditions NOT enforced!")
else:
    print("✅ Boundary conditions enforced")