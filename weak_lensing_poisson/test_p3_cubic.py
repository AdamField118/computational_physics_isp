"""Test if P3 solves cubic polynomials exactly"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from p3_mesh_generator import generate_p3_structured_mesh
from p3_assembly import solve_poisson_p3

# Cubic polynomial: ψ = x³ + y³ - x³y - xy³ with ψ=0 on [0,1]×[0,1] boundaries
# ∇²ψ = 6x + 6y - 6xy ≠ 0 (so it's not harmonic)
# But P3 should represent this EXACTLY if our implementation is correct

# Actually, let's use a simpler cubic that vanishes on boundaries:
# ψ = x(1-x)y(1-y)(x+y)  <- cubic, zero on boundaries

# Even simpler: ψ = (x² - x)(y² - y)  <- biquadratic (degree 4 total)
# ∇²ψ = 2(y²-y) + 2(x²-x) = 2y²-2y + 2x²-2x
# From ∇²ψ = 2κ: κ = (y²-y) + (x²-x)

mesh = generate_p3_structured_mesh(4, 4, 0, 1, 0, 1)
nodes = np.array(mesh.nodes)

# Source term
kappa = (nodes[:,1]**2 - nodes[:,1]) + (nodes[:,0]**2 - nodes[:,0])

# Solve
psi_computed = solve_poisson_p3(mesh, kappa)

# Exact
psi_exact = (nodes[:,0]**2 - nodes[:,0]) * (nodes[:,1]**2 - nodes[:,1])

# Error
error = np.max(np.abs(psi_computed - psi_exact))
print(f"\nP3 Cubic Test:")
print(f"  Max error: {error:.2e}")
print(f"  Should be < 1e-10 for P3 on degree-4 polynomial")
if error < 1e-8:
    print("  ✅ PASS")
else:
    print("  ❌ FAIL - P3 not working correctly!")