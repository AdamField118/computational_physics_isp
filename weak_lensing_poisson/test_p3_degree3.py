"""Test if P3 solves degree-3 polynomials exactly"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from p3_mesh_generator import generate_p3_structured_mesh
from p3_assembly import solve_poisson_p3

def psi_exact(x, y):
    """Degree-3 polynomial that vanishes on [0,1]×[0,1] boundaries"""
    return (x - x**2) * (y - y**2) * (x + y)

def kappa_exact(x, y):
    """Source from ∇²ψ = 2κ"""
    # ∇²ψ computed analytically
    laplacian = (y - y**2) * (2 - 6*x - 2*y) + (x - x**2) * (2 - 2*x - 6*y)
    return laplacian / 2.0

print("\n" + "="*70)
print("P3 DEGREE-3 POLYNOMIAL EXACTNESS TEST")
print("="*70)
print("Testing: ψ = (x-x²)(y-y²)(x+y)  [degree 3]")
print("P3 elements should solve this EXACTLY (error < 1e-10)")

mesh = generate_p3_structured_mesh(4, 4, 0, 1, 0, 1)
nodes = np.array(mesh.nodes)

kappa = kappa_exact(nodes[:,0], nodes[:,1])
psi_computed = solve_poisson_p3(mesh, kappa)
psi_exact_vals = psi_exact(nodes[:,0], nodes[:,1])

error_max = np.max(np.abs(psi_computed - psi_exact_vals))
error_l2 = np.sqrt(np.mean((psi_computed - psi_exact_vals)**2))

print(f"\nResults:")
print(f"  Max error: {error_max:.2e}")
print(f"  L² error:  {error_l2:.2e}")

if error_max < 1e-8:
    print("  ✅ PASS - P3 correctly solves degree-3 polynomials!")
else:
    print(f"  ❌ FAIL - P3 implementation has a bug")
    interior_mask = (nodes[:,0] > 0.01) & (nodes[:,0] < 0.99) & (nodes[:,1] > 0.01) & (nodes[:,1] < 0.99)
    interior_indices = np.where(interior_mask)[0][:5]
    print(f"\n  Sample interior node errors:")
    for i in interior_indices:
        print(f"    Node {i} ({nodes[i,0]:.3f},{nodes[i,1]:.3f}): err={psi_computed[i]-psi_exact_vals[i]:.6e}")

print("="*70)