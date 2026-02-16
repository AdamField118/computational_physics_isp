"""
Check multiple elements to find pattern in failures
"""

import jax
import jax.numpy as jnp
import numpy as np


def diagnose_multiple_elements():
    """
    Check first 10 elements to see which fail
    """
    print("=" * 70)
    print("DIAGNOSTIC: Checking multiple elements")
    print("=" * 70)
    
    from src.fem_solver import SinusoidalLens, solve_lensing_poisson, compute_p2_shape_functions
    from src.mesh_generator import generate_p2_structured_mesh
    from src.shear_computation_fixed import compute_shear_at_point_correct
    
    # Small mesh
    mesh = generate_p2_structured_mesh(5, 5, xmin=0, xmax=1, ymin=0, ymax=1)
    
    lens = SinusoidalLens()
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    sol = solve_lensing_poisson(mesh, kappa, verbose=False)
    
    print(f"\nChecking first 10 elements...")
    print(f"{'Elem':>4} {'Centroid':>20} {'γ₁':>10} {'Expected':>10} {'Error':>10}")
    print("-" * 70)
    
    xi_c, eta_c = 1.0/3.0, 1.0/3.0
    
    for elem_idx in range(min(10, mesh.n_elements)):
        nodes_idx = mesh.elements[elem_idx]
        coords = np.array(mesh.nodes[nodes_idx])
        psi_elem = np.array(sol.psi[nodes_idx])
        
        # Compute shear
        g1, g2 = compute_shear_at_point_correct(xi_c, eta_c, 
                                               jnp.array(coords), 
                                               jnp.array(psi_elem))
        
        # Centroid location
        N_c = compute_p2_shape_functions(xi_c, eta_c)
        x_c = np.dot(N_c, coords[:, 0])
        y_c = np.dot(N_c, coords[:, 1])
        
        # Expected γ₁ = 0
        error = abs(float(g1) - 0.0)
        
        status = "✓" if error < 0.01 else "✗"
        
        print(f"{elem_idx:4d} ({x_c:6.3f}, {y_c:6.3f})  {float(g1):10.4f}     0.0000  {error:10.4f} {status}")
    
    print("=" * 70)
    
    # Now let's check if there's a pattern with triangle orientation
    print("\nAnalyzing triangle orientations...")
    print("\nIn structured mesh, each square has 2 triangles:")
    print("  - Lower triangle (vertices at corners 0,1,2)")
    print("  - Upper triangle (vertices at corners 1,3,2)")
    print("\nLet's check if errors correlate with orientation...")
    
    # Elements 0, 2, 4, 6, 8 are lower triangles (even indices)
    # Elements 1, 3, 5, 7, 9 are upper triangles (odd indices)
    
    errors_lower = []
    errors_upper = []
    
    for elem_idx in range(min(20, mesh.n_elements)):
        nodes_idx = mesh.elements[elem_idx]
        coords = np.array(mesh.nodes[nodes_idx])
        psi_elem = np.array(sol.psi[nodes_idx])
        
        g1, g2 = compute_shear_at_point_correct(xi_c, eta_c,
                                               jnp.array(coords),
                                               jnp.array(psi_elem))
        
        error = abs(float(g1))
        
        if elem_idx % 2 == 0:
            errors_lower.append(error)
        else:
            errors_upper.append(error)
    
    print(f"\nLower triangles (even indices): mean error = {np.mean(errors_lower):.6f}")
    print(f"Upper triangles (odd indices):  mean error = {np.mean(errors_upper):.6f}")
    
    if np.mean(errors_lower) < 0.01 and np.mean(errors_upper) > 0.1:
        print("\n🔍 PATTERN FOUND: Upper triangles have errors, lower don't!")
        print("   → Bug is in how upper triangles are handled")
    elif np.mean(errors_upper) < 0.01 and np.mean(errors_lower) > 0.1:
        print("\n🔍 PATTERN FOUND: Lower triangles have errors, upper don't!")
        print("   → Bug is in how lower triangles are handled")
    else:
        print("\n⚠ No clear pattern with triangle orientation")
        print("   → Bug is more subtle")
    
    print("=" * 70)


if __name__ == "__main__":
    diagnose_multiple_elements()