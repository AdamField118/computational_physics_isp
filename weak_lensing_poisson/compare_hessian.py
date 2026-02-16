"""
Compare Hessian computation for element 0 (works) vs element 2 (fails)
"""

import jax
import jax.numpy as jnp
import numpy as np


def compare_hessians():
    """
    Detailed comparison of element 0 vs element 2
    """
    print("=" * 70)
    print("DETAILED HESSIAN COMPARISON: Element 0 vs Element 2")
    print("=" * 70)
    
    from src.fem_solver import (
        SinusoidalLens, solve_lensing_poisson, 
        compute_p2_shape_functions, compute_jacobian
    )
    from src.mesh_generator import generate_p2_structured_mesh
    
    mesh = generate_p2_structured_mesh(5, 5, xmin=0, xmax=1, ymin=0, ymax=1)
    
    lens = SinusoidalLens()
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    sol = solve_lensing_poisson(mesh, kappa, verbose=False)
    
    xi_c, eta_c = 1.0/3.0, 1.0/3.0
    
    for elem_idx in [0, 2]:
        print(f"\n{'='*70}")
        print(f"ELEMENT {elem_idx}:")
        print(f"{'='*70}")
        
        nodes_idx = mesh.elements[elem_idx]
        coords = mesh.nodes[nodes_idx]
        psi_elem = sol.psi[nodes_idx]
        
        # Centroid location
        N_c = compute_p2_shape_functions(xi_c, eta_c)
        x_c = np.dot(N_c, coords[:, 0])
        y_c = np.dot(N_c, coords[:, 1])
        
        print(f"\nCentroid: ({x_c:.6f}, {y_c:.6f})")
        
        # ψ values at nodes
        print(f"\nψ at element nodes:")
        for i in range(6):
            print(f"  Node {i}: ψ = {psi_elem[i]:.6f} at ({coords[i,0]:.3f}, {coords[i,1]:.3f})")
        
        # Exact ψ at centroid
        psi_exact = lens.psi(x_c, y_c)
        print(f"\nExact ψ at centroid: {psi_exact:.6f}")
        
        # Interpolated ψ at centroid
        psi_interp = np.dot(N_c, psi_elem)
        print(f"Interpolated ψ:       {psi_interp:.6f}")
        print(f"Interpolation error:  {abs(psi_interp - psi_exact):.6e}")
        
        # Compute Hessian in reference coords
        def psi_ref(xi_eta):
            xi, eta = xi_eta[0], xi_eta[1]
            N = compute_p2_shape_functions(xi, eta)
            return jnp.dot(N, psi_elem)
        
        xi_eta_pt = jnp.array([xi_c, eta_c])
        
        # First derivatives in reference coords
        grad_ref = jax.grad(psi_ref)(xi_eta_pt)
        print(f"\nGradient in reference coords:")
        print(f"  ψ_ξ = {grad_ref[0]:.6f}")
        print(f"  ψ_η = {grad_ref[1]:.6f}")
        
        # Hessian in reference coords
        H_ref = jax.hessian(psi_ref)(xi_eta_pt)
        print(f"\nHessian in reference coords:")
        print(f"  ψ_ξξ  = {H_ref[0,0]:.6f}")
        print(f"  ψ_ξη  = {H_ref[0,1]:.6f}")
        print(f"  ψ_ηη  = {H_ref[1,1]:.6f}")
        
        # Jacobian and its inverse
        J = compute_jacobian(xi_c, eta_c, coords)
        J_inv = jnp.linalg.inv(J)
        
        print(f"\nJacobian J:")
        print(f"  [[{J[0,0]:7.4f}, {J[0,1]:7.4f}]")
        print(f"   [{J[1,0]:7.4f}, {J[1,1]:7.4f}]]")
        
        print(f"\nInverse Jacobian J^(-1):")
        print(f"  [[{J_inv[0,0]:7.4f}, {J_inv[0,1]:7.4f}]")
        print(f"   [{J_inv[1,0]:7.4f}, {J_inv[1,1]:7.4f}]]")
        
        # Transform to physical coords
        xi_x = J_inv[0, 0]
        xi_y = J_inv[0, 1]
        eta_x = J_inv[1, 0]
        eta_y = J_inv[1, 1]
        
        psi_xi_xi = H_ref[0, 0]
        psi_xi_eta = H_ref[0, 1]
        psi_eta_eta = H_ref[1, 1]
        
        # Standard transformation (WITHOUT corrections - should be fine for affine)
        psi_xx = (psi_xi_xi * xi_x**2 + 
                  2 * psi_xi_eta * xi_x * eta_x + 
                  psi_eta_eta * eta_x**2)
        
        psi_yy = (psi_xi_xi * xi_y**2 + 
                  2 * psi_xi_eta * xi_y * eta_y + 
                  psi_eta_eta * eta_y**2)
        
        psi_xy = (psi_xi_xi * xi_x * xi_y + 
                  psi_xi_eta * (xi_x * eta_y + xi_y * eta_x) + 
                  psi_eta_eta * eta_x * eta_y)
        
        print(f"\nTransformed second derivatives:")
        print(f"  ψ_xx = {psi_xx:.6f}")
        print(f"  ψ_yy = {psi_yy:.6f}")
        print(f"  ψ_xy = {psi_xy:.6f}")
        
        # Exact second derivatives at centroid
        psi_xx_exact = -np.pi**2 * lens.psi(x_c, y_c)
        psi_yy_exact = -np.pi**2 * lens.psi(x_c, y_c)
        psi_xy_exact = np.pi**2 * np.cos(np.pi * x_c) * np.cos(np.pi * y_c)
        
        print(f"\nExact second derivatives:")
        print(f"  ψ_xx = {psi_xx_exact:.6f}")
        print(f"  ψ_yy = {psi_yy_exact:.6f}")
        print(f"  ψ_xy = {psi_xy_exact:.6f}")
        
        print(f"\nErrors in second derivatives:")
        print(f"  Δψ_xx = {psi_xx - psi_xx_exact:.6f}")
        print(f"  Δψ_yy = {psi_yy - psi_yy_exact:.6f}")
        print(f"  Δψ_xy = {psi_xy - psi_xy_exact:.6f}")
        
        # Shear
        gamma1 = 0.5 * (psi_xx - psi_yy)
        gamma2 = psi_xy
        
        gamma1_exact = 0.0  # By symmetry
        gamma2_exact = psi_xy_exact
        
        print(f"\nShear components:")
        print(f"  γ₁ = {gamma1:.6f}  (exact: {gamma1_exact:.6f}, error: {abs(gamma1):.6f})")
        print(f"  γ₂ = {gamma2:.6f}  (exact: {gamma2_exact:.6f}, error: {abs(gamma2 - gamma2_exact):.6f})")
        
        if abs(gamma1) > 0.1:
            print(f"\n  ✗ LARGE γ₁ ERROR!")
            print(f"    ψ_xx - ψ_yy = {psi_xx - psi_yy:.6f}")
            print(f"    Should be: {psi_xx_exact - psi_yy_exact:.6f}")
        else:
            print(f"\n  ✓ γ₁ is correct")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    compare_hessians()