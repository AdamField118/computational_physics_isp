"""
Diagnostic: Check if correction terms are the problem
"""

import jax
import jax.numpy as jnp
import numpy as np


def diagnose_correction_terms():
    """
    Check if correction terms are being computed correctly
    """
    print("=" * 70)
    print("DIAGNOSTIC: Checking correction terms")
    print("=" * 70)
    
    from src.fem_solver import SinusoidalLens, solve_lensing_poisson, compute_p2_shape_functions, compute_jacobian
    from src.mesh_generator import generate_p2_structured_mesh
    
    # Small mesh for debugging
    mesh = generate_p2_structured_mesh(5, 5, xmin=0, xmax=1, ymin=0, ymax=1)
    
    lens = SinusoidalLens()
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    sol = solve_lensing_poisson(mesh, kappa, verbose=False)
    
    print(f"\nChecking first element...")
    
    # First element
    elem_idx = 0
    nodes_idx = mesh.elements[elem_idx]
    coords = mesh.nodes[nodes_idx]
    psi_elem = sol.psi[nodes_idx]
    
    print(f"Element {elem_idx}:")
    print(f"  Node coords:\n{coords}")
    
    # Evaluate at centroid
    xi_c, eta_c = 1.0/3.0, 1.0/3.0
    
    # Check if coordinate transformation is affine
    def x_coord(xi_eta):
        xi, eta = xi_eta[0], xi_eta[1]
        N = compute_p2_shape_functions(xi, eta)
        return jnp.dot(N, coords[:, 0])
    
    def y_coord(xi_eta):
        xi, eta = xi_eta[0], xi_eta[1]
        N = compute_p2_shape_functions(xi, eta)
        return jnp.dot(N, coords[:, 1])
    
    xi_eta_pt = jnp.array([xi_c, eta_c])
    
    # Compute Hessians of coordinate transformation
    H_x = jax.hessian(x_coord)(xi_eta_pt)
    H_y = jax.hessian(y_coord)(xi_eta_pt)
    
    print(f"\nHessian of x(ξ,η):")
    print(H_x)
    print(f"Max |H_x|: {jnp.max(jnp.abs(H_x))}")
    
    print(f"\nHessian of y(ξ,η):")
    print(H_y)
    print(f"Max |H_y|: {jnp.max(jnp.abs(H_y))}")
    
    if jnp.max(jnp.abs(H_x)) < 1e-10 and jnp.max(jnp.abs(H_y)) < 1e-10:
        print("\n✓ Coordinate transformation IS affine (Hessians are zero)!")
        print("  → Correction terms SHOULD be zero")
        print("  → Problem is elsewhere (maybe in standard transformation)")
    else:
        print("\n⚠ Coordinate transformation is NOT affine")
        print("  → Correction terms are needed")
    
    # Now check the actual correction terms being computed
    from src.shear_computation_fixed import (
        compute_jacobian_derivatives,
        compute_inverse_jacobian_derivatives,
        compute_second_derivative_corrections
    )
    
    # Get ψ derivatives in reference coords
    def psi_ref(xi_eta):
        xi, eta = xi_eta[0], xi_eta[1]
        N = compute_p2_shape_functions(xi, eta)
        return jnp.dot(N, psi_elem)
    
    grad_ref = jax.grad(psi_ref)(xi_eta_pt)
    psi_xi = grad_ref[0]
    psi_eta = grad_ref[1]
    
    # Get Jacobian
    J = compute_jacobian(xi_c, eta_c, coords)
    J_inv = jnp.linalg.inv(J)
    
    # Compute correction term components
    dJ_dxi, dJ_deta = compute_jacobian_derivatives(xi_c, eta_c, coords)
    
    print(f"\n∂J/∂ξ:")
    print(dJ_dxi)
    print(f"Max |∂J/∂ξ|: {jnp.max(jnp.abs(dJ_dxi))}")
    
    dJinv_dxi, dJinv_deta = compute_inverse_jacobian_derivatives(J, dJ_dxi, dJ_deta)
    
    print(f"\n∂(J⁻¹)/∂ξ:")
    print(dJinv_dxi)
    print(f"Max |∂(J⁻¹)/∂ξ|: {jnp.max(jnp.abs(dJinv_dxi))}")
    
    corr_xx, corr_yy, corr_xy = compute_second_derivative_corrections(
        psi_xi, psi_eta, J_inv, dJinv_dxi, dJinv_deta
    )
    
    print(f"\nCorrection terms:")
    print(f"  corr_xx = {corr_xx}")
    print(f"  corr_yy = {corr_yy}")
    print(f"  corr_xy = {corr_xy}")
    
    # Now compute shear WITH and WITHOUT corrections
    H_ref = jax.hessian(psi_ref)(xi_eta_pt)
    psi_xi_xi = H_ref[0, 0]
    psi_xi_eta = H_ref[0, 1]
    psi_eta_eta = H_ref[1, 1]
    
    xi_x, xi_y = J_inv[0, 0], J_inv[0, 1]
    eta_x, eta_y = J_inv[1, 0], J_inv[1, 1]
    
    # WITHOUT corrections
    psi_xx_no_corr = psi_xi_xi * xi_x**2 + 2*psi_xi_eta * xi_x * eta_x + psi_eta_eta * eta_x**2
    psi_yy_no_corr = psi_xi_xi * xi_y**2 + 2*psi_xi_eta * xi_y * eta_y + psi_eta_eta * eta_y**2
    
    # WITH corrections
    psi_xx_with_corr = psi_xx_no_corr + corr_xx
    psi_yy_with_corr = psi_yy_no_corr + corr_yy
    
    gamma1_no_corr = 0.5 * (psi_xx_no_corr - psi_yy_no_corr)
    gamma1_with_corr = 0.5 * (psi_xx_with_corr - psi_yy_with_corr)
    
    print(f"\nShear γ₁:")
    print(f"  Without corrections: {gamma1_no_corr}")
    print(f"  With corrections:    {gamma1_with_corr}")
    print(f"  Expected:            0.0")
    
    print(f"\nDifference:")
    print(f"  |γ₁ - 0| without corr: {abs(gamma1_no_corr)}")
    print(f"  |γ₁ - 0| with corr:    {abs(gamma1_with_corr)}")
    
    # Check symmetry
    print(f"\nψ_xx vs ψ_yy (should be equal for this solution):")
    print(f"  ψ_xx (no corr):  {psi_xx_no_corr}")
    print(f"  ψ_yy (no corr):  {psi_yy_no_corr}")
    print(f"  Difference:      {psi_xx_no_corr - psi_yy_no_corr}")
    
    print(f"\n  ψ_xx (with corr): {psi_xx_with_corr}")
    print(f"  ψ_yy (with corr): {psi_yy_with_corr}")
    print(f"  Difference:       {psi_xx_with_corr - psi_yy_with_corr}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    diagnose_correction_terms()