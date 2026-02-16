"""
CORRECT P2 Shear Computation with Full Chain Rule

For P2 elements, coordinate transformation is QUADRATIC, not affine.
This requires correction terms in the Hessian transformation.

Mathematical derivation:
-----------------------
For ψ(x,y) where x=x(ξ,η), y=y(ξ,η) are quadratic:

ψ_xx = ψ_ξξ ξ_x² + 2ψ_ξη ξ_x η_x + ψ_ηη η_x² + ψ_ξ ξ_xx + ψ_η η_xx

The terms ξ_xx, η_xx are second derivatives of the inverse mapping.
These come from differentiating J^(-1) which depends on (ξ,η).
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, NamedTuple
import numpy as np


class ShearField(NamedTuple):
    """Container for shear field results"""
    gamma1: jnp.ndarray
    gamma2: jnp.ndarray
    gamma_mag: jnp.ndarray
    points: jnp.ndarray


def compute_jacobian_derivatives(xi: float, eta: float, coords: jnp.ndarray) -> Tuple:
    """
    Compute ∂J/∂ξ and ∂J/∂η for P2 coordinate transformation
    
    Since J = [[∂x/∂ξ, ∂y/∂ξ], [∂x/∂η, ∂y/∂η]]
    and x, y are P2 interpolations, we have:
    
    ∂J/∂ξ = [[∂²x/∂ξ², ∂²y/∂ξ²], [∂²x/∂ξ∂η, ∂²y/∂ξ∂η]]
    
    These come from second derivatives of P2 shape functions!
    """
    from src.fem_solver import compute_p2_shape_functions
    
    # Use JAX to compute second derivatives of coordinate mapping
    def x_coord(xi_eta):
        xi, eta = xi_eta[0], xi_eta[1]
        N = compute_p2_shape_functions(xi, eta)
        return jnp.dot(N, coords[:, 0])
    
    def y_coord(xi_eta):
        xi, eta = xi_eta[0], xi_eta[1]
        N = compute_p2_shape_functions(xi, eta)
        return jnp.dot(N, coords[:, 1])
    
    # Get Hessians of x(ξ,η) and y(ξ,η)
    xi_eta_pt = jnp.array([xi, eta])
    
    H_x = jax.hessian(x_coord)(xi_eta_pt)  # [[x_ξξ, x_ξη], [x_ηξ, x_ηη]]
    H_y = jax.hessian(y_coord)(xi_eta_pt)  # [[y_ξξ, y_ξη], [y_ηξ, y_ηη]]
    
    # ∂J/∂ξ
    dJ_dxi = jnp.array([
        [H_x[0, 0], H_y[0, 0]],  # [x_ξξ, y_ξξ]
        [H_x[0, 1], H_y[0, 1]]   # [x_ξη, y_ξη]
    ])
    
    # ∂J/∂η
    dJ_deta = jnp.array([
        [H_x[1, 0], H_y[1, 0]],  # [x_ηξ, y_ηξ]
        [H_x[1, 1], H_y[1, 1]]   # [x_ηη, y_ηη]
    ])
    
    return dJ_dxi, dJ_deta


def compute_inverse_jacobian_derivatives(J: jnp.ndarray, 
                                        dJ_dxi: jnp.ndarray,
                                        dJ_deta: jnp.ndarray) -> Tuple:
    """
    Compute ∂(J^(-1))/∂ξ and ∂(J^(-1))/∂η
    
    Using the formula: ∂(J^(-1))/∂s = -J^(-1) (∂J/∂s) J^(-1)
    """
    J_inv = jnp.linalg.inv(J)
    
    dJinv_dxi = -jnp.dot(jnp.dot(J_inv, dJ_dxi), J_inv)
    dJinv_deta = -jnp.dot(jnp.dot(J_inv, dJ_deta), J_inv)
    
    return dJinv_dxi, dJinv_deta


def compute_second_derivative_corrections(psi_xi: float, psi_eta: float,
                                         J_inv: jnp.ndarray,
                                         dJinv_dxi: jnp.ndarray,
                                         dJinv_deta: jnp.ndarray) -> Tuple:
    """
    Compute correction terms for second derivatives
    
    These are the ξ_xx, η_xx, etc. terms that arise from differentiating
    the inverse Jacobian.
    
    For ψ_xx, we need: ψ_ξ ξ_xx + ψ_η η_xx
    
    where ξ_xx = ∂²ξ/∂x² = ∂ξ_x/∂x
                         = (∂ξ_x/∂ξ) ξ_x + (∂ξ_x/∂η) η_x
    """
    # J_inv = [[ξ_x, ξ_y], [η_x, η_y]]
    xi_x = J_inv[0, 0]
    xi_y = J_inv[0, 1]
    eta_x = J_inv[1, 0]
    eta_y = J_inv[1, 1]
    
    # dJinv_dxi = [[∂ξ_x/∂ξ, ∂ξ_y/∂ξ], [∂η_x/∂ξ, ∂η_y/∂ξ]]
    dxi_x_dxi = dJinv_dxi[0, 0]
    dxi_y_dxi = dJinv_dxi[0, 1]
    deta_x_dxi = dJinv_dxi[1, 0]
    deta_y_dxi = dJinv_dxi[1, 1]
    
    # dJinv_deta = [[∂ξ_x/∂η, ∂ξ_y/∂η], [∂η_x/∂η, ∂η_y/∂η]]
    dxi_x_deta = dJinv_deta[0, 0]
    dxi_y_deta = dJinv_deta[0, 1]
    deta_x_deta = dJinv_deta[1, 0]
    deta_y_deta = dJinv_deta[1, 1]
    
    # Now compute ξ_xx = ∂ξ_x/∂x = (∂ξ_x/∂ξ) ξ_x + (∂ξ_x/∂η) η_x
    xi_xx = dxi_x_dxi * xi_x + dxi_x_deta * eta_x
    
    # η_xx = ∂η_x/∂x = (∂η_x/∂ξ) ξ_x + (∂η_x/∂η) η_x  
    eta_xx = deta_x_dxi * xi_x + deta_x_deta * eta_x
    
    # ξ_yy = ∂ξ_y/∂y = (∂ξ_y/∂ξ) ξ_y + (∂ξ_y/∂η) η_y
    xi_yy = dxi_y_dxi * xi_y + dxi_y_deta * eta_y
    
    # η_yy = ∂η_y/∂y = (∂η_y/∂ξ) ξ_y + (∂η_y/∂η) η_y
    eta_yy = deta_y_dxi * xi_y + deta_y_deta * eta_y
    
    # ξ_xy = ∂ξ_x/∂y = (∂ξ_x/∂ξ) ξ_y + (∂ξ_x/∂η) η_y
    xi_xy = dxi_x_dxi * xi_y + dxi_x_deta * eta_y
    
    # η_xy = ∂η_x/∂y = (∂η_x/∂ξ) ξ_y + (∂η_x/∂η) η_y
    eta_xy = deta_x_dxi * xi_y + deta_x_deta * eta_y
    
    # Correction terms for ψ_xx, ψ_yy, ψ_xy
    corr_xx = psi_xi * xi_xx + psi_eta * eta_xx
    corr_yy = psi_xi * xi_yy + psi_eta * eta_yy
    corr_xy = psi_xi * xi_xy + psi_eta * eta_xy
    
    return corr_xx, corr_yy, corr_xy


def compute_shear_at_point_correct(xi: float, eta: float,
                                   coords: jnp.ndarray,
                                   psi_vals: jnp.ndarray) -> Tuple[float, float]:
    """
    Compute shear with CORRECT P2 second derivative transformation
    
    Includes the correction terms that arise from non-affine mapping!
    """
    from src.fem_solver import compute_p2_shape_functions, compute_jacobian
    
    # Get ψ and its derivatives in reference coordinates
    def psi_ref(xi_eta):
        xi, eta = xi_eta[0], xi_eta[1]
        N = compute_p2_shape_functions(xi, eta)
        return jnp.dot(N, psi_vals)
    
    xi_eta_pt = jnp.array([xi, eta])
    
    # First derivatives
    grad_ref = jax.grad(psi_ref)(xi_eta_pt)  # [ψ_ξ, ψ_η]
    psi_xi = grad_ref[0]
    psi_eta = grad_ref[1]
    
    # Second derivatives (Hessian in reference coords)
    H_ref = jax.hessian(psi_ref)(xi_eta_pt)  # [[ψ_ξξ, ψ_ξη], [ψ_ηξ, ψ_ηη]]
    psi_xi_xi = H_ref[0, 0]
    psi_xi_eta = H_ref[0, 1]
    psi_eta_eta = H_ref[1, 1]
    
    # Jacobian and its inverse
    J = compute_jacobian(xi, eta, coords)
    J_inv = jnp.linalg.inv(J)
    
    xi_x = J_inv[0, 0]
    xi_y = J_inv[0, 1]
    eta_x = J_inv[1, 0]
    eta_y = J_inv[1, 1]
    
    # Compute ∂J/∂ξ and ∂J/∂η
    dJ_dxi, dJ_deta = compute_jacobian_derivatives(xi, eta, coords)
    
    # Compute ∂(J^(-1))/∂ξ and ∂(J^(-1))/∂η
    dJinv_dxi, dJinv_deta = compute_inverse_jacobian_derivatives(J, dJ_dxi, dJ_deta)
    
    # Compute correction terms
    corr_xx, corr_yy, corr_xy = compute_second_derivative_corrections(
        psi_xi, psi_eta, J_inv, dJinv_dxi, dJinv_deta
    )
    
    # Full second derivatives with correction terms!
    psi_xx = (psi_xi_xi * xi_x**2 + 
              2 * psi_xi_eta * xi_x * eta_x + 
              psi_eta_eta * eta_x**2 + 
              corr_xx)  # ← This is what we were missing!
    
    psi_yy = (psi_xi_xi * xi_y**2 + 
              2 * psi_xi_eta * xi_y * eta_y + 
              psi_eta_eta * eta_y**2 + 
              corr_yy)  # ← And this!
    
    psi_xy = (psi_xi_xi * xi_x * xi_y + 
              psi_xi_eta * (xi_x * eta_y + xi_y * eta_x) + 
              psi_eta_eta * eta_x * eta_y + 
              corr_xy)  # ← And this!
    
    # Shear components
    gamma1 = 0.5 * (psi_xx - psi_yy)
    gamma2 = psi_xy
    
    return gamma1, gamma2


def compute_shear_p2(mesh, psi: jnp.ndarray) -> ShearField:
    """
    Compute shear field using CORRECT P2 formulation
    """
    if mesh.elements.shape[1] != 6:
        raise ValueError("compute_shear_p2 requires P2 mesh (6 nodes per element)")
    
    n_elem = mesh.n_elements
    
    gamma1_vals = []
    gamma2_vals = []
    points_list = []
    
    print("Computing shear with correct P2 formulation...")
    
    # Centroid in reference triangle
    xi_c, eta_c = 1.0/3.0, 1.0/3.0
    
    for elem_idx in range(n_elem):
        if elem_idx % 500 == 0:
            print(f"  {elem_idx}/{n_elem}...", end='\r')
        
        # Get element data
        nodes_idx = mesh.elements[elem_idx]
        coords = np.array(mesh.nodes[nodes_idx])
        psi_elem = np.array(psi[nodes_idx])
        
        # Compute shear with full chain rule
        g1, g2 = compute_shear_at_point_correct(xi_c, eta_c, 
                                               jnp.array(coords), 
                                               jnp.array(psi_elem))
        
        # Centroid location
        from src.fem_solver import compute_p2_shape_functions
        N_c = compute_p2_shape_functions(xi_c, eta_c)
        x_c = np.dot(N_c, coords[:, 0])
        y_c = np.dot(N_c, coords[:, 1])
        
        gamma1_vals.append(float(g1))
        gamma2_vals.append(float(g2))
        points_list.append([x_c, y_c])
    
    print(f"  {n_elem}/{n_elem} ✓     ")
    
    gamma1 = jnp.array(gamma1_vals)
    gamma2 = jnp.array(gamma2_vals)
    points = jnp.array(points_list)
    gamma_mag = jnp.sqrt(gamma1**2 + gamma2**2)
    
    return ShearField(gamma1, gamma2, gamma_mag, points)


def test_correct_shear():
    """Test on sinusoidal manufactured solution"""
    print("=" * 70)
    print("TESTING CORRECT P2 SHEAR COMPUTATION")
    print("=" * 70)
    
    from src.fem_solver import SinusoidalLens, solve_lensing_poisson
    from src.mesh_generator import generate_p2_structured_mesh
    
    lens = SinusoidalLens(k=1)
    
    # Start with small mesh for debugging
    mesh = generate_p2_structured_mesh(10, 10, xmin=0, xmax=1, ymin=0, ymax=1)
    
    print(f"\nMesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # Solve
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    solution = solve_lensing_poisson(mesh, kappa, verbose=False)
    
    print(f"Solved: max|ψ| = {jnp.max(jnp.abs(solution.psi)):.6f}")
    
    # Compute shear with correct formulation
    shear = compute_shear_p2(mesh, solution.psi)
    
    print(f"\nShear field:")
    print(f"  γ₁ range: [{jnp.min(shear.gamma1):.6f}, {jnp.max(shear.gamma1):.6f}] (expect ~0)")
    print(f"  γ₂ range: [{jnp.min(shear.gamma2):.6f}, {jnp.max(shear.gamma2):.6f}]")
    print(f"  |γ| max: {jnp.max(shear.gamma_mag):.6f}")
    
    # Exact shear for sinusoidal solution
    g1_exact = np.array([0.0 for _ in shear.points])
    g2_exact = np.array([np.pi**2 * np.cos(np.pi*x) * np.cos(np.pi*y) 
                         for x, y in shear.points])
    
    err1 = np.sqrt(np.mean((shear.gamma1 - g1_exact)**2))
    err2 = np.sqrt(np.mean((shear.gamma2 - g2_exact)**2))
    
    print(f"\nErrors vs exact:")
    print(f"  RMS γ₁: {err1:.6e} (should be ~0)")
    print(f"  RMS γ₂: {err2:.6e}")
    
    # Debug: show first few values
    print(f"\nFirst 5 points:")
    print(f"  {'x':>6} {'y':>6} {'γ₁_num':>10} {'γ₁_ex':>10} {'γ₂_num':>10} {'γ₂_ex':>10}")
    print("  " + "-"*60)
    for i in range(min(5, len(shear.points))):
        x, y = shear.points[i]
        print(f"  {x:6.3f} {y:6.3f} {shear.gamma1[i]:10.4f} {g1_exact[i]:10.4f} "
              f"{shear.gamma2[i]:10.4f} {g2_exact[i]:10.4f}")
    
    # Pass/fail
    if err1 < 0.1 and err2 < 1.0:
        print("\n✓ TEST PASSED! Shear computation is now correct!")
        return True
    else:
        print(f"\n✗ TEST FAILED - errors still too large")
        print(f"   This might indicate a bug in the correction term computation")
        return False


if __name__ == "__main__":
    success = test_correct_shear()
    
    if success:
        print("\n" + "="*70)
        print("SUCCESS! Ready to replace src/shear_computation.py")
        print("="*70)