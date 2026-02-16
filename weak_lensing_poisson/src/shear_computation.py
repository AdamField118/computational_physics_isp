"""
Shear Computation for Weak Gravitational Lensing

Computes shear field γ = (γ₁, γ₂) from lensing potential ψ using P2 finite elements.

Key Physics:
- Shear components: γ₁ = (ψ_xx - ψ_yy)/2, γ₂ = ψ_xy
- Combined shear: γ = ∇²ψ - trace-free part of Hessian
- Observable: γ distorts background galaxy shapes

Why P2 is Essential:
- P1: ∇ψ is piecewise constant → ∇²ψ = 0 everywhere ❌
- P2: ∇ψ is piecewise linear → ∇²ψ is piecewise constant ✅
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple, Optional, NamedTuple


class ShearField(NamedTuple):
    """Container for shear field results"""
    gamma1: jnp.ndarray      # (n_points,) - γ₁ component
    gamma2: jnp.ndarray      # (n_points,) - γ₂ component
    gamma_mag: jnp.ndarray   # (n_points,) - |γ| = sqrt(γ₁² + γ₂²)
    points: jnp.ndarray      # (n_points, 2) - evaluation points
    kappa: jnp.ndarray       # (n_points,) - convergence (for reference)


@jit
def compute_p2_shape_second_derivatives_reference(xi: float, eta: float) -> jnp.ndarray:
    """
    Compute second derivatives of P2 shape functions w.r.t. reference coordinates
    
    Returns Hessian matrix for each shape function:
        H_i = [[∂²N_i/∂ξ², ∂²N_i/∂ξ∂η],
               [∂²N_i/∂ξ∂η, ∂²N_i/∂η²]]
    
    For P2 shape functions N_i(ξ, η), these are CONSTANT (since N_i is quadratic)
    
    Args:
        xi, eta: Point in reference triangle (not actually used - derivatives are constant!)
        
    Returns:
        (6, 2, 2) array where [i, :, :] is Hessian of N_i
    """
    # Barycentric coordinates: λ₁ = 1-ξ-η, λ₂ = ξ, λ₃ = η
    # P2 shape functions are quadratic in ξ, η
    
    # Vertex nodes: N_i = λ_i(2λ_i - 1)
    # After differentiation, second derivatives are CONSTANT
    
    # N₀ = (1-ξ-η)(2(1-ξ-η) - 1) = (1-ξ-η)(1-2ξ-2η)
    # ∂N₀/∂ξ = -(1-2ξ-2η) + (1-ξ-η)(-2) = -1 + 2ξ + 2η - 2 + 2ξ + 2η = 4ξ + 4η - 3
    # Wait, let me be more careful...
    
    # Actually, for N₀ = λ₁(2λ₁ - 1) where λ₁ = 1-ξ-η:
    # ∂N₀/∂ξ = ∂λ₁/∂ξ * (4λ₁ - 1) = (-1) * (4(1-ξ-η) - 1) = -4 + 4ξ + 4η + 1 = 4ξ + 4η - 3
    # ∂²N₀/∂ξ² = 4
    # ∂²N₀/∂ξ∂η = 4
    # ∂²N₀/∂η² = 4
    
    # Let me compute systematically:
    # For vertex i: N_i = λ_i(2λ_i - 1)
    # ∂N_i/∂ξ = ∂λ_i/∂ξ * (4λ_i - 1)
    # ∂²N_i/∂ξ² = (∂λ_i/∂ξ)² * 4
    
    # Gradient of barycentrics:
    # ∇λ₁ = (-1, -1), ∇λ₂ = (1, 0), ∇λ₃ = (0, 1)
    
    # Second derivatives of barycentrics are ZERO (they're linear)
    # So second derivatives of N_i come only from (∂λ_i/∂ξ)(∂λ_i/∂η) terms
    
    # For N₀ = λ₁(2λ₁ - 1):
    # ∂²N₀/∂ξ² = 4(∂λ₁/∂ξ)² = 4(-1)² = 4
    # ∂²N₀/∂ξ∂η = 4(∂λ₁/∂ξ)(∂λ₁/∂η) = 4(-1)(-1) = 4
    # ∂²N₀/∂η² = 4(∂λ₁/∂η)² = 4(-1)² = 4
    
    H0 = jnp.array([[4.0, 4.0],
                    [4.0, 4.0]])
    
    # For N₁ = λ₂(2λ₂ - 1):
    # ∂²N₁/∂ξ² = 4(∂λ₂/∂ξ)² = 4(1)² = 4
    # ∂²N₁/∂ξ∂η = 4(∂λ₂/∂ξ)(∂λ₂/∂η) = 4(1)(0) = 0
    # ∂²N₁/∂η² = 4(∂λ₂/∂η)² = 4(0)² = 0
    
    H1 = jnp.array([[4.0, 0.0],
                    [0.0, 0.0]])
    
    # For N₂ = λ₃(2λ₃ - 1):
    # ∂²N₂/∂ξ² = 4(∂λ₃/∂ξ)² = 4(0)² = 0
    # ∂²N₂/∂ξ∂η = 4(∂λ₃/∂ξ)(∂λ₃/∂η) = 4(0)(1) = 0
    # ∂²N₂/∂η² = 4(∂λ₃/∂η)² = 4(1)² = 4
    
    H2 = jnp.array([[0.0, 0.0],
                    [0.0, 4.0]])
    
    # For edge midpoints: N_i = 4λ_j λ_k
    # ∂²(λ_j λ_k)/∂ξ² = 2(∂λ_j/∂ξ)(∂λ_k/∂ξ)
    # (The λ's themselves have zero second derivatives)
    
    # N₃ = 4λ₁λ₂:
    # ∂²N₃/∂ξ² = 4 * 2(∂λ₁/∂ξ)(∂λ₂/∂ξ) = 8(-1)(1) = -8
    # ∂²N₃/∂ξ∂η = 4 * [(∂λ₁/∂ξ)(∂λ₂/∂η) + (∂λ₁/∂η)(∂λ₂/∂ξ)]
    #             = 4 * [(-1)(0) + (-1)(1)] = -4
    # ∂²N₃/∂η² = 4 * 2(∂λ₁/∂η)(∂λ₂/∂η) = 8(-1)(0) = 0
    
    H3 = jnp.array([[-8.0, -4.0],
                    [-4.0,  0.0]])
    
    # N₄ = 4λ₂λ₃:
    # ∂²N₄/∂ξ² = 4 * 2(∂λ₂/∂ξ)(∂λ₃/∂ξ) = 8(1)(0) = 0
    # ∂²N₄/∂ξ∂η = 4 * [(∂λ₂/∂ξ)(∂λ₃/∂η) + (∂λ₂/∂η)(∂λ₃/∂ξ)]
    #             = 4 * [(1)(1) + (0)(0)] = 4
    # ∂²N₄/∂η² = 4 * 2(∂λ₂/∂η)(∂λ₃/∂η) = 8(0)(1) = 0
    
    H4 = jnp.array([[0.0, 4.0],
                    [4.0, 0.0]])
    
    # N₅ = 4λ₃λ₁:
    # ∂²N₅/∂ξ² = 4 * 2(∂λ₃/∂ξ)(∂λ₁/∂ξ) = 8(0)(-1) = 0
    # ∂²N₅/∂ξ∂η = 4 * [(∂λ₃/∂ξ)(∂λ₁/∂η) + (∂λ₃/∂η)(∂λ₁/∂ξ)]
    #             = 4 * [(0)(-1) + (1)(-1)] = -4
    # ∂²N₅/∂η² = 4 * 2(∂λ₃/∂η)(∂λ₁/∂η) = 8(1)(-1) = -8
    
    H5 = jnp.array([[0.0, -4.0],
                    [-4.0, -8.0]])
    
    return jnp.array([H0, H1, H2, H3, H4, H5])


@jit
def compute_p2_shape_second_derivatives_physical(xi: float, eta: float, 
                                                coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute second derivatives of P2 shape functions in physical coordinates
    
    Uses chain rule to transform Hessian from reference to physical coordinates.
    
    For transformation x = x(ξ,η), y = y(ξ,η), we have:
        ∂²N/∂x² = ...complex chain rule involving J and H_ref...
    
    Actually, it's easier to compute via automatic differentiation!
    But for educational purposes, here's the analytical approach:
    
    Args:
        xi, eta: Point in reference triangle
        coords: (6, 2) physical coordinates of nodes
        
    Returns:
        (6, 3) array where [i, :] = [∂²N_i/∂x², ∂²N_i/∂y², ∂²N_i/∂x∂y]
    """
    # This is complex! Let's use a different approach in practice.
    # For now, compute at element centroid where it's simpler.
    
    # Get reference Hessians (constant for P2!)
    H_ref = compute_p2_shape_second_derivatives_reference(xi, eta)  # (6, 2, 2)
    
    # Get Jacobian and its inverse
    from .fem_solver import compute_jacobian
    J = compute_jacobian(xi, eta, coords)  # (2, 2)
    J_inv = jnp.linalg.inv(J)
    
    # Transform Hessians: H_phys = J_inv^T @ H_ref @ J_inv
    # But this isn't quite right for second derivatives...
    
    # Actually, the full formula is:
    # ∂²f/∂x² = (J⁻¹)₁₁² ∂²f/∂ξ² + (J⁻¹)₁₂² ∂²f/∂η² + 2(J⁻¹)₁₁(J⁻¹)₁₂ ∂²f/∂ξ∂η
    #           + (∂J⁻¹/∂ξ)₁ ∂f/∂ξ + (∂J⁻¹/∂η)₁ ∂f/∂η
    
    # This is getting messy. For P2 on affine triangles, the Jacobian is CONSTANT,
    # so the derivatives of J_inv are ZERO, simplifying things significantly!
    
    # For affine triangle (linear coordinate transformation):
    J11, J12 = J_inv[0, 0], J_inv[0, 1]
    J21, J22 = J_inv[1, 0], J_inv[1, 1]
    
    # Second derivatives in physical coords (assuming affine transformation)
    second_derivs = jnp.zeros((6, 3))  # [∂²N/∂x², ∂²N/∂y², ∂²N/∂x∂y]
    
    for i in range(6):
        H = H_ref[i]  # (2, 2) Hessian in reference coords
        
        # ∂²N/∂x²
        Nxx = J11**2 * H[0, 0] + J12**2 * H[1, 1] + 2 * J11 * J12 * H[0, 1]
        
        # ∂²N/∂y²
        Nyy = J21**2 * H[0, 0] + J22**2 * H[1, 1] + 2 * J21 * J22 * H[0, 1]
        
        # ∂²N/∂x∂y
        Nxy = J11 * J21 * H[0, 0] + J12 * J22 * H[1, 1] + (J11 * J22 + J12 * J21) * H[0, 1]
        
        second_derivs = second_derivs.at[i].set(jnp.array([Nxx, Nyy, Nxy]))
    
    return second_derivs


@jit
def compute_shear_at_point(xi: float, eta: float, 
                          coords: jnp.ndarray, 
                          psi_vals: jnp.ndarray) -> Tuple[float, float]:
    """
    Compute shear components (γ₁, γ₂) at a point within a P2 element
    
    Args:
        xi, eta: Point in reference triangle
        coords: (6, 2) physical coordinates of element nodes
        psi_vals: (6,) values of ψ at element nodes
        
    Returns:
        (γ₁, γ₂) tuple
    """
    # Get second derivatives of shape functions
    second_derivs = compute_p2_shape_second_derivatives_physical(xi, eta, coords)  # (6, 3)
    
    # Interpolate ψ second derivatives: ∂²ψ/∂x² = Σ ψ_i * ∂²N_i/∂x²
    psi_xx = jnp.dot(second_derivs[:, 0], psi_vals)
    psi_yy = jnp.dot(second_derivs[:, 1], psi_vals)
    psi_xy = jnp.dot(second_derivs[:, 2], psi_vals)
    
    # Shear components
    gamma1 = 0.5 * (psi_xx - psi_yy)
    gamma2 = psi_xy
    
    return gamma1, gamma2


def compute_shear_p2(mesh, psi: jnp.ndarray, 
                    eval_points: Optional[str] = 'centroids') -> ShearField:
    """
    Compute shear field from lensing potential for P2 elements
    
    Args:
        mesh: P2 Mesh object (6 nodes per element)
        psi: (n_nodes,) lensing potential at all nodes
        eval_points: Where to evaluate shear
            - 'centroids': Element centroids (default)
            - 'nodes': Mesh nodes (averaged from elements)
            - 'gauss': Gauss quadrature points
            
    Returns:
        ShearField with γ₁, γ₂, |γ|, and evaluation points
    """
    if mesh.elements.shape[1] != 6:
        raise ValueError("compute_shear_p2 requires P2 mesh (6 nodes per element)")
    
    if eval_points == 'centroids':
        return _compute_shear_at_centroids(mesh, psi)
    elif eval_points == 'nodes':
        return _compute_shear_at_nodes(mesh, psi)
    elif eval_points == 'gauss':
        return _compute_shear_at_gauss_points(mesh, psi)
    else:
        raise ValueError(f"Unknown eval_points: {eval_points}")


def _compute_shear_at_centroids(mesh, psi: jnp.ndarray) -> ShearField:
    """Evaluate shear at element centroids"""
    n_elem = mesh.n_elements
    
    gamma1_vals = jnp.zeros(n_elem)
    gamma2_vals = jnp.zeros(n_elem)
    points = jnp.zeros((n_elem, 2))
    kappa_vals = jnp.zeros(n_elem)
    
    # Centroid in reference triangle
    xi_c, eta_c = 1.0/3.0, 1.0/3.0
    
    def compute_elem_shear(carry, elem_idx):
        gamma1_arr, gamma2_arr, pts, kappa_arr = carry
        
        # Get element data
        nodes_idx = mesh.elements[elem_idx]  # (6,)
        coords = mesh.nodes[nodes_idx]  # (6, 2)
        psi_elem = psi[nodes_idx]  # (6,)
        
        # Compute shear at centroid
        g1, g2 = compute_shear_at_point(xi_c, eta_c, coords, psi_elem)
        
        # Physical coordinates of centroid
        from .fem_solver import compute_p2_shape_functions
        N = compute_p2_shape_functions(xi_c, eta_c)
        x_c = jnp.dot(N, coords[:, 0])
        y_c = jnp.dot(N, coords[:, 1])
        
        # Store results
        gamma1_arr = gamma1_arr.at[elem_idx].set(g1)
        gamma2_arr = gamma2_arr.at[elem_idx].set(g2)
        pts = pts.at[elem_idx].set(jnp.array([x_c, y_c]))
        
        # Also store convergence (for reference)
        # Convergence at element nodes (need to interpolate if needed)
        # For now, just average
        kappa_elem = jnp.mean(psi_elem)  # Placeholder
        kappa_arr = kappa_arr.at[elem_idx].set(kappa_elem)
        
        return (gamma1_arr, gamma2_arr, pts, kappa_arr), None
    
    (gamma1_vals, gamma2_vals, points, kappa_vals), _ = jax.lax.scan(
        compute_elem_shear,
        (gamma1_vals, gamma2_vals, points, kappa_vals),
        jnp.arange(n_elem)
    )
    
    gamma_mag = jnp.sqrt(gamma1_vals**2 + gamma2_vals**2)
    
    return ShearField(
        gamma1=gamma1_vals,
        gamma2=gamma2_vals,
        gamma_mag=gamma_mag,
        points=points,
        kappa=kappa_vals
    )


def _compute_shear_at_nodes(mesh, psi: jnp.ndarray) -> ShearField:
    """
    Evaluate shear at mesh nodes by averaging element contributions
    
    Each node is touched by multiple elements. Average shear from all.
    """
    n_nodes = mesh.n_nodes
    
    gamma1_sum = jnp.zeros(n_nodes)
    gamma2_sum = jnp.zeros(n_nodes)
    count = jnp.zeros(n_nodes)
    
    # For each element, compute shear at all 6 nodes
    xi_c, eta_c = 1.0/3.0, 1.0/3.0  # Use centroid for element evaluation
    
    def accumulate_node_shear(carry, elem_idx):
        g1_sum, g2_sum, cnt = carry
        
        nodes_idx = mesh.elements[elem_idx]
        coords = mesh.nodes[nodes_idx]
        psi_elem = psi[nodes_idx]
        
        # Compute shear at element centroid
        g1, g2 = compute_shear_at_point(xi_c, eta_c, coords, psi_elem)
        
        # Add to all nodes of this element
        for i in range(6):
            node_i = nodes_idx[i]
            g1_sum = g1_sum.at[node_i].add(g1)
            g2_sum = g2_sum.at[node_i].add(g2)
            cnt = cnt.at[node_i].add(1.0)
        
        return (g1_sum, g2_sum, cnt), None
    
    (gamma1_sum, gamma2_sum, count), _ = jax.lax.scan(
        accumulate_node_shear,
        (gamma1_sum, gamma2_sum, count),
        jnp.arange(mesh.n_elements)
    )
    
    # Average
    gamma1_vals = gamma1_sum / jnp.maximum(count, 1.0)
    gamma2_vals = gamma2_sum / jnp.maximum(count, 1.0)
    gamma_mag = jnp.sqrt(gamma1_vals**2 + gamma2_vals**2)
    
    return ShearField(
        gamma1=gamma1_vals,
        gamma2=gamma2_vals,
        gamma_mag=gamma_mag,
        points=mesh.nodes,
        kappa=jnp.zeros(n_nodes)  # Placeholder
    )


def _compute_shear_at_gauss_points(mesh, psi: jnp.ndarray) -> ShearField:
    """Evaluate shear at Gauss quadrature points within each element"""
    from .fem_solver import get_gauss_points_triangle
    
    gauss_points, gauss_weights = get_gauss_points_triangle(order=2)
    n_gauss = len(gauss_weights)
    n_eval = mesh.n_elements * n_gauss
    
    gamma1_vals = jnp.zeros(n_eval)
    gamma2_vals = jnp.zeros(n_eval)
    points = jnp.zeros((n_eval, 2))
    
    idx = 0
    for elem_idx in range(mesh.n_elements):
        nodes_idx = mesh.elements[elem_idx]
        coords = mesh.nodes[nodes_idx]
        psi_elem = psi[nodes_idx]
        
        for gp in gauss_points:
            xi, eta = gp[0], gp[1]
            
            g1, g2 = compute_shear_at_point(xi, eta, coords, psi_elem)
            
            # Physical coordinates
            from .fem_solver import compute_p2_shape_functions
            N = compute_p2_shape_functions(xi, eta)
            x_pt = jnp.dot(N, coords[:, 0])
            y_pt = jnp.dot(N, coords[:, 1])
            
            gamma1_vals = gamma1_vals.at[idx].set(g1)
            gamma2_vals = gamma2_vals.at[idx].set(g2)
            points = points.at[idx].set(jnp.array([x_pt, y_pt]))
            
            idx += 1
    
    gamma_mag = jnp.sqrt(gamma1_vals**2 + gamma2_vals**2)
    
    return ShearField(
        gamma1=gamma1_vals,
        gamma2=gamma2_vals,
        gamma_mag=gamma_mag,
        points=points,
        kappa=jnp.zeros(n_eval)
    )


# ============================================================================
# Validation Functions
# ============================================================================

def test_shear_on_manufactured_solution():
    """
    Test shear computation on sinusoidal manufactured solution
    
    For ψ = sin(πx)sin(πy):
    - ψ_xx = -π²sin(πx)sin(πy)
    - ψ_yy = -π²sin(πx)sin(πy)
    - ψ_xy = π²cos(πx)cos(πy)
    
    So:
    - γ₁ = (ψ_xx - ψ_yy)/2 = 0
    - γ₂ = ψ_xy = π²cos(πx)cos(πy)
    """
    print("=" * 70)
    print("SHEAR VALIDATION: Manufactured Solution")
    print("=" * 70)
    
    from .fem_solver import SinusoidalLens, solve_lensing_poisson
    from mesh_generator import generate_p2_structured_mesh
    
    lens = SinusoidalLens(k=1)
    
    # Create mesh
    nx = 20
    mesh = generate_p2_structured_mesh(nx, nx, xmin=0, xmax=1, ymin=0, ymax=1)
    print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # Solve for ψ
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    solution = solve_lensing_poisson(mesh, kappa, verbose=False)
    
    print(f"FEM solution: max|ψ| = {jnp.max(jnp.abs(solution.psi)):.6f}")
    
    # Compute shear
    shear = compute_shear_p2(mesh, solution.psi, eval_points='centroids')
    
    print(f"\nShear field computed at {len(shear.gamma1)} points")
    print(f"  γ₁ range: [{jnp.min(shear.gamma1):.6f}, {jnp.max(shear.gamma1):.6f}]")
    print(f"  γ₂ range: [{jnp.min(shear.gamma2):.6f}, {jnp.max(shear.gamma2):.6f}]")
    print(f"  |γ| max: {jnp.max(shear.gamma_mag):.6f}")
    
    # Compare with exact
    gamma1_exact = jnp.array([0.0 for x, y in shear.points])  # Should be ~0
    gamma2_exact = jnp.array([jnp.pi**2 * jnp.cos(jnp.pi*x) * jnp.cos(jnp.pi*y) 
                              for x, y in shear.points])
    
    err_gamma1 = jnp.max(jnp.abs(shear.gamma1 - gamma1_exact))
    err_gamma2 = jnp.max(jnp.abs(shear.gamma2 - gamma2_exact))
    
    print(f"\nComparison with exact shear:")
    print(f"  Max error in γ₁: {err_gamma1:.6e} (should be ~0)")
    print(f"  Max error in γ₂: {err_gamma2:.6e}")
    
    if err_gamma1 < 0.1 and err_gamma2 < 1.0:
        print("\n✓ Shear computation validated!")
    else:
        print("\n⚠ Large errors detected - investigate!")
    
    return shear


if __name__ == "__main__":
    # Test second derivatives
    print("Testing P2 second derivatives...")
    H = compute_p2_shape_second_derivatives_reference(0.5, 0.5)
    print(f"Shape: {H.shape}")
    print(f"H[0] (vertex):\n{H[0]}")
    print(f"H[3] (edge midpoint):\n{H[3]}")
    
    # Run validation
    shear = test_shear_on_manufactured_solution()
