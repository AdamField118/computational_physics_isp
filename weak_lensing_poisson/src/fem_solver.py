"""
JAX-based Finite Element Method solver for weak gravitational lensing
Solves:  nabla^2 psi = 2 kappa on 2D domains
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Tuple, Optional, NamedTuple
import numpy as np


class Mesh(NamedTuple):
    """Triangular mesh data structure"""
    nodes: jnp.ndarray      # (n_nodes, 2) - node coordinates
    elements: jnp.ndarray   # (n_elements, 3) - element connectivity (0-indexed)
    boundary: jnp.ndarray   # (n_boundary,) - boundary node indices
    
    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]
    
    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]


class FEMSolution(NamedTuple):
    """FEM solution container"""
    psi: jnp.ndarray           # (n_nodes,) - lensing potential
    alpha: jnp.ndarray         # (n_nodes, 2) - deflection angle
    convergence: jnp.ndarray   # (n_nodes,) - convergence field
    iterations: int            # CG iterations
    residual: float            # Final CG residual


@jit
def compute_element_area(coords: jnp.ndarray) -> float:
    """
    Compute area of a triangle given vertex coordinates
    
    Args:
        coords: (3, 2) array of vertex coordinates
        
    Returns:
        Triangle area
    """
    x1, x2, x3 = coords[:, 0]
    y1, y2, y3 = coords[:, 1]
    
    # Area = 0.5 * |det(B_K)|
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    return 0.5 * jnp.abs(det)


@jit
def compute_shape_gradients(coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute gradients of P1 basis functions on physical element
    
    Args:
        coords: (3, 2) array of vertex coordinates
        
    Returns:
        (3, 2) array where grad_N[i] =  nabla N_i = [ frac{ partial N_i}{ partial x},  frac{ partial N_i}{ partial y}]
    """
    x1, x2, x3 = coords[:, 0]
    y1, y2, y3 = coords[:, 1]
    
    # Compute 2*area
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    two_area = jnp.abs(det)
    
    # Gradients of barycentric coordinates
    #  nabla lambda_1 = 1/(2A) * [y2-y3, x3-x2]
    #  nabla lambda__2 = 1/(2A) * [y3-y1, x1-x3]
    #  nabla lambda__3 = 1/(2A) * [y1-y2, x2-x1]
    grad_N = jnp.array([
        [y2 - y3, x3 - x2],
        [y3 - y1, x1 - x3],
        [y1 - y2, x2 - x1]
    ]) / two_area
    
    return grad_N


@jit
def compute_element_stiffness(coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute element stiffness matrix K^e
    
    K^e[i,j] =  int_T  nabla N_i  cdot ∇N_j dA = Area(T) *  nabla N_i  cdot  nabla N_j
    
    Args:
        coords: (3, 2) array of vertex coordinates
        
    Returns:
        (3, 3) element stiffness matrix
    """
    area = compute_element_area(coords)
    grad_N = compute_shape_gradients(coords)
    
    # K^e[i,j] = area * grad_N[i]  cdot grad_N[j]
    K_elem = area * jnp.dot(grad_N, grad_N.T)
    
    return K_elem


@jit
def compute_element_load(coords: jnp.ndarray, kappa_vals: jnp.ndarray) -> jnp.ndarray:
    """
    Compute element load vector F^e for piecewise linear source
    
    F^e[i] =  int_T 2 kappa N_i dA
    
    For linear  kappa interpolation:  kappa =  Sigma  kappa_j N_j
    Then: F^e[i] = 2 *  Sigma_j  kappa_j *  int_T N_i N_j dA
    
    Using formula:  int_T N_i N_j dA = Area/12 * (1 +  delta_ij)
    
    Args:
        coords: (3, 2) array of vertex coordinates
        kappa_vals: (3,) convergence values at vertices
        
    Returns:
        (3,) element load vector
    """
    area = compute_element_area(coords)
    
    # Mass matrix entries: M[i,j] = area/12 if i≠j, area/6 if i=j
    # Diagonal: area/6, Off-diagonal: area/12
    M_elem = jnp.where(
        jnp.eye(3, dtype=bool),
        area / 6.0,
        area / 12.0
    )
    
    # F^e = -2 * M * κ
    F_elem = -2.0 * jnp.dot(M_elem, kappa_vals)
    
    return F_elem


def assemble_system(mesh, kappa: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Assemble global stiffness matrix K and load vector F
    
    Automatically detects element type (P1 or P2) from mesh.elements.shape[1]
    
    Args:
        mesh: Mesh object
        kappa: (n_nodes,) convergence field at nodes
        
    Returns:
        K: (n_nodes, n_nodes) stiffness matrix
        F: (n_nodes,) load vector
    """
    n = mesh.n_nodes
    nodes_per_elem = mesh.elements.shape[1]
    
    # Detect element type and route to appropriate assembly
    if nodes_per_elem == 3:
        print("  Element type: P1 (linear)")
        return assemble_system_p1(mesh, kappa)
    elif nodes_per_elem == 6:
        print("  Element type: P2 (quadratic)")
        return assemble_system_p2(mesh, kappa)
    else:
        raise ValueError(f"Unknown element type with {nodes_per_elem} nodes per element")


def assemble_system_p1(mesh, kappa: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Assemble system for P1 (linear) elements
    
    THIS IS YOUR EXISTING CODE - don't change it!
    Just rename your current assemble_system to this.
    """
    n = mesh.n_nodes
    
    # Initialize global arrays
    K_global = jnp.zeros((n, n))
    F_global = jnp.zeros(n)
    
    def assemble_element(carry, elem_idx):
        K_glob, F_glob = carry
        
        # Get element node indices
        nodes_idx = mesh.elements[elem_idx]  # (3,)
        
        # Get element coordinates
        coords = mesh.nodes[nodes_idx]  # (3, 2)
        
        # Get convergence at element nodes
        kappa_elem = kappa[nodes_idx]  # (3,)
        
        # Compute element matrices - THESE ARE YOUR P1 FUNCTIONS
        K_elem = compute_element_stiffness(coords)       # P1 version
        F_elem = compute_element_load(coords, kappa_elem)  # P1 version
        
        # Assembly: K[i,j] += K_elem[local_i, local_j]
        for i in range(3):  # P1 has 3 nodes
            glob_i = nodes_idx[i]
            for j in range(3):
                glob_j = nodes_idx[j]
                K_glob = K_glob.at[glob_i, glob_j].add(K_elem[i, j])
            
            F_glob = F_glob.at[glob_i].add(F_elem[i])
        
        return (K_glob, F_glob), None
    
    # Scan over elements
    (K_global, F_global), _ = jax.lax.scan(
        assemble_element,
        (K_global, F_global),
        jnp.arange(mesh.n_elements)
    )
    
    return K_global, F_global


def assemble_system_p2(mesh, kappa: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Assemble system for P2 (quadratic) elements
    
    This is NEW - uses P2 functions from p2_fem_additions.py
    """
    n = mesh.n_nodes
    
    # Initialize global arrays
    K_global = jnp.zeros((n, n))
    F_global = jnp.zeros(n)
    
    def assemble_element(carry, elem_idx):
        K_glob, F_glob = carry
        
        # Get element node indices
        nodes_idx = mesh.elements[elem_idx]  # (6,) for P2
        
        # Get element coordinates
        coords = mesh.nodes[nodes_idx]  # (6, 2)
        
        # Get convergence at element nodes
        kappa_elem = kappa[nodes_idx]  # (6,)
        
        # Compute element matrices - USE P2 FUNCTIONS
        K_elem = compute_element_stiffness_p2(coords, quad_order=2)  # P2 version!
        F_elem = compute_element_load_p2(coords, kappa_elem, quad_order=2)  # P2 version!
        
        # Assembly: K[i,j] += K_elem[local_i, local_j]
        for i in range(6):  # P2 has 6 nodes
            glob_i = nodes_idx[i]
            for j in range(6):
                glob_j = nodes_idx[j]
                K_glob = K_glob.at[glob_i, glob_j].add(K_elem[i, j])
            
            F_glob = F_glob.at[glob_i].add(F_elem[i])
        
        return (K_glob, F_glob), None
    
    # Scan over elements
    (K_global, F_global), _ = jax.lax.scan(
        assemble_element,
        (K_global, F_global),
        jnp.arange(mesh.n_elements)
    )
    
    return K_global, F_global


@jit
def apply_dirichlet_bc(K: jnp.ndarray, F: jnp.ndarray, 
                       boundary_nodes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply homogeneous Dirichlet boundary conditions:  psi = 0 on  partial Omega 

    Method: Set K[i,:] = 0, K[i,i] = 1, F[i] = 0 for boundary nodes
    
    Args:
        K: (n, n) stiffness matrix
        F: (n,) load vector
        boundary_nodes: indices of boundary nodes
        
    Returns:
        K_bc, F_bc: modified system
    """
    n = K.shape[0]
    
    # Create mask for boundary nodes
    is_boundary = jnp.zeros(n, dtype=bool)
    is_boundary = is_boundary.at[boundary_nodes].set(True)
    
    # Zero out rows and columns for boundary nodes
    K_bc = jnp.where(is_boundary[:, None] | is_boundary[None, :], 0.0, K)
    
    # Set diagonal to 1 for boundary nodes
    K_bc = jnp.where(jnp.diag(is_boundary), 1.0, K_bc)
    
    # Set RHS to 0 for boundary nodes
    F_bc = jnp.where(is_boundary, 0.0, F)
    
    return K_bc, F_bc


@partial(jit, static_argnames=['maxiter'])
def conjugate_gradient(K: jnp.ndarray, f: jnp.ndarray, 
                       x0: Optional[jnp.ndarray] = None,
                       tol: float = 1e-6, 
                       maxiter: int = 1000) -> Tuple[jnp.ndarray, int, float]:
    """
    Solve Kx = f using Conjugate Gradient method
    
    Optimized for symmetric positive definite matrices
    
    Args:
        K: (n, n) system matrix
        f: (n,) right-hand side
        x0: (n,) initial guess (default: zeros)
        tol: convergence tolerance
        maxiter: maximum iterations
        
    Returns:
        x: solution vector
        iterations: number of iterations
        residual: final residual norm
    """
    n = K.shape[0]
    
    if x0 is None:
        x = jnp.zeros(n)
    else:
        x = x0
    
    # Initial residual: r = f - Kx
    r = f - jnp.dot(K, x)
    p = r
    rsold = jnp.dot(r, r)
    
    def cg_step(carry, _):
        x, r, p, rsold = carry
        
        # Ap = K @ p
        Ap = jnp.dot(K, p)
        
        #  alpha = rsold / (p^T @ Ap)
        pAp = jnp.dot(p, Ap)
        alpha = rsold / (pAp + 1e-14)  # Regularize
        
        # x = x +  alpha*p
        x = x + alpha * p
        
        # r = r -  alpha*Ap
        r = r - alpha * Ap
        
        # rsnew = r^T @ r
        rsnew = jnp.dot(r, r)
        
        #  beta = rsnew / rsold
        beta = rsnew / (rsold + 1e-14)
        
        # p = r +  beta*p
        p = r + beta * p
        
        return (x, r, p, rsnew), rsnew
    
    # Run CG iterations
    (x, r, p, rsold), residuals = jax.lax.scan(
        cg_step,
        (x, r, p, rsold),
        None,
        length=maxiter
    )
    
    # Count iterations until convergence
    converged = residuals < tol**2
    iterations = jnp.argmax(converged).astype(int)
    iterations = jnp.where(jnp.any(converged), iterations, maxiter)
    
    final_residual = jnp.sqrt(rsold)
    
    return x, iterations, final_residual


@jit
def compute_deflection_p1(mesh: Mesh, psi: jnp.ndarray) -> jnp.ndarray:
    """
    Compute deflection angle  alpha =  nabla psi at mesh nodes
    
    For P1 elements, gradient is piecewise constant. We average
    contributions from all elements touching each node.
    
    Args:
        mesh: Mesh object
        psi: (n_nodes,) lensing potential
        
    Returns:
        alpha: (n_nodes, 2) deflection angle [ alpha_x,  alpha_y]
    """
    n = mesh.n_nodes
    
    # Accumulate gradient contributions
    alpha_sum = jnp.zeros((n, 2))
    alpha_count = jnp.zeros(n)
    
    def accumulate_gradient(carry, elem_idx):
        alpha_sum, alpha_count = carry
        
        # Get element data
        nodes_idx = mesh.elements[elem_idx]
        coords = mesh.nodes[nodes_idx]
        psi_elem = psi[nodes_idx]
        
        # Compute shape function gradients
        grad_N = compute_shape_gradients(coords)
        
        # Element gradient:  nabla psi =  Sigma  psi_i *  nabla N_i (constant on element)
        grad_psi = jnp.dot(grad_N.T, psi_elem)  # (2,) = (3,2).T @ (3,)
        
        # Add contribution to all three nodes of this element
        for i in range(3):
            node_i = nodes_idx[i]
            alpha_sum = alpha_sum.at[node_i].add(grad_psi)
            alpha_count = alpha_count.at[node_i].add(1.0)
        
        return (alpha_sum, alpha_count), None
    
    # Scan over all elements
    (alpha_sum, alpha_count), _ = jax.lax.scan(
        accumulate_gradient,
        (alpha_sum, alpha_count),
        jnp.arange(mesh.n_elements)
    )
    
    # Average contributions (avoid divide by zero)
    alpha = alpha_sum / jnp.maximum(alpha_count[:, None], 1.0)
    
    return alpha
    
@jit
def compute_deflection_p2(mesh, psi: jnp.ndarray) -> jnp.ndarray:
    """Compute deflection for P2 elements"""
    n = mesh.n_nodes
    alpha_sum = jnp.zeros((n, 2))
    alpha_count = jnp.zeros(n)
    
    def accumulate_gradient(carry, elem_idx):
        alpha_sum, alpha_count = carry
        nodes_idx = mesh.elements[elem_idx]  # (6,)
        coords = mesh.nodes[nodes_idx]
        psi_elem = psi[nodes_idx]
        
        # Compute gradient at centroid using P2 gradients
        xi, eta = 1.0/3.0, 1.0/3.0
        grad_N = compute_p2_shape_gradients_physical(xi, eta, coords)
        
        # ∇ψ = Σ ψ_i ∇N_i
        grad_psi = jnp.zeros(2)
        for i in range(6):
            grad_psi = grad_psi + psi_elem[i] * grad_N[i]
        
        # Add to all 6 nodes
        for i in range(6):
            alpha_sum = alpha_sum.at[nodes_idx[i]].add(grad_psi)
            alpha_count = alpha_count.at[nodes_idx[i]].add(1.0)
        
        return (alpha_sum, alpha_count), None
    
    (alpha_sum, alpha_count), _ = jax.lax.scan(
        accumulate_gradient,
        (alpha_sum, alpha_count),
        jnp.arange(mesh.n_elements)
    )
    
    return alpha_sum / jnp.maximum(alpha_count[:, None], 1.0)
    
@jit
def compute_deflection(mesh, psi: jnp.ndarray) -> jnp.ndarray:
    """Compute deflection - routes to P1 or P2"""
    nodes_per_elem = mesh.elements.shape[1]
    if nodes_per_elem == 3:
        return compute_deflection_p1(mesh, psi)
    elif nodes_per_elem == 6:
        return compute_deflection_p2(mesh, psi)
    else:
        raise ValueError(f"Unknown element type: {nodes_per_elem} nodes")


def solve_lensing_poisson(mesh: Mesh, 
                          kappa: jnp.ndarray,
                          tol: float = 1e-6,
                          maxiter: int = 1000,
                          verbose: bool = True) -> FEMSolution:
    """
    Complete FEM solver for lensing Poisson equation:  nabla^2  psi = 2 kappa
    
    Args:
        mesh: Mesh object
        kappa: (n_nodes,) convergence field at nodes
        tol: CG convergence tolerance
        maxiter: maximum CG iterations
        verbose: print solver info
        
    Returns:
        FEMSolution with potential, deflection, convergence, and solver stats
    """
    if verbose:
        print(f"Assembling FEM system...")
        print(f"  Nodes: {mesh.n_nodes}")
        print(f"  Elements: {mesh.n_elements}")
        print(f"  Boundary nodes: {len(mesh.boundary)}")
    
    # Assemble global system
    K, F = assemble_system(mesh, kappa)
    
    if verbose:
        print(f"  Matrix size: {K.shape[0]} × {K.shape[1]}")
        print(f"  Applying boundary conditions...")
    
    # Apply Dirichlet BC:  psi = 0 on boundary
    K_bc, F_bc = apply_dirichlet_bc(K, F, mesh.boundary)
    
    if verbose:
        print(f"  Solving with Conjugate Gradient...")
    
    # Solve K psi= F
    psi, iterations, residual = conjugate_gradient(K_bc, F_bc, tol=tol, maxiter=maxiter)
    
    if verbose:
        print(f"  CG iterations: {iterations}")
        print(f"  Final residual: {residual:.6e}")
        print(f"  Max | psi|: {jnp.max(jnp.abs(psi)):.6f}")
    
    # Compute deflection field
    if verbose:
        print(f"Computing deflection field  alpha = ∇ψ...")
    
    alpha = compute_deflection(mesh, psi)
    
    if verbose:
        alpha_mag = jnp.sqrt(jnp.sum(alpha**2, axis=1))
        print(f"  Max | alpha|: {jnp.max(alpha_mag):.6f}")
    
    return FEMSolution(
        psi=psi,
        alpha=alpha,
        convergence=kappa,
        iterations=int(iterations),
        residual=float(residual)
    )


# ============================================================================
# Helper functions for verification and visualization
# ============================================================================

@jit
def compute_errors_p1(mesh, psi_numerical: jnp.ndarray, 
                     psi_exact: jnp.ndarray) -> dict:
    """
    Compute L2 and L∞ errors for P1 elements
    
    Args:
        mesh: Mesh object
        psi_numerical: (n_nodes,) numerical solution
        psi_exact: (n_nodes,) exact solution at nodes
    
    Returns:
        dict with 'l2' and 'linf' errors
    """
    l2_sum = 0.0
    linf_error = 0.0
    
    def accumulate_error(carry, elem_idx):
        l2_sum, linf_error = carry
        
        nodes_idx = mesh.elements[elem_idx]  # (3,)
        coords = mesh.nodes[nodes_idx]  # (3, 2)
        
        # Error at each node
        error_vals = psi_numerical[nodes_idx] - psi_exact[nodes_idx]
        e1, e2, e3 = error_vals[0], error_vals[1], error_vals[2]
        
        # Element area
        area = compute_element_area(coords)
        
        # L2 contribution: ∫_T e² dA ≈ area/3 * (e1² + e2² + e3²)
        l2_contrib = (area / 3.0) * (e1**2 + e2**2 + e3**2)
        l2_sum = l2_sum + l2_contrib
        
        # L∞ error
        elem_linf = jnp.max(jnp.abs(error_vals))
        linf_error = jnp.maximum(linf_error, elem_linf)
        
        return (l2_sum, linf_error), None
    
    (l2_sum, linf_error), _ = jax.lax.scan(
        accumulate_error,
        (l2_sum, linf_error),
        jnp.arange(mesh.n_elements)
    )
    
    l2_error = jnp.sqrt(l2_sum)
    
    return {
        'l2': l2_error,
        'linf': linf_error
    }

@jit
def compute_errors_p2(mesh, psi_numerical: jnp.ndarray, 
                     psi_exact: jnp.ndarray) -> dict:
    """
    Compute L2 and L∞ errors for P2 elements using numerical integration
    
    Args:
        mesh: Mesh object
        psi_numerical: (n_nodes,) numerical solution
        psi_exact: (n_nodes,) exact solution at nodes
    
    Returns:
        dict with 'l2' and 'linf' errors
    """
    l2_sum = 0.0
    linf_error = 0.0
    
    def accumulate_error(carry, elem_idx):
        l2_sum, linf_error = carry
        
        nodes_idx = mesh.elements[elem_idx]  # (6,)
        coords = mesh.nodes[nodes_idx]  # (6, 2)
        
        # Get errors at all 6 nodes
        errors = psi_numerical[nodes_idx] - psi_exact[nodes_idx]
        
        # L∞ error (max at nodes)
        elem_linf = jnp.max(jnp.abs(errors))
        linf_error = jnp.maximum(linf_error, elem_linf)
        
        # L2 error via quadrature
        gauss_points, gauss_weights = get_gauss_points_triangle(2)
        
        l2_contrib = 0.0
        for gp, w in zip(gauss_points, gauss_weights):
            xi, eta = gp[0], gp[1]
            
            # Shape functions at quadrature point
            N = compute_p2_shape_functions(xi, eta)
            
            # Interpolate error at quadrature point
            error_at_gp = jnp.dot(N, errors)
            
            # Jacobian determinant
            J = compute_jacobian(xi, eta, coords)
            det_J = jnp.linalg.det(J)
            
            # Add contribution
            l2_contrib += w * det_J * error_at_gp**2
        
        l2_sum = l2_sum + l2_contrib
        
        return (l2_sum, linf_error), None
    
    (l2_sum, linf_error), _ = jax.lax.scan(
        accumulate_error,
        (l2_sum, linf_error),
        jnp.arange(mesh.n_elements)
    )
    
    l2_error = jnp.sqrt(l2_sum)
    
    return {
        'l2': l2_error,
        'linf': linf_error
    }

@jit
def compute_errors(mesh, psi_numerical: jnp.ndarray, 
                  psi_exact: jnp.ndarray) -> dict:
    """
    Compute L2 and L∞ errors - automatically detects P1 vs P2
    
    Args:
        mesh: Mesh object
        psi_numerical: (n_nodes,) numerical solution
        psi_exact: (n_nodes,) exact solution at nodes
    
    Returns:
        dict with 'l2' and 'linf' errors
    """
    nodes_per_elem = mesh.elements.shape[1]
    
    if nodes_per_elem == 3:
        return compute_errors_p1(mesh, psi_numerical, psi_exact)
    elif nodes_per_elem == 6:
        return compute_errors_p2(mesh, psi_numerical, psi_exact)
    else:
        raise ValueError(f"Unknown element type: {nodes_per_elem} nodes/elem")

# ============================================================================
# Example analytic lensing solutions for validation
# ============================================================================

class PointMassLens:
    """Point mass lens at origin"""
    
    def __init__(self, theta_E: float = 1.0):
        """
        Args:
            theta_E: Einstein radius
        """
        self.theta_E = theta_E
    
    def kappa(self, x: float, y: float) -> float:
        """Convergence (singular at origin)"""
        r = jnp.sqrt(x**2 + y**2) + 1e-6  # Regularize
        return self.theta_E**2 / (2 * jnp.pi * r**2)
    
    def psi(self, x: float, y: float) -> float:
        """Lensing potential"""
        r = jnp.sqrt(x**2 + y**2) + 1e-6
        return self.theta_E**2 * jnp.log(r)
    
    def alpha(self, x: float, y: float) -> tuple:
        """Deflection angle"""
        r = jnp.sqrt(x**2 + y**2) + 1e-6
        alpha_mag = self.theta_E**2 / r
        return (alpha_mag * x / r, alpha_mag * y / r)


class SISLens:
    """Singular Isothermal Sphere"""
    
    def __init__(self, theta_E: float = 1.0):
        self.theta_E = theta_E
    
    def kappa(self, x: float, y: float) -> float:
        r = jnp.sqrt(x**2 + y**2) + 1e-6
        return self.theta_E / (2 * r)
    
    def psi(self, x: float, y: float) -> float:
        r = jnp.sqrt(x**2 + y**2) + 1e-6
        return self.theta_E * r
    
    def alpha(self, x: float, y: float) -> tuple:
        """Constant deflection"""
        r = jnp.sqrt(x**2 + y**2) + 1e-6
        return (self.theta_E * x / r, self.theta_E * y / r)


class GaussianLens:
    """Gaussian mass distribution (smooth, good for testing)"""
    
    def __init__(self, amplitude: float = 1.0, sigma: float = 0.2):
        self.amplitude = amplitude
        self.sigma = sigma
    
    def kappa(self, x: float, y: float) -> float:
        r2 = x**2 + y**2
        return self.amplitude * jnp.exp(-r2 / (2 * self.sigma**2))
    
    def psi(self, x: float, y: float) -> float:
        """Approximate - for exact need hypergeometric function"""
        # For small r: psi ≈ A * sigma² * (1 - exp(-r²/2σ²))
        r2 = x**2 + y**2
        return self.amplitude * self.sigma**2 * (1 - jnp.exp(-r2 / (2 * self.sigma**2)))
    
    def alpha(self, x: float, y: float) -> tuple:
        """Deflection from Gaussian"""
        r2 = x**2 + y**2
        r = jnp.sqrt(r2) + 1e-6
        alpha_mag = self.amplitude * self.sigma**2 * (1 - jnp.exp(-r2 / (2 * self.sigma**2))) / r
        return (alpha_mag * x / r, alpha_mag * y / r)

class SinusoidalLens:
    """
    Manufactured solution for convergence testing
    
    Perfect for validation because:
    - Smooth (C^infty)
    - Satisfies homogeneous Dirichlet BC exactly
    - Known exact solution and source
    """
    
    def __init__(self, k: int = 1):
        """
        Args:
            k: Wavenumber (k=1 gives one wavelength across [0,1]x[0,1])
        """
        self.k = k
    
    def psi(self, x: float, y: float) -> float:
        """Exact lensing potential"""
        return jnp.sin(self.k * jnp.pi * x) * jnp.sin(self.k * jnp.pi * y)
    
    def kappa(self, x: float, y: float) -> float:
        """Convergence field (from nabla^2psi = 2kappa)"""
        # nabla^2psi = frac{partial^2psi}{partial x^2} + frac{partial^2psi}{partial y^2}
        #     = -k^2pi^2 sin(kpi x)sin(kpi y) - k^2pi^2 sin(kpi x)sin(kpi y)
        #     = -2k^2pi^2 sin(kpi x)sin(kpi y)
        # So: kappa = frac{nabla^2psi}{2} = -k^2pi^2 sin(kpi x)sin(kpi y)
        return -self.k**2 * jnp.pi**2 * jnp.sin(self.k * jnp.pi * x) * jnp.sin(self.k * jnp.pi * y)
    
    def alpha(self, x: float, y: float) -> tuple:
        """Deflection angle (gradient of psi)"""
        alpha_x = self.k * jnp.pi * jnp.cos(self.k * jnp.pi * x) * jnp.sin(self.k * jnp.pi * y)
        alpha_y = self.k * jnp.pi * jnp.sin(self.k * jnp.pi * x) * jnp.cos(self.k * jnp.pi * y)
        return (alpha_x, alpha_y)


class PolynomialLens:
    """
    Polynomial manufactured solution
    
    Domain: [-1, 1] x [-1, 1]
    Solution: psi = (1 - x^2)(1 - y^2)
    
    Vanishes on all four edges, smooth everywhere
    """
    
    def psi(self, x: float, y: float) -> float:
        """Exact lensing potential"""
        return (1 - x**2) * (1 - y**2)
    
    def kappa(self, x: float, y: float) -> float:
        """Convergence field"""
        # ∇²ψ = ∂²ψ/∂x² + ∂²ψ/∂y²
        # ∂ψ/∂x = -2x(1 - y²)  →  ∂²ψ/∂x² = -2(1 - y²)
        # ∂ψ/∂y = -2y(1 - x²)  →  ∂²ψ/∂y² = -2(1 - x²)
        # ∇²ψ = -2(1 - y²) - 2(1 - x²) = -2(2 - x² - y²)
        # κ = ∇²ψ/2 = -(2 - x² - y²)
        return -(2 - x**2 - y**2)
    
    def alpha(self, x: float, y: float) -> tuple:
        """Deflection angle"""
        alpha_x = -2 * x * (1 - y**2)
        alpha_y = -2 * y * (1 - x**2)
        return (alpha_x, alpha_y)


# ============================================================================
# P2 Shape Functions (Quadratic)
# ============================================================================

@jit
def compute_p2_shape_functions(xi: float, eta: float) -> jnp.ndarray:
    """
    Compute P2 shape functions at point (xi, eta) in reference triangle
    
    Reference triangle: (0,0), (1,0), (0,1)
    Barycentric coordinates: lambda1 = 1 - xi - eta, lambda2 = xi, lambda3 = eta
    
    P2 shape functions (quadratic):
        N0 = lambda1 * (2*lambda1 - 1)  [vertex at (0,0)]
        N1 = lambda2 * (2*lambda2 - 1)  [vertex at (1,0)]
        N2 = lambda3 * (2*lambda3 - 1)  [vertex at (0,1)]
        N3 = 4 * lambda1 * lambda2      [midpoint of edge 0-1]
        N4 = 4 * lambda2 * lambda3      [midpoint of edge 1-2]
        N5 = 4 * lambda3 * lambda1      [midpoint of edge 2-0]
    
    Args:
        xi, eta: Coordinates in reference triangle (xi, eta >= 0, xi+eta <= 1)
        
    Returns:
        (6,) array of shape function values
    """
    # Barycentric coordinates
    lam1 = 1.0 - xi - eta
    lam2 = xi
    lam3 = eta
    
    # Vertex shape functions (quadratic)
    N0 = lam1 * (2.0 * lam1 - 1.0)
    N1 = lam2 * (2.0 * lam2 - 1.0)
    N2 = lam3 * (2.0 * lam3 - 1.0)
    
    # Edge midpoint shape functions (quartic bubble functions)
    N3 = 4.0 * lam1 * lam2  # Midpoint of edge v0-v1
    N4 = 4.0 * lam2 * lam3  # Midpoint of edge v1-v2
    N5 = 4.0 * lam3 * lam1  # Midpoint of edge v2-v0
    
    return jnp.array([N0, N1, N2, N3, N4, N5])


@jit
def compute_p2_shape_gradients_reference(xi: float, eta: float) -> jnp.ndarray:
    """
    Compute gradients of P2 shape functions w.r.t. reference coordinates
    
    Returns:
        (6, 2) array where row i is [dN_i/dxi, dN_i/deta]
    """
    # Barycentric coordinates
    lam1 = 1.0 - xi - eta
    lam2 = xi
    lam3 = eta
    
    # Gradients of barycentric coordinates
    # dlam1/dxi = -1,  dlam1/deta = -1
    # dlam2/dxi =  1,  dlam2/deta =  0
    # dlam3/dxi =  0,  dlam3/deta =  1
    
    # Vertex nodes - using chain rule on N = lam * (2*lam - 1)
    # dN0/dxi = dlam1/dxi * (4*lam1 - 1)
    dN0_dxi = -1.0 * (4.0 * lam1 - 1.0)
    dN0_deta = -1.0 * (4.0 * lam1 - 1.0)
    
    dN1_dxi = 1.0 * (4.0 * lam2 - 1.0)
    dN1_deta = 0.0
    
    dN2_dxi = 0.0
    dN2_deta = 1.0 * (4.0 * lam3 - 1.0)
    
    # Edge midpoint nodes - using product rule on N = 4*lam_i*lam_j
    # N3 = 4*lam1*lam2
    dN3_dxi = 4.0 * (-1.0 * lam2 + lam1 * 1.0)  # 4*(dlam1/dxi*lam2 + lam1*dlam2/dxi)
    dN3_deta = 4.0 * (-1.0 * lam2 + lam1 * 0.0)
    
    # N4 = 4*lam2*lam3
    dN4_dxi = 4.0 * (1.0 * lam3 + lam2 * 0.0)
    dN4_deta = 4.0 * (0.0 * lam3 + lam2 * 1.0)
    
    # N5 = 4*lam3*lam1
    dN5_dxi = 4.0 * (0.0 * lam1 + lam3 * (-1.0))
    dN5_deta = 4.0 * (1.0 * lam1 + lam3 * (-1.0))
    
    return jnp.array([
        [dN0_dxi, dN0_deta],
        [dN1_dxi, dN1_deta],
        [dN2_dxi, dN2_deta],
        [dN3_dxi, dN3_deta],
        [dN4_dxi, dN4_deta],
        [dN5_dxi, dN5_deta]
    ])


@jit
def compute_jacobian(xi: float, eta: float, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Jacobian matrix for transformation from reference to physical triangle
    
    J = [dx/dxi   dy/dxi  ]
        [dx/deta  dy/deta ]
    
    where x, y are physical coordinates, xi, eta are reference coordinates
    
    Args:
        xi, eta: Point in reference triangle
        coords: (6, 2) physical coordinates of the 6 nodes
        
    Returns:
        (2, 2) Jacobian matrix
    """
    # Get shape function gradients in reference coordinates
    dN_dref = compute_p2_shape_gradients_reference(xi, eta)  # (6, 2)
    
    # x(xi, eta) = sum_i N_i(xi, eta) * x_i
    # dx/dxi = sum_i (dN_i/dxi) * x_i
    # dy/deta = sum_i (dN_i/deta) * y_i
    
    # J = coords.T @ dN_dref
    # coords.T is (2, 6), dN_dref is (6, 2) -> result is (2, 2)
    J = jnp.dot(coords.T, dN_dref)
    
    return J


@jit
def compute_p2_shape_gradients_physical(xi: float, eta: float, 
                                        coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute gradients of P2 shape functions in physical coordinates
    
    Uses chain rule: [dN/dx, dN/dy] = [dN/dxi, dN/deta] @ J^{-1}
    
    Args:
        xi, eta: Point in reference triangle
        coords: (6, 2) physical coordinates of the 6 nodes
        
    Returns:
        (6, 2) array where row i is [dN_i/dx, dN_i/dy]
    """
    # Gradients in reference coordinates
    dN_dref = compute_p2_shape_gradients_reference(xi, eta)  # (6, 2)
    
    # Jacobian
    J = compute_jacobian(xi, eta, coords)  # (2, 2)
    J_inv = jnp.linalg.inv(J)
    
    # Transform to physical coordinates
    # dN_dphys = dN_dref @ J_inv
    dN_dphys = jnp.dot(dN_dref, J_inv)  # (6, 2)
    
    return dN_dphys


# ============================================================================
# Gauss Quadrature Rules for Triangles
# ============================================================================

def get_gauss_points_triangle(order: int = 2):
    """
    Get Gauss quadrature points and weights for triangle
    
    Args:
        order: Quadrature order
            1: 1-point (order 1, exact for linear)
            2: 3-point (order 2, exact for quadratic) - DEFAULT
            3: 7-point (order 3, exact for cubic)
    
    Returns:
        points: (n_points, 2) array of (xi, eta) coordinates
        weights: (n_points,) array of weights (sum to 0.5, area of reference triangle)
    """
    if order == 1:
        # 1-point rule (centroid)
        points = jnp.array([[1.0/3.0, 1.0/3.0]])
        weights = jnp.array([0.5])  # Area of reference triangle
        
    elif order == 2:
        # 3-point rule (symmetric, order 2)
        a = 1.0 / 6.0
        b = 2.0 / 3.0
        points = jnp.array([
            [a, a],
            [b, a],
            [a, b]
        ])
        weights = jnp.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
        
    elif order == 3:
        # 7-point rule (order 3)
        a1 = 0.059715871789770
        a2 = 0.470142064105115
        b1 = 0.797426985353087
        b2 = 0.101286507323456
        
        points = jnp.array([
            [1.0/3.0, 1.0/3.0],  # Center
            [a1, a1],
            [a1, 1.0 - 2.0*a1],
            [1.0 - 2.0*a1, a1],
            [a2, a2],
            [a2, 1.0 - 2.0*a2],
            [1.0 - 2.0*a2, a2]
        ])
        
        w1 = 0.225
        w2 = 0.125939180544827
        w3 = 0.132394152788506
        weights = jnp.array([w1, w2, w2, w2, w3, w3, w3]) * 0.5
        
    else:
        raise ValueError(f"Quadrature order {order} not implemented")
    
    return points, weights


# ============================================================================
# P2 Element Assembly
# ============================================================================

@partial(jit, static_argnames=['quad_order'])
def compute_element_stiffness_p2(coords: jnp.ndarray, 
                                 quad_order: int = 2) -> jnp.ndarray:
    """
    Compute 6×6 element stiffness matrix for P2 element
    
    K[i,j] = ∫_T ∇N_i · ∇N_j dA
    
    Uses numerical integration (Gauss quadrature)
    
    Args:
        coords: (6, 2) array of node coordinates
        quad_order: Quadrature rule order (2 is sufficient for P2 stiffness)
        
    Returns:
        (6, 6) element stiffness matrix
    """
    gauss_points, gauss_weights = get_gauss_points_triangle(quad_order)
    
    K_elem = jnp.zeros((6, 6))
    
    # Loop over quadrature points
    for gp, w in zip(gauss_points, gauss_weights):
        xi, eta = gp[0], gp[1]
        
        # Shape function gradients at this point
        grad_N = compute_p2_shape_gradients_physical(xi, eta, coords)  # (6, 2)
        
        # Jacobian determinant (for area transformation)
        J = compute_jacobian(xi, eta, coords)
        det_J = jnp.linalg.det(J)
        
        # Add contribution: w * |det(J)| * grad_N_i · grad_N_j
        # This is an outer product: grad_N @ grad_N.T gives (6,6) matrix
        K_elem += w * det_J * jnp.dot(grad_N, grad_N.T)
    
    return K_elem


@partial(jit, static_argnames=['quad_order'])
def compute_element_load_p2(coords: jnp.ndarray, 
                            kappa_vals: jnp.ndarray,
                            quad_order: int = 2) -> jnp.ndarray:
    """
    Compute element load vector for P2 element
    
    F[i] = ∫_T 2κ N_i dA
    
    Assumes κ is interpolated using P2 basis: κ(xi,eta) = Σ κ_j N_j(xi,eta)
    
    Args:
        coords: (6, 2) node coordinates
        kappa_vals: (6,) convergence values at the 6 nodes
        quad_order: Quadrature rule order
        
    Returns:
        (6,) element load vector
    """
    gauss_points, gauss_weights = get_gauss_points_triangle(quad_order)
    
    F_elem = jnp.zeros(6)
    
    # Loop over quadrature points
    for gp, w in zip(gauss_points, gauss_weights):
        xi, eta = gp[0], gp[1]
        
        # Shape functions at this point
        N = compute_p2_shape_functions(xi, eta)  # (6,)
        
        # Jacobian determinant
        J = compute_jacobian(xi, eta, coords)
        det_J = jnp.linalg.det(J)
        
        # Interpolate κ at this quadrature point
        kappa_pt = jnp.dot(N, kappa_vals)
        
        # Add contribution: -2 * w * |det(J)| * κ * N_i
        F_elem += -2.0 * w * det_J * kappa_pt * N
    
    return F_elem


# ============================================================================
# Example/Test Functions
# ============================================================================

def test_p2_shape_functions():
    """
    Test that P2 shape functions have correct properties
    """
    print("Testing P2 shape functions...")
    
    # Reference triangle nodes
    ref_nodes = jnp.array([
        [0.0, 0.0],  # v0
        [1.0, 0.0],  # v1
        [0.0, 1.0],  # v2
        [0.5, 0.0],  # m01
        [0.5, 0.5],  # m12
        [0.0, 0.5],  # m20
    ])
    
    # Test 1: Partition of unity (sum of shape functions = 1)
    test_points = [
        (0.25, 0.25),
        (0.1, 0.2),
        (0.5, 0.3),
        (1.0/3.0, 1.0/3.0)  # centroid
    ]
    
    print("\nTest 1: Partition of unity")
    for xi, eta in test_points:
        N = compute_p2_shape_functions(xi, eta)
        sum_N = jnp.sum(N)
        print(f"  ({xi:.3f}, {eta:.3f}): sum(N) = {sum_N:.10f} (should be 1.0)")
        assert jnp.abs(sum_N - 1.0) < 1e-10, "Partition of unity failed!"
    
    # Test 2: Kronecker delta property N_i(x_j) = delta_ij
    print("\nTest 2: Kronecker delta property")
    for i in range(6):
        xi, eta = ref_nodes[i, 0], ref_nodes[i, 1]
        N = compute_p2_shape_functions(xi, eta)
        
        print(f"  Node {i}: N = {N}")
        assert jnp.abs(N[i] - 1.0) < 1e-10, f"N_{i}(x_{i}) should be 1.0!"
        for j in range(6):
            if i != j:
                assert jnp.abs(N[j]) < 1e-10, f"N_{j}(x_{i}) should be 0.0!"
    
    # Test 3: Gradient computation
    print("\nTest 3: Gradient computation at centroid")
    xi, eta = 1.0/3.0, 1.0/3.0
    
    # Physical triangle (equilateral for symmetry)
    phys_coords = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, jnp.sqrt(3.0)/2.0],
        [0.5, 0.0],
        [0.75, jnp.sqrt(3.0)/4.0],
        [0.25, jnp.sqrt(3.0)/4.0]
    ])
    
    grad_N = compute_p2_shape_gradients_physical(xi, eta, phys_coords)
    print(f"  Gradients shape: {grad_N.shape}")
    print(f"  Sum of gradient x-components: {jnp.sum(grad_N[:, 0]):.10f} (should be ~0)")
    print(f"  Sum of gradient y-components: {jnp.sum(grad_N[:, 1]):.10f} (should be ~0)")
    
    print("\n✓ All P2 shape function tests passed!")


if __name__ == "__main__":
    test_p2_shape_functions()
