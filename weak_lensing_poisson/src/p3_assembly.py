"""
P3 Finite Element Assembly for Cubic Elements

Assembles stiffness matrix and RHS for Poisson equation:
    ∇²ψ = 2κ in Ω
    ψ = 0 on ∂Ω

Uses 10-node cubic triangular elements with 7-point Gauss quadrature
for exact integration of degree-5 polynomials.
"""

import jax.numpy as jnp
import jax
from jax import jit
import numpy as np
from typing import Tuple
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from .p3_shape_functions import (
        compute_p3_shape_functions,
        compute_p3_shape_gradients_reference,
        compute_p3_shape_gradients_physical
    )
except ImportError:
    from p3_shape_functions import (
        compute_p3_shape_functions,
        compute_p3_shape_gradients_reference,
        compute_p3_shape_gradients_physical
    )


# ============================================================================
# Quadrature Rules for Triangles
# ============================================================================

def get_gauss_quadrature_triangle(order: int = 4):
    """
    Get Gauss quadrature points and weights for triangles
    
    Integration rule: ∫∫_T f(ξ,η) dξdη ≈ |T|/2 Σ wᵢ f(ξᵢ, ηᵢ)
    
    Order 4 (7 points): Exact for polynomials up to degree 5
    Order 5 (13 points): Exact for polynomials up to degree 7 - NEEDED FOR P3!
    
    Args:
        order: Quadrature order (5 recommended for P3)
        
    Returns:
        points: (nq, 2) array of (ξ, η) coordinates
        weights: (nq,) array of quadrature weights
    """
    if order == 1:
        # 1-point rule (degree 1 exact) - centroid
        points = jnp.array([[1/3, 1/3]])
        weights = jnp.array([1.0])
        
    elif order == 2:
        # 3-point rule (degree 2 exact) - vertices
        points = jnp.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3]
        ])
        weights = jnp.array([1/3, 1/3, 1/3])
        
    elif order == 3:
        # 4-point rule (degree 3 exact)
        a = 1/3
        points = jnp.array([
            [a, a],
            [0.6, 0.2],
            [0.2, 0.6],
            [0.2, 0.2]
        ])
        weights = jnp.array([-27/48, 25/48, 25/48, 25/48])
        
    elif order == 4:
        # 7-point rule (degree 5 exact)
        # Reference: Dunavant (1985)
        a1 = 0.059715871789770
        a2 = 0.797426985353087
        b1 = 0.470142064105115
        b2 = 0.101286507323456
        
        w1 = 0.225000000000000
        w2 = 0.132394152788506
        w3 = 0.125939180544827
        
        points = jnp.array([
            [1/3, 1/3],          # Center
            [a1, a1],            # Near vertex 0
            [a2, a1],            # Near vertex 1
            [a1, a2],            # Near vertex 2
            [b1, b1],            # Edge 0-1
            [b2, b1],            # Edge 1-2
            [b1, b2]             # Edge 2-0
        ])
        weights = jnp.array([w1, w2, w2, w2, w3, w3, w3])
        
    elif order == 5:
        # Dunavant degree-7 rule (13 points)
    
        a = 0.065130102902216
        b = 0.312865496004875
        c = 0.260345966079040
        d = 0.479308067841920
    
        w0 = -0.149570044467670
        w1 =  0.175615257433208
        w2 =  0.053347235608839
        w3 =  0.077113760890257
    
        points = jnp.array([
            # centroid
            [1/3, 1/3],
    
            # a-set
            [a, a],
            [1-2*a, a],
            [a, 1-2*a],
    
            # b-set
            [b, b],
            [1-2*b, b],
            [b, 1-2*b],
    
            # c,d set (6 permutations)
            [c, d],
            [d, c],
            [c, 1-c-d],
            [1-c-d, c],
            [d, 1-c-d],
            [1-c-d, d],
        ])
    
        weights = jnp.array([
            w0,
            w1, w1, w1,
            w2, w2, w2,
            w3, w3, w3, w3, w3, w3
        ])
        
    else:
        raise ValueError(f"Quadrature order {order} not implemented")
    
    return points, weights


# ============================================================================
# Element Stiffness Matrix
# ============================================================================

@jit
def compute_element_stiffness_p3(coords: jnp.ndarray,
                                 quad_points: jnp.ndarray,
                                 quad_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Compute 10×10 element stiffness matrix for P3 element
    
    Ke[i,j] = ∫_T ∇Nᵢ · ∇Nⱼ dA
    
    CRITICAL: Dunavant weights sum to 1.0, so integral = (detJ/2) * Σ wᵢ f(ξᵢ)
    """
    nq = len(quad_weights)
    Ke = jnp.zeros((10, 10))
    
    # SUBPARAMETRIC: Use only 3 vertex nodes for geometry
    vertex_coords = coords[:3, :]  # (3, 2)
    
    # Compute P1 geometry Jacobian (constant over element)
    x0, y0 = vertex_coords[0]
    x1, y1 = vertex_coords[1]
    x2, y2 = vertex_coords[2]
    
    J = jnp.array([[x1 - x0, y1 - y0],
                   [x2 - x0, y2 - y0]])
    detJ = jnp.linalg.det(J)
    J_inv = jnp.linalg.inv(J)
    
    # Physical element area (for quadrature with weights summing to 1.0)
    area_factor = jnp.abs(detJ) / 2.0
    
    # Loop over quadrature points
    for q in range(nq):
        xi, eta = quad_points[q]
        w = quad_weights[q]
        
        # Get P3 shape function gradients in reference coordinates
        dN_dxi = compute_p3_shape_gradients_reference(xi, eta)  # (10, 2)
        
        # Transform to physical coordinates
        dN_dxy = dN_dxi @ J_inv.T  # (10, 2)
        
        # Stiffness contribution with CORRECT area factor
        for i in range(10):
            for j in range(10):
                Ke = Ke.at[i, j].add(
                    w * area_factor * jnp.dot(dN_dxy[i], dN_dxy[j])
                )
    
    return Ke


@jit
def compute_element_load_p3(coords: jnp.ndarray,
                           source_values: jnp.ndarray,
                           quad_points: jnp.ndarray,
                           quad_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Compute 10×1 element load vector for P3 element
    
    Fe[i] = -2 ∫_T Nᵢ κ dA
    
    CRITICAL: Dunavant weights sum to 1.0, so integral = (detJ/2) * Σ wᵢ f(ξᵢ)
    """
    nq = len(quad_weights)
    Fe = jnp.zeros(10)
    
    # SUBPARAMETRIC: Use only 3 vertex nodes for geometry
    vertex_coords = coords[:3, :]
    
    # Compute P1 geometry Jacobian (constant)
    x0, y0 = vertex_coords[0]
    x1, y1 = vertex_coords[1]
    x2, y2 = vertex_coords[2]
    
    J = jnp.array([[x1 - x0, y1 - y0],
                   [x2 - x0, y2 - y0]])
    detJ = jnp.linalg.det(J)
    
    # Physical element area
    area_factor = jnp.abs(detJ) / 2.0
    
    for q in range(nq):
        xi, eta = quad_points[q]
        w = quad_weights[q]
        
        # P3 shape functions at quadrature point
        N = compute_p3_shape_functions(xi, eta)  # (10,)
        
        # Source term at quadrature point (interpolated from nodes)
        kappa_q = jnp.dot(N, source_values)
        
        # Load contribution with CORRECT area factor
        Fe += -2.0 * w * area_factor * N * kappa_q
    
    return Fe


# ============================================================================
# Global Assembly
# ============================================================================

def assemble_system_p3(mesh, kappa_values, use_jax: bool = False):
    """
    Assemble global stiffness matrix and load vector for P3 elements
    
    Args:
        mesh: P3 Mesh object with 10-node elements
        kappa_values: Source term κ evaluated at all nodes (n_nodes,)
        use_jax: If True, use JAX arrays (slower assembly but GPU-ready)
        
    Returns:
        K: Global stiffness matrix (sparse CSR)
        F: Global load vector (n_nodes,)
    """
    nodes = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    n_nodes = len(nodes)
    n_elements = len(elements)
    
    print(f"Assembling P3 system: {n_elements} elements, {n_nodes} DOFs...")
    
    # Get quadrature rule (7-point, order 4, exact for degree 5)
    # For SUBPARAMETRIC P3: Jacobian is constant, gradients are degree 2
    # Product ∇Ni·∇Nj is degree 4, so order 4 quadrature is sufficient
    quad_points, quad_weights = get_gauss_quadrature_triangle(order=5)
    
    # Preallocate sparse matrix storage (COO format)
    # Each element contributes 10×10 = 100 entries
    max_entries = n_elements * 100
    I = np.zeros(max_entries, dtype=np.int32)
    J = np.zeros(max_entries, dtype=np.int32)
    K_data = np.zeros(max_entries)
    F = np.zeros(n_nodes)
    
    entry_idx = 0
    
    # Loop over elements
    for elem_idx, elem in enumerate(elements):
        if elem_idx % 100 == 0:
            print(f"  Assembling element {elem_idx}/{n_elements}...", end='\r')
        
        # Get element node coordinates
        elem_coords = nodes[elem]  # (10, 2)
        elem_kappa = np.array(kappa_values[elem])  # (10,)
        
        # Convert to JAX for JIT-compiled functions
        elem_coords_jax = jnp.array(elem_coords)
        elem_kappa_jax = jnp.array(elem_kappa)
        quad_points_jax = jnp.array(quad_points)
        quad_weights_jax = jnp.array(quad_weights)
        
        # Compute element matrices
        Ke = compute_element_stiffness_p3(elem_coords_jax, 
                                         quad_points_jax, 
                                         quad_weights_jax)
        Fe = compute_element_load_p3(elem_coords_jax,
                                    elem_kappa_jax,
                                    quad_points_jax,
                                    quad_weights_jax)
        
        Ke = np.array(Ke)  # Convert back to numpy for assembly
        Fe = np.array(Fe)
        
        # Assemble into global system
        for i in range(10):
            global_i = elem[i]
            
            # Add to global load vector
            F[global_i] += Fe[i]
            
            # Add to global stiffness matrix
            for j in range(10):
                global_j = elem[j]
                
                I[entry_idx] = global_i
                J[entry_idx] = global_j
                K_data[entry_idx] = Ke[i, j]
                entry_idx += 1
    
    print(f"  Assembling element {n_elements}/{n_elements}... Done!")
    
    # Create sparse matrix (sum duplicate entries automatically)
    K = sp.coo_matrix((K_data[:entry_idx], (I[:entry_idx], J[:entry_idx])),
                      shape=(n_nodes, n_nodes))
    K = K.tocsr()
    
    print(f"  Global system: {n_nodes}×{n_nodes}, nnz={K.nnz}")
    
    if use_jax:
        F = jnp.array(F)
    
    return K, F


def apply_boundary_conditions_p3(K, F, mesh):
    """
    Apply Dirichlet boundary conditions (ψ = 0 on ∂Ω)
    
    Modifies K and F in-place using penalty method
    
    Args:
        K: Global stiffness matrix (sparse CSR)
        F: Global load vector
        mesh: P3 Mesh object with boundary nodes identified
        
    Returns:
        K_bc, F_bc: Modified system with BCs applied
    """
    boundary = np.array(mesh.boundary)
    n_boundary = len(boundary)
    
    print(f"Applying boundary conditions to {n_boundary} nodes...")
    
    # Penalty method: Set Kᵢᵢ = large number, Fᵢ = 0 for boundary nodes
    K_bc = K.tolil()  # Convert to LIL for efficient modification
    F_bc = F.copy()
    
    penalty = 1e8
    
    for node in boundary:
        K_bc[node, :] = 0
        K_bc[node, node] = penalty
        F_bc[node] = 0
    
    K_bc = K_bc.tocsr()
    
    return K_bc, F_bc


# ============================================================================
# Solver
# ============================================================================

def solve_p3_system(K, F, mesh):
    """
    Solve the P3 finite element system
    
    Args:
        K: Stiffness matrix (with BCs applied)
        F: Load vector (with BCs applied)
        mesh: P3 Mesh object
        
    Returns:
        psi: Solution vector (n_nodes,)
    """
    n_nodes = len(mesh.nodes)
    print(f"Solving {n_nodes}×{n_nodes} sparse linear system...")
    
    # Use sparse direct solver
    psi = spla.spsolve(K, F)
    
    print(f"  Solution complete, residual: {np.linalg.norm(K @ psi - F):.2e}")
    
    return psi


# ============================================================================
# Complete Solve Pipeline
# ============================================================================

def solve_poisson_p3(mesh, kappa_values):
    """
    Complete P3 FEM solution pipeline for Poisson equation
    
    Solves: -∇²ψ = κ with ψ=0 on boundary
    
    Args:
        mesh: P3 Mesh object
        kappa_values: Source term κ at all nodes (n_nodes,)
        
    Returns:
        psi: Lensing potential solution (n_nodes,)
    """
    print("\n" + "=" * 70)
    print("P3 POISSON SOLVER")
    print("=" * 70)
    
    # Assemble system
    K, F = assemble_system_p3(mesh, kappa_values)
    
    # Apply boundary conditions
    K_bc, F_bc = apply_boundary_conditions_p3(K, F, mesh)
    
    # Solve
    psi = solve_p3_system(K_bc, F_bc, mesh)
    
    print("=" * 70)
    print("✅ P3 SOLUTION COMPLETE")
    print("=" * 70 + "\n")
    
    return psi


# ============================================================================
# Test/Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🎯" * 35)
    print(" " * 22 + "P3 ASSEMBLY - TEST")
    print("🎯" * 35 + "\n")
    
    # Import mesh generator
    try:
        from .p3_mesh_generator import generate_p3_structured_mesh
    except ImportError:
        from p3_mesh_generator import generate_p3_structured_mesh
    
    # Test 1: Quadrature rules
    print("Testing quadrature rules...")
    for order in [1, 2, 3, 4, 5]:
        pts, wts = get_gauss_quadrature_triangle(order)
        print(f"  Order {order}: {len(pts)} points, Σw = {np.sum(wts):.10f} (should be 1.0)")
    
    # Test 2: Small mesh assembly
    print("\nTest: Small P3 mesh assembly")
    print("=" * 70)
    
    mesh = generate_p3_structured_mesh(3, 3, xmin=0, xmax=1, ymin=0, ymax=1)
    
    # Manufactured solution: ψ = sin(πx)sin(πy)
    # → ∇²ψ = -2π² sin(πx)sin(πy)
    # Strong form: ∇²ψ = 2κ
    # → -2π² sin(πx)sin(πy) = 2κ
    # → κ = -π² sin(πx)sin(πy)
    nodes = np.array(mesh.nodes)
    kappa = -np.pi**2 * np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    
    # Solve
    psi = solve_poisson_p3(mesh, kappa)
    
    # Compute error
    psi_exact = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    error_L2 = np.sqrt(np.mean((psi - psi_exact)**2))
    
    print(f"\nValidation:")
    print(f"  L² error: {error_L2:.6e}")
    print(f"  Max error: {np.max(np.abs(psi - psi_exact)):.6e}")
    
    # For 3×3 P3 mesh (h=1/3), expect O(h⁴) ≈ 1e-2
    if error_L2 < 2e-2:
        print(f"  ✅ Error excellent for P3 on coarse 3×3 mesh!")
    else:
        print(f"  ⚠️  Error higher than expected")
    
    print("\n" + "=" * 70)
    print("✅ P3 ASSEMBLY TEST COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. ✅ P3 shape functions")
    print("  2. ✅ P3 mesh generator")  
    print("  3. ✅ P3 assembly")
    print("  4. ⏳ P3 convergence study")
    print("  5. ⏳ P3 shear computation")
    print("=" * 70)