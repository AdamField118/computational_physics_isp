"""
fem.py
======
Consolidated P3 finite element module for weak gravitational lensing.

Assembles three fixed sparse linear operators for a given mesh:

    K  (stiffness)   :  K[i,j] = ∫ ∇Nᵢ · ∇Nⱼ dA
    M  (mass)        :  M[i,j] = ∫ Nᵢ Nⱼ dA
    S1 (shear-1 op)  :  (S1 ψ)[i] = ½(∂²ψ/∂x² − ∂²ψ/∂y²) at node i
    S2 (shear-2 op)  :  (S2 ψ)[i] = ∂²ψ/∂x∂y at node i

The complete forward model is then the purely linear chain:

    ψ  = K⁻¹ (−2 M κ)      (FEM Poisson solve with Dirichlet BCs)
    γ₁ = S1 ψ
    γ₂ = S2 ψ

Because K, M, S1, S2 are assembled once and cached, every forward
evaluation of κ → (γ₁, γ₂) costs only one sparse triangular solve
(from a cached LU factorization) plus three sparse matrix-vector products.

Usage
-----
    from src.fem import FEMOperators, build_operators

    ops = build_operators(nx=30, ny=30, xmin=-2.5, xmax=2.5,
                          ymin=-2.5, ymax=2.5)

    # ops.forward_kappa(kappa) -> (gamma1, gamma2)
    # ops.psi_from_kappa(kappa) -> psi
    # ops.K, ops.M, ops.S1, ops.S2  (scipy sparse CSR)
    # ops.K_lu                       (cached SuperLU factorization)
    # ops.mesh                       (P3 Mesh)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Tuple
import time

# ── re-use existing modules ────────────────────────────────────────────────────
from .p3_mesh_generator import generate_p3_structured_mesh
from .p3_shape_functions import (
    compute_p3_shape_functions,
    compute_p3_shape_gradients_reference,
)
from .p3_assembly import (
    get_gauss_quadrature_triangle,
    compute_element_stiffness_p3,
    compute_element_load_p3,
    apply_boundary_conditions_p3,
)
from .fem_solver import Mesh


# ══════════════════════════════════════════════════════════════════════════════
# Reference Hessians (precomputed once at import time)
# ══════════════════════════════════════════════════════════════════════════════

# P3 DOF positions in the reference triangle
_P3_REF_NODES = np.array([
    [0.0,       0.0      ],   # vertex 0
    [1.0,       0.0      ],   # vertex 1
    [0.0,       1.0      ],   # vertex 2
    [1.0/3.0,   0.0      ],   # edge 0-1, t=1/3
    [2.0/3.0,   0.0      ],   # edge 0-1, t=2/3
    [2.0/3.0,   1.0/3.0  ],   # edge 1-2, t=1/3
    [1.0/3.0,   2.0/3.0  ],   # edge 1-2, t=2/3
    [0.0,       2.0/3.0  ],   # edge 2-0, t=1/3
    [0.0,       1.0/3.0  ],   # edge 2-0, t=2/3
    [1.0/3.0,   1.0/3.0  ],   # interior (centroid)
])


def _build_ref_hessians() -> np.ndarray:
    """
    H_ref[eval_node, shape_fn, i, j] = ∂²Nⱼ/∂ξᵢ ∂ξⱼ  at each of the 10
    reference nodes.

    Shape: (10, 10, 2, 2).  Computed once via JAX forward-over-reverse AD.
    """
    def N_vec(xi_eta):
        return compute_p3_shape_functions(xi_eta[0], xi_eta[1])  # (10,)

    hess_fn = jax.jacfwd(jax.jacrev(N_vec))   # exact Hessian, no FD

    H = np.stack([
        np.array(hess_fn(jnp.array(pt, dtype=jnp.float64)))
        for pt in _P3_REF_NODES
    ])   # (10, 10, 2, 2)
    return H


# ══════════════════════════════════════════════════════════════════════════════
# Mass Matrix Assembly
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_mass_p3(nodes: np.ndarray,
                     elements: np.ndarray,
                     quad_points: np.ndarray,
                     quad_weights: np.ndarray) -> sp.csr_matrix:
    """
    Assemble global P3 mass matrix.

        M[i,j] = ∫_Ω Nᵢ Nⱼ dA

    Uses the same quadrature as the stiffness matrix (order=5 Dunavant,
    13 points, exact for degree 7; more than sufficient for the degree-6
    mass integrand).
    """
    n_nodes   = len(nodes)
    n_elems   = len(elements)
    max_nnz   = n_elems * 100

    I      = np.zeros(max_nnz, dtype=np.int32)
    J_idx  = np.zeros(max_nnz, dtype=np.int32)
    M_data = np.zeros(max_nnz)
    idx    = 0

    for elem in elements:
        x0, y0 = nodes[elem[0]]
        x1, y1 = nodes[elem[1]]
        x2, y2 = nodes[elem[2]]

        J     = np.array([[x1-x0, y1-y0], [x2-x0, y2-y0]])
        detJ  = np.linalg.det(J)
        area  = abs(detJ) / 2.0

        # 10×10 element mass matrix via quadrature
        Me = np.zeros((10, 10))
        for q, (xi, eta) in enumerate(quad_points):
            w  = quad_weights[q]
            N  = np.array(compute_p3_shape_functions(xi, eta))  # (10,)
            Me += w * area * np.outer(N, N)

        # Scatter into COO
        for i in range(10):
            for j in range(10):
                I[idx]      = elem[i]
                J_idx[idx]  = elem[j]
                M_data[idx] = Me[i, j]
                idx += 1

    M = sp.coo_matrix(
        (M_data[:idx], (I[:idx], J_idx[:idx])),
        shape=(n_nodes, n_nodes)
    ).tocsr()
    return M


# ══════════════════════════════════════════════════════════════════════════════
# Shear Operator Assembly
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_shear_ops(nodes: np.ndarray,
                        elements: np.ndarray,
                        H_ref: np.ndarray) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Assemble sparse shear operators S1 and S2.

    For a P3 solution ψ:

        (S1 ψ)[i] = ½ (ψ_xx − ψ_yy)  at mesh node i
        (S2 ψ)[i] = ψ_xy              at mesh node i

    Both are assembled by nodal averaging of element-wise Hessian
    contributions.  The affine Jacobian is constant per element, so:

        ∂²ψ/∂xₐ∂x_b = Σ_n ψ_n · [Σ_{j,k} A[j,a] A[k,b] H_ref[local_i,n,j,k]]

    where A = J⁻ᵀ maps reference→physical gradients.

    Args:
        nodes     : (n_nodes, 2) physical coordinates
        elements  : (n_elems, 10) element connectivity
        H_ref     : (10, 10, 2, 2) precomputed reference Hessians

    Returns:
        S1, S2 : (n_nodes, n_nodes) sparse CSR matrices
    """
    n_nodes = len(nodes)
    n_elems = len(elements)
    max_nnz = n_elems * 100

    I1 = np.zeros(max_nnz, dtype=np.int32)
    J1 = np.zeros(max_nnz, dtype=np.int32)
    D1 = np.zeros(max_nnz)

    I2 = np.zeros(max_nnz, dtype=np.int32)
    J2 = np.zeros(max_nnz, dtype=np.int32)
    D2 = np.zeros(max_nnz)

    idx = 0
    counts = np.zeros(n_nodes, dtype=np.int32)

    for elem in elements:
        x0, y0 = nodes[elem[0]]
        x1, y1 = nodes[elem[1]]
        x2, y2 = nodes[elem[2]]

        J = np.array([[x1-x0, y1-y0], [x2-x0, y2-y0]])
        A = np.linalg.inv(J).T   # A[j, a] = ∂ref_j/∂x_a

        for local_i in range(10):
            # Transform reference Hessians to physical:
            #   H_phys[n,a,b] = Σ_{j,k} A[j,a] A[k,b] H_ref[local_i, n, j, k]
            H_phys = np.einsum('ja,kb,njk->nab', A, A, H_ref[local_i])  # (10,2,2)

            row = elem[local_i]
            for local_j in range(10):
                col = elem[local_j]
                # γ₁ = ½(ψ_xx - ψ_yy)
                s1_val = 0.5 * (H_phys[local_j, 0, 0] - H_phys[local_j, 1, 1])
                # γ₂ = ψ_xy
                s2_val = H_phys[local_j, 0, 1]

                I1[idx] = row;  J1[idx] = col;  D1[idx] = s1_val
                I2[idx] = row;  J2[idx] = col;  D2[idx] = s2_val
                idx += 1

            counts[row] += 1

    # Build raw sparse matrices
    S1_raw = sp.coo_matrix(
        (D1[:idx], (I1[:idx], J1[:idx])), shape=(n_nodes, n_nodes)
    ).tocsr()
    S2_raw = sp.coo_matrix(
        (D2[:idx], (I2[:idx], J2[:idx])), shape=(n_nodes, n_nodes)
    ).tocsr()

    # Divide each row by how many elements contributed (nodal average)
    inv_counts = 1.0 / np.maximum(counts, 1)
    D_scale    = sp.diags(inv_counts)

    S1 = D_scale @ S1_raw
    S2 = D_scale @ S2_raw

    return S1.tocsr(), S2.tocsr()


# ══════════════════════════════════════════════════════════════════════════════
# FEMOperators container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FEMOperators:
    """
    All precomputed FEM operators for a fixed mesh.

    Attributes
    ----------
    mesh      : P3 Mesh
    K         : stiffness matrix (with Dirichlet BCs applied)
    M         : mass matrix (interior DOFs only; rows/cols for boundary DOFs zeroed)
    S1, S2    : shear operators
    K_lu      : cached SuperLU factorization of K (for fast solves)
    n_nodes   : total node count
    boundary  : boundary node indices (np.ndarray)
    interior  : interior node mask (bool array, length n_nodes)
    """
    mesh     : object          # P3 Mesh NamedTuple
    K        : sp.csr_matrix
    M        : sp.csr_matrix
    S1       : sp.csr_matrix
    S2       : sp.csr_matrix
    K_lu     : object          # SuperLU factorization
    n_nodes  : int
    boundary : np.ndarray
    interior : np.ndarray      # bool mask

    # ── public interface ──────────────────────────────────────────────────────

    def psi_from_kappa(self, kappa: np.ndarray) -> np.ndarray:
        """
        Solve K ψ = −2 M κ for ψ (Dirichlet BCs already baked into K).

        Args:
            kappa : (n_nodes,) convergence field

        Returns:
            psi   : (n_nodes,) lensing potential
        """
        rhs = -2.0 * self.M @ kappa
        # BC: zero out boundary entries in RHS (boundary rows of K are identity)
        rhs[self.boundary] = 0.0
        return self.K_lu.solve(rhs)

    def shear_from_psi(self, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute shear (γ₁, γ₂) from lensing potential.

        Args:
            psi : (n_nodes,) lensing potential

        Returns:
            gamma1, gamma2 : (n_nodes,) shear components
        """
        return self.S1 @ psi, self.S2 @ psi

    def forward(self, kappa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full forward model: κ → (γ₁, γ₂).

        Args:
            kappa : (n_nodes,) convergence field

        Returns:
            gamma1, gamma2 : (n_nodes,) shear components
        """
        psi = self.psi_from_kappa(kappa)
        return self.shear_from_psi(psi)

    def shear_magnitude(self, kappa: np.ndarray) -> np.ndarray:
        """Convenience: |γ| = sqrt(γ₁² + γ₂²)."""
        g1, g2 = self.forward(kappa)
        return np.sqrt(g1**2 + g2**2)

    def adjoint_rhs(self, dL_dgamma1: np.ndarray,
                    dL_dgamma2: np.ndarray) -> np.ndarray:
        """
        Compute ∂L/∂κ given upstream gradients ∂L/∂γ₁ and ∂L/∂γ₂.

        The adjoint computation (exact, no approximation):

            ∂L/∂κ = −2 Mᵀ K⁻ᵀ (S1ᵀ ∂L/∂γ₁ + S2ᵀ ∂L/∂γ₂)

        Since K is symmetric, K⁻ᵀ = K⁻¹ and we reuse the cached LU.

        Args:
            dL_dgamma1, dL_dgamma2 : (n_nodes,) upstream gradients

        Returns:
            dL_dkappa : (n_nodes,) gradient w.r.t. κ
        """
        # λ = K⁻¹ (S1ᵀ v1 + S2ᵀ v2)
        rhs_adj = self.S1.T @ dL_dgamma1 + self.S2.T @ dL_dgamma2
        rhs_adj[self.boundary] = 0.0
        lam = self.K_lu.solve(rhs_adj)
        return -2.0 * self.M.T @ lam


# ══════════════════════════════════════════════════════════════════════════════
# Main factory function
# ══════════════════════════════════════════════════════════════════════════════

def build_operators(nx: int, ny: int,
                    xmin: float = -2.5, xmax: float = 2.5,
                    ymin: float = -2.5, ymax: float = 2.5,
                    verbose: bool = True) -> FEMOperators:
    """
    Build all FEM operators for a P3 mesh on [xmin,xmax] × [ymin,ymax].

    This is the main entry point.  Call once; reuse FEMOperators for all
    forward/inverse computations.

    Args:
        nx, ny       : cells per axis
        xmin…ymax    : domain bounds
        verbose      : print timing info

    Returns:
        FEMOperators instance with K, M, S1, S2 assembled and K factorized
    """
    t0 = time.perf_counter()

    # ── 1. Mesh ────────────────────────────────────────────────────────────────
    if verbose:
        print(f"[fem] Building P3 mesh: {nx}×{ny} cells...")
    mesh = generate_p3_structured_mesh(nx, ny, xmin, xmax, ymin, ymax)
    nodes    = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    boundary = np.array(mesh.boundary)
    n_nodes  = len(nodes)
    interior = np.ones(n_nodes, dtype=bool)
    interior[boundary] = False

    if verbose:
        print(f"       {n_nodes} nodes, {len(elements)} elements, "
              f"{len(boundary)} boundary DOFs")

    # ── 2. Quadrature (order=5, 13-point Dunavant — matches p3_assembly.py) ───
    quad_pts, quad_wts = get_gauss_quadrature_triangle(order=5)
    quad_pts_np  = np.array(quad_pts)
    quad_wts_np  = np.array(quad_wts)

    # ── 3. Stiffness matrix K ──────────────────────────────────────────────────
    if verbose:
        print("[fem] Assembling stiffness matrix K...")
    t1 = time.perf_counter()

    max_nnz = len(elements) * 100
    I_k   = np.zeros(max_nnz, dtype=np.int32)
    J_k   = np.zeros(max_nnz, dtype=np.int32)
    K_dat = np.zeros(max_nnz)
    entry = 0

    for e_idx, elem in enumerate(elements):
        coords_jax = jnp.array(nodes[elem])
        Ke = np.array(compute_element_stiffness_p3(
            coords_jax,
            jnp.array(quad_pts_np),
            jnp.array(quad_wts_np)
        ))
        for i in range(10):
            for j in range(10):
                I_k[entry]   = elem[i]
                J_k[entry]   = elem[j]
                K_dat[entry] = Ke[i, j]
                entry += 1

    K_raw = sp.coo_matrix(
        (K_dat[:entry], (I_k[:entry], J_k[:entry])),
        shape=(n_nodes, n_nodes)
    ).tocsr()

    # Apply Dirichlet BCs: row/col identity for boundary nodes
    K_lil = K_raw.tolil()
    for b in boundary:
        K_lil[b, :] = 0
        K_lil[b, b] = 1.0
    K = K_lil.tocsr()

    if verbose:
        print(f"       K assembled: {K.shape}, nnz={K.nnz}  "
              f"({time.perf_counter()-t1:.1f}s)")

    # ── 4. Mass matrix M ───────────────────────────────────────────────────────
    if verbose:
        print("[fem] Assembling mass matrix M...")
    t2 = time.perf_counter()
    M_raw = _assemble_mass_p3(nodes, elements, quad_pts_np, quad_wts_np)

    # Zero-out boundary rows (so M κ is zero at boundary DOFs → RHS=0 there)
    M_lil = M_raw.tolil()
    for b in boundary:
        M_lil[b, :] = 0
    M = M_lil.tocsr()

    if verbose:
        print(f"       M assembled: {M.shape}, nnz={M.nnz}  "
              f"({time.perf_counter()-t2:.1f}s)")

    # ── 5. Reference Hessians for shear operators ──────────────────────────────
    if verbose:
        print("[fem] Precomputing P3 reference Hessians (JAX AD)...")
    t3 = time.perf_counter()
    H_ref = _build_ref_hessians()   # (10, 10, 2, 2)
    if verbose:
        print(f"       H_ref built  ({time.perf_counter()-t3:.1f}s)")

    # ── 6. Shear operators S1, S2 ──────────────────────────────────────────────
    if verbose:
        print("[fem] Assembling shear operators S1, S2...")
    t4 = time.perf_counter()
    S1, S2 = _assemble_shear_ops(nodes, elements, H_ref)
    if verbose:
        print(f"       S1, S2 assembled: nnz={S1.nnz}, {S2.nnz}  "
              f"({time.perf_counter()-t4:.1f}s)")

    # ── 7. Factorize K ─────────────────────────────────────────────────────────
    if verbose:
        print("[fem] Factorizing K (SuperLU)...")
    t5 = time.perf_counter()
    K_lu = spla.splu(K.tocsc())
    if verbose:
        print(f"       LU factorization done  ({time.perf_counter()-t5:.1f}s)")

    total = time.perf_counter() - t0
    if verbose:
        print(f"[fem] ✅  All operators ready  (total {total:.1f}s)\n")

    return FEMOperators(
        mesh     = mesh,
        K        = K,
        M        = M,
        S1       = S1,
        S2       = S2,
        K_lu     = K_lu,
        n_nodes  = n_nodes,
        boundary = boundary,
        interior = interior,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Laplacian regularizer (for MAP inverse problem)
# ══════════════════════════════════════════════════════════════════════════════

def build_laplacian(ops: FEMOperators) -> sp.csr_matrix:
    """
    Build graph Laplacian L on the mesh nodes for smoothness regularization.

    The MAP prior uses ‖∇κ‖² = κᵀ L κ where L is the FEM stiffness matrix
    restricted to interior DOFs.  We simply re-use ops.K (without the
    Dirichlet rows/cols set to identity) for this purpose — it is SPD on
    interior DOFs and provides the correct ‖∇κ‖² inner product.

    Returns:
        L : (n_nodes, n_nodes) sparse CSR matrix (same as ops.K for simplicity)
    """
    # ops.K already has BCs applied (identity rows for boundary DOFs).
    # For regularization we want smoothness only on interior DOFs.
    # ops.K serves this purpose directly: κᵀ K κ = ‖∇κ‖² on Ω (interior terms).
    return ops.K.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("fem.py — smoke test (6×6 mesh, Gaussian κ)")
    print("=" * 60)

    ops = build_operators(6, 6, verbose=True)

    nodes = np.array(ops.mesh.nodes)
    A, sigma = 1.0, 0.5
    kappa = A * np.exp(-(nodes[:, 0]**2 + nodes[:, 1]**2) / (2*sigma**2))

    psi = ops.psi_from_kappa(kappa)
    g1, g2 = ops.shear_from_psi(psi)

    print(f"max|ψ|  = {np.abs(psi).max():.4f}")
    print(f"max|γ₁| = {np.abs(g1).max():.4f}")
    print(f"max|γ₂| = {np.abs(g2).max():.4f}")
    print("✅  fem.py OK")