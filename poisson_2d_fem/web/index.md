---
title: "2D Poisson Equation: Hybrid Fortran-Python FEM Solver"
date: "2026-02-04"
tags: "Project"
snippet: "Finite element solver for the 2D Poisson equation with Dirichlet boundary conditions - achieving O(h²) convergence in L² norm."
---

## The Problem

We solve the **2D Poisson equation** on the unit square $\Omega = (0,1) \times (0,1)$:

$$-\Delta u = f \quad \text{in } \Omega$$
$$u = 0 \quad \text{on } \partial\Omega$$

where $\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$ is the Laplacian operator and $f(x,y)$ is a known source function.

**Physical Interpretation:**
- **Steady-state heat equation**: $u$ is temperature, $f$ is heat source
- **Electrostatics**: $u$ is electric potential, $f$ is charge distribution  
- **Membrane deflection**: $u$ is vertical displacement under load $f$

**Boundary Conditions:**
- **Homogeneous Dirichlet**: Temperature fixed at zero on all boundaries (e.g., edges held at 0°C)

## Mathematical Formulation

### Weak Form (Brenner & Scott §2.3)

Starting from the strong form $-\Delta u = f$ with $u = 0$ on $\partial\Omega$, multiply by test function $v \in H^1_0(\Omega)$ and integrate by parts:

$$\int_\Omega \nabla u \cdot \nabla v \, dx = \int_\Omega f v \, dx \quad \forall v \in H^1_0(\Omega)$$

**Finite Element Discretization:**

Choose finite-dimensional subspace $V_h \subset H^1_0(\Omega)$ (piecewise linear on triangulation). Find $u_h \in V_h$ such that:

$$a(u_h, v_h) = (f, v_h) \quad \forall v_h \in V_h$$

where:
- Bilinear form: $a(u, v) = \int_\Omega \nabla u \cdot \nabla v \, dx$
- Load functional: $(f, v) = \int_\Omega f v \, dx$

### Expected Convergence (Theorem 4.4.3)

For piecewise linear (P1) triangular elements with $h$ = maximum element diameter:

$$\|u - u_h\|_{L^2(\Omega)} \leq Ch^2 \|u\|_{H^2(\Omega)}$$
$$\|u - u_h\|_{H^1(\Omega)} \leq Ch \|u\|_{H^2(\Omega)}$$

**Goal**: Verify these theoretical convergence rates numerically.

## Interactive Convergence Results

[codeContainer](/poisson_2d_fem/web/scripts/poisson_convergence_viz.js)

## Verification Strategy

### Manufactured Solution

To verify correctness, we use a **manufactured solution** where we choose the exact answer:

$$u_{\text{exact}}(x,y) = \sin(\pi x) \sin(\pi y)$$

Compute the required source term by applying $-\Delta$:

$$f(x,y) = -\Delta u_{\text{exact}} = 2\pi^2 \sin(\pi x) \sin(\pi y)$$

**Verification procedure:**
1. Solve $-\Delta u_h = f$ numerically
2. Compare $u_h$ to $u_{\text{exact}}$ at mesh nodes
3. Compute error norms: $L^2$, $H^1$, $L\infty$
4. Repeat on successively refined meshes
5. Verify convergence rates match theory

## Implementation Architecture

### Hybrid Fortran-Python Design

**Fortran Backend** (Performance-Critical):
- Element-by-element assembly
- Affine element transformations
- Numerical quadrature  
- Linear system solve (LAPACK DPOSV)

**Python Frontend** (Convenience):
- Mesh generation (Triangle library)
- Manufactured solution evaluation
- Error computation and analysis
- Visualization (Matplotlib)

**Interface**: f2py wrapper for seamless integration

### Key Implementation Details

**Stiffness Matrix Assembly** (Fortran):

For each triangular element $K$:

1. **Affine map**: $F_K(\widehat{x}) = B_K \widehat{x} + b$ maps reference triangle to physical element
2. **Gradient transformation**: $\nabla \phi_{\text{phys}} = (B_K^{-1})^T \nabla \phi_{\text{ref}}$  
3. **Local stiffness**: 
$$K_{\text{elem}}(i,j) = \int_K \nabla \phi_i \cdot \nabla \phi_j \, dx = \frac{|det(B_K)|}{2} \cdot \nabla \phi_i^{\text{phys}} \cdot \nabla \phi_j^{\text{phys}}$$

**Load Vector Assembly** (Fortran):

$$F_i = \int_K f \phi_i \, dx \approx |det(B_K)| \sum_{q} w_q f(x_q) \phi_i(\xi_q)$$

Using 3-point Gaussian quadrature for degree-2 accuracy.

## Convergence Results

After some debugging and such, we achieved **theoretical convergence rates**:

| h      | Nodes | Elements | $L^2$ Error  | $L^2$ Rate | $H^1$ Error  | $H^1$ Rate |
|--------|-------|----------|-----------|---------|-----------|---------|
| 0.316  | 13    | 16       | 2.45e-03  | -       | 2.46e-02  | -       |
| 0.224  | 23    | 28       | 1.23e-03  | 2.0     | 1.74e-02  | 1.0     |
| 0.158  | 41    | 64       | 6.15e-04  | 2.0     | 1.23e-02  | 1.0     |
| 0.112  | 77    | 123      | 3.08e-04  | 2.0     | 8.70e-03  | 1.0     |
| 0.079  | 147   | 260      | 1.54e-04  | 2.0     | 6.15e-03  | 1.0     |

- $\boldsymbol{L^2}$ **convergence**: O(h²) as predicted by Theorem 4.4.3  
- $\boldsymbol{H^1}$ **convergence**: O(h) as predicted by theory

**Interpretation**: As we halve the mesh size $h$:
- $L^2$ error decreases by factor of $\sim$4 (quadratic)
- $H^1$ error decreases by factor of $\sim$2 (linear)

This confirms:
1. Implementation is mathematically correct
2. Numerical integration is sufficiently accurate  
3. Boundary conditions are properly enforced
4. No numerical instabilities present



## Future Extensions

### Immediate Next Steps

1. **Non-homogeneous Dirichlet BC**: $u = g(x,y)$ on $\partial\Omega$
2. **Different domains**: L-shaped region (singularity testing)
3. **Higher-order elements**: P2 triangles for O($h^3$) convergence

### Advanced Extensions

4. **Natural/Neumann BC**: $\frac{\partial u}{\partial n} = h$ on boundary
5. **Mixed formulations**: Coupled systems  
6. **Adaptive mesh refinement**: Error-driven h-refinement
7. **Iterative solvers**: CG with multigrid preconditioning
8. **Time-dependent**: Heat equation $\frac{\partial u}{\partial t} - \Delta u = f$



## Theoretical Foundation

This implementation closely follows:

**Brenner & Scott**: *The Mathematical Theory of Finite Element Methods* (3rd ed.)
- Chapter 2: Sobolev spaces and weak formulations
- Chapter 3: Finite element spaces  
- Chapter 4: Convergence theory (Theorem 4.4.3)

**Key Theoretical Results Used:**

1. **Céa's Lemma**: $\|u - u_h\|_E \leq C \inf_{v_h \in V_h} \|u - v_h\|_E$
2. **Approximation estimates**: $\inf_{v_h \in V_h} \|u - v_h\|_{H^1} \leq Ch \|u\|_{H^2}$
3. **Aubin-Nitsche duality**: Boosts to $\|u - u_h\|_{L^2} \leq Ch^2 \|u\|_{H^2}$



## Performance Characteristics

**Assembly complexity**: O($n_{\text{elements}}$)  
**Solve complexity**: O($n_{\text{nodes}}^3$) for dense LAPACK (could be improved with sparse solvers)

**Typical timings** (n=20,000 elements):
- Mesh generation: $\sim$0.5s (Triangle library)
- Fortran assembly: $\sim$0.05s
- LAPACK solve: $\sim$2s (dense solver - bottleneck!)
- Error computation: $\sim$0.1s

**Bottleneck**: Dense direct solver. For production, use:
- Sparse iterative solvers (CG, GMRES)
- Multigrid preconditioning
- Can reduce solve time to $\sim$0.01s for this problem size



## Comparison: 1D vs 2D FEM

| Aspect | 1D FEM (B&S Ch 0) | 2D FEM (B&S Ch 4) |
|--------|-------------------|-------------------|
| **Elements** | Line segments | Triangles |
| **DOFs** | n nodes | n_nodes (unstructured) |
| **Basis** | Piecewise linear (hat functions) | P1 on triangles |
| **Stiffness** | Tridiagonal | Sparse, irregular pattern |
| **Assembly** | O(n) simple | O($n_{text{elem}}$) with transformations |
| **Mesh** | 1D array | Triangle library |
| **Convergence** | O($h^2$) in $L^2$ | O($h^2$) in $L^2$ |

**New Challenges in 2D:**
- Affine element transformations $F_K$
- Gradient transformations $(B_K^{-1})^T$
- Numerical quadrature (3-point Gauss)
- Unstructured mesh data structures
- More complex boundary identification



## Acknowledgments

This project was developed as part of the Computational Physics Independent Study Project (ISP) at Worcester Polytechnic Institute. 

**Course**: PH 4000 - Computational Physics ISP  
**Institution**: Worcester Polytechnic Institute  
**Professor**: Dr. William Sanguinet  
**Date**: February 2026