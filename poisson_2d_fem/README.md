# 2D Poisson Equation FEM Solver

For a complete Linux toolchain, use the repository’s [Nix environment](../docs/NIX.md): `nix develop`, then `comphys-build`.

A P1 triangular-element solver with Fortran assembly and a Python driver.

---

## Mathematical Foundation

Solves the Poisson equation on 2D domains:

$$-\Delta u = f \text{ in } \Omega, \quad u = g \text{ on } \partial\Omega$$

**Weak Formulation** (Brenner & Scott §2.3):

Find $u \in H^1_g(\Omega)$ such that:
$$\int_\Omega \nabla u \cdot \nabla v \, dx = \int_\Omega f v \, dx \quad \forall v \in H^1_0(\Omega)$$

**Discretization:**
- P1 (piecewise linear) triangular elements
- Affine element transformations
- Numerical quadrature for load vector
- Direct solver (LAPACK DPOSV)

**Expected Convergence** (Theorem 4.4.3 from Brenner & Scott):
- $\|u - u_h\|_{L^2} = O(h^2)$
- $\|u - u_h\|_{H^1} = O(h)$

---

## Features

- **Arbitrary 2D domains** via Triangle mesh generator  
- **Manufactured solution verification**  
- **Hybrid architecture:** Fortran (assembly/solve) + Python (driver/visualization)  
- **Solution and convergence plots**
- **Convergence rate testing** with automatic refinement  

---

## Quick Start

Run these commands from `poisson_2d_fem/`.

### 1. Build
```bash
make build
```

### 2. Run Convergence Study
```bash
make test
```

### 3. Python API

Start Python in `poisson_2d_fem/python/`:

```python
from fem_solver import PoissonSolver2D
from manufactured_solutions import SineSolution

# Setup
mms = SineSolution()
solver = PoissonSolver2D('unit_square', max_area=0.01)

# Solve
u = solver.solve()

# Compute errors
L2_error, H1_error, Linf_error = solver.compute_errors(mms.u_exact)

# Visualize
from visualization import plot_solution, plot_mesh
plot_solution(solver.mesh, u)
plot_mesh(solver.mesh)
```

---

## Files

| Path | Contents |
|---|---|
| `fortran/` | P1 assembly, boundary conditions, LAPACK solver, and f2py signature |
| `python/` | Mesh generation, manufactured solutions, convergence study, and plots |
| [THEORY.md](THEORY.md) | Weak formulation and discretization |
| [notes/poisson.md](notes/poisson.md) | Derivation and convergence discussion |
| [results/convergence_snapshot.json](results/convergence_snapshot.json) | Saved numerical snapshot |

## Implementation Details

### Fortran assembly and solve

**Stiffness Matrix Assembly:**
```fortran
! For each element K:
K_elem(i,j) = Area(K) × ∇φᵢ · ∇φⱼ
            = |det(B_K)|/2 × (B_K⁻ᵀ ∇φᵢ_ref) · (B_K⁻ᵀ ∇φⱼ_ref)
```

**Load Vector Assembly:**
```fortran
! Numerical quadrature:
F_i = ∫_K f φᵢ dx ≈ |det(B_K)| × Σ wq f(xq) φᵢ(xq)
```

**Critical Details:**
- Element area = `|det(B_K)| / 2` (not `|det(B_K)|`)
- Gradient transformation: `∇φ_phys = (B_K⁻¹)ᵀ ∇φ_ref`
- 3-point Gauss quadrature for degree-2 accuracy

### Python Driver

- Mesh generation (Triangle library)
- Error computation (L², H¹, L∞ norms)
- Matplotlib visualizations
- Manufactured solution framework

---

## Verification Results

The saved table below has the expected L² and H¹ slopes, but `_compute_H1_seminorm` is a placeholder. Its H¹ values cannot verify convergence. See the [full snapshot](notes/convergence_snapshot.md) for the Linf values and limitations.

Convergence study with manufactured solution `u = sin(πx)sin(πy)`:

| h      | Nodes | L² error  | Rate | H¹ error  | Rate |
|--------|-------|-----------|------|-----------|------|
| 0.316  | 13    | 2.45e-03  | -    | 2.46e-02  | -    |
| 0.224  | 23    | 1.23e-03  | 2.0  | 1.74e-02  | 1.0  |
| 0.158  | 41    | 6.15e-04  | 2.0  | 1.23e-02  | 1.0  |
| 0.112  | 77    | 3.08e-04  | 2.0  | 8.70e-03  | 1.0  |
| 0.079  | 147   | 1.54e-04  | 2.0  | 6.15e-03  | 1.0  |

The theoretical targets for a sufficiently smooth solution are O(h²) in L² and O(h) in H¹.

---

## Manufactured Solutions

The manufactured-solutions module defines the following cases. The Fortran solve currently uses the sine source and homogeneous boundary data; the Python `f_source` and `g_boundary` arguments do not change that behavior.

**Sine Solution** (smooth):
```python
u(x,y) = sin(πx) sin(πy)
f(x,y) = 2π² sin(πx) sin(πy)
```

**Polynomial Solution**:
```python
u(x,y) = x(1-x) y(1-y)
f(x,y) = 2x(1-x) + 2y(1-y)
```

---

## Dependencies

**Python:**
- NumPy
- Matplotlib
- triangle (mesh generation)

**Fortran:**
- Modern Fortran compiler (gfortran, ifort)
- OpenBLAS or MKL (LAPACK)
- f2py (comes with NumPy)

---

## References

- **Brenner & Scott**: *The Mathematical Theory of Finite Element Methods* (3rd ed.)
- **Triangle**: Shewchuk's 2D mesh generator
- **LAPACK**: Linear algebra package

---

## Author

Adam Field  
Worcester Polytechnic Institute  
Computational Physics ISP
## Notes

- [Derivation and original results](notes/poisson.md)
- [Full saved convergence snapshot](notes/convergence_snapshot.md)
- [Weak formulation](THEORY.md)
