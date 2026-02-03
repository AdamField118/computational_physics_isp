# 2D Poisson Equation FEM Solver

Educational finite element solver for the 2D Poisson equation following *Brenner & Scott*.

## Mathematical Formulation

Solve: $-\Delta u = f$ in $\Omega$ with $u = g$ on $\partial\Omega$

**Weak form**: Find $u \in H^1_g(\Omega)$ such that
$$\int_\Omega \nabla u \cdot \nabla v \, dx = \int_\Omega f v \, dx$$

See [THEORY.md](THEORY.md) for complete derivation.

## Quick Start

### 1. Build
```bash
./build.sh
```

### 2. Run Example
```python
from python.fem_solver import PoissonSolver2D
from python.manufactured_solutions import SineSolution

mms = SineSolution()
solver = PoissonSolver2D(max_area=0.01)
u = solver.solve(mms.f_source)
solver.mesh.plot()
```

### 3. Convergence Study
```bash
cd python
python convergence_study.py
```

## Features
- P1 triangular elements
- Arbitrary 2D domains via Triangle
- Manufactured solution verification
- Convergence rate testing (O(h²) in L², O(h) in H¹)
- Direct solver (LAPACK)

## Project Structure
See [docs/implementation_guide.md](docs/implementation_guide.md)