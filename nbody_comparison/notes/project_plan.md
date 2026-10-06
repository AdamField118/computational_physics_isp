# N-Body Simulation: Multi-Language Performance Comparison

## Project Overview
Implement identical N-body gravitational simulations in JAX (GPU), Fortran, C++, and C.
Wrap all in Python, benchmark performance, and save plots and animations for comparison.

## Directory Structure

See the [project README](../README.md) for the current layout. The plan covers implementations, correctness tests, benchmark data, and visual analysis. It also proposed an exploratory notebook, a separate comparison script, theory notes, and a results write-up; these were planned outputs rather than files present in the original directory sketch.

## Implementation Plan

### Phase 1: Core Physics & JAX Implementation (Week 1)
-  Define N-body equations (Newton's law of gravitation)
-  Choose numerical integrator (Velocity Verlet or RK4)
-  Implement in JAX with JIT compilation
-  Create initial conditions generator (random, solar system, galaxy, etc.)
-  Basic visualization in Python

### Phase 2: Compiled Language Implementations (Week 2-3)
-  Fortran implementation with OpenMP parallelization
-  C implementation (baseline, then OpenMP)
-  C++ implementation with modern features
-  f2py wrapper for Fortran
-  pybind11 for C++
-  ctypes/CFFI for C

### Phase 3: Benchmarking & Validation (Week 4)
-  Accuracy tests (all implementations produce same results)
-  Performance benchmarks:
  - Vary N (particles): 100, 1000, 10000
  - Vary timesteps: 100, 1000, 10000
  - Time per step, total runtime, memory usage
-  Profile each implementation
-  Generate comparison plots

### Phase 4: Visualization (Week 5)
-  Python-based animation (matplotlib/plotly)
-  Offline 3D trajectory visualization
-  Figures and tables showing benchmark results
-  Parameter comparisons across saved runs

### Phase 5: Documentation & Writeup (Week 6)
-  Theory documentation
-  Code documentation (docstrings, comments)
-  Results analysis writeup
-  Create presentation/poster

## N-Body Physics Equations

### Gravitational Force
For particle i with mass m_i at position r_i:

F_i = G * Σ(j≠i) [ (m_i * m_j * (r_j - r_i)) / |r_j - r_i|³ ]

### Equations of Motion
dv_i/dt = F_i / m_i
dr_i/dt = v_i

### Numerical Integration (Velocity Verlet)
r(t + Δt) = r(t) + v(t)*Δt + 0.5*a(t)*Δt²
v(t + Δt) = v(t) + 0.5*(a(t) + a(t + Δt))*Δt

## Computational Complexity
- Direct summation: O(N²) per timestep
- Future optimization: Barnes-Hut tree O(N log N)

## Checks

Compare forces and trajectories at matched precision, measure energy drift as the timestep decreases, and locate the CPU/GPU crossover from timings.

## Questions to Explore
- At what N does GPU become advantageous?
- How do OpenMP parallel CPU implementations compare?
- Does compiler optimization matter? (gcc -O0 vs -O3)
- Memory bandwidth vs compute bound?
- Single vs double precision tradeoffs?