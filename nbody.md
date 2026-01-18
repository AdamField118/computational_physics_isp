# N-Body Simulation: Multi-Language Performance Comparison

## Project Overview
Implement identical N-body gravitational simulations in JAX (GPU), Fortran, C++, and C.
Wrap all in Python, benchmark performance, and visualize results with HTML frontend.

## Directory Structure
```
computational_physics_isp/
├── README.md
├── nbody/
│   ├── __init__.py
│   ├── jax/
│   │   ├── __init__.py
│   │   └── nbody_jax.py          # Pure JAX implementation
│   ├── fortran/
│   │   ├── nbody.f90              # Fortran implementation
│   │   ├── Makefile               # Build system
│   │   └── setup.py               # f2py wrapper config
│   ├── cpp/
│   │   ├── nbody.cpp              # C++ implementation
│   │   ├── nbody.h                # Header
│   │   ├── Makefile
│   │   └── setup.py               # pybind11 wrapper
│   ├── c/
│   │   ├── nbody.c                # C implementation
│   │   ├── nbody.h
│   │   ├── Makefile
│   │   └── setup.py               # ctypes/cffi wrapper
│   ├── python/
│   │   ├── __init__.py
│   │   └── nbody_python.py        # Pure Python (baseline)
│   └── benchmark/
│       ├── benchmark.py           # Performance testing
│       ├── visualize.py           # Matplotlib animations
│       └── compare.py             # Generate comparison plots
├── web/
│   ├── index.html                 # Main frontend
│   ├── js/
│   │   └── nbody_viz.js          # Three.js or WebGL or WebGPU visualization
│   ├── css/
│   │   └── styles.css
│   └── data/
│       └── benchmark_results.json # Generated from Python
├── tests/
│   └── test_accuracy.py           # Verify all implementations match
├── notebooks/
│   └── analysis.ipynb             # Jupyter notebook for exploration
└── docs/
    ├── theory.md                  # N-body physics & numerical methods
    └── results.md                 # Performance analysis writeup
```

## Implementation Plan

### Phase 1: Core Physics & JAX Implementation (Week 1)
- [ ] Define N-body equations (Newton's law of gravitation)
- [ ] Choose numerical integrator (Velocity Verlet or RK4)
- [ ] Implement in JAX with JIT compilation
- [ ] Create initial conditions generator (random, solar system, galaxy, etc.)
- [ ] Basic visualization in Python

### Phase 2: Compiled Language Implementations (Week 2-3)
- [ ] Fortran implementation with OpenMP parallelization
- [ ] C implementation (baseline, then OpenMP)
- [ ] C++ implementation with modern features
- [ ] f2py wrapper for Fortran
- [ ] pybind11 for C++
- [ ] ctypes/CFFI for C

### Phase 3: Benchmarking & Validation (Week 4)
- [ ] Accuracy tests (all implementations produce same results)
- [ ] Performance benchmarks:
  - Vary N (particles): 100, 1000, 10000
  - Vary timesteps: 100, 1000, 10000
  - Time per step, total runtime, memory usage
- [ ] Profile each implementation
- [ ] Generate comparison plots

### Phase 4: Visualization & Frontend (Week 5)
- [ ] Python-based animation (matplotlib/plotly)
- [ ] HTML/Three.js interactive 3D visualization
- [ ] Dashboard showing benchmark results
- [ ] Real-time parameter adjustment (if possible)

### Phase 5: Documentation & Writeup (Week 6)
- [ ] Theory documentation
- [ ] Code documentation (docstrings, comments)
- [ ] Results analysis writeup
- [ ] Create presentation/poster

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

## Success Metrics
1. All implementations produce numerically identical results (within floating-point error)
2. JAX GPU implementation is fastest for N > 1000
3. Fortran/C/C++ competitive for small N
4. Clean, documented code that demonstrates language strengths
5. Compelling visualizations showing chaotic dynamics

## Questions to Explore
- At what N does GPU become advantageous?
- How do OpenMP parallel CPU implementations compare?
- Does compiler optimization matter? (gcc -O0 vs -O3)
- Memory bandwidth vs compute bound?
- Single vs double precision tradeoffs?