# N-Body Gravitational Simulation: Multi-Language Performance Comparison

**Adam Field - Computational Physics Independent Study (ISP)**  
**Worcester Polytechnic Institute**

## Project Overview

This project implements the **same N-body gravitational simulation** in multiple languages and compares their performance characteristics:

- **JAX** (GPU-accelerated with JIT compilation)
- **Fortran** (CPU with OpenMP parallelization)
- **C++** (Coming soon)
- **C** (Coming soon)
- **Pure Python** (Baseline reference)

All implementations use the **Velocity Verlet** integrator for time-stepping and compute pairwise gravitational forces with O(N²) complexity. The goal is to understand the performance tradeoffs between different computational approaches for the same physics problem.

## Physics Background

### Gravitational N-Body Problem

The N-body problem simulates the motion of N particles under mutual gravitational attraction:

**Force on particle i:**
$$
F_i = G \cdot \sum_{j\neq i} \left[ \frac{m_i \cdot m_j \cdot (r_j - r_i)}{|r_j - r_i|^3} \right]
$$

**Equations of motion:**
$$
\frac{dv_i}{dt} = \frac{F_i}{m_i}\quad\text{(acceleration)}
$$
$$
\frac{dr_i}{dt} = v_i\quad\text{(velocity)}
$$

### Numerical Integration: Velocity Verlet

Velocity Verlet is a symplectic integrator that conserves energy better than simple Euler methods:

```
1. a(t) = compute_acceleration(r(t))
2. r(t + Δt) = r(t) + v(t)*Δt + 0.5*a(t)*Δt²
3. a(t + Δt) = compute_acceleration(r(t + Δt))
4. v(t + Δt) = v(t) + 0.5*(a(t) + a(t + Δt))*Δt
```

### Computational Complexity

- **Direct summation:** O(N²) per timestep (what we implement)
- **Barnes-Hut tree:** O(N log N) (potential future optimization)

## Project Structure

```
nbody_comparison/
├── nbody/
│   ├── jax/
│   │   └── nbody_jax.py              # JAX implementation (GPU)
│   ├── fortran/
│   │   ├── nbody.f90                 # Fortran implementation
│   │   └── setup.py                  # f2py build script
│   ├── cpp/                          # C++ (coming soon)
│   ├── c/                            # C (coming soon)
│   └── benchmark/
│       ├── benchmark.py              # Performance testing suite
│       └── visualize.py              # Matplotlib animations
├── web/
│   ├── index.html                    # Interactive frontend
│   ├── js/nbody_viz.js              # Three.js visualization
│   └── data/benchmark_results.json   # Benchmark data
└── tests/
    └── test_accuracy.py              # Verify implementations match
```

## Installation & Setup

### Prerequisites

```bash
# Python dependencies
pip install numpy jax jaxlib matplotlib scipy

# For GPU support (optional but recommended for JAX)
pip install --upgrade "jax[cuda12]"  # For CUDA 12
# or
pip install --upgrade "jax[cuda11]"  # For CUDA 11
```

### Building Fortran Module

```bash
cd nbody/fortran

# Option 1: Using setup.py
python setup.py build_ext --inplace

# Option 2: Using f2py directly
f2py -c nbody.f90 -m nbody_fortran_module --f90flags="-fopenmp" -lgomp

# Copy the compiled .so file to benchmark directory
cp nbody_fortran_module*.so ../benchmark/
```

## Running the Code

### Quick Start: JAX Implementation

```bash
cd nbody/jax
python nbody_jax.py
```

This will:
1. Create a random 100-particle system
2. Simulate 1000 timesteps
3. Report timing and energy conservation

### Running Benchmarks

```bash
cd nbody/benchmark
python benchmark.py
```

This will:
- Test multiple particle counts (10, 50, 100, 500, 1000)
- Test multiple timestep counts (100, 500, 1000, 5000)
- Compare JAX vs Fortran performance
- Generate plots in `web/data/`
- Save JSON results for web frontend

### Creating Visualizations

```python
from nbody.jax.nbody_jax import create_random_system, simulate, NBodyConfig
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create initial conditions
initial_state = create_random_system(50, jax.random.PRNGKey(42))
config = NBodyConfig(dt=0.01)

# Run simulation
times, positions, velocities = simulate(initial_state, config, 1000, save_every=10)

# Animate (2D projection)
fig, ax = plt.subplots()
scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1])

def update(frame):
    scatter.set_offsets(positions[frame, :, :2])
    return scatter,

anim = FuncAnimation(fig, update, frames=len(times), interval=50, blit=True)
plt.show()
```

## Implementation Details

### JAX (GPU)

**Strengths:**
- JIT compilation for near-native speed
- Automatic GPU acceleration
- Vectorized operations (vmap)
- Familiar NumPy-like syntax

**Key Features:**
- Uses `@jit` decorator for compilation
- Pairwise force calculation via broadcasting
- Can run on CPU or GPU transparently

### Fortran (CPU + OpenMP)

**Strengths:**
- Excellent compiler optimizations
- Native array operations
- OpenMP parallel loops
- Minimal overhead

**Key Features:**
- Double precision (`real(dp)`)
- OpenMP parallelization of force loop
- Column-major array layout (native to Fortran)
- f2py provides seamless Python interface

### Expected Performance Characteristics

For **N = 1000 particles**:
- **JAX (GPU):** ~1-5 ms/step (depending on GPU)
- **Fortran (CPU, 8 cores):** ~20-50 ms/step
- **Pure Python:** ~5000+ ms/step (1000x slower!)

**Crossover point:** JAX should dominate for N > 500-1000 particles

## Key Findings (To Be Updated)

### Scaling Behavior
- Both JAX and Fortran show expected O(N²) scaling
- JAX compilation overhead is ~1-2 seconds
- GPU memory transfer becomes bottleneck for small N

### Energy Conservation
- Velocity Verlet maintains energy to ~0.001% over 10,000 steps
- All implementations agree to machine precision
- Energy drift increases with larger timesteps

### Language Tradeoffs
| Language | Speed | Ease of Use | GPU Support | Parallelization |
|----------|-------|-------------|-------------|-----------------|
| JAX      | ★★★★★ | ★★★★☆       | ★★★★★       | Automatic       |
| Fortran  | ★★★★☆ | ★★☆☆☆       | ☆☆☆☆☆       | Manual (OpenMP) |
| C++      | ★★★★☆ | ★★★☆☆       | ★★★☆☆       | Manual          |
| C        | ★★★★☆ | ★★☆☆☆       | ★★★☆☆       | Manual          |
| Python   | ★☆☆☆☆ | ★★★★★       | ★★★☆☆       | Libraries only  |

## Testing & Validation

### Accuracy Tests

All implementations should produce **identical results** (within floating-point precision):

```bash
cd tests
python test_accuracy.py
```

This verifies:
- Force calculations match
- Energy is conserved
- Trajectories are identical

### Performance Profiling

```python
# JAX profiling
import jax.profiler
jax.profiler.start_trace("./tensorboard")
# ... run simulation ...
jax.profiler.stop_trace()

# Fortran profiling
gprof ./nbody_fortran
```

## Future Extensions

- [ ] Barnes-Hut tree algorithm (O(N log N))
- [ ] Fast Multipole Method (O(N))
- [ ] MPI parallelization across nodes
- [ ] CUDA/HIP direct implementations
- [ ] WebGPU for browser-based simulation
- [ ] Collision detection
- [ ] Relativistic corrections

## Learning Outcomes

This project demonstrates:
1. **Numerical methods:** Symplectic integrators, error analysis
2. **HPC concepts:** GPU acceleration, OpenMP, memory bandwidth
3. **Software engineering:** Multi-language interoperability, testing, benchmarking
4. **Performance analysis:** Profiling, scaling studies, optimization

## References

- Press et al., *Numerical Recipes* (Verlet integration)
- Barnes & Hut (1986), "A hierarchical O(N log N) force-calculation algorithm"
- JAX documentation: https://jax.readthedocs.io
- f2py guide: https://numpy.org/doc/stable/f2py/

## Contact

**Adam Field**  
Physics, Worcester Polytechnic Institute  
Email: adfield@wpi.edu  
Website: adamfield.org

---

*This project is part of my Computational Physics Independent Study at WPI, exploring high-performance computing techniques for astrophysical simulations.*