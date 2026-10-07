# N-Body Gravitational Simulation: Multi-Language Performance Comparison

For a complete Linux toolchain, use the repository’s [Nix environment](../docs/NIX.md): `nix develop`, then `comphys-build`.

**Adam Field - Computational Physics Independent Study (ISP)**  
**Worcester Polytechnic Institute**

## Implementations

The benchmark compares seven implementations of the same gravitational N-body problem:

- **JAX** (GPU-accelerated with JIT compilation)
- **Fortran** (CPU with OpenMP parallelization)
- **C++** (pybind11 wrapper)
- **C** (ctypes wrapper)
- **Rust** (PyO3 wrapper)
- **Julia** (Python wrappers)
- **Pure Python** (Baseline reference)

All implementations use the **Velocity Verlet** integrator for time-stepping and compute pairwise gravitational forces with O(N²) complexity. Timings include each implementation’s Python wrapper.

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
- **Barnes-Hut tree:** O(N log N)

## Project Structure

| Path | Contents |
|---|---|
| `nbody/` | Implementations and benchmark scripts |
| `tests/` | Cross-implementation accuracy checks |
| `results/` | Saved numerical data, plots, and summaries |
| `notes/benchmark.md` | Physics and benchmark discussion |
| `notes/numerical_setup.md` | Equations and benchmark comparisons |
| `notes/simulation_example.md` | Small-system demonstration algorithm |

## Build and run

Run from the repository root inside the Nix shell:

```bash
comphys-build nbody
comphys-build julia
python nbody_comparison/nbody/jax/nbody_jax.py
python nbody_comparison/tests/test_accuracy.py
```

The JAX example runs a seeded 100-particle system and reports timing and energy drift. JAX uses the CPU in the default shell; see [GPU setup](../docs/NIX.md#gpu-use) for CUDA.

To rerun the benchmark:

```bash
python nbody_comparison/nbody/benchmark/benchmark.py
```

The driver uses 10, 50, 100, 500, and 1000 particles at 1000 steps each. It runs the available implementations, reports missing modules, and replaces `nbody_comparison/results/benchmark_results.json`. Keep a copy of that file if you want to compare against the saved run.

Plot the saved data without rerunning the simulation:

```bash
python nbody_comparison/nbody/benchmark/visualize.py nbody_comparison/results/benchmark_results.json
python nbody_comparison/nbody/benchmark/analyze_benchmarks.py
```

### Creating Visualizations

Run this example from `nbody_comparison/`. To save an animation in the headless Nix shell, use `anim.save("trajectory.gif", writer="pillow")` in place of `plt.show()`.

```python
import jax
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

## Implementation details

JAX broadcasts the pairwise displacement array and compiles the force calculation with `jit`. It runs on either CPU or GPU. The Fortran implementation uses double precision and OpenMP over particles; f2py exposes its routines to Python. C uses ctypes, C++ uses pybind11, Rust uses PyO3, and Julia has subprocess and PyJulia wrappers.

## Saved Results

Timings and energy drift are recorded in `results/benchmark_results.json`. Compare implementations at the same particle count and number of steps; the GPU crossover depends on both the hardware and the CPU implementation.

## Numerical checks

`tests/test_accuracy.py` compares energies, single steps, and trajectories for the available Python, JAX, C, C++, and Fortran implementations. It also reports energy drift over 1000 steps. Read the available-implementation list: missing modules are skipped, and this script does not cover Rust or Julia.

## References

- Press et al., *Numerical Recipes* (Verlet integration)
- Barnes & Hut (1986), "A hierarchical O(N log N) force-calculation algorithm"
- JAX documentation: https://jax.readthedocs.io
- f2py guide: https://numpy.org/doc/stable/f2py/

## Contact

**Adam Field**  
Physics, Worcester Polytechnic Institute  
Email: adfield@wpi.edu
