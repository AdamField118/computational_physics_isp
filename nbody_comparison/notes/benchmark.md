# N-Body Gravitational Simulation

Plots are generated locally; see the [figure instructions](../../docs/GENERATED_FILES.md).

Date: 2026-01-20

Multi-language performance comparison of N-body gravitational simulations across JAX (GPU), Fortran, Rust, Julia, C++, C, and Python with timing and energy-drift measurements.

## Project Overview

This project compares seven implementations of the same gravitational N-body problem, including NumPy and JAX versions in Python.

### Objectives

- Implement the same N-body physics in multiple languages
- Compare raw performance across CPU and GPU architectures  
- Validate numerical accuracy across all implementations

### Implementations

- **JAX (GPU)** - GPU-accelerated with JIT compilation
- **Fortran (OpenMP)** - CPU parallel with OpenMP directives
- **Rust** - PyO3 wrapper
- **Julia** - PyJulia wrapper
- **C++** - Modern C++ with pybind11 wrapper
- **C** - Pure C with ctypes wrapper
- **Python (NumPy)** - Baseline pure Python/NumPy

### Results in the Original Write-Up

The tables below are from the original write-up and differ from the checked-in `../results/benchmark_results.json`. Use that file for comparisons from the saved run.

At **N=1000 particles**, the original table reports:
- JAX (GPU) is **1384× faster** than Python
- JAX (GPU) is **17× faster** than Fortran (OpenMP)
- The crossover depends on the CPU implementation and lies between the sampled particle counts

---

## Saved Results

[Measured results](../results/benchmark_results.json), [performance summary](../results/performance_summary.md), and [illustrative sample data](../results/illustrative_sample.json). The sample data are illustrative, not measurements.

---

## Physics & Theory

### The N-Body Problem

The gravitational N-body problem involves computing the trajectories of N particles under mutual gravitational attraction. For N ≥ 3, no general closed-form solution exists—numerical integration is required.

### Gravitational Force

Newton's Law of Universal Gravitation:

$$
\vec{F}_{ij} = -G \frac{m_i m_j}{r_{ij}^2} \hat{r}_{ij}
$$

Total force on particle $i$:

$$
\vec{F}_i = G m_i \sum_{j \neq i} \frac{m_j (\vec{r}_j - \vec{r}_i)}{|\vec{r}_j - \vec{r}_i|^3}
$$

### Equations of Motion

$$
\frac{d\vec{v}_i}{dt} = \vec{a}_i = \frac{\vec{F}_i}{m_i}
$$

$$
\frac{d\vec{r}_i}{dt} = \vec{v}_i
$$

This is a system of **6N coupled ODEs** (3N positions + 3N velocities).

### Velocity Verlet Integration

We use the **Velocity Verlet** algorithm—a symplectic, time-reversible integrator:

$$
\vec{r}_i(t + \Delta t) = \vec{r}_i(t) + \vec{v}_i(t) \Delta t + \frac{1}{2} \vec{a}_i(t) \Delta t^2
$$

$$
\vec{a}_i(t + \Delta t) = \text{compute\_acceleration}(\vec{r}(t + \Delta t))
$$

$$
\vec{v}_i(t + \Delta t) = \vec{v}_i(t) + \frac{1}{2} [\vec{a}_i(t) + \vec{a}_i(t + \Delta t)] \Delta t
$$

**Why Velocity Verlet?**
- Second-order accurate: $O(\Delta t^2)$
- Symplectic: preserves Hamiltonian structure
- Energy error should be checked as the timestep decreases
- Time-reversible: running backwards recovers initial conditions

### Softening Parameter

To avoid singularities when particles approach each other:

$$
\vec{F}_{ij} = -G \frac{m_i m_j (\vec{r}_j - \vec{r}_i)}{(|\vec{r}_j - \vec{r}_i|^2 + \epsilon^2)^{3/2}}
$$

where $\epsilon = 0.1$ is the softening length.

### Computational Complexity

Direct summation: **O(N²)** per timestep

For each particle, we compute forces from all other particles:
- N=100: 9,900 force pairs
- N=1000: 999,000 force pairs

The pairwise calculation exposes parallel work for the GPU.

---

## Performance Results

### Benchmark Configuration

**System:** WPI Turing Cluster
- CPU: 12 cores
- GPU: NVIDIA L40S
- Python: 3.12.12
- JAX: 0.7.2 with CUDA 12.9

**Parameters:**
- Timestep: $\Delta t = 0.01$
- Gravitational constant: $G = 1.0$
- Softening: $\epsilon = 0.1$
- Test cases: N = {10, 50, 100, 500, 1000}
- Timesteps per benchmark: 1000

### Performance Summary

| Implementation      | N=10  | N=50  | N=100 | N=500  | N=1000  |
|---------------------|-------|-------|-------|--------|---------|
| **JAX (GPU)**       | 0.084 | 0.096 | 0.125 | 0.088  | 0.080   |
| **Fortran (OpenMP)**| 0.015 | 0.020 | 0.034 | 0.394  | 1.373   |
| **Rust**            | 0.012 | 0.018 | 0.030 | 0.350  | 1.250   |
| **Julia**           | 0.013 | 0.019 | 0.032 | 0.370  | 1.300   |
| **C++**             | 0.001 | 0.017 | 0.101 | 1.705  | 6.622   |
| **C**               | 0.001 | 0.014 | 0.085 | 1.378  | 5.508   |
| **Python (NumPy)**  | 0.035 | 0.225 | 1.257 | 27.46  | 110.84  |

*Time per step in milliseconds (ms)*

### Speedup Analysis

| Implementation      | N=10 | N=50 | N=100 | N=500 | N=1000 |
|---------------------|------|------|-------|-------|--------|
| **JAX (GPU)**       | 0.4× | 2.3× | 10×   | 312×  | 1384×  |
| **Fortran (OpenMP)**| 2.3× | 11×  | 37×   | 70×   | 81×    |
| **Rust**            | 2.9× | 13×  | 42×   | 78×   | 89×    |
| **Julia**           | 2.7× | 12×  | 39×   | 74×   | 85×    |
| **C++**             | 54×  | 14×  | 12×   | 16×   | 17×    |
| **C**               | 63×  | 16×  | 15×   | 20×   | 20×    |

*Speedup factors relative to Python (NumPy)*

The GPU timing varies little over these problem sizes, while CPU timings rise with particle count. The crossover differs between CPU implementations; these measurements do not establish a hardware-independent threshold.

### Energy Conservation

The table reports relative energy drift:

| N    | JAX   | Fortran | Rust  | Julia | C++   | C     | Python |
|------|-------|---------|-------|-------|-------|-------|--------|
| 10   | 0.002%| 0.00002%| 0.00002%| 0.00002%| 0.00002%| 0.00002%| 0.00002%|
| 100  | 0.046%| 0.022%  | 0.022%| 0.022%| 0.022%| 0.022%| 0.022% |
| 1000 | 3.3%  | 3.9%    | 3.9%  | 3.2%  | 3.9%  | 4.3%  | 4.3%   |

The larger drift at N=1000 needs a timestep-convergence check before drawing conclusions about accuracy.

---

## Interpreting the Timings

These timings include each implementation's wrapper and execution strategy. They do not isolate compiler optimizations, memory traffic, or Python-call overhead. Separating those costs would require profiling the individual kernels and transfers.

## Trajectory Visualizations

### Energy Conservation Plot
Figure: Energy conservation over time. Generated locally as `../results/energy_conservation.png`.

*Kinetic, potential, and total energy over the integration.*

### Scaling Analysis
Figure: Scaling analysis across implementations. Generated locally as `../results/comprehensive_scaling.png`.

*Four-panel analysis: time per step, speedup, energy conservation, and performance at N=1000*

### GPU Crossover Analysis
Figure: GPU vs CPU crossover. Generated locally as `../results/crossover_analysis.png`.

*GPU and CPU timings across the sampled particle counts.*

---

## Small-System Demonstration

[Initial conditions and update algorithm](simulation_example.md) document the separate illustrative simulation.

---

## Implementation Details

### JAX (GPU) Implementation

```python
@jax.jit
def compute_forces(positions, masses, G, softening):
    """Vectorized force computation on GPU"""
    # Compute all pairwise displacements (N, N, 3)
    displacements = positions[None, :, :] - positions[:, None, :]
    
    # Distances with softening (N, N)
    distances = jnp.sqrt(
        jnp.sum(displacements**2, axis=2) + softening**2
    )
    
    # Forces (N, N, 3)
    forces = G * displacements / distances[:, :, None]**3
    
    # Total acceleration on each particle (N, 3)
    accelerations = jnp.sum(
        forces * masses[None, :, None], axis=1
    )
    
    return accelerations
```

**Key features:**
- `@jax.jit`: Just-in-time compilation to GPU kernels
- Vectorization: All force pairs computed simultaneously
- Automatic differentiation available (not used here)

### Fortran (OpenMP) Implementation

```fortran
subroutine compute_forces(positions, velocities, masses, &
                         n_particles, G, softening, accelerations)
    integer, intent(in) :: n_particles
    real(8), intent(in) :: positions(3, n_particles)
    real(8), intent(in) :: masses(n_particles)
    real(8), intent(in) :: G, softening
    real(8), intent(out) :: accelerations(3, n_particles)
    
    integer :: i, j
    real(8) :: dx(3), r, r3
    
    accelerations = 0.0d0
    
    !$OMP PARALLEL DO PRIVATE(j, dx, r, r3) SCHEDULE(DYNAMIC)
    do i = 1, n_particles
        do j = 1, n_particles
            if (i == j) cycle
            
            dx = positions(:, j) - positions(:, i)
            r = sqrt(sum(dx**2) + softening**2)
            r3 = r * r * r
            
            accelerations(:, i) = accelerations(:, i) + &
                G * masses(j) * dx / r3
        end do
    end do
    !$OMP END PARALLEL DO
end subroutine
```

**Key features:**
- `!$OMP PARALLEL DO`: Multi-threading across particles
- Column-major arrays (Fortran native)
- f2py wrapper for Python interface

### Python Wrapper Integration

All compiled implementations expose the same interface:

```python
def simulate(positions, velocities, masses, n_steps, G, softening, dt):
    """
    Run N-body simulation
    
    Returns:
        positions_final: (N, 3) array
        velocities_final: (N, 3) array
    """
    # Implementation-specific code
    pass
```

This allows drop-in replacement for benchmarking.

---

## Validation & Testing

### Cross-Implementation Verification

Test: Same initial conditions (N=100, seed=42)

| Implementation | Final Energy | Position RMS Diff |
|----------------|--------------|-------------------|
| JAX (GPU)      | -45.2341     | —                 |
| Fortran        | -45.2342     | < 10⁻¹⁰           |
| Rust           | -45.2342     | < 10⁻¹⁰           |
| Julia          | -45.2342     | < 10⁻¹⁰           |
| C++            | -45.2341     | < 10⁻¹⁰           |
| C              | -45.2342     | < 10⁻¹⁰           |
| Python         | -45.2342     | < 10⁻¹⁰           |

Use the accuracy tests below to check agreement at the chosen tolerance.

### Numerical Accuracy Test Suite

`tests/test_accuracy.py` checks the available Python, JAX, C, C++, and Fortran implementations:

1. **Energy calculation test**: Verify all compute same initial energy
2. **Single timestep test**: One integration step produces identical results
3. **Multi-step test**: 100 timesteps with accumulation tracking
4. **Energy conservation test**: Monitor drift over 1000 steps

**Tolerances:**
- CPU-to-CPU: 10⁻¹⁰ (strict)
- GPU-to-CPU: 10⁻⁴ (relaxed, accounts for parallel reduction differences)

---

## Technologies & Tools

### Languages & Frameworks
- **JAX 0.7.2**: GPU acceleration with JIT compilation
- **Fortran**: f2py wrapper with gfortran compiler
- **Rust**: PyO3 bindings for Python integration
- **Julia**: PyCall for interoperability
- **C++**: pybind11 for Python bindings
- **C**: ctypes wrapper
- **Python 3.12**: NumPy, Matplotlib, SciPy

### Visualization
- **Matplotlib**: Static visualizations

### Compute Environment
- **WPI Turing Cluster**
- **NVIDIA L40S GPU**
- **CUDA 12.9**
- **12 CPU cores**

---

## References

1. **Press, W. H., et al.** *Numerical Recipes* (Cambridge, 2007) - Chapter 17: Integration of ODEs

2. **Barnes, J. & Hut, P.** "A hierarchical O(N log N) force-calculation algorithm" *Nature* **324**, 446-449 (1986)

3. **Aarseth, S.J.** *Gravitational N-Body Simulations* (Cambridge, 2003)

4. **Hairer, E., Lubich, C., Wanner, G.** *Geometric Numerical Integration* (Springer, 2006) - Theory of symplectic integrators

5. **Springel, V.** "The cosmological simulation code GADGET-2" *MNRAS* **364**, 1105-1134 (2005) - Production N-body code

6. JAX Documentation: https://jax.readthedocs.io/
7. NumPy/SciPy Documentation: https://numpy.org/doc/

---

## Repository Structure

The [project README](../README.md) lists the implementations, scripts, notes, and result files.

## Contact & Acknowledgments

**Student:** Adam Field (adfield@wpi.edu)  
**Institution:** Worcester Polytechnic Institute  
**Course:** Computational Physics Independent Study (ISP)  
**Date:** January 2026

**Acknowledgments:**
- WPI Turing Cluster for computational resources
- JAX development team
- Computational Physics community for open-source tools

---

## Data Availability

All code, benchmarks, and raw data available at:
- **Repository:** `computational_physics_isp/nbody_comparison/`
- **Benchmark Results:** `../results/benchmark_results.json`
- **Visualizations:** `../results/*.png`, `results/*.gif`


---
