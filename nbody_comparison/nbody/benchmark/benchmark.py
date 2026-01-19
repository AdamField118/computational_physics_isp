"""
Benchmark script for comparing N-body simulation implementations

Compares:
- JAX (GPU)
- Fortran (CPU with OpenMP)

Adam Field - Computational Physics ISP
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append('../jax')
from nbody_jax import (
    create_random_system, NBodyState, NBodyConfig, 
    simulate as jax_simulate, compute_energy
)

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    print("Warning: JAX not available")
    HAS_JAX = False

try:
    import nbody_fortran_module as fortran_nbody
    HAS_FORTRAN = True
except ImportError:
    print("Warning: Fortran module not compiled")
    HAS_FORTRAN = False

try:
    sys.path.append('../python')
    from nbody_python import simulate as python_simulate, compute_energy as python_energy
    HAS_PYTHON = True
except ImportError:
    print("Warning: Pure Python module not available")
    HAS_PYTHON = False

try:
    sys.path.append('../c')
    import nbody_c_wrapper as c_nbody
    HAS_C = True
except Exception as e:
    print(f"Warning: C module not available: {e}")
    HAS_C = False

try:
    sys.path.append('../cpp')
    from nbody_cpp_module import NBodySimulator
    HAS_CPP = True
except Exception as e:
    print(f"Warning: C++ module not available: {e}")
    HAS_CPP = False

# Julia import (add after HAS_CPP block)
try:
    sys.path.append('../julia')
    import nbody_pyjulia_wrapper as julia_nbody
    HAS_JULIA = julia_nbody.check_pyjulia_available()
    if HAS_JULIA:
        print("✓ Using PyJulia (direct calling)")
except Exception as e:
    print(f"Warning: Julia not available: {e}")
    HAS_JULIA = False

# Rust import (add after Julia block)
try:
    sys.path.append('../rust')
    import nbody_rust_wrapper as rust_nbody
    if not rust_nbody.HAS_RUST:
        raise ImportError("Rust module not compiled")
    HAS_RUST = True
except Exception as e:
    print(f"Warning: Rust module not available: {e}")
    HAS_RUST = False

class BenchmarkResult:
    """Store results from a single benchmark run"""
    def __init__(self, implementation: str, n_particles: int, n_steps: int,
                 runtime: float, time_per_step: float, energy_drift: float = None):
        self.implementation = implementation
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.runtime = runtime
        self.time_per_step = time_per_step
        self.energy_drift = energy_drift
    
    def to_dict(self):
        return {
            'implementation': self.implementation,
            'n_particles': self.n_particles,
            'n_steps': self.n_steps,
            'runtime': self.runtime,
            'time_per_step': self.time_per_step,
            'energy_drift': self.energy_drift,
        }


def benchmark_jax(n_particles: int, n_steps: int, config: NBodyConfig) -> BenchmarkResult:
    """Benchmark JAX implementation"""
    print(f"  JAX: N={n_particles}, steps={n_steps}...", end=' ', flush=True)
    
    # Create initial conditions
    key = jax.random.PRNGKey(42)
    initial_state = create_random_system(
        n_particles, key, position_scale=10.0, velocity_scale=0.5
    )
    
    # Initial energy
    ke0, pe0, E0 = compute_energy(initial_state, config)
    
    # Warm-up run (JIT compilation)
    _ = jax_simulate(initial_state, config, 10, save_every=10)
    
    # Timed run
    start = time.perf_counter()
    times, positions, velocities = jax_simulate(initial_state, config, n_steps, save_every=n_steps)
    jax.block_until_ready(positions)
    elapsed = time.perf_counter() - start
    
    # Final energy
    final_state = NBodyState(
        positions=positions[-1], 
        velocities=velocities[-1],
        masses=initial_state.masses,
        time=times[-1]
    )
    ke_f, pe_f, E_f = compute_energy(final_state, config)
    energy_drift = abs(E_f - E0) / abs(E0) * 100
    
    print(f"{elapsed:.4f}s ({elapsed/n_steps*1000:.4f} ms/step, ΔE={energy_drift:.6f}%)")
    
    return BenchmarkResult(
        'JAX (GPU)', n_particles, n_steps, elapsed, 
        elapsed/n_steps, energy_drift
    )


def benchmark_fortran(n_particles: int, n_steps: int, config: NBodyConfig) -> BenchmarkResult:
    """Benchmark Fortran implementation"""
    print(f"  Fortran: N={n_particles}, steps={n_steps}...", end=' ', flush=True)
    
    # Create initial conditions (use NumPy, convert to Fortran order)
    np.random.seed(42)
    positions = np.random.uniform(-10, 10, (n_particles, 3)).astype(np.float64)
    velocities = 0.5 * np.random.randn(n_particles, 3).astype(np.float64)
    masses = np.random.uniform(0.1, 1.0, n_particles).astype(np.float64)
    
    # Fortran uses column-major order
    positions_f = np.asfortranarray(positions)
    velocities_f = np.asfortranarray(velocities)
    masses_f = np.asfortranarray(masses)
    
    # Initial energy
    ke0, pe0, E0 = fortran_nbody.nbody_fortran.compute_energy(
        positions_f, velocities_f, masses_f, config.G, config.softening
    )
    
    # Timed run
    start = time.perf_counter()
    pos_final, vel_final = fortran_nbody.nbody_fortran.simulate(
        positions_f, velocities_f, masses_f, n_steps, 
        config.G, config.softening, config.dt
    )
    elapsed = time.perf_counter() - start
    
    # Final energy
    ke_f, pe_f, E_f = fortran_nbody.nbody_fortran.compute_energy(
        pos_final, vel_final, masses_f, config.G, config.softening
    )
    energy_drift = abs(E_f - E0) / abs(E0) * 100
    
    print(f"{elapsed:.4f}s ({elapsed/n_steps*1000:.4f} ms/step, ΔE={energy_drift:.6f}%)")
    
    return BenchmarkResult(
        'Fortran (OpenMP)', n_particles, n_steps, elapsed,
        elapsed/n_steps, energy_drift
    )

def benchmark_python(n_particles: int, n_steps: int, config: NBodyConfig) -> BenchmarkResult:
    """Benchmark Pure Python implementation"""
    print(f"  Python: N={n_particles}, steps={n_steps}...", end=' ', flush=True)
    
    np.random.seed(42)
    positions = np.random.uniform(-10, 10, (n_particles, 3))
    velocities = 0.5 * np.random.randn(n_particles, 3)
    masses = np.random.uniform(0.1, 1.0, n_particles)
    
    # Initial energy
    ke0, pe0, E0 = python_energy(positions, velocities, masses, config.G, config.softening)
    
    # Timed run
    start = time.perf_counter()
    pos_final, vel_final = python_simulate(
        positions, velocities, masses, n_steps, config.G, config.softening, config.dt
    )
    elapsed = time.perf_counter() - start
    
    # Final energy
    ke_f, pe_f, E_f = python_energy(pos_final, vel_final, masses, config.G, config.softening)
    energy_drift = abs(E_f - E0) / abs(E0) * 100
    
    print(f"{elapsed:.4f}s ({elapsed/n_steps*1000:.4f} ms/step, ΔE={energy_drift:.6f}%)")
    
    return BenchmarkResult(
        'Python (NumPy)', n_particles, n_steps, elapsed, elapsed/n_steps, energy_drift
    )


def benchmark_c(n_particles: int, n_steps: int, config: NBodyConfig) -> BenchmarkResult:
    """Benchmark C implementation"""
    print(f"  C: N={n_particles}, steps={n_steps}...", end=' ', flush=True)
    
    np.random.seed(42)
    positions = np.random.uniform(-10, 10, (n_particles, 3))
    velocities = 0.5 * np.random.randn(n_particles, 3)
    masses = np.random.uniform(0.1, 1.0, n_particles)
    
    # Initial energy
    ke0, pe0, E0 = c_nbody.compute_energy(positions, velocities, masses, config.G, config.softening)
    
    # Timed run
    start = time.perf_counter()
    pos_final, vel_final = c_nbody.simulate(
        positions, velocities, masses, n_steps, config.G, config.softening, config.dt
    )
    elapsed = time.perf_counter() - start
    
    # Final energy
    ke_f, pe_f, E_f = c_nbody.compute_energy(pos_final, vel_final, masses, config.G, config.softening)
    energy_drift = abs(E_f - E0) / abs(E0) * 100
    
    print(f"{elapsed:.4f}s ({elapsed/n_steps*1000:.4f} ms/step, ΔE={energy_drift:.6f}%)")
    
    return BenchmarkResult(
        'C', n_particles, n_steps, elapsed, elapsed/n_steps, energy_drift
    )


def benchmark_cpp(n_particles: int, n_steps: int, config: NBodyConfig) -> BenchmarkResult:
    """Benchmark C++ implementation"""
    print(f"  C++: N={n_particles}, steps={n_steps}...", end=' ', flush=True)
    
    np.random.seed(42)
    positions = np.random.uniform(-10, 10, (n_particles, 3))
    velocities = 0.5 * np.random.randn(n_particles, 3)
    masses = np.random.uniform(0.1, 1.0, n_particles)
    
    # Flatten for C++ (expects N*3 flat array)
    pos_flat = positions.flatten()
    vel_flat = velocities.flatten()
    
    # Create simulator
    sim = NBodySimulator(n_particles, config.G, config.softening, config.dt)
    
    # Initial energy
    ke0, pe0, E0 = sim.compute_energy(pos_flat, vel_flat, masses)
    
    # Timed run
    start = time.perf_counter()
    pos_final, vel_final = sim.simulate(pos_flat, vel_flat, masses, n_steps)
    elapsed = time.perf_counter() - start
    
    # Final energy
    ke_f, pe_f, E_f = sim.compute_energy(
        pos_final.flatten(), vel_final.flatten(), masses
    )
    energy_drift = abs(E_f - E0) / abs(E0) * 100
    
    print(f"{elapsed:.4f}s ({elapsed/n_steps*1000:.4f} ms/step, ΔE={energy_drift:.6f}%)")
    
    return BenchmarkResult(
        'C++', n_particles, n_steps, elapsed, elapsed/n_steps, energy_drift
    )

def benchmark_julia(n_particles: int, n_steps: int, config: NBodyConfig) -> BenchmarkResult:
    """Benchmark Julia implementation"""
    print(f"  Julia: N={n_particles}, steps={n_steps}...", end=' ', flush=True)
    
    np.random.seed(42)
    positions = np.random.uniform(-10, 10, (n_particles, 3))
    velocities = 0.5 * np.random.randn(n_particles, 3)
    masses = np.random.uniform(0.1, 1.0, n_particles)
    
    # Initial energy
    ke0, pe0, E0 = julia_nbody.compute_energy(positions, velocities, masses, 
                                              config.G, config.softening)
    
    # Warm-up (Julia JIT compilation) - ONLY NEEDED ONCE with PyJulia
    if n_particles == 10:  # Only warm up on first benchmark
        _ = julia_nbody.simulate(positions.copy(), velocities.copy(), masses, 10,
                                 config.G, config.softening, config.dt)
    
    # Timed run
    start = time.perf_counter()
    pos_final, vel_final = julia_nbody.simulate(
        positions, velocities, masses, n_steps,
        config.G, config.softening, config.dt
    )
    elapsed = time.perf_counter() - start
    
    # Final energy
    ke_f, pe_f, E_f = julia_nbody.compute_energy(pos_final, vel_final, masses,
                                                 config.G, config.softening)
    energy_drift = abs(E_f - E0) / abs(E0) * 100
    
    print(f"{elapsed:.4f}s ({elapsed/n_steps*1000:.4f} ms/step, ΔE={energy_drift:.6f}%)")
    
    return BenchmarkResult(
        'Julia', n_particles, n_steps, elapsed, elapsed/n_steps, energy_drift
    )


def benchmark_rust(n_particles: int, n_steps: int, config: NBodyConfig) -> BenchmarkResult:
    """Benchmark Rust implementation"""
    print(f"  Rust: N={n_particles}, steps={n_steps}...", end=' ', flush=True)
    
    np.random.seed(42)
    positions = np.random.uniform(-10, 10, (n_particles, 3))
    velocities = 0.5 * np.random.randn(n_particles, 3)
    masses = np.random.uniform(0.1, 1.0, n_particles)
    
    # Initial energy
    ke0, pe0, E0 = rust_nbody.compute_energy(positions, velocities, masses,
                                             config.G, config.softening)
    
    # Timed run
    start = time.perf_counter()
    pos_final, vel_final = rust_nbody.simulate(
        positions, velocities, masses, n_steps,
        config.G, config.softening, config.dt
    )
    elapsed = time.perf_counter() - start
    
    # Final energy
    ke_f, pe_f, E_f = rust_nbody.compute_energy(pos_final, vel_final, masses,
                                                config.G, config.softening)
    energy_drift = abs(E_f - E0) / abs(E0) * 100
    
    print(f"{elapsed:.4f}s ({elapsed/n_steps*1000:.4f} ms/step, ΔE={energy_drift:.6f}%)")
    
    return BenchmarkResult(
        'Rust', n_particles, n_steps, elapsed, elapsed/n_steps, energy_drift
    )

def run_benchmark_suite() -> List[BenchmarkResult]:
    """Run comprehensive benchmark suite"""
    print("=" * 70)
    print("N-Body Simulation Benchmark Suite")
    print("=" * 70)
    
    config = NBodyConfig(G=1.0, softening=0.1, dt=0.01)
    results = []
    
    # Vary N (number of particles)
    particle_counts = [10, 50, 100, 500, 1000]
    n_steps = 1000
    
    print(f"\nBenchmark: Varying N (particles), fixed steps={n_steps}")
    print("-" * 70)
    
    for N in particle_counts:
        if HAS_PYTHON:
            results.append(benchmark_python(N, n_steps, config))
        
        if HAS_C:
            results.append(benchmark_c(N, n_steps, config))
        
        if HAS_CPP:
            results.append(benchmark_cpp(N, n_steps, config))
        
        if HAS_FORTRAN:
            results.append(benchmark_fortran(N, n_steps, config))
        
        if HAS_JAX:
            results.append(benchmark_jax(N, n_steps, config))

        if HAS_JULIA:
            results.append(benchmark_julia(N, n_steps, config))
        
        if HAS_RUST:
            results.append(benchmark_rust(N, n_steps, config))
    
    return results


def plot_results(results: List[BenchmarkResult], output_dir: str = '../../web/data'):
    """Generate comparison plots"""
    print("\nGenerating plots...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Organize results
    implementations = list(set(r.implementation for r in results))
    
    # Plot: Runtime vs N
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for impl in implementations:
        impl_results = [r for r in results if r.implementation == impl]
        N_vals = [r.n_particles for r in impl_results]
        times = [r.runtime for r in impl_results]
        ax1.loglog(N_vals, times, 'o-', label=impl, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Particles (N)', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Scaling with Problem Size', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time per step vs N
    for impl in implementations:
        impl_results = [r for r in results if r.implementation == impl]
        N_vals = [r.n_particles for r in impl_results]
        time_per_step = [r.time_per_step * 1000 for r in impl_results]
        ax2.loglog(N_vals, time_per_step, 'o-', label=impl, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of Particles (N)', fontsize=12)
    ax2.set_ylabel('Time per Step (ms)', fontsize=12)
    ax2.set_title('Per-Step Performance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/scaling_comparison.png")
    plt.close()


def save_results_json(results: List[BenchmarkResult], output_dir: str = '../../web/data'):
    """Save results as JSON"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file = f'{output_dir}/benchmark_results.json'
    
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': [r.to_dict() for r in results]
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved results to: {output_file}")


def print_summary(results: List[BenchmarkResult]):
    """Print summary table"""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Implementation':<20} {'N':<8} {'Steps':<8} {'Runtime':<12} {'ms/step':<12} {'ΔE (%)':<10}")
    print("-" * 70)
    
    for r in results:
        energy_str = f"{r.energy_drift:.6f}" if r.energy_drift is not None else "N/A"
        print(f"{r.implementation:<20} {r.n_particles:<8} {r.n_steps:<8} "
              f"{r.runtime:<12.4f} {r.time_per_step*1000:<12.4f} {energy_str:<10}")


if __name__ == "__main__":
    # Run benchmarks
    results = run_benchmark_suite()
    
    # Print summary
    print_summary(results)
    
    # Generate plots
    plot_results(results)
    
    # Save JSON
    save_results_json(results)
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)