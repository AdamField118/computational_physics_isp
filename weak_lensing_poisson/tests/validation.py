"""
Validation and convergence testing for JAX FEM weak lensing solver

Tests against analytic solutions:
- Gaussian lens (smooth, good for convergence studies)
- Point mass lens
- SIS lens
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fem_solver import (
    solve_lensing_poisson, compute_errors,
    GaussianLens, PointMassLens, SISLens
)
from src.mesh_generator import generate_structured_mesh


def convergence_study_gaussian(mesh_sizes: List[int] = [10, 20, 40, 80],
                               save_plot: bool = True) -> dict:
    """
    Convergence rate study using Gaussian lens
    
    Expected: O(h^2) in L^2 norm for P1 elements
    
    Args:
        mesh_sizes: List of mesh resolutions (nx = ny)
        save_plot: Whether to save convergence plot
        
    Returns:
        dict with convergence data
    """
    print("=" * 70)
    print("CONVERGENCE STUDY: Gaussian Lens")
    print("=" * 70)
    
    lens = GaussianLens(amplitude=1.0, sigma=0.2)
    
    results = {
        'h': [],
        'n_nodes': [],
        'L2_error': [],
        'H1_error': [],
        'Linf_error': []
    }
    
    print(f"\n{'h':>10} {'Nodes':>8} {'L^2 Error':>12} {'Rate':>8} {'Linf Error':>12}")
    print("-" * 70)
    
    for i, nx in enumerate(mesh_sizes):
        # Generate mesh
        mesh = generate_structured_mesh(nx, nx, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
        h = 2.0 / nx
        
        # Create convergence field at nodes
        kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
        
        # Solve
        solution = solve_lensing_poisson(mesh, kappa, verbose=False)
        
        # Compute errors
        errors = compute_errors(mesh, solution.psi, lens.psi)
        
        # Store results
        results['h'].append(h)
        results['n_nodes'].append(mesh.n_nodes)
        results['L2_error'].append(errors['L2'])
        results['H1_error'].append(errors['H1'])
        results['Linf_error'].append(errors['Linf'])
        
        # Compute convergence rate
        if i > 0:
            L2_rate = np.log(results['L2_error'][i-1] / errors['L2']) / np.log(2.0)
        else:
            L2_rate = 0.0
        
        print(f"{h:10.5f} {mesh.n_nodes:8d} {errors['L2']:12.6e} {L2_rate:8.2f} {errors['Linf']:12.6e}")
    
    print("=" * 70)
    print(f"Expected L^2 rate: 2.0 (O(h^2) for P1 elements)")
    print("=" * 70)
    
    # Plot convergence
    if save_plot:
        plot_convergence_rates(results)
    
    return results


def plot_convergence_rates(results: dict, 
                          filename: str = 'convergence_gaussian.png'):
    """
    Plot convergence rates with reference slopes
    """
    h = np.array(results['h'])
    L2 = np.array(results['L2_error'])
    Linf = np.array(results['Linf_error'])
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot errors
    ax.loglog(h, L2, 'o-', label='L^2 error', linewidth=2.5, markersize=9,
              color='#00ff41', markeredgecolor='white', markeredgewidth=0.5)
    ax.loglog(h, Linf, 's-', label='Linf error', linewidth=2.5, markersize=9,
              color='#00aaff', markeredgecolor='white', markeredgewidth=0.5)
    
    # Reference slopes
    h_ref = np.array([h[1], h[-1]])
    
    # O(h^2) reference
    L2_ref_scale = L2[1] / h[1]**2
    ax.loglog(h_ref, L2_ref_scale * h_ref**2, '--',
             label='O(h^2)', alpha=0.7, linewidth=2, color='white')
    
    ax.set_xlabel('Mesh size h', fontsize=15, fontweight='bold')
    ax.set_ylabel('Error', fontsize=15, fontweight='bold')
    ax.set_title('Convergence Study: JAX FEM Weak Lensing (Gaussian Lens)',
                 fontsize=17, color='#00ff41', fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.25, which='both', linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"\nConvergence plot saved: {filename}")
    plt.close()


def test_gaussian_lens(nx: int = 40):
    """
    Single test with Gaussian lens - visualize solution
    """
    print("\n" + "=" * 70)
    print("TEST: Gaussian Lens")
    print("=" * 70)
    
    # Setup
    lens = GaussianLens(amplitude=2.0, sigma=0.15)
    mesh = generate_structured_mesh(40, 40, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
    
    # Convergence field
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    
    # Solve
    solution = solve_lensing_poisson(mesh, kappa, verbose=True)
    
    # Compute errors
    errors = compute_errors(mesh, solution.psi, lens.psi)
    
    print(f"\nErrors vs. analytic solution:")
    print(f"  L^2 error:  {errors['L2']:.6e}")
    print(f"  Linf error:  {errors['Linf']:.6e}")
    
    # Visualize
    visualize_solution(mesh, solution, lens, save_prefix='gaussian')
    
    return solution


def test_point_mass_lens(nx: int = 40):
    """
    Test with point mass lens (singular at origin)
    """
    print("\n" + "=" * 70)
    print("TEST: Point Mass Lens")
    print("=" * 70)
    
    lens = PointMassLens(theta_E=0.5)
    mesh = generate_structured_mesh(40, 40, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
    
    # Convergence (regularized)
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    
    # Solve
    solution = solve_lensing_poisson(mesh, kappa, verbose=True)
    
    # Visualize
    visualize_solution(mesh, solution, lens, save_prefix='point_mass')
    
    return solution


def visualize_solution(mesh, solution, lens=None, save_prefix='solution'):
    """
    Create visualization of solution components
    """
    from matplotlib.tri import Triangulation
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Create triangulation
    triang = Triangulation(
        np.array(mesh.nodes[:, 0]),
        np.array(mesh.nodes[:, 1]),
        np.array(mesh.elements)
    )
    
    # 1. Convergence kappa
    ax = axes[0, 0]
    tcf = ax.tricontourf(triang, np.array(solution.convergence), levels=20, cmap='hot')
    ax.set_title('Convergence kappa', fontsize=14, color='#00ff41')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')
    plt.colorbar(tcf, ax=ax, label='kappa')
    
    # 2. Lensing potential psi
    ax = axes[0, 1]
    tcf = ax.tricontourf(triang, np.array(solution.psi), levels=20, cmap='viridis')
    ax.set_title('Lensing Potential psi', fontsize=14, color='#00ff41')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')
    plt.colorbar(tcf, ax=ax, label='psi')
    
    # 3. Deflection magnitude |alpha|
    ax = axes[1, 0]
    alpha_mag = np.sqrt(np.sum(np.array(solution.alpha)**2, axis=1))
    tcf = ax.tricontourf(triang, alpha_mag, levels=20, cmap='plasma')
    ax.set_title('Deflection Magnitude |alpha|', fontsize=14, color='#00ff41')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')
    plt.colorbar(tcf, ax=ax, label='|alpha|')
    
    # 4. Deflection field (quiver)
    ax = axes[1, 1]
    # Subsample for visualization
    skip = max(1, mesh.n_nodes // 400)
    nodes_sub = mesh.nodes[::skip]
    alpha_sub = solution.alpha[::skip]
    
    ax.quiver(
        np.array(nodes_sub[:, 0]),
        np.array(nodes_sub[:, 1]),
        np.array(alpha_sub[:, 0]),
        np.array(alpha_sub[:, 1]),
        alpha_mag[::skip],
        cmap='plasma',
        scale=5.0,
        width=0.003
    )
    ax.set_title('Deflection Field alpha', fontsize=14, color='#00ff41')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    filename = f'{save_prefix}_solution.png'
    plt.savefig(filename, dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"Solution visualization saved: {filename}")
    plt.close()


def benchmark_solver_performance():
    """
    Benchmark solver performance for different mesh sizes
    """
    import time
    
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    lens = GaussianLens()
    mesh_sizes = [20, 40, 80, 160]
    
    print(f"\n{'Mesh':>10} {'Nodes':>10} {'Elements':>10} {'Time (s)':>12} {'Iter':>6}")
    print("-" * 70)
    
    for nx in mesh_sizes:
        mesh = generate_structured_mesh(nx, nx, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
        kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
        
        # Warm-up (JIT compilation)
        if nx == mesh_sizes[0]:
            _ = solve_lensing_poisson(mesh, kappa, verbose=False)
        
        # Time solve
        start = time.time()
        solution = solve_lensing_poisson(mesh, kappa, verbose=False)
        elapsed = time.time() - start
        
        print(f"{nx}×{nx:>4} {mesh.n_nodes:10d} {mesh.n_elements:10d} {elapsed:12.6f} {solution.iterations:6d}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Run convergence study
    results = convergence_study_gaussian(mesh_sizes=[10, 20, 40, 80])
    
    # Test individual lenses
    print("\n")
    test_gaussian_lens(nx=40)
    
    print("\n")
    test_point_mass_lens(nx=40)
    
    # Benchmark
    print("\n")
    benchmark_solver_performance()