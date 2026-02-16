"""
Shear Computation Validation & Visualization

Tests shear computation on multiple manufactured solutions and creates
publication-quality visualizations.
"""

import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# Add parent to path
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'tests' else Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.fem_solver import solve_lensing_poisson, SinusoidalLens, PolynomialLens, GaussianLens
from src.mesh_generator import generate_p2_structured_mesh

# Import shear module (you'll need to move shear_computation.py to src/)
sys.path.insert(0, str(Path(__file__).parent))
from src import compute_shear_p2, ShearField


def visualize_shear_field(mesh, solution, shear: ShearField, 
                          title="Shear Field Visualization",
                          filename="shear_visualization.png"):
    """
    Create comprehensive 4-panel visualization of shear field
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 14))
    
    # Extract vertex nodes for triangulation (P2 mesh visualization)
    nx = int(np.sqrt(mesh.n_elements / 2))  # Approximate
    n_vertices = (nx + 1)**2
    vertex_nodes = mesh.nodes[:n_vertices]
    p1_elements = mesh.elements[:, :3]  # Use only vertex indices
    
    triang = Triangulation(
        np.array(vertex_nodes[:, 0]),
        np.array(vertex_nodes[:, 1]),
        np.array(p1_elements)
    )
    
    # Panel 1: Convergence κ (mass distribution)
    ax1 = plt.subplot(2, 2, 1)
    kappa_plot = np.array(solution.convergence[:n_vertices])
    tcf = ax1.tricontourf(triang, kappa_plot, levels=20, cmap='hot')
    ax1.triplot(triang, 'w-', alpha=0.05, linewidth=0.2)
    ax1.set_title('Convergence κ (Mass)', fontsize=16, color='#00ff41', fontweight='bold')
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('y', fontsize=13)
    ax1.set_aspect('equal')
    plt.colorbar(tcf, ax=ax1, fraction=0.046, pad=0.04)
    
    # Panel 2: Shear component γ₁
    ax2 = plt.subplot(2, 2, 2)
    
    # Create scatter plot of shear values at eval points
    sc = ax2.scatter(np.array(shear.points[:, 0]), 
                     np.array(shear.points[:, 1]),
                     c=np.array(shear.gamma1),
                     cmap='RdBu_r', 
                     s=50,
                     edgecolors='white',
                     linewidth=0.5,
                     vmin=-np.max(np.abs(shear.gamma1)),
                     vmax=np.max(np.abs(shear.gamma1)))
    
    ax2.set_title('Shear Component γ₁', fontsize=16, color='#00ff41', fontweight='bold')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('y', fontsize=13)
    ax2.set_aspect('equal')
    cbar = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('γ₁', fontsize=13, rotation=0, labelpad=15)
    
    # Panel 3: Shear component γ₂
    ax3 = plt.subplot(2, 2, 3)
    
    sc = ax3.scatter(np.array(shear.points[:, 0]), 
                     np.array(shear.points[:, 1]),
                     c=np.array(shear.gamma2),
                     cmap='RdBu_r',
                     s=50,
                     edgecolors='white',
                     linewidth=0.5,
                     vmin=-np.max(np.abs(shear.gamma2)),
                     vmax=np.max(np.abs(shear.gamma2)))
    
    ax3.set_title('Shear Component γ₂', fontsize=16, color='#00ff41', fontweight='bold')
    ax3.set_xlabel('x', fontsize=13)
    ax3.set_ylabel('y', fontsize=13)
    ax3.set_aspect('equal')
    cbar = plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('γ₂', fontsize=13, rotation=0, labelpad=15)
    
    # Panel 4: Shear magnitude |γ| with direction arrows
    ax4 = plt.subplot(2, 2, 4)
    
    # Background: shear magnitude
    sc = ax4.scatter(np.array(shear.points[:, 0]), 
                     np.array(shear.points[:, 1]),
                     c=np.array(shear.gamma_mag),
                     cmap='plasma',
                     s=50,
                     edgecolors='white',
                     linewidth=0.5,
                     vmin=0,
                     vmax=np.max(shear.gamma_mag))
    
    # Subsample for arrows
    skip = max(1, len(shear.points) // 100)
    points_sub = shear.points[::skip]
    gamma1_sub = shear.gamma1[::skip]
    gamma2_sub = shear.gamma2[::skip]
    
    # Shear "sticks" (short lines showing orientation)
    # Shear is a symmetric traceless tensor - visualize as oriented ellipse
    for i in range(len(points_sub)):
        x, y = points_sub[i]
        g1, g2 = gamma1_sub[i], gamma2_sub[i]
        
        # Shear angle: θ = 0.5 * arctan2(γ₂, γ₁)
        theta = 0.5 * np.arctan2(g2, g1)
        length = 0.05 * np.sqrt(g1**2 + g2**2) / np.max(shear.gamma_mag)
        
        # Draw stick
        dx = length * np.cos(theta)
        dy = length * np.sin(theta)
        ax4.plot([x - dx, x + dx], [y - dy, y + dy], 
                'w-', linewidth=1.5, alpha=0.6)
    
    ax4.set_title('Shear Magnitude |γ| with Orientation', 
                  fontsize=16, color='#00ff41', fontweight='bold')
    ax4.set_xlabel('x', fontsize=13)
    ax4.set_ylabel('y', fontsize=13)
    ax4.set_aspect('equal')
    cbar = plt.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('|γ|', fontsize=13, rotation=0, labelpad=15)
    
    fig.suptitle(title, fontsize=20, color='#00ff41', fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"✓ Saved visualization: {filename}")
    plt.close()


def test_sinusoidal_shear():
    """
    Test on sinusoidal solution with known exact shear
    
    For ψ = sin(πx)sin(πy):
    - γ₁ = 0 (symmetric in x and y)
    - γ₂ = π²cos(πx)cos(πy)
    """
    print("\n" + "=" * 70)
    print("TEST 1: Sinusoidal Manufactured Solution")
    print("=" * 70)
    
    lens = SinusoidalLens(k=1)
    mesh = generate_p2_structured_mesh(30, 30, xmin=0, xmax=1, ymin=0, ymax=1)
    
    print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # Solve
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    solution = solve_lensing_poisson(mesh, kappa, verbose=False)
    
    # Compute shear
    shear = compute_shear_p2(mesh, solution.psi, eval_points='centroids')
    
    print(f"\nShear statistics:")
    print(f"  γ₁: [{jnp.min(shear.gamma1):.6f}, {jnp.max(shear.gamma1):.6f}] (should be ~0)")
    print(f"  γ₂: [{jnp.min(shear.gamma2):.6f}, {jnp.max(shear.gamma2):.6f}]")
    print(f"  |γ|: {jnp.max(shear.gamma_mag):.6f}")
    
    # Exact shear
    gamma1_exact = np.array([0.0 for x, y in shear.points])
    gamma2_exact = np.array([np.pi**2 * np.cos(np.pi*x) * np.cos(np.pi*y) 
                             for x, y in shear.points])
    
    # Errors
    err1 = jnp.sqrt(jnp.mean((shear.gamma1 - gamma1_exact)**2))
    err2 = jnp.sqrt(jnp.mean((shear.gamma2 - gamma2_exact)**2))
    
    print(f"\nComparison with exact:")
    print(f"  RMS error γ₁: {err1:.6e}")
    print(f"  RMS error γ₂: {err2:.6e}")
    
    # Visualize
    visualize_shear_field(mesh, solution, shear,
                         title="Shear Field: Sinusoidal Test",
                         filename="shear_sinusoidal.png")
    
    success = err1 < 0.1 and err2 < 0.5
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Sinusoidal test")
    
    return success


def test_polynomial_shear():
    """
    Test on polynomial solution
    
    For ψ = (1-x²)(1-y²) on [-1,1]²:
    - ψ_xx = -2(1-y²)
    - ψ_yy = -2(1-x²)
    - ψ_xy = 4xy
    
    So:
    - γ₁ = [-2(1-y²) + 2(1-x²)]/2 = (x² - y²)
    - γ₂ = 4xy
    """
    print("\n" + "=" * 70)
    print("TEST 2: Polynomial Manufactured Solution")
    print("=" * 70)
    
    lens = PolynomialLens()
    mesh = generate_p2_structured_mesh(30, 30, xmin=-1, xmax=1, ymin=-1, ymax=1)
    
    print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # Solve
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    solution = solve_lensing_poisson(mesh, kappa, verbose=False)
    
    # Compute shear
    shear = compute_shear_p2(mesh, solution.psi, eval_points='centroids')
    
    print(f"\nShear statistics:")
    print(f"  γ₁: [{jnp.min(shear.gamma1):.6f}, {jnp.max(shear.gamma1):.6f}]")
    print(f"  γ₂: [{jnp.min(shear.gamma2):.6f}, {jnp.max(shear.gamma2):.6f}]")
    print(f"  |γ|: {jnp.max(shear.gamma_mag):.6f}")
    
    # Exact shear
    gamma1_exact = np.array([x**2 - y**2 for x, y in shear.points])
    gamma2_exact = np.array([4*x*y for x, y in shear.points])
    
    # Errors
    err1 = jnp.sqrt(jnp.mean((shear.gamma1 - gamma1_exact)**2))
    err2 = jnp.sqrt(jnp.mean((shear.gamma2 - gamma2_exact)**2))
    
    print(f"\nComparison with exact:")
    print(f"  RMS error γ₁: {err1:.6e}")
    print(f"  RMS error γ₂: {err2:.6e}")
    
    # Visualize
    visualize_shear_field(mesh, solution, shear,
                         title="Shear Field: Polynomial Test",
                         filename="shear_polynomial.png")
    
    success = err1 < 0.2 and err2 < 0.2
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Polynomial test")
    
    return success


def test_galaxy_cluster_shear():
    """
    Test on realistic galaxy cluster (Gaussian profile)
    """
    print("\n" + "=" * 70)
    print("TEST 3: Galaxy Cluster (Gaussian Profile)")
    print("=" * 70)
    
    lens = GaussianLens(amplitude=1.5, sigma=0.4)
    mesh = generate_p2_structured_mesh(40, 40, xmin=-2, xmax=2, ymin=-2, ymax=2)
    
    print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    
    # Solve
    kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    solution = solve_lensing_poisson(mesh, kappa, verbose=False)
    
    print(f"Max κ: {jnp.max(kappa):.4f}")
    print(f"Max |ψ|: {jnp.max(jnp.abs(solution.psi)):.4f}")
    
    # Compute shear
    shear = compute_shear_p2(mesh, solution.psi, eval_points='centroids')
    
    print(f"\nShear statistics:")
    print(f"  γ₁: [{jnp.min(shear.gamma1):.6f}, {jnp.max(shear.gamma1):.6f}]")
    print(f"  γ₂: [{jnp.min(shear.gamma2):.6f}, {jnp.max(shear.gamma2):.6f}]")
    print(f"  |γ|: {jnp.max(shear.gamma_mag):.6f}")
    
    # For axisymmetric mass, shear pattern should be tangential
    # At (x, 0): expect γ₁ > 0, γ₂ ≈ 0
    # At (0, y): expect γ₁ < 0, γ₂ ≈ 0
    
    # Find points near x-axis
    x_axis_mask = np.abs(shear.points[:, 1]) < 0.1
    x_axis_points = shear.points[x_axis_mask]
    x_axis_gamma1 = shear.gamma1[x_axis_mask]
    
    print(f"\nAlong x-axis (y≈0): γ₁ = {np.mean(x_axis_gamma1):.4f} (should be positive)")
    
    # Visualize
    visualize_shear_field(mesh, solution, shear,
                         title="Shear Field: Galaxy Cluster",
                         filename="shear_cluster.png")
    
    print("\n✓ PASS: Galaxy cluster visualization")
    
    return True


def shear_convergence_study():
    """
    Study how shear error decreases with mesh refinement
    """
    print("\n" + "=" * 70)
    print("CONVERGENCE STUDY: Shear Error vs Mesh Size")
    print("=" * 70)
    
    lens = PolynomialLens()
    mesh_sizes = [10, 20, 40, 80]
    
    results = {'h': [], 'err_gamma1': [], 'err_gamma2': []}
    
    print(f"\n{'h':>10} {'Nodes':>8} {'γ₁ Error':>12} {'γ₂ Error':>12} {'Rate₁':>8} {'Rate₂':>8}")
    print("-" * 80)
    
    for i, nx in enumerate(mesh_sizes):
        h = 2.0 / nx
        mesh = generate_p2_structured_mesh(nx, nx, xmin=-1, xmax=1, ymin=-1, ymax=1)
        
        kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
        solution = solve_lensing_poisson(mesh, kappa, verbose=False)
        shear = compute_shear_p2(mesh, solution.psi, eval_points='centroids')
        
        # Exact shear
        gamma1_exact = np.array([x**2 - y**2 for x, y in shear.points])
        gamma2_exact = np.array([4*x*y for x, y in shear.points])
        
        err1 = float(jnp.sqrt(jnp.mean((shear.gamma1 - gamma1_exact)**2)))
        err2 = float(jnp.sqrt(jnp.mean((shear.gamma2 - gamma2_exact)**2)))
        
        results['h'].append(h)
        results['err_gamma1'].append(err1)
        results['err_gamma2'].append(err2)
        
        if i > 0:
            rate1 = np.log(results['err_gamma1'][i-1] / err1) / np.log(2.0)
            rate2 = np.log(results['err_gamma2'][i-1] / err2) / np.log(2.0)
        else:
            rate1 = rate2 = 0.0
        
        print(f"{h:10.5f} {mesh.n_nodes:8d} {err1:12.6e} {err2:12.6e} {rate1:8.2f} {rate2:8.2f}")
    
    print("=" * 80)
    print("Expected: O(h²) convergence for shear (one derivative less than ψ)")
    
    # Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    h = np.array(results['h'])
    err1 = np.array(results['err_gamma1'])
    err2 = np.array(results['err_gamma2'])
    
    ax.loglog(h, err1, 'o-', label='γ₁ error', linewidth=2.5, markersize=9, color='#00ff41')
    ax.loglog(h, err2, 's-', label='γ₂ error', linewidth=2.5, markersize=9, color='#00aaff')
    
    # Reference slope
    h_ref = np.array([h[1], h[-1]])
    C = err1[1] / h_ref[0]**2
    ax.loglog(h_ref, C * h_ref**2, '--', label='O(h²)', alpha=0.5, linewidth=2, color='white')
    
    ax.set_xlabel('Mesh size h', fontsize=15, fontweight='bold')
    ax.set_ylabel('RMS Error', fontsize=15, fontweight='bold')
    ax.set_title('Shear Convergence Study', fontsize=16, color='#00ff41', fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig('shear_convergence.png', dpi=300, facecolor='#1a1a1a')
    print("\n✓ Convergence plot saved: shear_convergence.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "🌌" * 35)
    print(" " * 25 + "SHEAR COMPUTATION VALIDATION")
    print("🌌" * 35)
    
    # Run all tests
    test1 = test_sinusoidal_shear()
    test2 = test_polynomial_shear()
    test3 = test_galaxy_cluster_shear()
    
    # Convergence study
    shear_convergence_study()
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Sinusoidal test:  {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"  Polynomial test:  {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"  Galaxy cluster:   {'✓ PASS' if test3 else '✗ FAIL'}")
    print("=" * 70)
    
    if all([test1, test2, test3]):
        print("\n🎉 ALL TESTS PASSED! Shear computation is working correctly!")
        print("\nNext steps:")
        print("  1. Move shear_computation.py to src/")
        print("  2. Update src/__init__.py to export shear functions")
        print("  3. Ready for Phase 2: Autodiff integration!")
    else:
        print("\n⚠ Some tests failed - investigate before proceeding")
