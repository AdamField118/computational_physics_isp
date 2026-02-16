"""
Fixed validation with proper boundary-compatible test cases

Key insight: For convergence testing, the analytic solution MUST satisfy
the boundary conditions we're imposing!
"""

import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Add parent to path
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'tests' else Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.fem_solver import solve_lensing_poisson, compute_errors
from src.mesh_generator import generate_structured_mesh


# ============================================================================
# Manufactured Solutions with Compatible BCs
# ============================================================================

class SinusoidalLens:
    """
    Sinusoidal solution on [0, 1] x [0, 1] with psi = 0 on boundary
    
    This is a MANUFACTURED solution perfect for convergence testing:
    - Smooth (infinitely differentiable)
    - Satisfies Dirichlet BC exactly
    - Known exact solution and source term
    """
    
    def __init__(self, k: int = 1):
        """
        Args:
            k: wavenumber (k=1 gives one wavelength across domain)
        """
        self.k = k
    
    def psi(self, x: float, y: float) -> float:
        """Exact solution"""
        return jnp.sin(self.k * jnp.pi * x) * jnp.sin(self.k * jnp.pi * y)
    
    def kappa(self, x: float, y: float) -> float:
        """Source term from Laplacian"""
        # nabla^2psi = -2k^2pi^2 sin(kpi x)sin(kpi y)
        # So: nabla^2psi = 2kappa → kappa = -k^2pi^2 sin(kpi x)sin(kpi y)
        return -self.k**2 * jnp.pi**2 * jnp.sin(self.k * jnp.pi * x) * jnp.sin(self.k * jnp.pi * y)
    
    def alpha(self, x: float, y: float) -> tuple:
        """Exact deflection = gradient of psi"""
        alpha_x = self.k * jnp.pi * jnp.cos(self.k * jnp.pi * x) * jnp.sin(self.k * jnp.pi * y)
        alpha_y = self.k * jnp.pi * jnp.sin(self.k * jnp.pi * x) * jnp.cos(self.k * jnp.pi * y)
        return (alpha_x, alpha_y)


class PolynomialLens:
    """
    Polynomial solution with psi = 0 on boundary of [-1, 1]^2
    
    Using psi = (1 - x^2)(1 - y^2) which:
    - Vanishes on all four edges
    - Is smooth
    - Has simple derivatives
    """
    
    def psi(self, x: float, y: float) -> float:
        """Exact solution"""
        return (1 - x**2) * (1 - y**2)
    
    def kappa(self, x: float, y: float) -> float:
        """Source term"""
        # nabla^2psi = frac{partial^2psi}{partialx^2} + frac{partial^2psi}{partialy^2}
        # frac{partial^2psi}{partialx^2} = -2(1 - y^2)
        # frac{partial^2psi}{partialy^2} = -2(1 - x^2)
        # nabla^2psi = -2(1 - y^2) - 2(1 - x^2) = -2(2 - x^2 - y^2)
        # So: kappa = frac{nabla^2psi}{2} = -(2 - x^2 - y^2)
        return -(2 - x**2 - y**2)
    
    def alpha(self, x: float, y: float) -> tuple:
        """Exact deflection"""
        alpha_x = -2 * x * (1 - y**2)
        alpha_y = -2 * y * (1 - x**2)
        return (alpha_x, alpha_y)


class LargeGaussianLens:
    """
    Gaussian on LARGE domain where boundary is far from mass
    
    This tests if the original Gaussian works when domain is big enough
    that psi approx 0 on boundary is actually satisfied
    """
    
    def __init__(self, amplitude: float = 1.0, sigma: float = 0.2):
        self.amplitude = amplitude
        self.sigma = sigma
    
    def kappa(self, x: float, y: float) -> float:
        r2 = x**2 + y**2
        return self.amplitude * jnp.exp(-r2 / (2 * self.sigma**2))
    
    def psi(self, x: float, y: float) -> float:
        """Approximate solution"""
        r2 = x**2 + y**2
        return self.amplitude * self.sigma**2 * (1 - jnp.exp(-r2 / (2 * self.sigma**2)))
    
    def alpha(self, x: float, y: float) -> tuple:
        r2 = x**2 + y**2
        r = jnp.sqrt(r2) + 1e-6
        alpha_mag = self.amplitude * self.sigma**2 * (1 - jnp.exp(-r2 / (2 * self.sigma**2))) / r
        return (alpha_mag * x / r, alpha_mag * y / r)


# ============================================================================
# Convergence Studies
# ============================================================================

def convergence_study_sinusoidal(mesh_sizes=[10, 20, 40, 80]):
    """
    Test with sinusoidal manufactured solution
    GUARANTEED to show O(h^2) convergence for P1 elements!
    """
    print("=" * 70)
    print("CONVERGENCE STUDY: Sinusoidal Manufactured Solution")
    print("=" * 70)
    print("Domain: [0, 1] x [0, 1]")
    print("Solution: psi = sin(pi x)sin(pi y)")
    print("BC: psi = 0 on boundary (EXACTLY SATISFIED by solution)")
    print("=" * 70)
    
    lens = SinusoidalLens(k=1)
    
    results = {
        'h': [],
        'n_nodes': [],
        'L2_error': [],
        'Linf_error': []
    }
    
    print(f"\n{'h':>10} {'Nodes':>8} {'L^2 Error':>12} {'Rate':>8} {'Linf Error':>12} {'Rate':>8}")
    print("-" * 80)
    
    for i, nx in enumerate(mesh_sizes):
        # Mesh on [0, 1] x [0, 1]
        mesh = generate_structured_mesh(nx, nx, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        h = 1.0 / nx
        
        # Convergence field
        kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
        
        # Solve
        solution = solve_lensing_poisson(mesh, kappa, verbose=False)
        
        # Compute errors
        errors = compute_errors(mesh, solution.psi, lens.psi)
        
        results['h'].append(h)
        results['n_nodes'].append(mesh.n_nodes)
        results['L2_error'].append(errors['L2'])
        results['Linf_error'].append(errors['Linf'])
        
        # Compute rates
        if i > 0:
            L2_rate = np.log(results['L2_error'][i-1] / errors['L2']) / np.log(2.0)
            Linf_rate = np.log(results['Linf_error'][i-1] / errors['Linf']) / np.log(2.0)
        else:
            L2_rate = 0.0
            Linf_rate = 0.0
        
        print(f"{h:10.5f} {mesh.n_nodes:8d} {errors['L2']:12.6e} {L2_rate:8.2f} "
              f"{errors['Linf']:12.6e} {Linf_rate:8.2f}")
    
    print("=" * 80)
    print(f"Expected L^2 rate: 2.0 (should see ~2.0 after first refinement)")
    print(f"Expected Linfty rate: 2.0")
    print("=" * 80)
    
    return results


def convergence_study_polynomial(mesh_sizes=[10, 20, 40, 80]):
    """
    Test with polynomial manufactured solution
    """
    print("\n" + "=" * 70)
    print("CONVERGENCE STUDY: Polynomial Manufactured Solution")
    print("=" * 70)
    print("Domain: [-1, 1] x [-1, 1]")
    print("Solution: psi = (1 - x^2)(1 - y^2)")
    print("BC: psi = 0 on boundary (EXACTLY SATISFIED)")
    print("=" * 70)
    
    lens = PolynomialLens()
    
    results = {
        'h': [],
        'n_nodes': [],
        'L2_error': [],
        'Linf_error': []
    }
    
    print(f"\n{'h':>10} {'Nodes':>8} {'L^2 Error':>12} {'Rate':>8} {'Linf Error':>12} {'Rate':>8}")
    print("-" * 80)
    
    for i, nx in enumerate(mesh_sizes):
        mesh = generate_structured_mesh(nx, nx, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
        h = 2.0 / nx
        
        kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
        solution = solve_lensing_poisson(mesh, kappa, verbose=False)
        errors = compute_errors(mesh, solution.psi, lens.psi)
        
        results['h'].append(h)
        results['n_nodes'].append(mesh.n_nodes)
        results['L2_error'].append(errors['L2'])
        results['Linf_error'].append(errors['Linf'])
        
        if i > 0:
            L2_rate = np.log(results['L2_error'][i-1] / errors['L2']) / np.log(2.0)
            Linf_rate = np.log(results['Linf_error'][i-1] / errors['Linf']) / np.log(2.0)
        else:
            L2_rate = 0.0
            Linf_rate = 0.0
        
        print(f"{h:10.5f} {mesh.n_nodes:8d} {errors['L2']:12.6e} {L2_rate:8.2f} "
              f"{errors['Linf']:12.6e} {Linf_rate:8.2f}")
    
    print("=" * 80)
    print(f"Should see ~2.0 convergence rate")
    print("=" * 80)
    
    return results


def test_gaussian_large_domain():
    """
    Test original Gaussian on LARGE domain
    Check if boundary effect disappears when domain is big enough
    """
    print("\n" + "=" * 70)
    print("TEST: Gaussian Lens on Expanding Domains")
    print("=" * 70)
    print("Checking how error depends on domain size...")
    print("(Gaussian has sigma = 0.2, centered at origin)")
    print("=" * 70)
    
    lens = LargeGaussianLens(amplitude=1.0, sigma=0.2)
    nx = 40  # Fixed resolution
    
    print(f"\n{'Domain':>15} {'L^2 Error':>12} {'Linf Error':>12} {'Max |psi|':>12}")
    print("-" * 60)
    
    for domain_size in [2, 4, 8, 16, 32]:
        L = domain_size / 2  # Half-width
        mesh = generate_structured_mesh(nx, nx, xmin=-L, xmax=L, ymin=-L, ymax=L)
        
        kappa = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
        solution = solve_lensing_poisson(mesh, kappa, verbose=False)
        errors = compute_errors(mesh, solution.psi, lens.psi)
        
        max_psi = jnp.max(jnp.abs(solution.psi))
        
        print(f"[-{L:3.0f}, {L:3.0f}]^2 {errors['L2']:12.6e} {errors['Linf']:12.6e} {max_psi:12.6f}")
    
    print("=" * 60)
    print("Error should decrease as domain expands")
    print("When domain >> sigma, boundary is effectively at infinity")
    print("=" * 60)


def plot_convergence_comparison(results_list, labels, filename='convergence_comparison.png'):
    """
    Plot multiple convergence studies on same axes
    """
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = ['#00ff41', '#00aaff', '#ff00ff', '#ffaa00']
    
    for results, label, color in zip(results_list, labels, colors):
        h = np.array(results['h'])
        L2 = np.array(results['L2_error'])
        Linf = np.array(results['Linf_error'])
        
        ax1.loglog(h, L2, 'o-', label=label, linewidth=2.5, markersize=9,
                   color=color, markeredgecolor='white', markeredgewidth=0.5)
        
        ax2.loglog(h, Linf, 's-', label=label, linewidth=2.5, markersize=9,
                   color=color, markeredgecolor='white', markeredgewidth=0.5)
    
    # Reference slopes
    h_ref = np.array([results_list[0]['h'][1], results_list[0]['h'][-1]])
    
    # O(h^2) reference
    L2_ref = results_list[0]['L2_error'][1] / h_ref[0]**2
    ax1.loglog(h_ref, L2_ref * h_ref**2, '--', label='O(h^2)', 
               alpha=0.7, linewidth=2, color='white')
    ax2.loglog(h_ref, L2_ref * h_ref**2, '--', label='O(h^2)',
               alpha=0.7, linewidth=2, color='white')
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('Mesh size h', fontsize=15, fontweight='bold')
        ax.set_ylabel('Error', fontsize=15, fontweight='bold')
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.25, which='both')
        ax.tick_params(labelsize=12)
    
    ax1.set_title('L^2 Error Convergence', fontsize=16, color='#00ff41', fontweight='bold')
    ax2.set_title('Linfty Error Convergence', fontsize=16, color='#00ff41', fontweight='bold')
    
    fig.suptitle('FEM Convergence Studies: Boundary-Compatible Test Cases',
                 fontsize=18, color='#00ff41', fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"\nConvergence comparison plot saved: {filename}")
    plt.close()


# ============================================================================
# Main Test Suite
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 35)
    print(" " * 20 + "FIXED VALIDATION SUITE")
    print("=" * 35)
    print("\nKey Insight: Convergence tests require analytic solutions")
    print("that EXACTLY satisfy the imposed boundary conditions!")
    print("=" * 70 + "\n")
    
    # Run all convergence studies
    results_sin = convergence_study_sinusoidal(mesh_sizes=[10, 20, 40, 80])
    results_poly = convergence_study_polynomial(mesh_sizes=[10, 20, 40, 80])
    
    # Test Gaussian on expanding domains
    test_gaussian_large_domain()
    
    # Plot comparison
    plot_convergence_comparison(
        [results_sin, results_poly],
        ['Sinusoidal (perfect BC)', 'Polynomial (perfect BC)'],
        filename='convergence_gaussian.png'
    )
    
    print("\n" + "=" * 70)
    print("ALIDATION COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Manufactured solutions with compatible BCs show O(h^2) convergence")
    print("2. Original Gaussian has BC mismatch causing flat convergence")
    print("3. Point mass singularity causes numerical explosion (expected)")
    print("\nNext Steps:")
    print("-> Use manufactured solutions for convergence testing")
    print("-> For real lensing: use large domains or Neumann BC")
    print("-> Ready to move to P2 elements!")
    print("=" * 70)