"""
P3 Convergence Study

Validates that P3 cubic elements achieve O(h⁴) convergence for potential
by comparing against manufactured solutions with exact boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from p3_mesh_generator import generate_p3_structured_mesh
from p3_assembly import solve_poisson_p3


# ============================================================================
# Manufactured Solutions
# ============================================================================

class ManufacturedSolution:
    """Base class for manufactured solutions"""
    
    def psi(self, x, y):
        """Exact potential"""
        raise NotImplementedError
    
    def kappa(self, x, y):
        """Source term from ∇²ψ = 2κ"""
        raise NotImplementedError
    
    def name(self):
        """Solution name"""
        raise NotImplementedError


class SinusoidalSolution(ManufacturedSolution):
    """
    ψ = sin(πx)sin(πy) on [0,1]×[0,1]
    
    ∇²ψ = -2π²sin(πx)sin(πy)
    ∇²ψ = 2κ → κ = -π²sin(πx)sin(πy)
    
    Satisfies ψ=0 on all boundaries
    """
    
    def psi(self, x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def kappa(self, x, y):
        return -np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def name(self):
        return "sin(πx)sin(πy)"


class PolynomialSolution(ManufacturedSolution):
    """
    ψ = (x²-1)(y²-1) on [-1,1]×[-1,1]
    
    ∇²ψ = 2(x²-1) + 2(y²-1) = 2x² + 2y² - 4
    ∇²ψ = 2κ → κ = x² + y² - 2
    
    Satisfies ψ=0 on all boundaries
    """
    
    def psi(self, x, y):
        return (x**2 - 1) * (y**2 - 1)
    
    def kappa(self, x, y):
        return x**2 + y**2 - 2
    
    def name(self):
        return "(x²-1)(y²-1)"


class BiquadraticSolution(ManufacturedSolution):
    """
    ψ = x(1-x)y(1-y) on [0,1]×[0,1]
    
    ∇²ψ = -2y(1-y) - 2x(1-x)
    ∇²ψ = 2κ → κ = -y(1-y) - x(1-x)
    
    Satisfies ψ=0 on all boundaries
    """
    
    def psi(self, x, y):
        return x * (1 - x) * y * (1 - y)
    
    def kappa(self, x, y):
        return -y * (1 - y) - x * (1 - x)
    
    def name(self):
        return "x(1-x)y(1-y)"


# ============================================================================
# Convergence Study
# ============================================================================

def compute_errors_p3(mesh, psi_numerical, psi_exact):
    """
    Compute L² and L∞ errors
    
    Args:
        mesh: P3 mesh
        psi_numerical: Computed solution at nodes
        psi_exact: Exact solution at nodes
        
    Returns:
        L2_error, Linf_error
    """
    diff = psi_numerical - psi_exact
    
    L2_error = np.sqrt(np.mean(diff**2))
    Linf_error = np.max(np.abs(diff))
    
    return L2_error, Linf_error


def convergence_study_p3(solution: ManufacturedSolution,
                        mesh_sizes: list,
                        domain: tuple = (0, 1, 0, 1)):
    """
    Run convergence study for P3 elements
    
    Args:
        solution: Manufactured solution object
        mesh_sizes: List of (nx, ny) tuples
        domain: (xmin, xmax, ymin, ymax)
        
    Returns:
        results: Dict with mesh sizes, errors, convergence rates
    """
    xmin, xmax, ymin, ymax = domain
    
    print("\n" + "=" * 70)
    print(f"P3 CONVERGENCE STUDY: {solution.name()}")
    print(f"Domain: [{xmin},{xmax}] × [{ymin},{ymax}]")
    print("=" * 70)
    
    h_values = []
    L2_errors = []
    Linf_errors = []
    n_dofs = []
    
    for nx, ny in mesh_sizes:
        print(f"\n--- Mesh: {nx}×{ny} ---")
        
        # Generate mesh
        mesh = generate_p3_structured_mesh(nx, ny, xmin, xmax, ymin, ymax)
        nodes = np.array(mesh.nodes)
        n_nodes = len(nodes)
        n_dofs.append(n_nodes)
        
        # Compute mesh size
        h = max((xmax - xmin) / nx, (ymax - ymin) / ny)
        h_values.append(h)
        
        # Evaluate source term
        kappa_values = solution.kappa(nodes[:, 0], nodes[:, 1])
        
        # Solve
        print(f"Solving {n_nodes} DOF system...")
        psi_numerical = solve_poisson_p3(mesh, kappa_values)
        
        # Exact solution
        psi_exact = solution.psi(nodes[:, 0], nodes[:, 1])
        
        # Compute errors
        L2_error, Linf_error = compute_errors_p3(mesh, psi_numerical, psi_exact)
        L2_errors.append(L2_error)
        Linf_errors.append(Linf_error)
        
        print(f"L² error:  {L2_error:.6e}")
        print(f"L∞ error:  {Linf_error:.6e}")
    
    # Compute convergence rates
    print("\n" + "=" * 70)
    print("CONVERGENCE RATES")
    print("=" * 70)
    print(f"{'h':>10} {'DOFs':>8} {'L² Error':>12} {'Rate':>6} {'L∞ Error':>12} {'Rate':>6}")
    print("-" * 70)
    
    for i in range(len(mesh_sizes)):
        if i == 0:
            print(f"{h_values[i]:10.4f} {n_dofs[i]:8d} {L2_errors[i]:12.3e}   --   {Linf_errors[i]:12.3e}   --")
        else:
            L2_rate = np.log(L2_errors[i] / L2_errors[i-1]) / np.log(h_values[i] / h_values[i-1])
            Linf_rate = np.log(Linf_errors[i] / Linf_errors[i-1]) / np.log(h_values[i] / h_values[i-1])
            print(f"{h_values[i]:10.4f} {n_dofs[i]:8d} {L2_errors[i]:12.3e} {L2_rate:6.2f} {Linf_errors[i]:12.3e} {Linf_rate:6.2f}")
    
    # Expected O(h⁴) for P3
    print("\n" + "=" * 70)
    avg_L2_rate = np.mean([np.log(L2_errors[i] / L2_errors[i-1]) / np.log(h_values[i] / h_values[i-1]) 
                           for i in range(1, len(mesh_sizes))])
    print(f"Average L² convergence rate: {avg_L2_rate:.2f}")
    print(f"Theoretical rate (P3):       4.00")
    
    if avg_L2_rate > 3.5:
        print("✅ P3 achieves expected O(h⁴) convergence!")
    elif avg_L2_rate > 2.5:
        print("⚠️  Convergence rate between O(h³) and O(h⁴)")
    else:
        print("❌ Convergence rate below O(h³) - check implementation")
    
    print("=" * 70)
    
    return {
        'h': h_values,
        'L2_errors': L2_errors,
        'Linf_errors': Linf_errors,
        'n_dofs': n_dofs,
        'solution_name': solution.name()
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_convergence_p3(results_list, filename='p3_convergence.png'):
    """
    Plot convergence curves for multiple solutions
    
    Args:
        results_list: List of results dicts from convergence_study_p3
        filename: Output filename
    """
    print(f"\nGenerating convergence plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#ff4444', '#4444ff', '#44ff44', '#ffaa44']
    markers = ['o', 's', '^', 'D']
    
    for i, results in enumerate(results_list):
        h = np.array(results['h'])
        L2 = np.array(results['L2_errors'])
        Linf = np.array(results['Linf_errors'])
        name = results['solution_name']
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # L² errors
        ax1.loglog(h, L2, marker=marker, color=color, linewidth=2, 
                  markersize=8, label=f'{name}')
        
        # L∞ errors
        ax2.loglog(h, Linf, marker=marker, color=color, linewidth=2,
                  markersize=8, label=f'{name}')
    
    # Reference lines
    h_ref = np.array([results_list[0]['h'][0], results_list[0]['h'][-1]])
    
    # O(h²), O(h³), O(h⁴) reference slopes
    C = results_list[0]['L2_errors'][0] / h_ref[0]**4
    
    ax1.loglog(h_ref, C * h_ref**2, 'k--', linewidth=1.5, alpha=0.5, label='O(h²)')
    ax1.loglog(h_ref, C * h_ref**3, 'k-.', linewidth=1.5, alpha=0.5, label='O(h³)')
    ax1.loglog(h_ref, C * h_ref**4, 'k-', linewidth=2, alpha=0.7, label='O(h⁴)')
    
    ax2.loglog(h_ref, C * h_ref**2, 'k--', linewidth=1.5, alpha=0.5, label='O(h²)')
    ax2.loglog(h_ref, C * h_ref**3, 'k-.', linewidth=1.5, alpha=0.5, label='O(h³)')
    ax2.loglog(h_ref, C * h_ref**4, 'k-', linewidth=2, alpha=0.7, label='O(h⁴)')
    
    # Formatting
    ax1.set_xlabel('Mesh size h', fontsize=13)
    ax1.set_ylabel('L² Error', fontsize=13)
    ax1.set_title('L² Convergence (P3 Elements)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2.set_xlabel('Mesh size h', fontsize=13)
    ax2.set_ylabel('L∞ Error', fontsize=13)
    ax2.set_title('L∞ Convergence (P3 Elements)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


# ============================================================================
# Main Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🎯" * 35)
    print(" " * 18 + "P3 CONVERGENCE VALIDATION")
    print("🎯" * 35)
    
    # Mesh refinement sequence
    mesh_sizes = [
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32)
    ]
    
    results_list = []
    
    # Test 1: Sinusoidal solution
    print("\n" + "🔥" * 35)
    print("TEST 1: SINUSOIDAL SOLUTION")
    print("🔥" * 35)
    
    sol1 = SinusoidalSolution()
    results1 = convergence_study_p3(sol1, mesh_sizes, domain=(0, 1, 0, 1))
    results_list.append(results1)
    
    # Test 2: Polynomial solution
    print("\n" + "🔥" * 35)
    print("TEST 2: POLYNOMIAL SOLUTION")
    print("🔥" * 35)
    
    sol2 = PolynomialSolution()
    results2 = convergence_study_p3(sol2, mesh_sizes, domain=(-1, 1, -1, 1))
    results_list.append(results2)
    
    # Test 3: Biquadratic solution
    print("\n" + "🔥" * 35)
    print("TEST 3: BIQUADRATIC SOLUTION")
    print("🔥" * 35)
    
    sol3 = BiquadraticSolution()
    results3 = convergence_study_p3(sol3, mesh_sizes, domain=(0, 1, 0, 1))
    results_list.append(results3)
    
    # Generate convergence plot
    plot_convergence_p3(results_list)
    
    print("\n" + "=" * 70)
    print("✅ P3 CONVERGENCE VALIDATION COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print("  - All solutions should show O(h⁴) convergence")
    print("  - Convergence plot saved to: p3_convergence.png")
    print("  - Ready for P1 vs P3 comparison!")
    print("=" * 70)