"""
Clean convergence verification for documentation
Generates high-quality plots
"""
import numpy as np
import matplotlib.pyplot as plt
from fem_solver import PoissonSolver2D
from manufactured_solutions import SineSolution
from visualization import plot_solution, plot_mesh, plot_error_distribution

# Dark theme
plt.style.use('dark_background')

def main():
    """Run complete convergence study with visualizations"""
    
    print("=" * 70)
    print("2D POISSON FEM SOLVER - CONVERGENCE VERIFICATION")
    print("=" * 70)
    
    # Manufactured solution
    mms = SineSolution()
    
    # Mesh refinement sequence
    mesh_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    h_vals = []
    L2_errors = []
    H1_errors = []
    Linf_errors = []
    
    print(f"\n{'h':>10} {'Nodes':>8} {'L² error':>12} {'Rate':>8} {'H¹ error':>12} {'Rate':>8}")
    print("-" * 70)
    
    for i, max_area in enumerate(mesh_sizes):
        # Solve
        solver = PoissonSolver2D('unit_square', max_area=max_area)
        solver.solve()
        
        # Compute errors
        L2, H1, Linf = solver.compute_errors(mms.u_exact)
        h = np.sqrt(max_area)
        
        # Convergence rates
        if i > 0:
            L2_rate = np.log(L2_errors[-1] / L2) / np.log(h_vals[-1] / h)
            H1_rate = np.log(H1_errors[-1] / H1) / np.log(h_vals[-1] / h)
        else:
            L2_rate = H1_rate = 0.0
        
        h_vals.append(h)
        L2_errors.append(L2)
        H1_errors.append(H1)
        Linf_errors.append(Linf)
        
        print(f"{h:10.5f} {solver.mesh.nodes.shape[0]:8d} {L2:12.6e} {L2_rate:8.2f} "
              f"{H1:12.6e} {H1_rate:8.2f}")
    
    print("=" * 70)
    print("✓ Expected: L² rate → 2.0, H¹ rate → 1.0")
    print("=" * 70)
    
    # Generate visualizations
    plot_convergence_results(h_vals, L2_errors, H1_errors, Linf_errors)
    plot_solution_example(mms)
    
    print("\nPlots saved to ../results/")
    print("  - convergence_rates.png")
    print("  - solution_example.png")
    print("  - error_distribution.png")


def plot_convergence_results(h, L2, H1, Linf):
    """Publication-quality convergence plot"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot errors
    ax.loglog(h, L2, 'o-', label='L² error', linewidth=2.5, markersize=9, 
              color='#00ff41', markeredgecolor='white', markeredgewidth=0.5)
    ax.loglog(h, H1, 's-', label='H¹ error', linewidth=2.5, markersize=9,
              color='#00aaff', markeredgecolor='white', markeredgewidth=0.5)
    ax.loglog(h, Linf, '^-', label='L∞ error', linewidth=2.5, markersize=9,
              color='#ffaa00', markeredgecolor='white', markeredgewidth=0.5)
    
    # Reference slopes
    h_ref = np.array([h[0], h[-1]])
    
    # O(h²) - match L² error
    ref_scale = L2[1] / h[1]**2
    ax.loglog(h_ref, ref_scale * h_ref**2, '--', 
             label='O(h²)', alpha=0.7, linewidth=2, color='white')
    
    # O(h) - match H¹ error
    ref_scale = H1[1] / h[1]
    ax.loglog(h_ref, ref_scale * h_ref, ':', 
             label='O(h)', alpha=0.7, linewidth=2, color='white')
    
    ax.set_xlabel('Mesh size h', fontsize=15, fontweight='bold')
    ax.set_ylabel('Error', fontsize=15, fontweight='bold')
    ax.set_title('Convergence Study: 2D Poisson Equation (P1 FEM)', 
                 fontsize=17, color='#00ff41', fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.25, which='both', linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig('../results/convergence_rates.png', dpi=300, 
                facecolor='#1a1a1a', bbox_inches='tight')
    plt.close()


def plot_solution_example(mms):
    """Plot numerical solution and exact solution side-by-side"""
    # Create fine mesh for nice visualization
    solver = PoissonSolver2D('unit_square', max_area=0.005)
    solver.solve()
    
    from visualization import plot_error_distribution
    plot_error_distribution(solver.mesh, solver.solution, mms.u_exact,
                           save_path='../results/error_distribution.png')
    
    # Also 3D plot
    from visualization import plot_3d_solution
    plot_3d_solution(solver.mesh, solver.solution, 
                    title='Numerical Solution: u(x,y) = sin(πx)sin(πy)',
                    save_path='../results/solution_3d.png')


if __name__ == '__main__':
    main()