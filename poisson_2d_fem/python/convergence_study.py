"""
Convergence rate verification
Tests O(h^2) in L2, O(h) in H1
"""
import numpy as np
import matplotlib.pyplot as plt
from fem_solver import PoissonSolver2D
from manufactured_solutions import SineSolution

def convergence_test():
    """Run mesh refinement study"""
    
    # Manufactured solution
    mms = SineSolution()
    
    # Mesh sizes to test
    mesh_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    h_vals = []
    L2_errors = []
    H1_errors = []
    Linf_errors = []
    
    print("Running convergence study...")
    print(f"{'h':>10} {'L2 error':>12} {'H1 error':>12} {'Linf error':>12}")
    print("-" * 50)
    
    for max_area in mesh_sizes:
        # Solve
        solver = PoissonSolver2D('unit_square', max_area=max_area)
        solver.solve(mms.f_source, mms.g_boundary)
        
        # Compute errors
        L2, H1, Linf = solver.compute_errors(mms.u_exact)
        
        # Approximate h
        h = np.sqrt(max_area)
        
        h_vals.append(h)
        L2_errors.append(L2)
        H1_errors.append(H1)
        Linf_errors.append(Linf)
        
        print(f"{h:10.5f} {L2:12.6e} {H1:12.6e} {Linf:12.6e}")
    
    # Plot convergence
    plot_convergence_rates(h_vals, L2_errors, H1_errors, Linf_errors)
    
def plot_convergence_rates(h, L2, H1, Linf):
    """Plot errors vs h with reference slopes"""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Log-log plot
    ax.loglog(h, L2, 'o-', label='L² error', linewidth=2, markersize=8)
    ax.loglog(h, H1, 's-', label='H¹ error', linewidth=2, markersize=8)
    ax.loglog(h, Linf, '^-', label='L∞ error', linewidth=2, markersize=8)
    
    # Reference slopes
    h_ref = np.array([h[0], h[-1]])
    ax.loglog(h_ref, 0.1*h_ref**2, 'k--', label='O(h²)', alpha=0.5)
    ax.loglog(h_ref, h_ref, 'k:', label='O(h)', alpha=0.5)
    
    ax.set_xlabel('Mesh size h', fontsize=14)
    ax.set_ylabel('Error', fontsize=14)
    ax.set_title('Convergence Study: 2D Poisson Equation', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../docs/results/convergence.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    convergence_test()