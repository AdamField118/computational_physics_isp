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
    
    print("=" * 60)
    print("Convergence Study: 2D Poisson Equation")
    print("=" * 60)
    print(f"{'h':>10} {'L2 error':>12} {'H1 error':>12} {'Linf error':>12}")
    print("-" * 60)
    
    for max_area in mesh_sizes:
        # Solve (note: currently uses hardcoded sine source in Fortran)
        solver = PoissonSolver2D('unit_square', max_area=max_area)
        solver.solve()  # Uses default sine source
        
        # Compute errors
        L2, H1, Linf = solver.compute_errors(mms.u_exact)
        
        # Approximate h
        h = np.sqrt(max_area)
        
        h_vals.append(h)
        L2_errors.append(L2)
        H1_errors.append(H1)
        Linf_errors.append(Linf)
        
        print(f"{h:10.5f} {L2:12.6e} {H1:12.6e} {Linf:12.6e}")
    
    print("=" * 60)
    
    # Compute convergence rates
    compute_rates(h_vals, L2_errors, H1_errors, Linf_errors)
    
    # Plot convergence
    plot_convergence_rates(h_vals, L2_errors, H1_errors, Linf_errors)
    

def compute_rates(h, L2, H1, Linf):
    """Compute experimental convergence rates"""
    print("\nConvergence Rates:")
    print("-" * 60)
    
    # Compute rates between consecutive refinements
    for i in range(1, len(h)):
        ratio = h[i-1] / h[i]
        
        L2_rate = np.log(L2[i-1] / L2[i]) / np.log(ratio)
        H1_rate = np.log(H1[i-1] / H1[i]) / np.log(ratio)
        Linf_rate = np.log(Linf[i-1] / Linf[i]) / np.log(ratio)
        
        print(f"h: {h[i]:.5f}  |  L2: {L2_rate:.2f}  |  H1: {H1_rate:.2f}  |  Linf: {Linf_rate:.2f}")
    
    print("-" * 60)
    print("Expected rates: L2 ~ 2.0, H1 ~ 1.0")
    print("=" * 60)


def plot_convergence_rates(h, L2, H1, Linf):
    """Plot errors vs h with reference slopes"""
    
    # Set dark theme
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Log-log plot
    ax.loglog(h, L2, 'o-', label='L² error', linewidth=2, markersize=8, color='#00ff41')
    ax.loglog(h, H1, 's-', label='H¹ error', linewidth=2, markersize=8, color='#00aaff')
    ax.loglog(h, Linf, '^-', label='L∞ error', linewidth=2, markersize=8, color='#ffaa00')
    
    # Reference slopes
    h_ref = np.array([h[0], h[-1]])
    
    # O(h²) reference
    ref_scale = L2[0] / h[0]**2
    ax.loglog(h_ref, ref_scale * h_ref**2, 'w--', label='O(h²)', alpha=0.5, linewidth=1.5)
    
    # O(h) reference
    ref_scale = H1[0] / h[0]
    ax.loglog(h_ref, ref_scale * h_ref, 'w:', label='O(h)', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Mesh size h', fontsize=14)
    ax.set_ylabel('Error', fontsize=14)
    ax.set_title('Convergence Study: 2D Poisson Equation', fontsize=16, color='#00ff41')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('../docs/results/convergence.png', dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    print("\nPlot saved to: ../docs/results/convergence.png")
    plt.show()


if __name__ == '__main__':
    convergence_test()