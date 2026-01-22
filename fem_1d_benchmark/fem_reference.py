"""
1D Finite Element Method - Python Reference Implementation
Based on Chapter 0, Section 0.4 of Brenner & Scott

This serves as the reference implementation for correctness verification.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def manufactured_solution(x):
    """Exact solution: u(x) = x^2 - x^3"""
    return x**2 - x**3


def manufactured_solution_derivative(x):
    """Exact derivative: u'(x) = 2x - 3x^2"""
    return 2*x - 3*x**2


def source_term(x):
    """Source term: f(x) = -u''(x) = 2 - 6x"""
    return 2 - 6*x


def assemble_system(n, f_vals):
    """
    Assemble stiffness matrix K and load vector F.
    
    Parameters:
    -----------
    n : int
        Number of elements (mesh has n+1 nodes)
    f_vals : ndarray
        Values of source function at nodes (length n+1)
    
    Returns:
    --------
    K : ndarray (n, n)
        Stiffness matrix (with BC applied)
    F : ndarray (n,)
        Load vector (with BC applied)
    """
    h = 1.0 / n
    
    # Allocate arrays (we'll work with the reduced system, excluding u_0 = 0)
    K = np.zeros((n, n))
    F = np.zeros(n)
    
    # Assemble stiffness matrix element by element
    # Element e connects nodes e-1 and e (in 0-based indexing)
    for e in range(1, n+1):
        # Local stiffness matrix for element e
        # [ 1 -1]
        # [-1  1] / h
        k_local = np.array([[1.0, -1.0], 
                           [-1.0, 1.0]]) / h
        
        # Global node indices (0-based)
        i_left = e - 1
        i_right = e
        
        # Map to reduced system (excluding u_0)
        # If i_left = 0, we skip it (BC u_0 = 0)
        # Otherwise, map to index i_left - 1 in reduced system
        
        if i_left > 0:
            # Contribution to K[i_left-1, i_left-1]
            K[i_left-1, i_left-1] += k_local[0, 0]
            # Contribution to K[i_left-1, i_right-1]
            K[i_left-1, i_right-1] += k_local[0, 1]
        
        # Contribution to K[i_right-1, i_right-1]
        K[i_right-1, i_right-1] += k_local[1, 1]
        
        if i_left > 0:
            # Contribution to K[i_right-1, i_left-1]
            K[i_right-1, i_left-1] += k_local[1, 0]
    
    # Assemble load vector using trapezoidal rule
    # (f, phi_i) ≈ (h/2)(f_{i-1} + f_{i+1}) for interior nodes
    for i in range(1, n):  # Nodes 1 to n-1 (0-based: 1 to n-1)
        # In reduced system, this is index i-1
        F[i-1] = (h / 2.0) * (f_vals[i-1] + f_vals[i+1])
    
    # Last node (index n in original, n-1 in reduced)
    # Only left element contributes
    F[n-1] = (h / 2.0) * f_vals[n-1]
    
    return K, F


def assemble_system_sparse(n, f_vals):
    """Same as assemble_system but returns sparse matrix"""
    h = 1.0 / n
    
    K = lil_matrix((n, n))
    F = np.zeros(n)
    
    # Assembly loop
    for e in range(1, n+1):
        k_local = 1.0 / h
        
        i_left = e - 1
        i_right = e
        
        if i_left > 0:
            K[i_left-1, i_left-1] += k_local
            K[i_left-1, i_right-1] -= k_local
        
        K[i_right-1, i_right-1] += k_local
        
        if i_left > 0:
            K[i_right-1, i_left-1] -= k_local
    
    # Load vector
    for i in range(1, n):
        F[i-1] = (h / 2.0) * (f_vals[i-1] + f_vals[i+1])
    F[n-1] = (h / 2.0) * f_vals[n-1]
    
    return K.tocsr(), F


def solve_fem(n, f_vals, use_sparse=True):
    """
    Complete FEM solve: assemble and solve the system.
    
    Parameters:
    -----------
    n : int
        Number of elements
    f_vals : ndarray
        Source function values at nodes
    use_sparse : bool
        Use sparse matrix solver
    
    Returns:
    --------
    U : ndarray (n+1,)
        Solution at all nodes (including u_0 = 0)
    """
    if use_sparse:
        K, F = assemble_system_sparse(n, f_vals)
        U_reduced = spsolve(K, F)
    else:
        K, F = assemble_system(n, f_vals)
        U_reduced = np.linalg.solve(K, F)
    
    # Add back the boundary condition u_0 = 0
    U = np.zeros(n + 1)
    U[1:] = U_reduced
    
    return U


def compute_errors(n, U):
    """
    Compute L2, energy, and max norm errors.
    
    Parameters:
    -----------
    n : int
        Number of elements
    U : ndarray (n+1,)
        Numerical solution at nodes
    
    Returns:
    --------
    dict with keys: 'L2', 'energy', 'max'
    """
    h = 1.0 / n
    x = np.linspace(0, 1, n+1)
    u_exact = manufactured_solution(x)
    u_prime_exact = manufactured_solution_derivative(x)
    
    # Max norm error (at nodes)
    error_max = np.max(np.abs(U - u_exact))
    
    # L2 and energy errors (integrated over elements)
    error_L2_squared = 0.0
    error_energy_squared = 0.0
    
    for e in range(n):
        x_left = e * h
        x_right = (e + 1) * h
        
        # Numerical solution is linear on element
        U_left = U[e]
        U_right = U[e+1]
        
        # Exact solution values
        u_left = u_exact[e]
        u_right = u_exact[e+1]
        
        # L2 error integral using quadrature
        # ∫ (u - u_h)^2 dx on [x_left, x_right]
        # For linear u_h, use Simpson's rule or exact integration
        
        # Exact integration for piecewise linear u_h:
        # u_h(x) = U_left + (U_right - U_left) * (x - x_left) / h
        # u(x) = u_left + (u_right - u_left) * (x - x_left) / h + higher order
        
        # For simplicity, use 3-point Gauss quadrature
        gauss_points = np.array([0.0, 0.5, 1.0])
        gauss_weights = np.array([1.0/6.0, 4.0/6.0, 1.0/6.0])
        
        for xi, w in zip(gauss_points, gauss_weights):
            x_quad = x_left + xi * h
            
            # Evaluate numerical solution at quadrature point
            u_h = U_left + (U_right - U_left) * xi
            
            # Evaluate exact solution at quadrature point
            u_ex = manufactured_solution(x_quad)
            
            # Add to L2 error
            error_L2_squared += w * h * (u_h - u_ex)**2
        
        # Energy error: derivative is constant on element for piecewise linear
        u_h_prime = (U_right - U_left) / h
        
        # For energy error, derivative is constant, so integration is exact
        # Average exact derivative on element (using midpoint)
        x_mid = (x_left + x_right) / 2.0
        u_ex_prime = manufactured_solution_derivative(x_mid)
        
        error_energy_squared += h * (u_h_prime - u_ex_prime)**2
    
    return {
        'L2': np.sqrt(error_L2_squared),
        'energy': np.sqrt(error_energy_squared),
        'max': error_max
    }


def convergence_study(n_values, verbose=True):
    """
    Run convergence study to verify O(h^2) in L2, O(h) in energy.
    
    Parameters:
    -----------
    n_values : list of int
        Mesh sizes to test
    verbose : bool
        Print results
    
    Returns:
    --------
    DataFrame with convergence data
    """
    import pandas as pd
    
    results = []
    
    for n in n_values:
        h = 1.0 / n
        x = np.linspace(0, 1, n+1)
        f_vals = source_term(x)
        
        # Solve
        U = solve_fem(n, f_vals)
        
        # Compute errors
        errors = compute_errors(n, U)
        
        results.append({
            'n': n,
            'h': h,
            'L2_error': errors['L2'],
            'energy_error': errors['energy'],
            'max_error': errors['max']
        })
        
        if verbose:
            print(f"n={n:6d}, h={h:.6f}: "
                  f"L2={errors['L2']:.6e}, "
                  f"E={errors['energy']:.6e}, "
                  f"max={errors['max']:.6e}")
    
    df = pd.DataFrame(results)
    
    # Compute convergence rates
    if len(n_values) > 1:
        df['L2_rate'] = -np.log(df['L2_error'] / df['L2_error'].shift(1)) / np.log(df['h'] / df['h'].shift(1))
        df['energy_rate'] = -np.log(df['energy_error'] / df['energy_error'].shift(1)) / np.log(df['h'] / df['h'].shift(1))
        df['max_rate'] = -np.log(df['max_error'] / df['max_error'].shift(1)) / np.log(df['h'] / df['h'].shift(1))
        
        if verbose:
            print("\nConvergence rates:")
            print(df[['n', 'L2_rate', 'energy_rate', 'max_rate']].to_string(index=False))
            print("\nExpected: L2_rate ≈ 2.0, energy_rate ≈ 1.0, max_rate ≈ 2.0")
    
    return df


if __name__ == '__main__':
    print("=" * 60)
    print("1D FEM Reference Implementation")
    print("=" * 60)
    
    # Quick test
    print("\nQuick test (n=10):")
    n = 10
    x = np.linspace(0, 1, n+1)
    f_vals = source_term(x)
    U = solve_fem(n, f_vals)
    errors = compute_errors(n, U)
    print(f"L2 error: {errors['L2']:.6e}")
    print(f"Energy error: {errors['energy']:.6e}")
    print(f"Max error: {errors['max']:.6e}")
    
    # Convergence study
    print("\n" + "=" * 60)
    print("Convergence Study")
    print("=" * 60)
    n_values = [10, 20, 40, 80, 160, 320]
    df = convergence_study(n_values)
    
    # Plot convergence
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error vs h (log-log)
        ax1.loglog(df['h'], df['L2_error'], 'o-', label='L² error')
        ax1.loglog(df['h'], df['energy_error'], 's-', label='Energy error')
        ax1.loglog(df['h'], df['max_error'], '^-', label='Max error')
        ax1.loglog(df['h'], df['h']**2, 'k--', alpha=0.5, label='O(h²)')
        ax1.loglog(df['h'], df['h'], 'k:', alpha=0.5, label='O(h)')
        ax1.set_xlabel('Mesh size h')
        ax1.set_ylabel('Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Convergence: Error vs h')
        
        # Convergence rates
        ax2.plot(df['n'][1:], df['L2_rate'][1:], 'o-', label='L² rate')
        ax2.plot(df['n'][1:], df['energy_rate'][1:], 's-', label='Energy rate')
        ax2.plot(df['n'][1:], df['max_rate'][1:], '^-', label='Max rate')
        ax2.axhline(2.0, color='k', linestyle='--', alpha=0.5, label='Rate = 2')
        ax2.axhline(1.0, color='k', linestyle=':', alpha=0.5, label='Rate = 1')
        ax2.set_xlabel('Number of elements n')
        ax2.set_ylabel('Convergence rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Convergence Rates')
        
        plt.tight_layout()
        plt.savefig('fem_convergence.png', dpi=150)
        print("\nPlot saved to: fem_convergence.png")
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available, skipping plot")