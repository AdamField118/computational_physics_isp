"""
1D Finite Element Method - Python Reference Implementation
Based on Chapter 0, Section 0.4 of Brenner & Scott
"""

import numpy as np


def manufactured_solution(x):
    """Exact solution: u(x) = x^2 - x^3"""
    return x**2 - x**3


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
    
    # Allocate arrays
    K = np.zeros((n, n))
    F = np.zeros(n)
    
    # Assemble stiffness matrix element by element
    for e in range(1, n+1):
        k_local = 1.0 / h
        
        i_left = e - 1
        i_right = e
        
        if i_left > 0:
            idx_left = i_left - 1
            idx_right = i_right - 1
            
            K[idx_left, idx_left] += k_local
            K[idx_left, idx_right] += -k_local
            K[idx_right, idx_left] += -k_local
        
        idx_right = i_right - 1
        K[idx_right, idx_right] += k_local
    
    # Assemble load vector using trapezoidal rule
    for i in range(1, n):
        F[i-1] = (h / 2.0) * (f_vals[i-1] + f_vals[i+1])
    
    F[n-1] = (h / 2.0) * f_vals[n-1]
    
    return K, F


if __name__ == '__main__':
    # Quick test
    n = 10
    x = np.linspace(0, 1, n+1)
    f_vals = source_term(x)
    K, F = assemble_system(n, f_vals)
    print(f"K shape: {K.shape}, F shape: {F.shape}")
    print(f"K[0,0] = {K[0,0]:.6f} (expected: {2.0/0.1:.6f})")
    print(f"K[0,1] = {K[0,1]:.6f} (expected: {-1.0/0.1:.6f})")
    print(f"F[0] = {F[0]:.6f}")