"""
Julia FEM Wrapper for Python
Provides Python interface to Julia FEM implementations
"""

import numpy as np
import os
import sys

# Initialize Julia on first import
_julia_initialized = False
_Main = None


def _init_julia():
    """Initialize Julia and load FEM code"""
    global _julia_initialized, _Main
    
    if _julia_initialized:
        return _Main
    
    try:
        # Import PyJulia
        from julia import Main
        
        # Load the Julia FEM code
        julia_dir = os.path.dirname(os.path.abspath(__file__))
        julia_file = os.path.join(julia_dir, 'fem_assembly.jl')
        
        if not os.path.exists(julia_file):
            raise FileNotFoundError(f"Julia file not found: {julia_file}")
        
        # Load the Julia code
        Main.include(julia_file)
        
        _Main = Main
        _julia_initialized = True
        
        print(f"✓ Julia initialized with {Main.nthreads()} threads")
        
        return _Main
        
    except ImportError:
        raise ImportError(
            "PyJulia not found. Install with:\n"
            "  pip install julia\n"
            "  python3 -c 'import julia; julia.install()'"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Julia: {e}")


def assemble_system(n, f_vals):
    """
    Parallel FEM assembly using Julia
    
    Parameters
    ----------
    n : int
        Number of elements
    f_vals : array_like
        Source term values at nodes (length n+1)
        
    Returns
    -------
    K : ndarray, shape (n, n)
        Stiffness matrix
    F : ndarray, shape (n,)
        Load vector
    """
    Main = _init_julia()
    
    # Ensure correct dtype
    f_vals = np.asarray(f_vals, dtype=np.float64)
    
    if len(f_vals) != n + 1:
        raise ValueError(f"f_vals must have length n+1={n+1}, got {len(f_vals)}")
    
    # Call Julia function
    K_julia, F_julia = Main.assemble_system(int(n), f_vals)
    
    # Convert to NumPy arrays
    K = np.array(K_julia, dtype=np.float64)
    F = np.array(F_julia, dtype=np.float64)
    
    return K, F


def assemble_system_serial(n, f_vals):
    """
    Serial FEM assembly using Julia
    
    Parameters
    ----------
    n : int
        Number of elements
    f_vals : array_like
        Source term values at nodes (length n+1)
        
    Returns
    -------
    K : ndarray, shape (n, n)
        Stiffness matrix
    F : ndarray, shape (n,)
        Load vector
    """
    Main = _init_julia()
    
    f_vals = np.asarray(f_vals, dtype=np.float64)
    
    if len(f_vals) != n + 1:
        raise ValueError(f"f_vals must have length n+1={n+1}, got {len(f_vals)}")
    
    K_julia, F_julia = Main.assemble_system_serial(int(n), f_vals)
    
    K = np.array(K_julia, dtype=np.float64)
    F = np.array(F_julia, dtype=np.float64)
    
    return K, F


def assemble_system_optimized(n, f_vals):
    """
    Optimized parallel FEM assembly using Julia
    
    Parameters
    ----------
    n : int
        Number of elements
    f_vals : array_like
        Source term values at nodes (length n+1)
        
    Returns
    -------
    K : ndarray, shape (n, n)
        Stiffness matrix
    F : ndarray, shape (n,)
        Load vector
    """
    Main = _init_julia()
    
    f_vals = np.asarray(f_vals, dtype=np.float64)
    
    if len(f_vals) != n + 1:
        raise ValueError(f"f_vals must have length n+1={n+1}, got {len(f_vals)}")
    
    K_julia, F_julia = Main.assemble_system_optimized(int(n), f_vals)
    
    K = np.array(K_julia, dtype=np.float64)
    F = np.array(F_julia, dtype=np.float64)
    
    return K, F


if __name__ == "__main__":
    # Test the wrapper
    print("Testing Julia FEM wrapper...")
    
    # Simple test
    n = 10
    x = np.linspace(0, 1, n+1)
    f_vals = 2 - 6*x  # Source term
    
    print(f"\nTest: n = {n}")
    print(f"f_vals shape: {f_vals.shape}")
    
    # Parallel version
    K, F = assemble_system(n, f_vals)
    print(f"\nParallel version:")
    print(f"  K shape: {K.shape}")
    print(f"  F shape: {F.shape}")
    print(f"  K[0,0] = {K[0,0]:.6f}")
    print(f"  F[0] = {F[0]:.6f}")
    
    # Serial version
    K_s, F_s = assemble_system_serial(n, f_vals)
    print(f"\nSerial version:")
    print(f"  K shape: {K_s.shape}")
    print(f"  F shape: {F_s.shape}")
    
    # Check they match
    k_diff = np.max(np.abs(K - K_s))
    f_diff = np.max(np.abs(F - F_s))
    print(f"\nParallel vs Serial:")
    print(f"  Max K difference: {k_diff:.2e}")
    print(f"  Max F difference: {f_diff:.2e}")
    
    if k_diff < 1e-12 and f_diff < 1e-12:
        print("\n✓ Test PASSED!")
    else:
        print("\n✗ Test FAILED!")