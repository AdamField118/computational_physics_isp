#!/usr/bin/env python3
"""
Debug script for Fortran FEM implementation
Tests the f2py interface and array handling
"""

import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fortran'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

print("="*70)
print("FORTRAN FEM DEBUG SCRIPT")
print("="*70)

# Try to import reference
try:
    from fem_reference import source_term, assemble_system as ref_assemble
    print("✓ Reference implementation imported")
except ImportError as e:
    print(f"✗ Failed to import reference: {e}")
    sys.exit(1)

# Try to import Fortran module
print("\nAttempting to import Fortran module...")
try:
    import fem_fortran
    print("✓ Fortran module imported successfully")
    print(f"  Module location: {fem_fortran.__file__}")
except ImportError as e:
    print(f"✗ Failed to import Fortran module: {e}")
    print("\nTo build Fortran module:")
    print("  cd fortran/")
    print("  f2py -c -m fem_fortran fem_assembly.f90 --f90flags='-fopenmp -O3' -lgomp")
    sys.exit(1)

# Check what functions are available
print("\nAvailable functions in fem_fortran:")
for attr in dir(fem_fortran):
    if not attr.startswith('_'):
        print(f"  - {attr}")

print("\n" + "="*70)
print("TESTING WITH DIFFERENT VALUES OF n")
print("="*70)

def test_fortran(n):
    """Test Fortran implementation for a given n"""
    print(f"\n{'─'*70}")
    print(f"Testing n = {n}")
    print(f"{'─'*70}")
    
    # Create test data
    x = np.linspace(0, 1, n+1)
    f_vals = source_term(x)
    
    print(f"  x shape: {x.shape}")
    print(f"  f_vals shape: {f_vals.shape}")
    print(f"  f_vals dtype: {f_vals.dtype}")
    print(f"  f_vals range: [{f_vals.min():.3f}, {f_vals.max():.3f}]")
    
    # Check if f_vals is C-contiguous
    print(f"  f_vals C-contiguous: {f_vals.flags['C_CONTIGUOUS']}")
    print(f"  f_vals F-contiguous: {f_vals.flags['F_CONTIGUOUS']}")
    
    # Get reference solution
    try:
        K_ref, F_ref = ref_assemble(n, f_vals)
        print(f"\n  Reference solution:")
        print(f"    K shape: {K_ref.shape}")
        print(f"    F shape: {F_ref.shape}")
        print(f"    K[0,0] = {K_ref[0,0]:.6f}")
        print(f"    F[0] = {F_ref[0]:.6f}")
    except Exception as e:
        print(f"  ✗ Reference failed: {e}")
        return False
    
    # Test Fortran with exact array that reference uses
    print(f"\n  Testing Fortran with f_vals...")
    
    # Try to call the function
    try:
        # Method 1: Direct call with positional args
        print(f"    Calling: fem_fortran.assemble_system(n={n}, f_vals=...)")
        K, F = fem_fortran.assemble_system(n=n, f_vals=f_vals)
        
        print(f"    ✓ Call succeeded!")
        print(f"    K shape: {K.shape}")
        print(f"    F shape: {F.shape}")
        print(f"    K dtype: {K.dtype}")
        print(f"    F dtype: {F.dtype}")
        
        # Compare with reference
        k_diff = np.max(np.abs(K - K_ref))
        f_diff = np.max(np.abs(F - F_ref))
        
        print(f"\n  Comparison with reference:")
        print(f"    Max K difference: {k_diff:.2e}")
        print(f"    Max F difference: {f_diff:.2e}")
        
        if k_diff < 1e-12 and f_diff < 1e-12:
            print(f"    ✓ PASS - Results match reference!")
            return True
        else:
            print(f"    ✗ FAIL - Results differ from reference")
            print(f"\n  K difference matrix (max 5x5):")
            K_diff = K - K_ref
            print(K_diff[:5, :5])
            print(f"\n  F difference vector (first 10):")
            print((F - F_ref)[:10])
            return False
            
    except Exception as e:
        print(f"    ✗ Call failed!")
        print(f"    Error type: {type(e).__name__}")
        print(f"    Error message: {e}")
        
        # Try to get more info
        import traceback
        print(f"\n  Full traceback:")
        traceback.print_exc()
        
        # Try alternate calling methods
        print(f"\n  Trying alternate calling methods...")
        
        # Try with positional args only
        try:
            print(f"    Method 2: fem_fortran.assemble_system(n, f_vals)")
            K, F = fem_fortran.assemble_system(n, f_vals)
            print(f"    ✓ Method 2 worked!")
            return True
        except Exception as e2:
            print(f"    ✗ Method 2 failed: {e2}")
        
        # Try with explicit copy
        try:
            print(f"    Method 3: with np.asfortranarray(f_vals)")
            f_vals_f = np.asfortranarray(f_vals)
            K, F = fem_fortran.assemble_system(n, f_vals_f)
            print(f"    ✓ Method 3 worked!")
            return True
        except Exception as e3:
            print(f"    ✗ Method 3 failed: {e3}")
        
        return False

# Test with increasing n values
test_cases = [2, 3, 5, 10, 100]
results = {}

for n in test_cases:
    success = test_fortran(n)
    results[n] = success

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

all_pass = all(results.values())
for n, passed in results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  n = {n:4d}: {status}")

if all_pass:
    print("\n✓ All tests PASSED!")
    sys.exit(0)
else:
    print("\n✗ Some tests FAILED")
    sys.exit(1)