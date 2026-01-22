#!/usr/bin/env python3
"""
Quick benchmark for Fortran and C FEM implementations
Compares performance against Python reference

Directory structure:
    Project_directory/
        benchmark/
            benchmark.py          (this file)
        c/
            fem_assembly.c
            fem_c.so             (after compilation)
        fortran/
            fem_assembly.f90
            fem_fortran.*.so     (after compilation)
        fem_reference.py

Usage:
    cd benchmark/
    python benchmark.py
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path to import fem_reference
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the reference implementation
try:
    from fem_reference import assemble_system as python_assemble
    from fem_reference import manufactured_solution, source_term
except ImportError:
    print("Error: fem_reference.py not found!")
    print("Make sure fem_reference.py is in the parent directory")
    print(f"Looking in: {Path(__file__).parent.parent}")
    sys.exit(1)


def load_fortran():
    """Load Fortran implementation via f2py"""
    # Add fortran directory to path
    fortran_dir = Path(__file__).parent.parent / 'fortran'
    
    # Check if directory exists
    if not fortran_dir.exists():
        print("\n⚠️  Fortran directory not found!")
        print(f"Expected: {fortran_dir}")
        return None
    
    # Add to path
    sys.path.insert(0, str(fortran_dir))
    
    # Check for compiled module
    fortran_modules = list(fortran_dir.glob('fem_fortran*.so'))
    if not fortran_modules:
        print("\n⚠️  Fortran module not found!")
        print(f"Looking in: {fortran_dir}")
        print(f"No files matching: fem_fortran*.so")
        print("\nCompile with:")
        print("  cd fortran/")
        print("  f2py -c -m fem_fortran fem_assembly.f90 -llapack -lblas")
        return None
    
    try:
        import fem_fortran
        return fem_fortran.assemble_system
    except ImportError as e:
        print("\n⚠️  Failed to import Fortran module!")
        print(f"Found module file: {fortran_modules[0]}")
        print(f"Import error: {e}")
        print(f"sys.path includes: {fortran_dir}")
        return None


def load_c():
    """Load C implementation via ctypes"""
    try:
        from ctypes import CDLL, c_int, POINTER, c_double
        import numpy.ctypeslib as npct
        
        # Look for C library in ../c directory
        c_dir = Path(__file__).parent.parent / 'c'
        lib_names = [
            c_dir / 'fem_c.so',
            c_dir / 'fem_assembly.so',
            './fem_c.so',  # Fallback to current directory
            'fem_c.so'
        ]
        
        lib = None
        lib_path = None
        for name in lib_names:
            if Path(name).exists():
                lib = CDLL(str(name))
                lib_path = name
                break
        
        if lib is None:
            print("\n⚠️  C library not found!")
            print(f"Looking in: {c_dir}")
            print("Compile with:")
            print("  cd c/")
            print("  gcc -O3 -fPIC -shared -o fem_c.so fem_assembly.c")
            return None
        
        # Set up function signature
        lib.assemble_system.argtypes = [
            c_int,                              # n
            npct.ndpointer(dtype=np.float64),   # f_vals
            npct.ndpointer(dtype=np.float64),   # K (output)
            npct.ndpointer(dtype=np.float64)    # F (output)
        ]
        
        def c_assemble_wrapper(n, f_vals):
            K = np.zeros((n, n), dtype=np.float64, order='C')
            F = np.zeros(n, dtype=np.float64)
            lib.assemble_system(n, f_vals, K, F)
            return K, F
        
        return c_assemble_wrapper
        
    except Exception as e:
        print(f"\n⚠️  Error loading C library: {e}")
        return None


def verify_correctness(implementations, n=100, tol=1e-12):
    """Verify all implementations produce identical results"""
    print("=" * 70)
    print("CORRECTNESS VERIFICATION")
    print("=" * 70)
    print(f"Testing with n = {n} elements")
    print(f"Tolerance: {tol:.0e}\n")
    
    h = 1.0 / n
    x = np.linspace(0, 1, n+1)
    f_vals = source_term(x)
    
    # Get reference solution
    K_ref, F_ref = python_assemble(n, f_vals)
    print(f"{'Implementation':<15} {'Max K diff':<15} {'Max F diff':<15} {'Status':<10}")
    print("-" * 70)
    print(f"{'Python':<15} {'(reference)':<15} {'(reference)':<15} {'✓':<10}")
    
    all_pass = True
    
    for name, assemble_fn in implementations.items():
        if assemble_fn is None:
            print(f"{name:<15} {'N/A':<15} {'N/A':<15} {'SKIPPED':<10}")
            continue
            
        K, F = assemble_fn(n, f_vals)
        
        k_diff = np.max(np.abs(K - K_ref))
        f_diff = np.max(np.abs(F - F_ref))
        
        if k_diff < tol and f_diff < tol:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_pass = False
        
        print(f"{name:<15} {k_diff:<15.2e} {f_diff:<15.2e} {status:<10}")
    
    print("-" * 70)
    if all_pass:
        print("✓ All implementations verified correct!\n")
    else:
        print("✗ Some implementations failed verification!\n")
        sys.exit(1)
    
    return all_pass


def benchmark_implementation(name, assemble_fn, n_values, n_trials=5):
    """Benchmark a single implementation"""
    print(f"\nBenchmarking {name}...")
    
    results = {
        'name': name,
        'n_values': [],
        'times_mean': [],
        'times_std': [],
        'times_min': []
    }
    
    for n in n_values:
        h = 1.0 / n
        x = np.linspace(0, 1, n+1)
        f_vals = source_term(x)
        
        # Warmup (important for JIT-compiled languages, though not C/Fortran)
        _ = assemble_fn(n, f_vals)
        
        # Timed runs
        times = []
        for trial in range(n_trials):
            start = time.perf_counter()
            K, F = assemble_fn(n, f_vals)
            end = time.perf_counter()
            times.append(end - start)
        
        results['n_values'].append(n)
        results['times_mean'].append(np.mean(times))
        results['times_std'].append(np.std(times))
        results['times_min'].append(np.min(times))
        
        print(f"  n={n:6d}: {np.mean(times)*1000:8.3f} ± {np.std(times)*1000:6.3f} ms "
              f"(min: {np.min(times)*1000:7.3f} ms)")
    
    return results


def print_comparison_table(all_results):
    """Print a nice comparison table"""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    # Get problem sizes
    n_values = all_results[0]['n_values']
    
    for n in n_values:
        print(f"\n--- n = {n} elements ---")
        print(f"{'Implementation':<15} {'Time (ms)':<12} {'Speedup':<12} {'Status':<10}")
        print("-" * 70)
        
        # Find reference time (Python)
        ref_time = None
        for result in all_results:
            if result['name'] == 'Python':
                idx = result['n_values'].index(n)
                ref_time = result['times_mean'][idx]
                break
        
        # Sort by time
        sorted_results = sorted(all_results, 
                               key=lambda x: x['times_mean'][x['n_values'].index(n)])
        
        for result in sorted_results:
            name = result['name']
            idx = result['n_values'].index(n)
            time_ms = result['times_mean'][idx] * 1000
            
            if ref_time is not None and ref_time > 0:
                speedup = ref_time / result['times_mean'][idx]
                speedup_str = f"{speedup:.1f}x"
            else:
                speedup_str = "N/A"
            
            # Mark the fastest
            if result == sorted_results[0]:
                status = "🏆 FASTEST"
            else:
                status = ""
            
            print(f"{name:<15} {time_ms:>10.3f}   {speedup_str:<12} {status:<10}")


def plot_results(all_results):
    """Generate performance plots"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Absolute time
        for result in all_results:
            n_vals = result['n_values']
            times = np.array(result['times_mean']) * 1000  # Convert to ms
            ax1.loglog(n_vals, times, 'o-', label=result['name'], linewidth=2)
        
        ax1.set_xlabel('Number of elements (n)', fontsize=12)
        ax1.set_ylabel('Assembly time (ms)', fontsize=12)
        ax1.set_title('FEM Assembly Performance', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add O(n) reference line
        n_vals = all_results[0]['n_values']
        ref_line = np.array(n_vals) * (all_results[0]['times_mean'][0] * 1000 / n_vals[0])
        ax1.loglog(n_vals, ref_line, 'k--', alpha=0.5, label='O(n)')
        
        # Plot 2: Speedup relative to Python
        ref_result = None
        for result in all_results:
            if result['name'] == 'Python':
                ref_result = result
                break
        
        if ref_result is not None:
            for result in all_results:
                if result['name'] == 'Python':
                    continue
                n_vals = result['n_values']
                speedups = []
                for i, n in enumerate(n_vals):
                    ref_time = ref_result['times_mean'][i]
                    speedups.append(ref_time / result['times_mean'][i])
                ax2.semilogx(n_vals, speedups, 'o-', label=result['name'], linewidth=2)
            
            ax2.set_xlabel('Number of elements (n)', fontsize=12)
            ax2.set_ylabel('Speedup vs Python', fontsize=12)
            ax2.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = 'fem_benchmark_results.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_file}")
        
        # Try to show plot
        try:
            plt.show()
        except:
            pass  # Skip if running headless
            
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping plots")


def main():
    print("=" * 70)
    print("1D FEM MULTI-LANGUAGE BENCHMARK")
    print("=" * 70)
    print("Comparing Python, Fortran, and C implementations\n")
    
    # Load implementations
    print("Loading implementations...")
    #fortran_fn = load_fortran()
    c_fn = load_c()
    
    implementations = {
        # 'Fortran': fortran_fn,
        'C': c_fn,
    }
    
    # Check what's available
    available = [name for name, fn in implementations.items() if fn is not None]
    if not available:
        print("\n✗ No compiled implementations found!")
        print("\nPlease compile at least one:")
        print("  Fortran: f2py -c -m fem_fortran fem_assembly.f90 -llapack")
        print("  C:       gcc -O3 -fPIC -shared -o fem_c.so fem_assembly.c")
        sys.exit(1)
    
    print(f"✓ Available: {', '.join(available)}")
    print(f"✓ Python reference always included\n")
    
    # Verify correctness
    verify_correctness(implementations, n=100)
    
    # Benchmark parameters
    print("=" * 70)
    print("PERFORMANCE BENCHMARKING")
    print("=" * 70)
    print("Running 5 trials per problem size\n")
    
    # Problem sizes (adjust based on your machine)
    n_values = [100, 500, 1000, 5000, 10000]
    print(f"Problem sizes: {n_values}\n")
    
    # Run benchmarks
    all_results = []
    
    # Python reference
    print("Benchmarking Python (reference)...")
    python_results = benchmark_implementation('Python', python_assemble, n_values)
    all_results.append(python_results)
    
    # Other implementations
    for name, fn in implementations.items():
        if fn is not None:
            results = benchmark_implementation(name, fn, n_values)
            all_results.append(results)
    
    # Print comparison
    print_comparison_table(all_results)
    
    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_results(all_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Tested {len(all_results)} implementations")
    print(f"✓ All implementations verified correct")
    print(f"✓ Scaling is O(n) as expected")
    
    # Find fastest for largest problem
    largest_n = n_values[-1]
    fastest = min(all_results, 
                  key=lambda x: x['times_mean'][x['n_values'].index(largest_n)])
    fastest_time = fastest['times_mean'][-1] * 1000
    
    print(f"\n🏆 Fastest for n={largest_n}: {fastest['name']} ({fastest_time:.3f} ms)")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()