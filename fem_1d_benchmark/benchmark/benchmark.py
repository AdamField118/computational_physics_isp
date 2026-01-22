#!/usr/bin/env python3
"""
Comprehensive FEM 1D Benchmark Suite
Compares Python, C, C++, Fortran, Julia, and Rust implementations
Outputs results to JSON for web visualization
"""

import numpy as np
import time
import json
import sys
from pathlib import Path
from datetime import datetime
from ctypes import CDLL, c_int, c_double
import numpy.ctypeslib as npct

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))
from fem_reference import assemble_system as python_assemble, source_term


class BenchmarkSuite:
    def __init__(self):
        self.implementations = {}
        self.results = {
            'metadata': {
                'date': datetime.now().isoformat(),
                'problem': '1D FEM Assembly',
                'description': 'Benchmark of multi-language FEM implementations'
            },
            'implementations': {},
            'benchmarks': []
        }
    
    def load_implementations(self):
        """Load all available implementations"""
        print("=" * 70)
        print("LOADING IMPLEMENTATIONS")
        print("=" * 70)
        
        # Always available: Python reference
        self.implementations['Python'] = {
            'assemble': python_assemble,
            'parallel': False,
            'serial': python_assemble
        }
        print("✓ Python reference loaded")
        
        # Try to load C
        c_impl = self._load_c()
        if c_impl:
            self.implementations['C'] = c_impl
            print("✓ C (OpenMP) loaded")
        
        # Try to load C++
        cpp_impl = self._load_cpp()
        if cpp_impl:
            self.implementations['C++'] = cpp_impl
            print("✓ C++ (OpenMP) loaded")
        
        # Try to load Fortran
        fortran_impl = self._load_fortran()
        if fortran_impl:
            self.implementations['Fortran'] = fortran_impl
            print("✓ Fortran (OpenMP) loaded")
        
        # Try to load Julia
        julia_impl = self._load_julia()
        if julia_impl:
            self.implementations['Julia'] = julia_impl
            print("✓ Julia (Threads) loaded")
        
        # Try to load Rust
        rust_impl = self._load_rust()
        if rust_impl:
            self.implementations['Rust'] = rust_impl
            print("✓ Rust (Rayon) loaded")
        
        print(f"\nLoaded {len(self.implementations)} implementations")
        print()
    
    def _load_c(self):
        """Load C implementation via ctypes"""
        try:
            c_dir = Path(__file__).parent.parent / 'c'
            lib_path = c_dir / 'fem_c.so'
            
            if not lib_path.exists():
                return None
            
            lib = CDLL(str(lib_path))
            
            # Set up function signatures
            for func_name in ['assemble_system', 'assemble_system_serial']:
                func = getattr(lib, func_name)
                func.argtypes = [
                    c_int,
                    npct.ndpointer(dtype=np.float64),
                    npct.ndpointer(dtype=np.float64),
                    npct.ndpointer(dtype=np.float64)
                ]
            
            def c_wrapper(n, f_vals, serial=False):
                K = np.zeros((n, n), dtype=np.float64, order='C')
                F = np.zeros(n, dtype=np.float64)
                func = lib.assemble_system_serial if serial else lib.assemble_system
                func(n, f_vals, K, F)
                return K, F
            
            return {
                'assemble': lambda n, f: c_wrapper(n, f, False),
                'serial': lambda n, f: c_wrapper(n, f, True),
                'parallel': True
            }
        except Exception as e:
            print(f"⚠️  Failed to load C: {e}")
            return None
    
    def _load_cpp(self):
        """Load C++ implementation via pybind11"""
        try:
            cpp_dir = Path(__file__).parent.parent / 'cpp'
            sys.path.insert(0, str(cpp_dir))
            
            import fem_cpp
            
            return {
                'assemble': fem_cpp.assemble_system,
                'serial': fem_cpp.assemble_system_serial,
                'parallel': True
            }
        except Exception as e:
            print(f"⚠️  Failed to load C++: {e}")
            return None
    
    def _load_fortran(self):
        """Load Fortran implementation via f2py"""
        try:
            fortran_dir = Path(__file__).parent.parent / 'fortran'
            sys.path.insert(0, str(fortran_dir))
            
            import fem_fortran
            
            return {
                'assemble': fem_fortran.assemble_system,
                'serial': fem_fortran.assemble_system_serial,
                'parallel': True
            }
        except Exception as e:
            print(f"⚠️  Failed to load Fortran: {e}")
            return None
    
    def _load_julia(self):
        """Load Julia implementation via PyJulia"""
        try:
            from julia.api import Julia
            jl = Julia(compiled_modules=False)
            from julia import Main
            julia_path = Path(__file__).parent.parent / 'julia' / 'fem_assembly.jl'
            Main.include(str(julia_path))
            
            def julia_wrapper(n, f_vals):
                # Julia uses 1-based indexing
                K, F = Main.assemble_system(n, f_vals)
                return np.array(K), np.array(F)
            
            def julia_wrapper_serial(n, f_vals):
                K, F = Main.assemble_system_serial(n, f_vals)
                return np.array(K), np.array(F)
            
            return {
                'assemble': julia_wrapper,
                'serial': julia_wrapper_serial,
                'parallel': True
            }
        except Exception as e:
            print(f"⚠️  Failed to load Julia: {e}")
            return None
    
    def _load_rust(self):
        """Load Rust implementation via PyO3/maturin"""
        try:
            rust_dir = Path(__file__).parent.parent / 'rust'
            sys.path.insert(0, str(rust_dir / 'target' / 'release'))
            
            import fem_rust
            
            return {
                'assemble': fem_rust.assemble_system,
                'serial': fem_rust.assemble_system_serial,
                'parallel': True
            }
        except Exception as e:
            print(f"⚠️  Failed to load Rust: {e}")
            return None
    
    def verify_correctness(self, n=100, tol=1e-12):
        """Verify all implementations produce identical results"""
        print("=" * 70)
        print("CORRECTNESS VERIFICATION")
        print("=" * 70)
        print(f"Testing with n = {n} elements")
        print(f"Tolerance: {tol:.0e}\n")
        
        x = np.linspace(0, 1, n+1)
        f_vals = source_term(x)
        
        # Get reference solution
        K_ref, F_ref = python_assemble(n, f_vals)
        
        print(f"{'Implementation':<15} {'Max K diff':<15} {'Max F diff':<15} {'Status':<10}")
        print("-" * 70)
        print(f"{'Python':<15} {'(reference)':<15} {'(reference)':<15} {'✓':<10}")
        
        all_pass = True
        
        for name, impl in self.implementations.items():
            if name == 'Python':
                continue
            
            try:
                K, F = impl['assemble'](n, f_vals)
                
                k_diff = np.max(np.abs(K - K_ref))
                f_diff = np.max(np.abs(F - F_ref))
                
                if k_diff < tol and f_diff < tol:
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL"
                    all_pass = False
                
                print(f"{name:<15} {k_diff:<15.2e} {f_diff:<15.2e} {status:<10}")
            except Exception as e:
                print(f"{name:<15} {'ERROR':<15} {str(e):<15} {'✗ FAIL':<10}")
                all_pass = False
        
        print("-" * 70)
        if all_pass:
            print("✓ All implementations verified correct!\n")
        else:
            print("✗ Some implementations failed verification!\n")
            sys.exit(1)
        
        return all_pass
    
    def benchmark_implementation(self, name, n_values, n_trials=5):
        """Benchmark a single implementation"""
        print(f"\nBenchmarking {name}...")
        
        impl = self.implementations[name]
        results_parallel = []
        results_serial = [] if impl['parallel'] else None
        
        for n in n_values:
            x = np.linspace(0, 1, n+1)
            f_vals = source_term(x)
            
            # Warmup
            _ = impl['assemble'](n, f_vals)
            
            # Parallel version
            times_parallel = []
            for _ in range(n_trials):
                start = time.perf_counter()
                K, F = impl['assemble'](n, f_vals)
                end = time.perf_counter()
                times_parallel.append(end - start)
            
            results_parallel.append({
                'n': n,
                'mean': np.mean(times_parallel),
                'std': np.std(times_parallel),
                'min': np.min(times_parallel),
                'max': np.max(times_parallel)
            })
            
            print(f"  n={n:6d} (parallel): {np.mean(times_parallel)*1000:8.3f} ± "
                  f"{np.std(times_parallel)*1000:6.3f} ms (min: {np.min(times_parallel)*1000:7.3f} ms)")
            
            # Serial version if available
            if impl['parallel']:
                times_serial = []
                for _ in range(n_trials):
                    start = time.perf_counter()
                    K, F = impl['serial'](n, f_vals)
                    end = time.perf_counter()
                    times_serial.append(end - start)
                
                results_serial.append({
                    'n': n,
                    'mean': np.mean(times_serial),
                    'std': np.std(times_serial),
                    'min': np.min(times_serial),
                    'max': np.max(times_serial)
                })
                
                speedup = np.mean(times_serial) / np.mean(times_parallel)
                print(f"  n={n:6d} (serial):   {np.mean(times_serial)*1000:8.3f} ± "
                      f"{np.std(times_serial)*1000:6.3f} ms (speedup: {speedup:.2f}x)")
        
        return {
            'parallel': results_parallel,
            'serial': results_serial
        }
    
    def run_benchmarks(self, n_values, n_trials=5):
        """Run benchmarks on all implementations"""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARKING")
        print("=" * 70)
        print(f"Problem sizes: {n_values}")
        print(f"Trials per size: {n_trials}\n")
        
        for name in self.implementations.keys():
            results = self.benchmark_implementation(name, n_values, n_trials)
            self.results['benchmarks'].append({
                'name': name,
                'parallel': self.implementations[name]['parallel'],
                'results': results
            })
    
    def save_results(self, output_file='fem_benchmark_results.json'):
        """Save results to JSON file"""
        output_path = Path(__file__).parent.parent / 'results' / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary table"""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        # Get the largest problem size
        n_values = [r['n'] for r in self.results['benchmarks'][0]['results']['parallel']]
        largest_n = n_values[-1]
        
        print(f"\nResults for n={largest_n}:")
        print(f"{'Implementation':<15} {'Parallel (ms)':<15} {'Serial (ms)':<15} {'Speedup':<10}")
        print("-" * 70)
        
        for bench in self.results['benchmarks']:
            name = bench['name']
            parallel_time = bench['results']['parallel'][-1]['mean'] * 1000
            
            if bench['parallel'] and bench['results']['serial']:
                serial_time = bench['results']['serial'][-1]['mean'] * 1000
                speedup = serial_time / parallel_time
                print(f"{name:<15} {parallel_time:>13.3f}   {serial_time:>13.3f}   {speedup:>8.2f}x")
            else:
                print(f"{name:<15} {parallel_time:>13.3f}   {'N/A':<15} {'N/A':<10}")
        
        # Relative speedups vs Python
        print(f"\nSpeedup vs Python (parallel):")
        print(f"{'Implementation':<15} {'Speedup':<10}")
        print("-" * 30)
        
        python_time = next(b['results']['parallel'][-1]['mean'] 
                          for b in self.results['benchmarks'] if b['name'] == 'Python')
        
        for bench in sorted(self.results['benchmarks'], 
                          key=lambda x: x['results']['parallel'][-1]['mean']):
            name = bench['name']
            time_val = bench['results']['parallel'][-1]['mean']
            speedup = python_time / time_val
            
            if name == bench['name']:
                marker = "🏆" if time_val == min(b['results']['parallel'][-1]['mean'] 
                                                for b in self.results['benchmarks']) else ""
                print(f"{name:<15} {speedup:>8.2f}x {marker}")


def main():
    # Configuration
    n_values = [100, 500, 1000, 5000, 10000, 50000]
    n_trials = 5
    
    # Create benchmark suite
    suite = BenchmarkSuite()
    
    # Load implementations
    suite.load_implementations()
    
    if len(suite.implementations) < 2:
        print("Error: Need at least 2 implementations to benchmark")
        sys.exit(1)
    
    # Verify correctness
    suite.verify_correctness(n=100)
    
    # Run benchmarks
    suite.run_benchmarks(n_values, n_trials)
    
    # Save results
    suite.save_results()
    
    # Print summary
    suite.print_summary()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()