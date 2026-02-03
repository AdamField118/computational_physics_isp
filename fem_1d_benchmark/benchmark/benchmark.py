#!/usr/bin/env python3
"""
FEM 1D Benchmark Suite - Uniform Interface (Julia Fix)
ALL languages receive pre-allocated numpy arrays

Note: Julia allocates internally (PyCall limitation) then copies back
"""

import numpy as np
import time
import json
import sys
import gc
from pathlib import Path
from datetime import datetime
from ctypes import CDLL, c_int, c_double
import numpy.ctypeslib as npct

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))
from fem_reference import assemble_system as python_assemble, source_term

# Global Julia initialization flag
_julia_initialized = False
_julia_Main = None


class BenchmarkSuite:
    def __init__(self):
        self.implementations = {}
        self.results = {
            'metadata': {
                'date': datetime.now().isoformat(),
                'problem': '1D FEM Assembly (Uniform Interface)',
                'description': 'All languages receive pre-allocated arrays - fair comparison'
            },
            'benchmarks': []
        }
    
    def load_implementations(self):
        """Load all available implementations"""
        print("=" * 70)
        print("LOADING IMPLEMENTATIONS (Uniform Interface)")
        print("=" * 70)
        print("All languages receive pre-allocated numpy arrays\n")
        
        # Python reference
        def python_wrapper(n, f_vals, K, F):
            """Python fills pre-allocated arrays"""
            K_temp, F_temp = python_assemble(n, f_vals)
            K[:] = K_temp
            F[:] = F_temp
        
        self.implementations['Python'] = {
            'assemble': python_wrapper,
            'order': 'C'
        }
        print("✓ Python reference loaded")
        
        # C
        c_impl = self._load_c()
        if c_impl:
            self.implementations['C'] = c_impl
            print("✓ C (OpenMP) loaded")
        
        # C++
        cpp_impl = self._load_cpp()
        if cpp_impl:
            self.implementations['C++'] = cpp_impl
            print("✓ C++ (OpenMP) loaded")
        
        # Fortran
        fortran_impl = self._load_fortran()
        if fortran_impl:
            self.implementations['Fortran'] = fortran_impl
            print("✓ Fortran (OpenMP) loaded")
        
        # Julia
        julia_impl = self._load_julia()
        if julia_impl:
            self.implementations['Julia'] = julia_impl
            print("✓ Julia (Threads) loaded")
        
        # Rust
        rust_impl = self._load_rust()
        if rust_impl:
            self.implementations['Rust'] = rust_impl
            print("✓ Rust (Rayon) loaded")
        
        print(f"\nLoaded {len(self.implementations)} implementations\n")
    
    def _load_c(self):
        """Load C implementation via ctypes"""
        try:
            c_dir = Path(__file__).parent.parent / 'c'
            lib_path = c_dir / 'fem_c.so'
            
            if not lib_path.exists():
                return None
            
            lib = CDLL(str(lib_path))
            lib.assemble_system.argtypes = [
                c_int,
                npct.ndpointer(dtype=np.float64),
                npct.ndpointer(dtype=np.float64),
                npct.ndpointer(dtype=np.float64)
            ]
            
            def c_wrapper(n, f_vals, K, F):
                """C fills pre-allocated arrays"""
                lib.assemble_system(n, f_vals, K, F)
            
            return {
                'assemble': c_wrapper,
                'order': 'C'
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
            
            def cpp_wrapper(n, f_vals, K, F):
                """C++ fills pre-allocated arrays"""
                fem_cpp.assemble_system(n, f_vals, K, F)
            
            return {
                'assemble': cpp_wrapper,
                'order': 'C'
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
            
            def fortran_wrapper(n, f_vals, K, F):
                """Fortran fills pre-allocated arrays"""
                fem_fortran.assemble_system(n=n, f_vals=f_vals, k=K, f=F)
            
            return {
                'assemble': fortran_wrapper,
                'order': 'F'
            }
        except Exception as e:
            print(f"⚠️  Failed to load Fortran: {e}")
            return None
    
    def _load_julia(self):
        """Load Julia implementation via PyJulia"""
        global _julia_initialized, _julia_Main
        
        try:
            # Initialize Julia only once
            if not _julia_initialized:
                from julia.api import Julia
                jl = Julia(compiled_modules=False)
                from julia import Main
                julia_path = Path(__file__).parent.parent / 'julia' / 'fem_assembly.jl'
                Main.include(str(julia_path))
                _julia_Main = Main
                _julia_initialized = True
            
            Main = _julia_Main
            
            def julia_wrapper(n, f_vals, K, F):
                """Julia allocates internally, then copies to pre-allocated arrays"""
                # PyCall doesn't reliably support in-place modification
                # So Julia allocates and returns, we copy back
                f_vals = np.asarray(f_vals, dtype=np.float64)
                
                # Julia allocates and returns
                K_julia, F_julia = Main.assemble_system(int(n), f_vals)
                
                # Copy back to pre-allocated arrays
                K[:] = np.array(K_julia, dtype=np.float64)
                F[:] = np.array(F_julia, dtype=np.float64)
                
                # Clean up Julia objects
                del K_julia, F_julia
            
            return {
                'assemble': julia_wrapper,
                'order': 'F'  # Julia uses column-major
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
            
            def rust_wrapper(n, f_vals, K, F):
                """Rust fills pre-allocated arrays"""
                fem_rust.assemble_system(n, f_vals, K, F)
            
            return {
                'assemble': rust_wrapper,
                'order': 'C'
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
        K_ref = np.zeros((n, n), dtype=np.float64, order='C')
        F_ref = np.zeros(n, dtype=np.float64)
        self.implementations['Python']['assemble'](n, f_vals, K_ref, F_ref)
        
        print(f"{'Implementation':<15} {'Max K diff':<15} {'Max F diff':<15} {'Status':<10}")
        print("-" * 70)
        print(f"{'Python':<15} {'(reference)':<15} {'(reference)':<15} {'✓':<10}")
        
        all_pass = True
        
        for name, impl in self.implementations.items():
            if name == 'Python':
                continue
            
            try:
                # Allocate arrays with correct order for this language
                order = impl['order']
                K = np.zeros((n, n), dtype=np.float64, order=order)
                F = np.zeros(n, dtype=np.float64)
                
                # Fill arrays
                impl['assemble'](n, f_vals, K, F)
                
                # Compare
                k_diff = np.max(np.abs(K - K_ref))
                f_diff = np.max(np.abs(F - F_ref))
                
                if k_diff < tol and f_diff < tol:
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL"
                    all_pass = False
                
                print(f"{name:<15} {k_diff:<15.2e} {f_diff:<15.2e} {status:<10}")
                
                del K, F
                gc.collect()
                
            except Exception as e:
                print(f"{name:<15} {'ERROR':<15} {str(e):<15} {'✗ FAIL':<10}")
                all_pass = False
        
        del K_ref, F_ref
        gc.collect()
        
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
        order = impl['order']
        results = []
        
        for n in n_values:
            gc.collect()
            
            x = np.linspace(0, 1, n+1)
            f_vals = source_term(x)
            
            # Pre-allocate arrays with correct order
            K = np.zeros((n, n), dtype=np.float64, order=order)
            F = np.zeros(n, dtype=np.float64)
            
            # Warmup
            impl['assemble'](n, f_vals, K, F)
            
            # Timed runs
            times = []
            for _ in range(n_trials):
                K.fill(0.0)
                F.fill(0.0)
                
                gc.collect()
                start = time.perf_counter()
                impl['assemble'](n, f_vals, K, F)
                end = time.perf_counter()
                times.append(end - start)
            
            results.append({
                'n': n,
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            })
            
            print(f"  n={n:6d}: {np.mean(times)*1000:8.3f} ± {np.std(times)*1000:6.3f} ms "
                  f"(min: {np.min(times)*1000:7.3f} ms)")
            
            del K, F, x, f_vals
            gc.collect()
        
        return results
    
    def save_results(self, output_file='fem_benchmark_results.json'):
        """Save results to JSON file"""
        output_path = Path(__file__).parent.parent / 'results' / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
        return output_path
    
    def run_benchmarks(self, n_values, n_trials=5):
        """Run benchmarks on all implementations"""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARKING")
        print("=" * 70)
        print(f"Problem sizes: {n_values}")
        print(f"Trials per size: {n_trials}")
        print("Note: All languages receive pre-allocated arrays\n")
        
        for name in self.implementations.keys():
            results = self.benchmark_implementation(name, n_values, n_trials)
            self.results['benchmarks'].append({
                'name': name,
                'results': results
            })
            
            self.save_results()
            print(f"  ✓ Saved after {name}")
            gc.collect()
    
    def print_summary(self):
        """Print summary table"""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        n_values = [r['n'] for r in self.results['benchmarks'][0]['results']]
        largest_n = n_values[-1]
        
        print(f"\nResults for n={largest_n}:")
        print(f"{'Implementation':<15} {'Time (ms)':<15} {'Speedup vs Python':<20}")
        print("-" * 70)
        
        python_time = next(b['results'][-1]['mean'] 
                          for b in self.results['benchmarks'] if b['name'] == 'Python')
        
        for bench in sorted(self.results['benchmarks'], 
                          key=lambda x: x['results'][-1]['mean']):
            name = bench['name']
            time_val = bench['results'][-1]['mean'] * 1000
            speedup = python_time / bench['results'][-1]['mean']
            
            marker = "🏆 " if time_val == min(b['results'][-1]['mean'] * 1000
                                              for b in self.results['benchmarks']) else ""
            
            print(f"{marker}{name:<15} {time_val:>13.3f}   {speedup:>8.2f}x")
        
        print("\n" + "=" * 70)


def main():
    n_values = [500, 1000, 5000, 10000, 20000]
    n_trials = 5
    
    suite = BenchmarkSuite()
    suite.load_implementations()
    
    if len(suite.implementations) < 2:
        print("Error: Need at least 2 implementations to benchmark")
        sys.exit(1)
    
    suite.verify_correctness(n=100)
    suite.run_benchmarks(n_values, n_trials)
    suite.print_summary()
    
    print("\nBENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
