#!/usr/bin/env python3
"""
Comprehensive multi-language FEM benchmark
Saves results as JSON for analysis and visualization

Usage:
    cd benchmark/
    python benchmark_all.py
"""

import numpy as np
import time
import sys
import json
from pathlib import Path
from datetime import datetime
import platform
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import reference implementation
try:
    from fem_reference import assemble_system as python_assemble
    from fem_reference import manufactured_solution, source_term
except ImportError as e:
    print(f"Error: fem_reference.py not found! {e}")
    print("Make sure fem_reference.py is in the parent directory")
    sys.exit(1)


class BenchmarkRunner:
    """Run FEM benchmarks across multiple languages"""
    
    def __init__(self, n_values, n_trials=5):
        self.n_values = n_values
        self.n_trials = n_trials
        self.implementations = {}
        self.results = {
            'metadata': self._get_metadata(),
            'implementations': {},
            'n_values': n_values,
            'n_trials': n_trials
        }
        
    def _get_metadata(self):
        """Collect system information"""
        try:
            cpu_info = subprocess.check_output(
                "lscpu | grep 'Model name' | cut -d: -f2 | xargs",
                shell=True
            ).decode().strip()
        except:
            cpu_info = "Unknown"
            
        try:
            cpu_count = subprocess.check_output(
                "nproc",
                shell=True
            ).decode().strip()
        except:
            cpu_count = "Unknown"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'platform': platform.system(),
            'cpu': cpu_info,
            'cpu_count': cpu_count,
            'python_version': platform.python_version(),
        }
    
    def load_implementations(self):
        """Load all available implementations"""
        print("=" * 70)
        print("LOADING IMPLEMENTATIONS")
        print("=" * 70)
        
        # Python (always available)
        self.implementations['Python'] = {
            'function': python_assemble,
            'parallel': False,
            'available': True
        }
        print("✓ Python (reference)")
        
        # C
        c_fn = self._load_c()
        if c_fn:
            self.implementations['C'] = {
                'function': c_fn,
                'parallel': True,
                'available': True
            }
            print("✓ C (OpenMP)")
        
        # C (serial version)
        c_serial_fn = self._load_c_serial()
        if c_serial_fn:
            self.implementations['C (serial)'] = {
                'function': c_serial_fn,
                'parallel': False,
                'available': True
            }
            print("✓ C (serial)")
        
        # Fortran
        fortran_fn = self._load_fortran()
        if fortran_fn:
            self.implementations['Fortran'] = {
                'function': fortran_fn,
                'parallel': True,
                'available': True
            }
            print("✓ Fortran (OpenMP)")
        
        # Fortran (serial)
        fortran_serial_fn = self._load_fortran_serial()
        if fortran_serial_fn:
            self.implementations['Fortran (serial)'] = {
                'function': fortran_serial_fn,
                'parallel': False,
                'available': True
            }
            print("✓ Fortran (serial)")
        
        # C++
        cpp_fn = self._load_cpp()
        if cpp_fn:
            self.implementations['C++'] = {
                'function': cpp_fn,
                'parallel': True,
                'available': True
            }
            print("✓ C++ (OpenMP)")
        
        # C++ (serial)
        cpp_serial_fn = self._load_cpp_serial()
        if cpp_serial_fn:
            self.implementations['C++ (serial)'] = {
                'function': cpp_serial_fn,
                'parallel': False,
                'available': True
            }
            print("✓ C++ (serial)")
        
        # Julia
        julia_fn = self._load_julia()
        if julia_fn:
            self.implementations['Julia'] = {
                'function': julia_fn,
                'parallel': True,
                'available': True
            }
            print("✓ Julia (multi-threaded)")
        
        # Julia (serial)
        julia_serial_fn = self._load_julia_serial()
        if julia_serial_fn:
            self.implementations['Julia (serial)'] = {
                'function': julia_serial_fn,
                'parallel': False,
                'available': True
            }
            print("✓ Julia (serial)")
        
        # Rust
        rust_fn = self._load_rust()
        if rust_fn:
            self.implementations['Rust'] = {
                'function': rust_fn,
                'parallel': True,
                'available': True
            }
            print("✓ Rust (Rayon)")
        
        # Rust (serial)
        rust_serial_fn = self._load_rust_serial()
        if rust_serial_fn:
            self.implementations['Rust (serial)'] = {
                'function': rust_serial_fn,
                'parallel': False,
                'available': True
            }
            print("✓ Rust (serial)")
        
        print(f"\n✓ Loaded {len(self.implementations)} implementations\n")
        
        return len(self.implementations) > 1
    
    def _load_c(self):
        """Load C implementation via ctypes"""
        try:
            from ctypes import CDLL, c_int, c_double
            import numpy.ctypeslib as npct
            
            c_dir = Path(__file__).parent.parent / 'c'
            lib_path = c_dir / 'fem_c.so'
            
            if not lib_path.exists():
                print(f"⚠️  C library not found at {lib_path}")
                return None
            
            lib = CDLL(str(lib_path))
            lib.assemble_system.argtypes = [
                c_int,
                npct.ndpointer(dtype=np.float64),
                npct.ndpointer(dtype=np.float64),
                npct.ndpointer(dtype=np.float64)
            ]
            
            def c_assemble_wrapper(n, f_vals):
                K = np.zeros((n, n), dtype=np.float64, order='C')
                F = np.zeros(n, dtype=np.float64)
                lib.assemble_system(n, f_vals, K, F)
                return K, F
            
            return c_assemble_wrapper
        except Exception as e:
            print(f"⚠️  Error loading C library: {e}")
            return None
    
    def _load_c_serial(self):
        """Load C serial implementation"""
        try:
            from ctypes import CDLL, c_int, c_double
            import numpy.ctypeslib as npct
            
            c_dir = Path(__file__).parent.parent / 'c'
            lib_path = c_dir / 'fem_c.so'
            
            if not lib_path.exists():
                return None
            
            lib = CDLL(str(lib_path))
            lib.assemble_system_serial.argtypes = [
                c_int,
                npct.ndpointer(dtype=np.float64),
                npct.ndpointer(dtype=np.float64),
                npct.ndpointer(dtype=np.float64)
            ]
            
            def c_serial_wrapper(n, f_vals):
                K = np.zeros((n, n), dtype=np.float64, order='C')
                F = np.zeros(n, dtype=np.float64)
                lib.assemble_system_serial(n, f_vals, K, F)
                return K, F
            
            return c_serial_wrapper
        except:
            return None
    
    def _load_fortran(self):
        """Load Fortran implementation via f2py"""
        fortran_dir = Path(__file__).parent.parent / 'fortran'
        
        if not fortran_dir.exists():
            print(f"⚠️  Fortran directory not found at {fortran_dir}")
            return None
        
        sys.path.insert(0, str(fortran_dir))
        
        fortran_modules = list(fortran_dir.glob('fem_fortran*.so'))
        if not fortran_modules:
            print(f"⚠️  Fortran module not found in {fortran_dir}")
            return None
        
        try:
            import fem_fortran
            return fem_fortran.assemble_system
        except ImportError as e:
            print(f"⚠️  Failed to import Fortran module: {e}")
            return None
    
    def _load_fortran_serial(self):
        """Load Fortran serial implementation"""
        try:
            import fem_fortran
            return fem_fortran.assemble_system_serial
        except:
            return None
    
    def _load_cpp(self):
        """Load C++ implementation via pybind11"""
        try:
            cpp_dir = Path(__file__).parent.parent / 'cpp'
            sys.path.insert(0, str(cpp_dir))
            
            import fem_cpp
            return fem_cpp.assemble_system
        except ImportError as e:
            print(f"⚠️  C++ module not found: {e}")
            return None
    
    def _load_cpp_serial(self):
        """Load C++ serial implementation"""
        try:
            import fem_cpp
            return fem_cpp.assemble_system_serial
        except:
            return None
    
    def _load_julia(self):
        """Load Julia implementation via PyJulia wrapper"""
        try:
            julia_dir = Path(__file__).parent.parent / 'julia'
            sys.path.insert(0, str(julia_dir))
            
            import fem_julia
            
            def julia_wrapper(n, f_vals):
                K, F = fem_julia.assemble_system(n, f_vals)
                return np.array(K), np.array(F)
            
            return julia_wrapper
        except Exception as e:
            print(f"⚠️  Failed to load Julia: {e}")
            return None
    
    def _load_julia_serial(self):
        """Load Julia serial implementation"""
        try:
            import fem_julia
            
            def julia_serial_wrapper(n, f_vals):
                K, F = fem_julia.assemble_system_serial(n, f_vals)
                return np.array(K), np.array(F)
            
            return julia_serial_wrapper
        except:
            return None
    
    def _load_rust(self):
        """Load Rust implementation via PyO3"""
        try:
            rust_dir = Path(__file__).parent.parent / 'rust'
            sys.path.insert(0, str(rust_dir))
            
            import fem_rust
            
            def rust_wrapper(n, f_vals):
                K, F = fem_rust.assemble_system(n, f_vals)
                return np.array(K), np.array(F)
            
            return rust_wrapper
        except ImportError as e:
            print(f"⚠️  Rust module not found: {e}")
            return None
    
    def _load_rust_serial(self):
        """Load Rust serial implementation"""
        try:
            import fem_rust
            
            def rust_serial_wrapper(n, f_vals):
                K, F = fem_rust.assemble_system_serial(n, f_vals)
                return np.array(K), np.array(F)
            
            return rust_serial_wrapper
        except:
            return None
    
    def verify_correctness(self, n=100, tol=1e-12):
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
        
        print(f"{'Implementation':<25} {'Max K diff':<15} {'Max F diff':<15} {'Status':<10}")
        print("-" * 70)
        print(f"{'Python (reference)':<25} {'—':<15} {'—':<15} {'✓':<10}")
        
        all_pass = True
        
        for name, impl in self.implementations.items():
            if name == 'Python':
                continue
                
            try:
                K, F = impl['function'](n, f_vals)
                
                k_diff = np.max(np.abs(K - K_ref))
                f_diff = np.max(np.abs(F - F_ref))
                
                if k_diff < tol and f_diff < tol:
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL"
                    all_pass = False
                
                print(f"{name:<25} {k_diff:<15.2e} {f_diff:<15.2e} {status:<10}")
                
                # Store verification results
                if name not in self.results['implementations']:
                    self.results['implementations'][name] = {}
                self.results['implementations'][name]['verified'] = (k_diff < tol and f_diff < tol)
                self.results['implementations'][name]['k_diff'] = float(k_diff)
                self.results['implementations'][name]['f_diff'] = float(f_diff)
                
            except Exception as e:
                print(f"{name:<25} {'ERROR':<15} {'ERROR':<15} {'✗ ERROR':<10}")
                print(f"  Error: {e}")
                all_pass = False
        
        print("-" * 70)
        if all_pass:
            print("✓ All implementations verified correct!\n")
        else:
            print("✗ Some implementations failed verification!\n")
            sys.exit(1)
        
        return all_pass
    
    def benchmark_implementation(self, name, impl_info):
        """Benchmark a single implementation"""
        print(f"\nBenchmarking {name}...")
        
        assemble_fn = impl_info['function']
        results = {
            'n_values': [],
            'times_mean': [],
            'times_std': [],
            'times_min': [],
            'times_max': [],
            'trials': [],
            'parallel': impl_info['parallel']
        }
        
        for n in self.n_values:
            x = np.linspace(0, 1, n+1)
            f_vals = source_term(x)
            
            # Warmup
            _ = assemble_fn(n, f_vals)
            
            # Timed runs
            times = []
            for trial in range(self.n_trials):
                start = time.perf_counter()
                K, F = assemble_fn(n, f_vals)
                end = time.perf_counter()
                times.append(end - start)
            
            results['n_values'].append(n)
            results['times_mean'].append(float(np.mean(times)))
            results['times_std'].append(float(np.std(times)))
            results['times_min'].append(float(np.min(times)))
            results['times_max'].append(float(np.max(times)))
            results['trials'].append([float(t) for t in times])
            
            print(f"  n={n:7d}: {np.mean(times)*1000:8.3f} ± {np.std(times)*1000:6.3f} ms "
                  f"(min: {np.min(times)*1000:7.3f} ms, max: {np.max(times)*1000:7.3f} ms)")
        
        return results
    
    def run_benchmarks(self):
        """Run all benchmarks"""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARKING")
        print("=" * 70)
        print(f"Running {self.n_trials} trials per problem size\n")
        print(f"Problem sizes: {self.n_values}\n")
        
        for name, impl_info in self.implementations.items():
            try:
                results = self.benchmark_implementation(name, impl_info)
                self.results['implementations'][name] = {
                    **self.results['implementations'].get(name, {}),
                    **results
                }
            except Exception as e:
                print(f"✗ Error benchmarking {name}: {e}")
                import traceback
                traceback.print_exc()
    
    def save_results(self, output_file='benchmark_results.json'):
        """Save results to JSON file"""
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path.absolute()}")
    
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        
        # Find Python reference time for largest n
        largest_n = max(self.n_values)
        python_time = None
        
        if 'Python' in self.results['implementations']:
            python_data = self.results['implementations']['Python']
            n_idx = python_data['n_values'].index(largest_n)
            python_time = python_data['times_mean'][n_idx]
        
        print(f"\n--- Performance at n = {largest_n} ---")
        print(f"{'Implementation':<25} {'Time (ms)':<12} {'Speedup':<12} {'Type':<12}")
        print("-" * 70)
        
        # Collect and sort results
        results_list = []
        for name, data in self.results['implementations'].items():
            if 'times_mean' in data:
                n_idx = data['n_values'].index(largest_n)
                time_ms = data['times_mean'][n_idx] * 1000
                results_list.append((name, time_ms, data.get('parallel', False)))
        
        results_list.sort(key=lambda x: x[1])
        
        for name, time_ms, is_parallel in results_list:
            if python_time and python_time > 0:
                speedup = python_time / (time_ms / 1000)
                speedup_str = f"{speedup:.1f}x"
            else:
                speedup_str = "—"
            
            parallel_str = "Parallel" if is_parallel else "Serial"
            marker = "🏆 " if time_ms == results_list[0][1] else ""
            
            print(f"{marker}{name:<25} {time_ms:>10.3f}   {speedup_str:<12} {parallel_str:<12}")
        
        print("\n" + "=" * 70)


def main():
    # Benchmark parameters
    n_values = [100, 500, 1000, 5000, 10000, 50000, 100000]
    n_trials = 5
    
    # Create runner
    runner = BenchmarkRunner(n_values, n_trials)
    
    # Load implementations
    if not runner.load_implementations():
        print("\n✗ Failed to load implementations")
        sys.exit(1)
    
    # Verify correctness
    runner.verify_correctness(n=100)
    
    # Run benchmarks
    runner.run_benchmarks()
    
    # Save results
    output_file = Path(__file__).parent / 'benchmark_results.json'
    runner.save_results(output_file)
    
    # Print summary
    runner.print_summary()


if __name__ == '__main__':
    main()