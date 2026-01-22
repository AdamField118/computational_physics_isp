---
title: "1D Finite Element Method: Multi-Language Benchmark Project"
date: "2026-01-21"
tags: "Project"
snippet: "Implement and benchmark piecewise linear FEM across Fortran, C, C++, Julia, and Rust"
---

# Project: 1D FEM Multi-Language Benchmark

## Overview

Implement the piecewise linear finite element method from Chapter 0, Section 0.4 in multiple languages and benchmark their performance. This project focuses on the **computational kernel** (matrix assembly) which is simple enough to implement quickly but representative of real FEM performance characteristics.

## Mathematical Problem

Solve the boundary value problem:
$$-u''(x) = f(x) \quad \text{on } (0,1)$$
$$u(0) = 0, \quad u'(1) = 0$$

**Manufactured Solution** (for verification):
$$u_{\text{exact}}(x) = x^2(1-x) = x^2 - x^3$$
$$f(x) = -u''(x) = 2 - 6x$$

This allows us to compute exact errors and verify all implementations produce identical results.

## Core Algorithm

From Section 0.4, implement piecewise linear finite element method:

1. **Mesh Generation**: Uniform mesh with `n` elements
   - Nodes: $x_i = i \cdot h$ where $h = 1/n$, $i = 0, 1, \ldots, n$

2. **Stiffness Matrix Assembly** (the performance kernel):
   ```
   For each element e = 1 to n:
       K_local = (1/h) * [[1, -1], [-1, 1]]
       Add K_local to global K at positions [e-1:e, e-1:e]
   ```

3. **Load Vector Assembly**:
   ```
   For i = 1 to n-1:
       F[i] = (h/2) * (f(x_i-1) + f(x_i+1))  # trapezoidal rule
   F[n] = (h/2) * f(x_n-1)
   ```

4. **Apply Boundary Conditions**:
   - u(0) = 0: Remove first row/column
   - u'(1) = 0: Natural BC (already incorporated)

5. **Solve Linear System**: $KU = F$

6. **Compute Errors**:
   - $L^2$ error: $\|u - u_h\|_{L^2} = \sqrt{\sum_{i=1}^n \int_{x_{i-1}}^{x_i} (u - u_h)^2 dx}$
   - Energy error: $\|u - u_h\|_E = \sqrt{\sum_{i=1}^n \int_{x_{i-1}}^{x_i} (u' - u_h')^2 dx}$
   - Max error: $\|u - u_h\|_\infty = \max_i |u(x_i) - U_i|$

## Project Structure

```
fem_1d_benchmark/
├── src/
│   ├── fortran/
│   │   ├── fem_assembly.f90
│   │   └── Makefile
│   ├── c/
│   │   ├── fem_assembly.c
│   │   └── Makefile
│   ├── cpp/
│   │   ├── fem_assembly.cpp
│   │   └── Makefile
│   ├── julia/
│   │   └── fem_assembly.jl
│   └── rust/
│       ├── src/lib.rs
│       └── Cargo.toml
├── python/
│   ├── fem_driver.py          # Main benchmarking script
│   ├── fem_reference.py       # Pure Python reference implementation
│   └── visualize.py           # Plotting/analysis
├── tests/
│   └── test_correctness.py    # Verify all implementations match
└── results/
    ├── performance_plots.png
    └── benchmark_data.csv
```

## Implementation Specifications

### Common Interface (to be wrapped in Python)

Each language must implement these functions:

1. **`assemble_system(n, f_vals)`**
   - Input: `n` (number of elements), `f_vals` (array of f(x) at nodes)
   - Output: Stiffness matrix `K` (n×n), load vector `F` (n×1)
   - This is the **performance kernel** to benchmark

2. **`solve_fem(n, f_vals)`** (optional, for full benchmarking)
   - Input: Same as above
   - Output: Solution vector `U` (n×1)
   - Calls `assemble_system` then solves the linear system

### Language-Specific Implementation Notes

#### **Fortran (f2py wrapper)**
```fortran
subroutine assemble_system(n, f_vals, K, F)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: f_vals(0:n)
    real(8), intent(out) :: K(n,n)
    real(8), intent(out) :: F(n)
    
    ! Implementation here
    ! Focus: column-major order, explicit loops
end subroutine
```

#### **C (ctypes wrapper)**
```c
void assemble_system(int n, double* f_vals, double* K, double* F) {
    // Implementation here
    // Focus: row-major order, pointer arithmetic
}
```

#### **C++ (pybind11 wrapper)**
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

py::tuple assemble_system(int n, py::array_t<double> f_vals) {
    // Implementation using Eigen or raw arrays
    // Return tuple of (K, F) as numpy arrays
}
```

#### **Julia (PyJulia wrapper)**
```julia
function assemble_system(n::Int, f_vals::Vector{Float64})
    K = zeros(n, n)
    F = zeros(n)
    
    # Implementation here
    # Focus: column-major like Fortran, but with better syntax
    
    return K, F
end
```

#### **Rust (PyO3/maturin wrapper)**
```rust
use pyo3::prelude::*;
use numpy::{PyArray2, PyArray1};

#[pyfunction]
fn assemble_system(n: usize, f_vals: Vec<f64>) 
    -> (Py<PyArray2<f64>>, Py<PyArray1<f64>>) {
    // Implementation here
    // Focus: zero-cost abstractions, ownership
}
```

## Benchmarking Plan

### Test Cases

Run each implementation on:
- Small: n = 100
- Medium: n = 1,000
- Large: n = 10,000
- Very Large: n = 100,000

### Metrics to Track

1. **Assembly Time**: Time to build K and F (the main kernel)
2. **Total Solve Time**: Including solve step
3. **Memory Usage**: Peak memory during assembly
4. **Accuracy**: Verify all implementations produce identical results (to machine precision)
5. **Scaling**: Plot time vs. n on log-log scale

### Expected Results

Based on typical performance characteristics:
- **Fortran**: Baseline, excellent performance with simple loops
- **C**: Similar to Fortran, possibly slightly faster with optimizations
- **C++**: Comparable to C, depends on abstraction overhead
- **Rust**: Similar to C++, possibly faster with better optimizations
- **Julia**: Close to Fortran/C after JIT warmup, may have startup overhead
- **Python (reference)**: Much slower, included for comparison

## Python Driver Script

```python
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# Import wrapped implementations
from src.fortran import fem_fortran
from src.c import fem_c
from src.cpp import fem_cpp
from src.julia import fem_julia
from src.rust import fem_rust
from python import fem_reference

def manufactured_solution(x):
    """Exact solution: u(x) = x^2 - x^3"""
    return x**2 - x**3

def source_term(x):
    """Source: f(x) = 2 - 6x"""
    return 2 - 6*x

def benchmark_implementation(impl_name, assemble_fn, n_values, n_trials=5):
    """Benchmark a single implementation"""
    results = []
    
    for n in n_values:
        h = 1.0 / n
        x = np.linspace(0, 1, n+1)
        f_vals = source_term(x)
        
        times = []
        for trial in range(n_trials):
            start = time.perf_counter()
            K, F = assemble_fn(n, f_vals)
            end = time.perf_counter()
            times.append(end - start)
        
        results.append({
            'implementation': impl_name,
            'n': n,
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_min': np.min(times)
        })
    
    return pd.DataFrame(results)

def verify_correctness(implementations, n=100):
    """Verify all implementations produce identical results"""
    h = 1.0 / n
    x = np.linspace(0, 1, n+1)
    f_vals = source_term(x)
    
    reference_K, reference_F = None, None
    
    for name, assemble_fn in implementations.items():
        K, F = assemble_fn(n, f_vals)
        
        if reference_K is None:
            reference_K, reference_F = K, F
            print(f"{name}: Reference implementation")
        else:
            k_diff = np.max(np.abs(K - reference_K))
            f_diff = np.max(np.abs(F - reference_F))
            print(f"{name}: Max diff in K = {k_diff:.2e}, F = {f_diff:.2e}")
            
            assert k_diff < 1e-12, f"{name} K matrix differs!"
            assert f_diff < 1e-12, f"{name} F vector differs!"
    
    print("✓ All implementations verified correct!")

def main():
    implementations = {
        'Fortran': fem_fortran.assemble_system,
        'C': fem_c.assemble_system,
        'C++': fem_cpp.assemble_system,
        'Julia': fem_julia.assemble_system,
        'Rust': fem_rust.assemble_system,
        'Python': fem_reference.assemble_system,
    }
    
    # Verify correctness first
    print("=== Correctness Verification ===")
    verify_correctness(implementations)
    
    # Run benchmarks
    print("\n=== Performance Benchmarking ===")
    n_values = [100, 1000, 10000, 100000]
    all_results = []
    
    for name, fn in implementations.items():
        print(f"Benchmarking {name}...")
        results = benchmark_implementation(name, fn, n_values)
        all_results.append(results)
    
    # Combine and analyze
    df = pd.concat(all_results, ignore_index=True)
    df.to_csv('results/benchmark_data.csv', index=False)
    
    # Plot results
    plot_performance(df)
    
    # Print summary
    print("\n=== Summary (n=10000) ===")
    summary = df[df['n'] == 10000].sort_values('time_mean')
    for _, row in summary.iterrows():
        print(f"{row['implementation']:10s}: {row['time_mean']*1000:8.3f} ms")

if __name__ == '__main__':
    main()
```

## Deliverables

### Code Deliverables
1. ✅ Working implementation in each language
2. ✅ Python wrappers for all implementations
3. ✅ Test suite verifying correctness
4. ✅ Benchmarking driver script

### Analysis Deliverables
1. 📊 Performance comparison plot (time vs. n)
2. 📊 Speedup plot (relative to Python)
3. 📊 Scaling plot (log-log to verify O(n) complexity)
4. 📝 Short report (1-2 pages) discussing:
   - Performance results
   - Language-specific optimizations used
   - Surprises or interesting findings
   - Memory usage comparison

### Bonus Extensions (if time permits)

1. **GPU Acceleration**: Add a CUDA or OpenCL version
2. **Parallel Assembly**: Use OpenMP (Fortran/C++), Rayon (Rust), or threads (Julia)
3. **Higher-Order Elements**: Implement piecewise quadratics (Exercise 0.x.4)
4. **Adaptive Mesh**: Implement the adaptive algorithm from Section 0.8
5. **2D Extension**: Extend to 2D Poisson equation with triangular elements

## Learning Objectives

After completing this project, you will:
- ✓ Understand FEM assembly process at a low level
- ✓ Know how to wrap compiled code in Python across multiple languages
- ✓ Gain insights into performance characteristics of different languages
- ✓ Practice verification (matching exact solution) and validation (comparing implementations)
- ✓ Build a foundation for more complex FEM implementations

## Estimated Time

- **Quick version** (just assembly kernel): 4-6 hours
  - 30-45 min per language implementation
  - 1 hour for Python driver and testing
  - 30 min for benchmarking and plots

- **Full version** (with solve, error computation): 8-12 hours
  - Add linear solver calls
  - Implement error computations
  - More comprehensive benchmarking

## Tips for Success

1. **Start with Python reference**: Get the algorithm right first
2. **Fortran next**: Usually easiest to wrap with f2py
3. **C after Fortran**: Very similar, practice with ctypes
4. **Verify incrementally**: Test each language against reference immediately
5. **Watch for indexing**: 0-based (C/C++/Rust/Python) vs 1-based (Fortran/Julia)
6. **Profile first**: Use Python's cProfile to find actual bottlenecks
7. **Optimize later**: Get it working correctly first, then optimize

## Success Criteria

- ✅ All implementations produce identical results (< 1e-12 difference)
- ✅ Error norms match theoretical predictions: O(h²) in L² norm, O(h) in energy norm
- ✅ Timing results are reproducible (< 5% variation across runs)
- ✅ Clear performance ranking among languages
- ✅ Scaling is O(n) as theory predicts

## References

- Chapter 0, Section 0.4: Piecewise Polynomial Spaces
- Chapter 0, Section 0.6: Computer Implementation
- Your N-body ISP code for multi-language wrapping patterns

---

This project perfectly combines the theory from Chapter 0 with practical implementation skills, and fits your existing workflow from the N-body simulations. The assembly kernel is the perfect benchmarking target - it's simple but representative of real FEM performance!