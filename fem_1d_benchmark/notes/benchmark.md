# 1D Finite Element Method: Multi-Language Performance Analysis

Date: 2026-02-02
Topics: Project

Assembly timings for a 1D finite-element problem in Python, C, C++, Fortran, Julia, and Rust.

## Summary
This project implements and benchmarks the piecewise linear finite element method from Brenner & Scott Chapter 0, Section 0.4 across **six programming languages**: Python, C, C++, Fortran, Julia, and Rust. 
**Key Results:**
- **Rust & Fortran**: Tied for fastest at 0.531 ms (294× faster than Python for n=20,000)
- **C++**: 0.547 ms (286× speedup)
- **C**: 0.566 ms (276× speedup)
- **Python**: 156 ms baseline
- **Julia**: 1845 ms, including internal allocation and copying into NumPy arrays
The saved timings are in `fem_1d_benchmark/results/fem_benchmark_results.json`. The benchmark checks assembled matrices and load vectors against the Python reference at a tolerance of 10⁻¹².
## Mathematical Problem
We solve the boundary value problem:
$$-u''(x) = f(x) \quad \text{on } (0,1)$$
$$u(0) = 0, \quad u'(1) = 0$$
The reference currently defines:
$$u_{\text{exact}}(x) = x^2 - x^3$$
$$f(x) = 2 - 6x$$
These definitions need to be reconciled before a convergence test: the stated solution has $-u''=-2+6x$ and $u'(1)=-1$. The benchmark checks agreement between assembly implementations, not agreement with this analytic solution.
## Timing Tables and Figures

[Saved timings](../results/fem_benchmark_results.json) and [full timing table](../results/benchmark_summary.md).

![Assembly timings and speedup](../results/assembly_comparison.png)
## Performance Results
### Summary Table (n = 20,000 elements)
| Language | Assembly Time | Speedup vs Python | Relative to Fastest |
|----------|---------------|-------------------|---------------------|
| **Rust**  | 0.531 ± 0.008 ms | **294.26×** | 1.00× |
| **Fortran** | 0.531 ± 0.004 ms | **294.09×** | 1.00× |
| **C++** | 0.547 ± 0.014 ms | **285.89×** | 0.97× |
| **C** | 0.566 ± 0.008 ms | **276.32×** | 0.94× |
| **Python** | 156.295 ± 0.414 ms | 1.00× | 0.003× |
| **Julia** | 1845.302 ± 27.807 ms | 0.08× | 0.0003× |
### Scaling Analysis
Assembly timings increase with mesh size:
| n | Python (ms) | C (ms) | C++ (ms) | Fortran (ms) | Rust (ms) |
|---|-------------|--------|----------|--------------|-----------|
| 500 | 0.594 | 0.023 | 0.010 | 0.006 | 0.006 |
| 1,000 | 1.806 | 0.024 | 0.011 | 0.008 | 0.007 |
| 5,000 | 18.515 | 0.086 | 0.069 | 0.056 | 0.077 |
| 10,000 | 52.075 | 0.221 | 0.205 | 0.188 | 0.207 |
| 20,000 | 156.295 | 0.566 | 0.547 | 0.531 | 0.531 |
At n=20,000, C, C++, Fortran, and Rust each take less than 0.6 ms in the timed assembly call.
## Implementation Highlights
### Core Algorithm
All implementations follow the same mathematical procedure from Brenner & Scott Section 0.4:
**Element-wise Assembly Loop:**
```
For each element e = 1 to n:
    Compute local stiffness: K_local = (1/h) * [[1, -1], [-1, 1]]
    Compute local load: F_local = (h/2) * [f(x_{e-1}) + f(x_e)]
    Add K_local to global K at positions [e-1:e, e-1:e]
    Add F_local to global F
```
Here h = 1/n. The element loop performs O(n) work, but the dense stiffness matrix occupies O(n²) storage.
### Language-Specific Implementations
### **Fortran**: Natural Column-Major Order
```fortran
! Fortran's 1-based indexing and column-major storage
do e = 2, n
    i = e - 1
    K(i, i)     = K(i, i) + k_local
    K(i+1, i)   = K(i+1, i) - k_local
    K(i, i+1)   = K(i, i+1) - k_local
    K(i+1, i+1) = K(i+1, i+1) + k_local
enddo
```
The Fortran wrapper receives column-major NumPy arrays.
**Performance**: 0.531 ms (tied for fastest)
### **Rust**: Memory-Safe Systems Programming
```rust
let k_data = k_array.as_slice_mut().unwrap();
for e in 2..=n {
    let i = e - 2;
    let row_i = i * n;
    k_data[row_i + i] += k_local;
    k_data[row_i + (i + 1)] -= k_local;
    // Symmetric entries...
}
```
The Rust extension accesses NumPy arrays through PyO3 and the numpy crate.
**Performance**: 0.531 ms (tied for fastest)
### **C**: Explicit Low-Level Control
```c
double* Kprev = K;
double* Kcur = K + n;
for (int e = 2; e <= n; e++) {
    int i = e - 2;
    Kprev[i]   += k_local;
    Kprev[i+1] -= k_local;
    // Advance row pointers
    Kprev = Kcur;
    Kcur += n;
}
```
The C function writes into arrays supplied by its ctypes wrapper.
**Performance**: 0.566 ms
### **C++**: High-Level with pybind11
```cpp
auto K_buf = K_array.request();
double* K_ptr = static_cast<double*>(K_buf.ptr);

for (int e = 2; e <= n; e++) {
    int i = e - 2;
    K_ptr[i*n + i] += k_local;
    // Symmetric assembly...
}
```
The C++ extension uses pybind11 to access the supplied NumPy buffers.
**Performance**: 0.547 ms
### **Python**: NumPy Arrays and Python Loops
```python
for e in range(1, n+1):
    i_left = e - 1
    i_right = e
    
    if i_left > 0:
        idx_left = i_left - 1
        idx_right = i_right - 1
        K[idx_left, idx_left] += k_local
        K[idx_left, idx_right] -= k_local
        K[idx_right, idx_left] -= k_local
```
The Python reference allocates its own arrays and assembles them in Python loops. Its benchmark wrapper then copies the results into the supplied buffers.
**Performance**: 156.295 ms (baseline for comparison)
## Timing Scope

The wrappers accept preallocated arrays, but their internal work differs. C, C++, Fortran, and Rust fill the supplied buffers. Python and Julia allocate temporary arrays and copy the results back. Those costs are included in their timings.

`benchmark_implementation` makes a warmup call before timing. Allocation and buffer reset for the supplied arrays happen outside the timed interval. The Julia result therefore needs a separate allocation/copy benchmark before attributing its runtime to the language or its compiler.

At n=20,000, the C, C++, Fortran, and Rust times are close. From n=10,000 to 20,000 they grow by about 2.6–2.8 times, rather than exactly twice. The O(n) element loop does not imply that the measured wrapper time scales perfectly linearly.

## Correctness Verification
The benchmark compares each loaded implementation with the Python reference using a maximum absolute difference of 10⁻¹²:
```python
# Verification results for n=100
Fortran: Max diff in K = 0.00e+00, F = 0.00e+00
C:       Max diff in K = 0.00e+00, F = 0.00e+00
C++:     Max diff in K = 0.00e+00, F = 0.00e+00
Rust:    Max diff in K = 0.00e+00, F = 0.00e+00
Julia:   Max diff in K = 0.00e+00, F = 0.00e+00

All implementations verified correct!
```
### Convergence Study
For a compatible smooth manufactured solution, the expected P1 rates are:
- **L² error**: $\|u - u_h\|_{L^2} = O(h^2)$
- **Energy error**: $\|u - u_h\|_E = O(h)$
- **Max error**: $\|u - u_h\|_\infty = O(h^2)$
## Building & Running
### Quick Start
```bash
# Build all implementations
make build

# Run benchmarks
make benchmark

# Generate static figures and Markdown tables
make plots

# Run correctness tests
make test
```
### Individual Language Builds
```bash
# C
cd c && gcc -O3 -fPIC -shared -fopenmp -o fem_c.so fem_assembly.c -lgomp

# C++
cd cpp && c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp \
    $(python3 -m pybind11 --includes) \
    fem_assembly.cpp -o fem_cpp$(python3-config --extension-suffix) -lgomp

# Fortran
cd fortran && f2py -c -m fem_fortran fem_assembly.f90 \
    --f90flags="-fopenmp -O3" -lgomp

# Rust
cd rust && maturin develop --release

# Julia (setup)
pip install julia
python3 -c "import julia; julia.install()"
```
## Future Extensions
### Potential Next Steps
1. **Parallel Assembly**: Add OpenMP versions for C/C++/Fortran
2. **GPU Acceleration**: Compare CUDA/HIP assembly with the CPU kernels
3. **2D Extension**: Triangular elements for Poisson equation
4. **Higher-Order Elements**: Piecewise quadratic basis functions
5. **Adaptive Refinement**: Implement Section 0.8 algorithms
6. **Sparse Matrix Formats**: CSR/COO for efficiency at scale
## What the Comparison Shows

At n=20,000, the saved C, C++, Fortran, and Rust timings are about 276–294 times faster than the Python wrapper. This is a comparison of these implementations and their allocation/copy behavior. It does not establish a general ranking of the languages.

The next useful comparison would give every implementation the same allocation policy, then time the solve separately from assembly.

## Technical Specifications
**Hardware**: WPI Turing Supercomputing Cluster  
**OS**: Ubuntu 24.04  
**Compilers**:
- GCC 11.4.0 (C/C++/Fortran)
- Rust 1.75.0
- Julia 1.9.3
**Benchmark Parameters**:
- Problem sizes: n ∈ {500, 1000, 5000, 10000, 20000}
- Trials per size: 5 (10 for n ≤ 1000)
- Timing method: Python's `time.perf_counter()`
- Statistical analysis: Mean ± StdDev reported
## References
1. **Brenner, S. C., & Scott, L. R.** (2008). *The Mathematical Theory of Finite Element Methods* (3rd ed.). Springer. Chapter 0: Basic Concepts.
2. **NumPy Documentation**: Array allocation and memory management - [numpy.org](https://numpy.org/doc/stable/)
3. **f2py Documentation**: Fortran to Python interface - [numpy.org/f2py](https://numpy.org/doc/stable/f2py/)
4. **pybind11 Documentation**: C++/Python bindings - [pybind11.readthedocs.io](https://pybind11.readthedocs.io/)
5. **PyO3 Documentation**: Rust/Python bindings - [pyo3.rs](https://pyo3.rs/)
6. **PyJulia Documentation**: Julia/Python integration - [pyjulia.readthedocs.io](https://pyjulia.readthedocs.io/)
## Acknowledgments
Developed for the Computational Physics independent study at WPI.
**Course**: PH 4000 - Computational Physics  
**Institution**: Worcester Polytechnic Institute  
**Advisor**: Dr. William Sanguinet  
**Date**: February 2026
