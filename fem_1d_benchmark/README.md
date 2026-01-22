# 1D FEM Multi-Language Benchmark

Implementation and benchmarking of 1D finite element method from Brenner & Scott Chapter 0 across multiple programming languages.

## Quick Start

### 1. Test Python Reference Implementation

```bash
python fem_reference.py
```

This will:
- Run a quick test with n=10
- Perform convergence study
- Generate a convergence plot
- Verify O(h²) convergence in L² norm

### 2. Build Fortran Extension

```bash
# Using f2py
f2py -c -m fem_fortran fem_assembly.f90 -llapack -lblas

# Test in Python
python -c "from fem_fortran import assemble_system; print('Fortran module loaded successfully!')"
```

### 3. Build C Extension

```bash
# Compile as shared library
gcc -O3 -fPIC -shared -o fem_c.so fem_assembly.c

# For testing standalone
gcc -O3 -DTEST_MAIN -o fem_c_test fem_assembly.c
./fem_c_test
```

Python wrapper example:
```python
import numpy as np
from ctypes import CDLL, c_int, POINTER, c_double
import numpy.ctypeslib as npct

# Load library
lib = CDLL('./fem_c.so')

# Set up function signature
lib.assemble_system.argtypes = [
    c_int,                              # n
    npct.ndpointer(dtype=np.float64),   # f_vals
    npct.ndpointer(dtype=np.float64),   # K (output)
    npct.ndpointer(dtype=np.float64)    # F (output)
]

def assemble_system(n, f_vals):
    K = np.zeros((n, n), dtype=np.float64)
    F = np.zeros(n, dtype=np.float64)
    lib.assemble_system(n, f_vals, K, F)
    return K, F

# Test
n = 10
x = np.linspace(0, 1, n+1)
f_vals = 2 - 6*x
K, F = assemble_system(n, f_vals)
print(f"K shape: {K.shape}, F shape: {F.shape}")
```

### 4. Build C++ Extension (with pybind11)

```bash
# Install pybind11 if needed
pip install pybind11

# Compile
c++ -O3 -Wall -shared -std=c++11 -fPIC \
    $(python3 -m pybind11 --includes) \
    fem_assembly.cpp -o fem_cpp$(python3-config --extension-suffix)
```

### 5. Julia (using PyJulia)

```bash
# Install PyJulia
pip install julia

# In Python, initialize Julia
python -c "import julia; julia.install()"
```

### 6. Rust (using PyO3/maturin)

```bash
# Install maturin
pip install maturin

# In the Rust project directory
maturin develop --release
```

## Project Structure

```
.
├── README.md                    # This file
├── fem_reference.py            # Python reference implementation
├── fem_assembly.f90            # Fortran implementation
├── fem_assembly.c              # C implementation
├── fem_assembly.cpp            # C++ implementation (TODO)
├── fem_assembly.jl             # Julia implementation (TODO)
├── fem_assembly_rust/          # Rust project (TODO)
│   ├── Cargo.toml
│   └── src/lib.rs
├── benchmark.py                # Main benchmarking script (TODO)
└── results/
    ├── convergence.png
    └── benchmark_results.csv
```

## Mathematical Problem

Solve: $-u''(x) = f(x)$ on $(0,1)$ with $u(0) = 0$, $u'(1) = 0$

**Manufactured solution**: $u(x) = x^2 - x^3$, so $f(x) = 2 - 6x$

## Implementation Checklist

- [x] Python reference implementation
- [x] Fortran with f2py wrapper
- [x] C with ctypes wrapper
- [ ] C++ with pybind11 wrapper
- [ ] Julia with PyJulia wrapper
- [ ] Rust with PyO3 wrapper
- [ ] Benchmarking script
- [ ] Correctness tests
- [ ] Performance plots

## Expected Performance Characteristics

Based on typical performance:
1. **Fortran/C/Rust**: ~1x (baseline)
2. **C++ (optimized)**: ~1-1.2x
3. **Julia (after JIT)**: ~1-1.5x
4. **Python**: ~50-100x slower

Assembly complexity: O(n) where n is number of elements

## Verification

All implementations should produce:
- Identical stiffness matrices and load vectors (< 1e-12 difference)
- L² error: O(h²) convergence
- Energy error: O(h) convergence
- Max error: O(h²) convergence

## Next Steps

1. **Implement remaining languages** (C++, Julia, Rust)
2. **Create benchmark.py** that:
   - Loads all implementations
   - Runs correctness tests
   - Times assembly for various n
   - Generates comparison plots
3. **Add parallel versions** (OpenMP, Rayon, etc.)
4. **Extend to 2D** (optional)

## Tips

- **Indexing**: Watch for 0-based (C/C++/Rust/Python) vs 1-based (Fortran/Julia)
- **Memory layout**: Row-major (C/C++/Python) vs column-major (Fortran/Julia)
- **Testing**: Use small n first, check against reference
- **Debugging**: Print K[0,0], K[0,1], F[0] and compare with reference
- **Profiling**: Use language-specific profilers before optimizing

## References

- Brenner & Scott, "Mathematical Theory of Finite Element Methods", Chapter 0
- f2py documentation: https://numpy.org/doc/stable/f2py/
- ctypes tutorial: https://docs.python.org/3/library/ctypes.html
- pybind11 docs: https://pybind11.readthedocs.io/
- PyJulia: https://pyjulia.readthedocs.io/
- PyO3: https://pyo3.rs/

## Sample Benchmark Results (Placeholder)

| Language | n=1000 (ms) | n=10000 (ms) | n=100000 (ms) | Speedup |
|----------|-------------|--------------|---------------|---------|
| Fortran  | 0.15        | 1.2          | 12           | 1.0x    |
| C        | 0.14        | 1.1          | 11           | 1.1x    |
| C++      | 0.15        | 1.2          | 12           | 1.0x    |
| Rust     | 0.13        | 1.0          | 10           | 1.2x    |
| Julia    | 0.18        | 1.4          | 14           | 0.9x    |
| Python   | 8.5         | 85           | 850          | 0.01x   |

*Note: These are placeholder values - actual results will vary by machine*