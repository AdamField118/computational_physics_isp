# 1D FEM Multi-Language Benchmark

For a complete Linux toolchain, use the repository’s [Nix environment](../docs/NIX.md): `nix develop`, then `comphys-build`.

Assembly of a one-dimensional finite-element stiffness matrix and load vector in Python, C, C++, Fortran, Julia, and Rust, following Brenner & Scott, Chapter 0.

## Run

From `fem_1d_benchmark/`, run the Python reference check:

```bash
python python/fem_reference.py
```

This prints matrix and load-vector entries for n=10. It does not run a convergence study.

The Makefile provides the compiled-language workflow:

```bash
make build
make test
make benchmark
make plots
```

`make build` calls `build.sh`. The compiled extensions require their respective compilers and Python bindings: f2py for Fortran, pybind11 for C++, PyO3/maturin for Rust, and PyJulia for Julia. The C wrapper uses ctypes.

## Files

| Path | Purpose |
|---|---|
| `python/fem_reference.py` | NumPy reference assembly |
| `c/fem_assembly.c` | C implementation |
| `cpp/fem_assembly.cpp` | C++ implementation |
| `fortran/fem_assembly.f90` | Fortran implementation |
| `julia/fem_assembly.jl` | Julia implementation |
| `rust/src/lib.rs` | Rust implementation |
| `benchmark/benchmark.py` | Benchmark used by the Makefile |
| `benchmark/benchmark_all.py` | Alternate benchmark driver |
| `benchmark/visualize.py` | Static plots and Markdown timing tables |
| `results/fem_benchmark_results.json` | Saved timings |
| `notes/benchmark.md` | Benchmark write-up |

## What Is Timed

The main driver compares matrices and load vectors with the Python reference, warms up each implementation, and times repeated assembly calls. It reports mean, standard deviation, minimum, and maximum runtime.

The wrappers receive preallocated arrays. Python and Julia also allocate internally and copy their output into those arrays, so their timings include work that the other implementations avoid. The element loop is O(n), while dense matrix storage is O(n²).

The saved run contains n = 500, 1000, 5000, 10000, and 20000. Use the JSON results for numerical comparisons.

## Verification Scope

Agreement with the Python reference checks cross-language assembly consistency. It does not establish convergence to an analytic solution. The manufactured solution and source term in the reference need to be reconciled before using them for that purpose: for `u = x**2 - x**3`, `-u'' = -2 + 6*x`, whereas `source_term` returns `2 - 6*x`; this `u` also has `u'(1) = -1`, not zero.

## References

- Brenner & Scott, "Mathematical Theory of Finite Element Methods", Chapter 0
- f2py documentation: https://numpy.org/doc/stable/f2py/
- ctypes tutorial: https://docs.python.org/3/library/ctypes.html
- pybind11 docs: https://pybind11.readthedocs.io/
- PyJulia: https://pyjulia.readthedocs.io/
- PyO3: https://pyo3.rs/

## Notes and figures

- [Benchmark discussion](notes/benchmark.md)
- [Full timing table](results/benchmark_summary.md)
- [Timing figure](results/assembly_comparison.png)
