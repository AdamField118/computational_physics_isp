#!/usr/bin/env python3
"""Check dependencies and, with --built, exercise every compiled backend."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("COMPHYS_PYTHON", sys.executable)


def check(name, code, directory):
    print(f"\nChecking {name}...", flush=True)
    marker = "COMPHYS_CHECK_COMPLETED"
    result = subprocess.run([PYTHON, "-c", textwrap.dedent(code) + f"\nprint({marker!r}, flush=True)"],
                            cwd=ROOT / directory, stdout=subprocess.PIPE, text=True)
    print(result.stdout.replace(marker + "\n", ""), end="", flush=True)
    result.check_returncode()
    if marker not in result.stdout.splitlines():
        raise RuntimeError(f"{name} exited before its checks finished (for example, Fortran STOP)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--built", action="store_true")
    args = parser.parse_args()
    for name in ["gcc", "g++", "gfortran", "make", "cargo", "rustc", "julia", "meson", "ninja", "pkg-config"]:
        if not shutil.which(name):
            raise SystemExit(f"Missing {name}: enter nix develop first")
    check("Python, JAX execution, meshing and headless rendering", '''
        import io
        import numpy as np
        import scipy, pandas, pybind11, julia, triangle, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import jax, jax.numpy as jnp
        np.testing.assert_allclose(np.asarray(jax.jit(lambda x: x*x)(jnp.arange(4.))), [0,1,4,9])
        mesh = triangle.triangulate({"vertices": np.array([[0.,0.],[1.,0.],[0.,1.]])}, "Q")
        assert len(mesh["triangles"]) == 1
        plt.plot([0, 1], [0, 1]); plt.savefig(io.BytesIO(), format="png"); plt.close()
        print("Python dependencies OK; JAX devices:", jax.devices())
    ''', ".")
    check("weak-lensing P3 basis and JAX differentiation", '''
        import numpy as np
        import jax
        from src.p3_shape_functions import compute_p3_shape_functions
        from src.fem import build_operators
        from src.forward import DifferentiableForward
        from src.inverse import MAPReconstructor
        np.testing.assert_allclose(np.asarray(compute_p3_shape_functions(.2, .3)).sum(), 1., atol=1e-12)
        assert np.isfinite(jax.grad(lambda x: compute_p3_shape_functions(x, .3).sum())(.2))
        ops = build_operators(2, 2, verbose=False)
        kappa = np.exp(-np.sum(np.asarray(ops.mesh.nodes)**2, axis=1))
        g1, g2 = ops.forward(kappa)
        assert np.isfinite(g1).all() and np.isfinite(g2).all()
        result = DifferentiableForward(ops).validate_gradients(kappa, g1, g2, n_checks=2, verbose=False)
        assert result["passed"], result
        print("P3 forward model and gradients OK")
    ''', "weak_lensing_poisson")
    if args.built:
        # Separate processes prevent fem_fortran and Julia Main collisions.
        check("1D FEM: all six implementations", '''
            from benchmark import BenchmarkSuite
            suite = BenchmarkSuite()
            suite.load_implementations()
            assert set(suite.implementations) == {"Python", "C", "C++", "Fortran", "Rust", "Julia"}, suite.implementations.keys()
            assert suite.verify_correctness(n=8)
        ''', "fem_1d_benchmark/benchmark")
        check("alternate FEM harness", """
            from benchmark_all import BenchmarkRunner
            runner = BenchmarkRunner([8], n_trials=1)
            runner.load_implementations()
            assert {"Python", "C", "C++", "Fortran", "Rust", "Julia"} <= set(runner.implementations)
            assert runner.verify_correctness(n=8)
        """, "fem_1d_benchmark/benchmark")
        check("N-body: C/C++/Fortran/Rust/Julia against Python", '''
            import sys
            from pathlib import Path
            for part in ["c", "cpp", "fortran", "rust", "python", "julia"]:
                sys.path.insert(0, str(Path.cwd() / part))
            import numpy as np
            import nbody_python as ref, nbody_c_wrapper as c, nbody_rust_wrapper as rust
            import nbody_pyjulia_wrapper as julia
            import nbody_julia_wrapper as subprocess_julia
            from nbody_fortran_module import nbody_fortran as f
            from nbody_cpp_module import NBodySimulator
            p = np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.]])
            v = np.array([[0.,.1,0.],[0.,-.1,0.],[.05,0.,0.]])
            m = np.array([1.,2.,.5])
            reference = ref.simulate(p.copy(), v.copy(), m, 2)
            outputs = [c.simulate(p,v,m,2), rust.simulate(p,v,m,2), julia.simulate(p,v,m,2),
                       subprocess_julia.simulate(p,v,m,2),
                       f.simulate(np.asfortranarray(p), np.asfortranarray(v), m, 2, 1., .1, .01),
                       NBodySimulator(3,1.,.1,.01).simulate(p.ravel(),v.ravel(),m,2)]
            for result in outputs:
                np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-12)
            for energy in [c.compute_energy(p,v,m), rust.compute_energy(p,v,m), julia.compute_energy(p,v,m),
                           f.compute_energy(p,v,m,1.,.1)]:
                np.testing.assert_allclose(energy, ref.compute_energy(p,v,m), rtol=1e-12)
            print("All native and Julia N-body backends agree")
        ''', "nbody_comparison/nbody")
        check("N-body JAX vs NumPy", '''
            import sys
            from pathlib import Path
            for part in ["jax", "python"]:
                sys.path.insert(0, str(Path.cwd().parent / part))
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            import numpy as np
            from nbody_jax import NBodyState, NBodyConfig, simulate
            from nbody_python import simulate as reference
            p=np.array([[0.,0.,0.],[1.,0.,0.]])
            v=np.zeros_like(p); m=np.ones(2)
            result=simulate(NBodyState(jnp.array(p),jnp.array(v),jnp.array(m),0.),NBodyConfig(),2)
            np.testing.assert_allclose([result[1][-1], result[2][-1]], reference(p,v,m,2), atol=1e-12)
            print("JAX N-body execution OK")
        ''', "nbody_comparison/nbody/benchmark")
        check("Poisson Fortran interface and OpenBLAS", '''
            import numpy as np
            from fem_solver import PoissonSolver2D
            solver = PoissonSolver2D(max_area=.1)
            u = solver.solve()
            assert u.dtype == np.float64 and np.isfinite(u).all()
            assert np.max(np.abs(u)) > 0
            np.testing.assert_allclose(u[solver.mesh.boundary - 1], 0., atol=1e-12)
            print("Poisson solve OK")
        ''', "poisson_2d_fem/python")
        subprocess.run([str(ROOT / "learning_fortran/helloworld")], check=True)
    print("\nRequested environment checks passed.")


if __name__ == "__main__":
    main()
