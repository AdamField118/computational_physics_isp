#!/usr/bin/env python3
"""Build native extensions in place without pip installs or a virtualenv."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("COMPHYS_PYTHON", sys.executable)
SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")


def run(args, cwd):
    print(f"[{Path(cwd).relative_to(ROOT)}] {shlex.join(map(str, args))}", flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def compiler(var, fallback):
    return shlex.split(os.environ.get(var, fallback))


def native(project, language):
    directory = ROOT / project / language
    fem = project == "fem_1d_benchmark"
    if language == "c":
        source, module = ("fem_assembly.c", "fem_c") if fem else ("nbody.c", "nbody_c")
        run(compiler("CC", "gcc") + ["-O3", "-fPIC", "-shared", "-fopenmp", source,
                                     "-o", module + ".so", "-lm"], directory)
    elif language == "cpp":
        import pybind11
        source, module = ("fem_assembly.cpp", "fem_cpp") if fem else ("nbody.cpp", "nbody_cpp_module")
        run(compiler("CXX", "g++") + ["-O3", "-std=c++11", "-shared", "-fPIC", "-fopenmp",
            "-I" + pybind11.get_include(), "-I" + sysconfig.get_path("include"),
            source, "-o", module + SUFFIX], directory)
    elif language == "fortran":
        source, module = ("fem_assembly.f90", "fem_fortran") if fem else ("nbody.f90", "nbody_fortran_module")
        run([PYTHON, "-m", "numpy.f2py", "-c", "--backend", "meson", "-m", module,
             source, '--f90flags=-O3 -fopenmp', "-lgomp"], directory)
    elif language == "rust":
        module = "fem_rust" if fem else "nbody_rust_module"
        # Copy the extension to its source directory: no writable site-packages needed.
        run(["cargo", "build", "--release", "--locked", "--target-dir", directory / "target"], directory)
        shutil.copy2(directory / "target/release" / ("lib" + module + ".so"),
                     directory / (module + SUFFIX))


def poisson():
    directory = ROOT / "poisson_2d_fem/fortran"
    # A narrow signature avoids exposing mesh_t and maps dp to C double explicitly.
    sources = ["types_module.f90", "reference_element.f90", "assembly.f90",
               "boundary_conditions.f90", "solver.f90", "python_interface.f90"]
    with tempfile.TemporaryDirectory(prefix="build-", dir=directory) as tmp:
        # f2py parses source filenames internally; use bare names so checkout
        # paths with spaces are never re-split as compiler arguments.
        for name in ["fem_fortran.pyf", *sources]:
            shutil.copy2(directory / name, Path(tmp) / name)
        run([PYTHON, "-m", "numpy.f2py", "-c", "--backend", "meson",
             "fem_fortran.pyf", *sources, "--f90flags=-O3", "-lopenblas"], tmp)
        shutil.copy2(Path(tmp) / ("fem_fortran" + SUFFIX),
                     ROOT / "poisson_2d_fem/python" / ("fem_fortran" + SUFFIX))


def julia():
    env = os.environ.copy()
    env["PYTHON"] = PYTHON
    env["JULIA_PROJECT"] = str(ROOT / "nix/julia")
    env["JULIA_DEPOT_PATH"] = str(ROOT / ".cache/julia") + ":"
    # Rebuild even if packages exist: PyCall records an absolute Python path.
    subprocess.run([os.environ.get("COMPHYS_JULIA", "julia"), "--startup-file=no",
                    "-e", 'using Pkg; Pkg.instantiate(); Pkg.build("PyCall"); using PyCall, JSON; '
                    'println("Julia/Python bridge ready: ", PyCall.python)'],
                   cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", default="all",
                        choices=["all", "native", "fem", "nbody", "poisson", "fortran", "julia"])
    parser.add_argument("--language", choices=["c", "cpp", "fortran", "rust"],
                        help="Build one language of the fem or nbody project")
    args = parser.parse_args()
    if args.language and args.target not in ("fem", "nbody"):
        parser.error("--language requires fem or nbody")
    if sys.platform != "linux":
        parser.error("These build targets currently require Linux (GNU OpenMP and .so libraries).")
    if args.target in ("all", "native", "fem", "nbody"):
        for target, project in [("fem", "fem_1d_benchmark"), ("nbody", "nbody_comparison/nbody")]:
            if args.target in ("all", "native", target):
                for lang in ([args.language] if args.language else ["c", "cpp", "fortran", "rust"]):
                    native(project, lang)
    if args.target in ("all", "native", "poisson"):
        poisson()
    if args.target in ("all", "native", "fortran"):
        run(compiler("FC", "gfortran") + ["helloworld.f90", "-o", "helloworld"], ROOT / "learning_fortran")
    if args.target in ("all", "julia"):
        julia()


if __name__ == "__main__":
    main()
