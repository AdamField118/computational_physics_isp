# Nix development environment

From the repository root:

```bash
nix develop
comphys-build
```

`nix develop` supplies the tools and Python packages. `comphys-build` compiles the native modules in place and initializes Julia's Python bridge. It stops on the first failed build.

The first entry downloads a substantial compiler/scientific-library environment. Later entries reuse the Nix store. The first Rust and Julia builds also need network access for their locked dependencies. Entering the shell itself does not run `pip`, alter a global Julia environment, or compile project code.

## Included software

- Python 3.12 with NumPy, SciPy, JAX, Matplotlib, Pandas, Triangle, PyJulia, pybind11, Pillow, pytest, and Python build tools.
- GCC/G++, GNU Fortran, OpenMP, OpenBLAS, Make, Meson, Ninja, CMake, and pkg-config.
- Rust/Cargo, maturin, Julia 1.10, Git, and headless FFmpeg.

`flake.lock` pins Nixpkgs. Python is fixed at 3.12 to keep the compiled extension ABI consistent. The two Cargo lockfiles pin Rust dependencies; the Julia manifest pins PyCall and JSON dependencies. Triangle and PyJulia are separately hash-pinned because they are not packaged in the selected Nixpkgs Python set. Triangle retains its upstream license restrictions; see [Triangle's license](https://www.cs.cmu.edu/~quake/triangle.html).

The supported platform is **x86_64 Linux**, including NixOS and Linux with Nix installed. The current build scripts use GNU OpenMP and Linux shared-library names, and Triangle's pinned wheel is x86_64. This flake does not claim support for macOS, Windows, or ARM. On Windows, use an x86_64 Linux WSL installation.

## Build and run individual projects

| Project | Build | Small check or example |
|---|---|---|
| 1D FEM | `comphys-build fem` | `make -C fem_1d_benchmark test` |
| N-body | `comphys-build nbody` | `python nbody_comparison/tests/test_accuracy.py` |
| 2D Poisson | `comphys-build poisson` | `make -C poisson_2d_fem test` |
| Weak lensing | No native build | `bash weak_lensing_poisson/run.sh tests/test_pipeline.py` |
| Fortran exercise | `comphys-build fortran` | `./learning_fortran/helloworld` |
| Julia bridge | `comphys-build julia` | `python -c 'from julia.api import Julia; Julia(compiled_modules=False); from julia import Main; print(Main.eval("1 + 1"))'` |

`comphys-build fem --language rust` builds just that extension; `c`, `cpp`, and `fortran` are also accepted. `comphys-build native` builds both benchmark projects, Poisson, and the Fortran exercise without initializing Julia. Set up Julia before running either benchmark's Julia backend. Several validation files are standalone scripts with their own `main`; use the commands above rather than assuming `pytest .` is their entry point.

Standalone plots need no native build:

```bash
python textbook_notes/generate_figures.py
python weak_lensing_poisson/notes/generate_examples.py
make -C fem_1d_benchmark plots
```

The Burgers and shallow-water directories contain method notes and code sketches. They do not yet have executable solvers.

You can also run without an interactive shell:

```bash
nix develop --command comphys-build
```

Run these from the checkout. The helper commands locate the root through Git and work from its subdirectories. Paths containing spaces are supported. Weak-lensing scripts should use `run.sh`, which establishes their project import path and forwards all arguments.

## GPU use

The default shell forces JAX onto the CPU, so it works without an NVIDIA driver. On x86_64 NixOS with a working NVIDIA driver:

```bash
nix develop .#cuda
python -c 'import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices())'
```

This shell selects JAX's CUDA 12 plugin and its matching runtime dependencies. It also exposes NixOS's driver libraries at `/run/opengl-driver/lib`. The CUDA dependencies are unfree and may require a large download or local builds if substitutes are unavailable. CPU compilers do not require the CUDA shell.

Nix cannot install or repair the host kernel driver from a development shell. An older driver, an unsupported GPU, a container without GPU access, or a scheduler allocation without a GPU will still prevent GPU execution. On non-NixOS Linux, expose the host NVIDIA driver libraries using that system's Nix GPU integration; the NixOS driver path alone is insufficient. The device assertion above detects a silent CPU fallback. Some existing benchmark labels say “JAX (GPU)” even when JAX uses the CPU; check the devices before interpreting those timings.

## Rebuilds and isolation

- Exit Conda or an unrelated virtualenv before entering. Do not install another NumPy or JAX over this environment. Python's user site is disabled. An externally set `PYTHONPATH` or `PYTHONHOME` can still interfere; unset it if you see unexpected imports.
- After changing the lockfile, switching CPU/CUDA shells, or moving the checkout, run `comphys-build` again. Compiled modules are ignored by Git and are not portable between Python ABIs or operating systems. PyCall also records an absolute Python path, so `comphys-build julia` rebuilds its binding rather than assuming an existing depot is compatible.
- Julia's writable depot is `.cache/julia`; the project is `nix/julia`. It does not install packages into your normal Julia project. PyJulia uses `compiled_modules=False` in the existing wrappers.
- Rust extensions are copied beside their source with Python's extension suffix. They do not need `maturin develop`, a writable Nix store, or an activated virtualenv.
- Fortran uses NumPy's Meson backend rather than removed `numpy.distutils` APIs. The N-body kind map and Poisson signature file preserve 64-bit floating-point arguments. The LP64 OpenBLAS variant is supplied through Nix, matching the solver’s 32-bit Fortran integers; the default ILP64 library has an incompatible LAPACK calling convention. No Conda prefix is assumed.
- Avoid launching an interactive Python session from `nbody_comparison/nbody`: its local `jax/` package can shadow the installed JAX library. Run the documented drivers from the repository root or their benchmark directory.
- Both FEM projects export a module called `fem_fortran`. Run their drivers in separate Python processes. A global `PYTHONPATH` containing both module directories would make the result import-order dependent; the shell deliberately does not add either.
- The default plotting backend is `Agg`, suitable for SSH, CI, and saved figures. `plt.show()` does not open a window with this backend. To use a GUI, select a supported installed backend before entering, for example `MPLBACKEND=TkAgg nix develop`, and provide a working display.
- Thread counts default to one unless already set, avoiding accidental CPU oversubscription. For performance runs, set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `JULIA_NUM_THREADS`, and `RAYON_NUM_THREADS` deliberately. These are not interchangeable with JAX/XLA's thread settings.
- Do not run two builds of the same target in the same checkout concurrently. Use separate worktrees for simultaneous CPU/CUDA experiments.
- `nix develop --offline` only works once the necessary Nix paths are cached. Cargo and Julia also need their own cached dependencies; lockfiles alone are not enough for an offline build.

If flakes are disabled in your Nix configuration:

```bash
nix --extra-experimental-features 'nix-command flakes' develop
```

For NixOS, the persistent setting is `nix.settings.experimental-features = [ "nix-command" "flakes" ];`. New flake files must be tracked by Git to appear in a Git-backed flake.

## Documentation

Sources for the environment mechanisms: [Nix development shells](https://nix.dev/manual/nix/latest/command-ref/new-cli/nix3-develop), [NumPy f2py Meson support](https://numpy.org/doc/stable/f2py/buildtools/meson.html), and [PyJulia installation](https://pyjulia.readthedocs.io/en/latest/installation.html).
