# Computational Physics Independent Study

Code, notes, and numerical experiments from my Computational Physics independent study (PH3999) at WPI. The projects cover finite-element and finite-volume methods, weak lensing, and N-body integration.

## Projects and notes

| Project | Code and instructions | Notes |
|---|---|---|
| 1D FEM benchmark | [README](fem_1d_benchmark/README.md) | [Benchmark analysis](fem_1d_benchmark/notes/benchmark.md) |
| N-body comparison | [README](nbody_comparison/README.md) | [Physics and results](nbody_comparison/notes/benchmark.md), [numerical setup](nbody_comparison/notes/numerical_setup.md) |
| 2D Poisson FEM | [README](poisson_2d_fem/README.md) | [Derivation and results](poisson_2d_fem/notes/poisson.md), [theory](poisson_2d_fem/THEORY.md) |
| Weak-lensing FEM | [README](weak_lensing_poisson/README.md) | [Lensing notes](weak_lensing_poisson/notes/fem_lensing.md), [P1 examples](weak_lensing_poisson/notes/examples.md) |
| 1D Burgers FVM | [Method notes and code sketches](burger_1d_fvm/README.md) | Shock capturing, reconstruction, and time integration |
| 2D shallow water | [Method notes and code sketches](2d_shallow_water/README.md) | Source balancing and wet/dry fronts |
| Textbook notes | [Reading index](textbook_notes/README.md) | Chapters 0, 3, and 4 of Brenner & Scott |
| Fortran exercises | [Hello world](learning_fortran/helloworld.f90) | Language practice |

Markdown files can be read directly. Numerical tables are stored alongside their projects. Plot images and compiled outputs are generated locally and excluded from Git; see [generated files](docs/GENERATED_FILES.md).

## Environment

The Nix environment includes Python packages and the C, C++, Fortran, Rust, and Julia toolchains:

```bash
nix develop
comphys-build
```

See [Nix setup and troubleshooting](docs/NIX.md) for GPU support, individual builds, and platform requirements. For Conda, use the pinned Linux/CUDA environment:

```bash
conda env create -f environment.yml
conda activate comphys
```

This Conda export pins Python and numerical libraries for x86_64 Linux with CUDA 12.9. It does not include the full build toolchain, Pandas, Triangle, pybind11, or PyJulia. Supply those separately for the corresponding projects; the Nix shell includes them. Outside Nix, the build entry point is `python scripts/build.py`.

See each project's README for its numerical examples. The standalone note figures require NumPy and Matplotlib:

```bash
python textbook_notes/generate_figures.py
python weak_lensing_poisson/notes/generate_examples.py
python fem_1d_benchmark/benchmark/visualize.py
```
