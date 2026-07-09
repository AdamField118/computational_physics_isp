# computational_physics_isp

Code and written work from my Computational Physics independent study (PH3999)
— Finite Volume and Finite Element methods, and related numerical experiments.

This repository is **code and work only**. The website that used to live at
`comphys.adamfield.org` has moved into my main site:

- Write-ups and notes → **Courses** and **Blog** at <https://www.adamfield.org>
- `comphys.adamfield.org` now redirects there.

## Contents

| Path | What it is |
|---|---|
| `2d_shallow_water/` | 2D shallow-water solver |
| `burger_1d_fvm/` | 1D Burgers' equation, finite volume |
| `fem_1d_benchmark/` | 1D finite-element benchmark |
| `poisson_2d_fem/` | 2D Poisson, finite element |
| `weak_lensing_poisson/` | Weak-lensing Poisson problem |
| `nbody_comparison/` | N-body integrator comparison |
| `learning_fortran/` | Fortran exercises |
| `textbook_notes/` | Reading notes (chapters 0, 3, 4) |
| `nbody.md` | N-body write-up |
| `environment.yml` | Conda environment for the Python code |

## Environment

```bash
conda env create -f environment.yml
```
