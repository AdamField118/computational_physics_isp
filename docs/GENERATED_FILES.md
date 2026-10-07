# Generated files

The repository stores code, notes, and text-format numerical results. Plots and compiled libraries are generated locally. The notes give the output paths for their figures.

Enter `nix develop` before using the commands below. Build compiled solvers with `comphys-build` when needed. Generated binary outputs are ignored by Git; the source and saved JSON/CSV/text results remain tracked.

## Figures from saved data or explicit examples

Run from the repository root:

```bash
python textbook_notes/generate_figures.py
python weak_lensing_poisson/notes/generate_examples.py
python fem_1d_benchmark/benchmark/visualize.py
python nbody_comparison/nbody/benchmark/visualize.py nbody_comparison/results/benchmark_results.json
python nbody_comparison/nbody/benchmark/analyze_benchmarks.py
```

The textbook script creates nine chapter figures. The weak-lensing notes script creates two P1 figures and the element-matrix JSON. The FEM plotter reads the saved timing JSON and writes a plot and Markdown timing table. The N-body scripts read saved measurements; they also regenerate text summaries, so review changes to those summaries before committing.

## Figures requiring a solver run

Run the Poisson and weak-lensing calculations with:

```bash
comphys-build poisson
(cd poisson_2d_fem/python && python generate_results.py)
bash weak_lensing_poisson/run.sh tests/example.py
bash weak_lensing_poisson/run.sh tests/validation.py
bash weak_lensing_poisson/run.sh tests/validation_p3.py
bash weak_lensing_poisson/run.sh tests/demo_p3_pipeline.py
bash weak_lensing_poisson/run.sh src/inverse.py
```

These rerun numerical calculations and can take longer. The P3 basis and mesh illustration routines are in `weak_lensing_poisson/src/p3_shape_functions.py` and `p3_mesh_generator.py`.

Timings depend on the machine, thread counts, and selected JAX device. Re-running a benchmark replaces its saved result files; copy those files elsewhere first if you want to compare runs.
