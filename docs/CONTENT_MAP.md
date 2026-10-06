# Content map

The `humanization` branch at commit `5443666d27890783856899f20bc63c2a2fc0e9ed` preserves the complete tree before this restructuring. The `offline-notes` branch contains the reorganized working tree. Both branches are included in the Git bundle; no history was rewritten.

## Notes and results

| Previous location | Current location |
|---|---|
| `fem_1d_benchmark/web/fem_1d_benchmark_project.md` | [FEM benchmark notes](../fem_1d_benchmark/notes/benchmark.md) |
| `nbody_comparison/web/index.md` | [N-body notes](../nbody_comparison/notes/benchmark.md) |
| `poisson_2d_fem/web/index.md` | [Poisson notes](../poisson_2d_fem/notes/poisson.md) |
| `weak_lensing_poisson/web/index.md` | [Weak-lensing notes](../weak_lensing_poisson/notes/fem_lensing.md) |
| `nbody.md` | [N-body project plan](../nbody_comparison/notes/project_plan.md) |
| `nbody_comparison/web/data/` | [N-body results](../nbody_comparison/results) |
| `fem_1d_benchmark/benchmark/fem_benchmark_results.png` | [Original FEM timing figure](../fem_1d_benchmark/results/fem_benchmark_results.png) |

All original numerical data and image files are retained byte-for-byte. Textbook derivations and exercises remain in their chapter folders. Publication dates, topic labels, and description text are retained as ordinary Markdown rather than metadata for a page renderer.

## Demonstration content

The following names identify the old scripts in the checkpoint. Their scientific content is now accessible without executing presentation code.

| Original demonstration | Preserved content |
|---|---|
| `ex_9_rectangle` | [Rectangle formulas, normalization note, and figure](../textbook_notes/chapter_3/examples.md#rectangle) |
| `ex_10_triangle` | [Six quadratic basis functions and figure](../textbook_notes/chapter_3/examples.md#quadratic-triangle) |
| `ex_14_nonconforming` | [Conforming/CR spaces, mesh, and DOF counts](../textbook_notes/chapter_3/examples.md#nonconforming-elements) |
| `ex_19_lagrange` | [Barycentric lattice construction, counts, and figures](../textbook_notes/chapter_3/examples.md#lagrange-node-counts) |
| `ex_11_reference_triangle` | [All stiffness entries and gradients](../textbook_notes/chapter_4/examples.md#reference-stiffness) |
| `ex_10_angle_checker` | [Angle/inradius formulas, quality thresholds, and figures](../textbook_notes/chapter_4/examples.md#mesh-quality) |
| `ex_5_homogeneity` | [Scaling relations and figures](../textbook_notes/chapter_4/examples.md#homogeneity) |
| `ex_17_condition_number` | [Exact eigenvalues, conditioning, and solver interpretation](../textbook_notes/chapter_4/examples.md#conditioning) |
| `ex_21_quad_mapping` | [Q1 mapping, Jacobian, sampling rule, and examples](../textbook_notes/chapter_4/examples.md#quadrilateral-mapping) |
| `fem-mesh-demo` | [Mesh connectivity, basis support, and refinement](../weak_lensing_poisson/notes/examples.md#mesh-and-basis) |
| `fem-assembly-demo` | [Original mesh, assembly rule, matrices, and figures](../weak_lensing_poisson/notes/examples.md#element-assembly) |
| `fem-solver-demo` | [Preset sources, load convention, CG, and deflection averaging](../weak_lensing_poisson/notes/examples.md#p1-solver) |
| `threejs_simulation` | [Initial distributions, parameters, and actual update rule](../nbody_comparison/notes/simulation_example.md) |
| `performance_dashboard` | [Measured data](../nbody_comparison/results/benchmark_results.json) and separate [illustrative fallback values](../nbody_comparison/results/illustrative_sample.json) |
| `poisson_convergence_viz` | [All original arrays, including Linf](../poisson_2d_fem/results/convergence_snapshot.json), and [table with rates](../poisson_2d_fem/notes/convergence_snapshot.md) |
| `fem_benchmark_viz` | [Saved numerical arrays](../fem_1d_benchmark/results/fem_benchmark_results.json), [alternate descriptive metadata](../fem_1d_benchmark/results/alternate_metadata.json), and [full timing table](../fem_1d_benchmark/results/benchmark_summary.md) |
| `fem_dashboard_iframe` | Embedded the same FEM results; no independent numerical content |

The standalone FEM HTML output contained exactly the same dataset as the saved JSON. Its generator now produces a static figure and a Markdown table containing every timing row, including minimum and maximum values.

The Burgers and shallow-water plans retain their scientific content. Their display-code sketches have been replaced with descriptions of the parameter comparisons, saved fields, and animation outputs they proposed.

## Boundaries of this conversion

Browser controls, camera settings, styling, and page-renderer hooks were removed. Mathematical content and numerical examples were retained in notes or executable Python figure scripts. Known mistakes encountered during conversion are called out in the worked examples; this was not a full mathematical review of all the original solutions. The exact original implementations remain available in the checkpoint history.
