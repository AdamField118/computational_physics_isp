# Git history cleanup

Deleting a binary in a new commit leaves its old contents in Git history. This cleanup rewrote every retained branch and historical commit, then added a housekeeping commit for ignore rules, note references, and repository checks.

## Scope and preservation

The audit examined 62 commits and 1,437 distinct file blobs. It removed all 394 binary blobs (404,648,359 uncompressed bytes across historical versions), along with generated content at 1,700 distinct paths. These sets overlap: many binaries were inside build directories. Rust `target/` directories accounted for most of the old repository size.

Binary detection inspected the full contents of every blob, including extensionless files: NUL bytes, invalid UTF-8, and non-text control characters were rejected. Generated paths were removed independently, so text files inside build trees, caches, and package metadata were removed too. No arbitrary size threshold was used.

All 62 historical commits were retained, including commits that became empty. For every rewritten commit, its surviving file paths, modes, and blob IDs were compared against the original tree. Every non-generated text file survived byte-for-byte. The housekeeping commit then updated the current documentation and ignore rules. Source code, dependency lockfiles, equations, and text-format numerical datasets remain; the FEM plotting script only changed its generated image reference.

The pre-cleanup current tree lost 37 raster images and five generated `egg-info` files. Notes now refer to local figures, with [generation instructions](GENERATED_FILES.md). Binary files are not retained on an archive branch, since that would make them downloadable again during cloning.

## Rewritten checkpoints

| Checkpoint | Previous commit | Rewritten commit |
|---|---|---|
| Humanization | `5443666d27890783856899f20bc63c2a2fc0e9ed` | `37f12d500133f24d608eaeca46f1cb5dd4f6b502` |
| Standalone notes | `eb6c8911126b97e53e9f094c9766ab6955d9faf0` | `4df6ee7ff3347efc57fd54cd483b454e948c33a9` |
| Nix environment | `627d3b86b4ecfbfc51a8fae9d77964ce1ea8da34` | `4f96f86c8c9626aad5f0ae58180c193b22353d9b` |

The bundle includes rewritten `humanization` and `offline-notes` branches. `offline-notes` also includes the subsequent housekeeping commit. The old IDs above are text references only; their objects are not included in the cleaned bundle. No changes have been pushed to GitHub.

## Using the bundle

Use a fresh directory:

```bash
git clone -b offline-notes computational_physics_isp.bundle computational_physics_isp
cd computational_physics_isp
python3 scripts/check_repository.py --all-history
nix develop
```

Do not merge an old clone's branches into the rewritten history: that would reconnect the removed objects. If the cleaned history is later published, the affected remote branches will need a deliberate history replacement, and other working copies should be freshly cloned. No force-push is performed by this bundle.

Before a new commit, `python3 scripts/check_repository.py` checks the Git index, including staged file contents. With `--all-history`, it checks every reachable commit on all local refs, including remote-tracking branches. It does not inspect ignored local builds, which are safe to regenerate. `.gitignore` also excludes common image, archive, binary-data, compiled-output, and cache formats while keeping scientific text results and lockfiles.

## Files removed from the pre-cleanup current tree

- `assets/nbody_comparison.jpg`
- `fem_1d_benchmark/results/assembly_comparison.png`
- `fem_1d_benchmark/results/fem_benchmark_results.png`
- `nbody_comparison/results/comprehensive_scaling.png`
- `nbody_comparison/results/crossover_analysis.png`
- `nbody_comparison/results/efficiency_heatmap.png`
- `nbody_comparison/results/energy_conservation.png`
- `nbody_comparison/results/energy_drift.png`
- `nbody_comparison/results/performance_summary.png`
- `nbody_comparison/results/scaling_comparison.png`
- `nbody_comparison/results/speedup_analysis.png`
- `poisson_2d_fem/results/convergence.png`
- `poisson_2d_fem/results/convergence_rates.png`
- `poisson_2d_fem/results/error_distribution.png`
- `poisson_2d_fem/results/solution_3d.png`
- `textbook_notes/chapter_3/figures/lagrange_nodes.png`
- `textbook_notes/chapter_3/figures/nonconforming.png`
- `textbook_notes/chapter_3/figures/quadratic_triangle.png`
- `textbook_notes/chapter_3/figures/rectangle.png`
- `textbook_notes/chapter_4/figures/conditioning.png`
- `textbook_notes/chapter_4/figures/homogeneity.png`
- `textbook_notes/chapter_4/figures/mesh_quality.png`
- `textbook_notes/chapter_4/figures/quadrilateral_mapping.png`
- `textbook_notes/chapter_4/figures/reference_stiffness.png`
- `weak_lensing_poisson/cluster_example.png`
- `weak_lensing_poisson/convergence_p1.png`
- `weak_lensing_poisson/map_reconstruction.png`
- `weak_lensing_poisson/notes/figures/assembly.png`
- `weak_lensing_poisson/notes/figures/mesh_basis.png`
- `weak_lensing_poisson/p3_basis_functions.png`
- `weak_lensing_poisson/p3_convergence.png`
- `weak_lensing_poisson/p3_element_detail.png`
- `weak_lensing_poisson/p3_mesh_structure.png`
- `weak_lensing_poisson/tests/p1_vs_p3_shear.png`
- `weak_lensing_poisson/tests/p3_pipeline_gaussian.png`
- `weak_lensing_poisson/tests/p3_shear_validation.png`
- `weak_lensing_poisson/tests/p3_two_cluster.png`
- `weak_lensing_poisson/weak_lensing_poisson.egg-info/PKG-INFO`
- `weak_lensing_poisson/weak_lensing_poisson.egg-info/SOURCES.txt`
- `weak_lensing_poisson/weak_lensing_poisson.egg-info/dependency_links.txt`
- `weak_lensing_poisson/weak_lensing_poisson.egg-info/requires.txt`
- `weak_lensing_poisson/weak_lensing_poisson.egg-info/top_level.txt`
