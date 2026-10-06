# Weak-lensing Poisson experiments

Finite-element experiments for the potential solve and shear-to-mass reconstruction.

- [FEM derivation and lensing notes](notes/fem_lensing.md)
- [P1 mesh, assembly, and solver examples](notes/examples.md)
- [Source code](src)
- [Validation scripts and examples](tests)

The elementary notes describe P1 assembly. The source also contains the later P3 forward model and inverse reconstruction; the notes are not a complete description of those later developments.

Generate the small explanatory figures from the repository root:

```bash
python weak_lensing_poisson/notes/generate_examples.py
```

Existing numerical outputs are retained in this directory and in `tests/`, including `cluster_example.png`, `convergence_p1.png`, `p3_convergence.png`, `map_reconstruction.png`, and the P3 basis and mesh diagrams.
