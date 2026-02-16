"""
JAX-based FEM for Weak Gravitational Lensing

Solves the lensing Poisson equation
"""

from .fem_solver import (
    Mesh,
    FEMSolution,
    solve_lensing_poisson,
    compute_errors,
    GaussianLens,
    PointMassLens,
    SISLens,
)

from .mesh_generator import (
    generate_structured_mesh,
    generate_masked_structured_mesh,
    generate_unstructured_mesh,
    refine_mesh_uniform,
)

from .shear_computation import (
    compute_shear_p2,
    ShearField,
)

from .autodiff_integration import (
    forward_model_potential,
    forward_model_shear,
    compute_gradient_loss,
    compute_value_and_gradient,
    validate_gradients,
)

__version__ = "0.1.0"

__all__ = [
    # Core solver
    "Mesh",
    "FEMSolution",
    "solve_lensing_poisson",
    "compute_errors",
    
    # Lens models
    "GaussianLens",
    "PointMassLens",
    "SISLens",
    "SinusoidalLens",
    "PolynomialLens",
    
    # Mesh generation
    "generate_structured_mesh",
    "generate_masked_structured_mesh",
    "generate_unstructured_mesh",
    "refine_mesh_uniform",

    # Shear computation
    "compute_shear_p2",
    "ShearField",
    
    # Autodiff
    "forward_model_potential",
    "forward_model_shear",
    "compute_gradient_loss",
    "compute_value_and_gradient",
    "validate_gradients",
]
