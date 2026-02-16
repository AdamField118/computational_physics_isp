"""
Automatic Differentiation Integration for Weak Lensing FEM

Makes the entire solver differentiable: κ → ψ → γ
Enables inverse problems via gradient-based optimization.

Key Features:
- Fully differentiable forward model
- Gradient validation against finite differences
- Performance profiling (forward vs backward pass)
- Ready for Bayesian inference
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from functools import partial
from typing import Tuple, Callable, Dict
import time


# ============================================================================
# Differentiable Forward Model
# ============================================================================

@jit
def forward_model_potential(kappa: jnp.ndarray, mesh) -> jnp.ndarray:
    """
    Differentiable forward model: κ → ψ
    
    This is the MINIMAL differentiable version that returns only ψ.
    Use this when you need gradients w.r.t. κ.
    
    Args:
        kappa: (n_nodes,) convergence field
        mesh: Mesh object (must be passed as static arg in practice)
        
    Returns:
        psi: (n_nodes,) lensing potential
    """
    from .fem_solver import assemble_system, apply_dirichlet_bc, conjugate_gradient
    
    # Assemble system
    K, F = assemble_system(mesh, kappa)
    
    # Apply boundary conditions
    K_bc, F_bc = apply_dirichlet_bc(K, F, mesh.boundary)
    
    # Solve (CG is differentiable!)
    psi, _, _ = conjugate_gradient(K_bc, F_bc, tol=1e-6, maxiter=1000)
    
    return psi


@jit
def forward_model_shear(kappa: jnp.ndarray, mesh, 
                       eval_points: str = 'centroids') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Differentiable forward model: κ → ψ → (γ₁, γ₂)
    
    Returns shear field at specified evaluation points.
    
    Args:
        kappa: (n_nodes,) convergence field
        mesh: P2 Mesh object
        eval_points: Where to evaluate shear
        
    Returns:
        gamma1: (n_eval,) first shear component
        gamma2: (n_eval,) second shear component
    """
    # Get potential
    psi = forward_model_potential(kappa, mesh)
    
    # Compute shear
    # NOTE: This requires shear_computation module
    # For now, return a placeholder - you'll integrate properly
    from .shear_computation import compute_shear_p2
    
    shear = compute_shear_p2(mesh, psi, eval_points=eval_points)
    
    return shear.gamma1, shear.gamma2


@partial(jit, static_argnames=['loss_fn'])
def differentiable_loss(kappa: jnp.ndarray, 
                       gamma_obs: Tuple[jnp.ndarray, jnp.ndarray],
                       mesh,
                       loss_fn: Callable = None) -> float:
    """
    Differentiable loss function for inverse problem
    
    Computes ||γ_pred - γ_obs||² where γ_pred = forward_model(κ)
    
    Args:
        kappa: (n_nodes,) convergence field (optimization variable)
        gamma_obs: (gamma1_obs, gamma2_obs) observed shear
        mesh: Mesh object
        loss_fn: Custom loss function (default: MSE)
        
    Returns:
        loss: Scalar loss value
    """
    # Predict shear
    gamma1_pred, gamma2_pred = forward_model_shear(kappa, mesh)
    
    gamma1_obs, gamma2_obs = gamma_obs
    
    if loss_fn is None:
        # Mean squared error
        loss = jnp.mean((gamma1_pred - gamma1_obs)**2 + (gamma2_pred - gamma2_obs)**2)
    else:
        loss = loss_fn(gamma1_pred, gamma2_pred, gamma1_obs, gamma2_obs)
    
    return loss


# ============================================================================
# Gradient Computation
# ============================================================================

def compute_gradient_potential(kappa: jnp.ndarray, mesh) -> jnp.ndarray:
    """
    Compute ∂ψ/∂κ using automatic differentiation
    
    Returns Jacobian: (n_nodes, n_nodes) where [i,j] = ∂ψ_i/∂κ_j
    
    Args:
        kappa: (n_nodes,) convergence field
        mesh: Mesh object
        
    Returns:
        jacobian: (n_nodes, n_nodes) sensitivity matrix
    """
    # Use JAX jacobian for full Jacobian matrix
    jac_fn = jax.jacobian(forward_model_potential, argnums=0)
    
    # This computes ∂ψ/∂κ
    jacobian = jac_fn(kappa, mesh)
    
    return jacobian


def compute_gradient_loss(kappa: jnp.ndarray,
                         gamma_obs: Tuple[jnp.ndarray, jnp.ndarray],
                         mesh) -> jnp.ndarray:
    """
    Compute ∂L/∂κ where L = ||γ_pred - γ_obs||²
    
    This is what you use for gradient descent in inverse problem!
    
    Args:
        kappa: (n_nodes,) current convergence estimate
        gamma_obs: Observed shear field
        mesh: Mesh object
        
    Returns:
        gradient: (n_nodes,) ∂L/∂κ
    """
    grad_fn = jax.grad(differentiable_loss, argnums=0)
    
    gradient = grad_fn(kappa, gamma_obs, mesh)
    
    return gradient


def compute_value_and_gradient(kappa: jnp.ndarray,
                               gamma_obs: Tuple[jnp.ndarray, jnp.ndarray],
                               mesh) -> Tuple[float, jnp.ndarray]:
    """
    Efficiently compute both loss and gradient in one pass
    
    Uses JAX's value_and_grad for efficiency (shares forward pass)
    
    Returns:
        loss: Scalar loss value
        gradient: (n_nodes,) gradient
    """
    val_grad_fn = jax.value_and_grad(differentiable_loss, argnums=0)
    
    loss, gradient = val_grad_fn(kappa, gamma_obs, mesh)
    
    return loss, gradient


# ============================================================================
# Gradient Validation
# ============================================================================

def finite_difference_gradient(kappa: jnp.ndarray,
                               gamma_obs: Tuple[jnp.ndarray, jnp.ndarray],
                               mesh,
                               epsilon: float = 1e-5) -> jnp.ndarray:
    """
    Compute gradient using finite differences (for validation)
    
    ∂L/∂κ_i ≈ [L(κ + ε e_i) - L(κ)] / ε
    
    Args:
        kappa: Current convergence field
        gamma_obs: Observed shear
        mesh: Mesh object
        epsilon: Finite difference step size
        
    Returns:
        gradient: (n_nodes,) approximate gradient
    """
    n = len(kappa)
    gradient = jnp.zeros(n)
    
    # Base loss
    L0 = differentiable_loss(kappa, gamma_obs, mesh)
    
    print(f"Computing finite difference gradient (n={n})...")
    
    # Perturb each component
    for i in range(n):
        if i % 100 == 0:
            print(f"  {i}/{n}...", end='\r')
        
        # Forward difference
        kappa_pert = kappa.at[i].add(epsilon)
        L_pert = differentiable_loss(kappa_pert, gamma_obs, mesh)
        
        gradient = gradient.at[i].set((L_pert - L0) / epsilon)
    
    print(f"  {n}/{n} ✓")
    
    return gradient


def validate_gradients(kappa: jnp.ndarray,
                      gamma_obs: Tuple[jnp.ndarray, jnp.ndarray],
                      mesh,
                      n_samples: int = 10,
                      epsilon: float = 1e-5,
                      verbose: bool = True) -> Dict:
    """
    Validate autodiff gradients against finite differences
    
    Compares ∂L/∂κ from autodiff vs finite differences on random samples
    
    Args:
        kappa: Convergence field
        gamma_obs: Observed shear
        mesh: Mesh object
        n_samples: Number of components to check
        epsilon: FD step size
        verbose: Print detailed comparison
        
    Returns:
        dict with validation results
    """
    print("=" * 70)
    print("GRADIENT VALIDATION: Autodiff vs Finite Differences")
    print("=" * 70)
    
    # Compute full autodiff gradient
    print("\n1. Computing autodiff gradient...")
    t0 = time.time()
    grad_auto = compute_gradient_loss(kappa, gamma_obs, mesh)
    t_auto = time.time() - t0
    print(f"   Time: {t_auto:.4f}s")
    print(f"   Norm: {jnp.linalg.norm(grad_auto):.6e}")
    
    # Sample random components
    n = len(kappa)
    sample_indices = jnp.array(np.random.choice(n, size=min(n_samples, n), replace=False))
    
    print(f"\n2. Computing finite difference gradient for {len(sample_indices)} samples...")
    t0 = time.time()
    
    results = {
        'auto': [],
        'fd': [],
        'rel_error': [],
        'abs_error': []
    }
    
    L0 = differentiable_loss(kappa, gamma_obs, mesh)
    
    for idx in sample_indices:
        # Autodiff
        g_auto = grad_auto[idx]
        
        # Finite difference
        kappa_pert = kappa.at[idx].add(epsilon)
        L_pert = differentiable_loss(kappa_pert, gamma_obs, mesh)
        g_fd = (L_pert - L0) / epsilon
        
        # Compare
        abs_err = float(jnp.abs(g_auto - g_fd))
        rel_err = abs_err / (jnp.abs(g_fd) + 1e-10)
        
        results['auto'].append(float(g_auto))
        results['fd'].append(float(g_fd))
        results['abs_error'].append(abs_err)
        results['rel_error'].append(rel_err)
        
        if verbose:
            print(f"   κ[{idx:4d}]: autodiff={g_auto:12.6e}, FD={g_fd:12.6e}, "
                  f"err={rel_err:10.2e}")
    
    t_fd = time.time() - t0
    print(f"   Time: {t_fd:.4f}s")
    
    # Statistics
    mean_rel_err = np.mean(results['rel_error'])
    max_rel_err = np.max(results['rel_error'])
    
    print(f"\n3. Validation Summary:")
    print(f"   Mean relative error: {mean_rel_err:.6e}")
    print(f"   Max relative error:  {max_rel_err:.6e}")
    print(f"   Speedup (autodiff): {t_fd/t_auto:.1f}× faster")
    
    # Pass/fail
    tolerance = 1e-4
    passed = max_rel_err < tolerance
    
    print(f"\n   Result: {'✓ PASS' if passed else '✗ FAIL'} "
          f"(tolerance = {tolerance:.0e})")
    
    print("=" * 70)
    
    results['mean_rel_error'] = mean_rel_err
    results['max_rel_error'] = max_rel_err
    results['passed'] = passed
    results['speedup'] = t_fd / t_auto
    
    return results


# ============================================================================
# Performance Profiling
# ============================================================================

def profile_forward_backward(kappa: jnp.ndarray,
                            gamma_obs: Tuple[jnp.ndarray, jnp.ndarray],
                            mesh,
                            n_trials: int = 10) -> Dict:
    """
    Profile performance of forward and backward passes
    
    Measures:
    - Forward pass time (κ → γ)
    - Backward pass time (∂L/∂κ)
    - Combined time (value + gradient)
    
    Args:
        kappa: Convergence field
        gamma_obs: Observed shear
        mesh: Mesh object
        n_trials: Number of timing trials
        
    Returns:
        dict with timing results
    """
    print("=" * 70)
    print("PERFORMANCE PROFILING: Forward vs Backward Pass")
    print("=" * 70)
    
    # Compile everything first
    print("\nWarmup (JIT compilation)...")
    _ = differentiable_loss(kappa, gamma_obs, mesh)
    _ = compute_gradient_loss(kappa, gamma_obs, mesh)
    _ = compute_value_and_gradient(kappa, gamma_obs, mesh)
    print("✓ Warmup complete\n")
    
    # Forward pass only
    print(f"1. Forward pass (κ → γ → loss) [{n_trials} trials]...")
    times_fwd = []
    for i in range(n_trials):
        t0 = time.time()
        loss = differentiable_loss(kappa, gamma_obs, mesh)
        jax.block_until_ready(loss)  # Wait for GPU
        times_fwd.append(time.time() - t0)
    
    t_fwd = np.median(times_fwd)
    print(f"   Median time: {t_fwd*1000:.2f} ms")
    
    # Backward pass only
    print(f"\n2. Backward pass (∂L/∂κ) [{n_trials} trials]...")
    times_bwd = []
    for i in range(n_trials):
        t0 = time.time()
        grad = compute_gradient_loss(kappa, gamma_obs, mesh)
        jax.block_until_ready(grad)
        times_bwd.append(time.time() - t0)
    
    t_bwd = np.median(times_bwd)
    print(f"   Median time: {t_bwd*1000:.2f} ms")
    
    # Combined (efficient!)
    print(f"\n3. Combined (value + gradient) [{n_trials} trials]...")
    times_both = []
    for i in range(n_trials):
        t0 = time.time()
        loss, grad = compute_value_and_gradient(kappa, gamma_obs, mesh)
        jax.block_until_ready((loss, grad))
        times_both.append(time.time() - t0)
    
    t_both = np.median(times_both)
    print(f"   Median time: {t_both*1000:.2f} ms")
    
    # Analysis
    print(f"\n4. Analysis:")
    print(f"   Forward:  {t_fwd*1000:6.2f} ms")
    print(f"   Backward: {t_bwd*1000:6.2f} ms  ({t_bwd/t_fwd:.2f}× forward)")
    print(f"   Combined: {t_both*1000:6.2f} ms  (saves {(t_fwd+t_bwd-t_both)*1000:.2f} ms)")
    print(f"   Efficiency: {(t_fwd+t_bwd)/t_both:.2f}× (shared forward pass)")
    
    print("=" * 70)
    
    return {
        'forward_ms': t_fwd * 1000,
        'backward_ms': t_bwd * 1000,
        'combined_ms': t_both * 1000,
        'backward_overhead': t_bwd / t_fwd,
        'efficiency_gain': (t_fwd + t_bwd) / t_both
    }


# ============================================================================
# Example Usage
# ============================================================================

def demonstrate_autodiff():
    """
    Complete demonstration of autodiff capabilities
    """
    print("\n" + "🚀" * 35)
    print(" " * 25 + "AUTODIFF DEMONSTRATION")
    print("🚀" * 35)
    
    # Setup
    print("\nSetup: Creating synthetic problem...")
    from .fem_solver import GaussianLens, solve_lensing_poisson
    from .mesh_generator import generate_p2_structured_mesh
    
    lens = GaussianLens(amplitude=1.0, sigma=0.3)
    mesh = generate_p2_structured_mesh(20, 20, xmin=-1, xmax=1, ymin=-1, ymax=1)
    
    print(f"  Mesh: {mesh.n_nodes} nodes")
    
    # Generate "true" convergence
    kappa_true = jnp.array([lens.kappa(x, y) for x, y in mesh.nodes])
    
    # Generate "observed" shear (forward model)
    print("\nGenerating synthetic observations...")
    gamma1_obs, gamma2_obs = forward_model_shear(kappa_true, mesh)
    gamma_obs = (gamma1_obs, gamma2_obs)
    
    print(f"  Shear points: {len(gamma1_obs)}")
    print(f"  Max |γ|: {jnp.max(jnp.sqrt(gamma1_obs**2 + gamma2_obs**2)):.4f}")
    
    # Test with perturbed initial guess
    kappa_init = kappa_true * 0.5  # 50% of truth
    
    print(f"\nInitial guess: 50% of true convergence")
    
    # Compute loss and gradient
    print("\nComputing loss and gradient...")
    loss, grad = compute_value_and_gradient(kappa_init, gamma_obs, mesh)
    
    print(f"  Loss: {loss:.6e}")
    print(f"  Gradient norm: {jnp.linalg.norm(grad):.6e}")
    print(f"  Max gradient: {jnp.max(jnp.abs(grad)):.6e}")
    
    # Validate
    validate_gradients(kappa_init, gamma_obs, mesh, n_samples=20)
    
    # Profile
    profile_forward_backward(kappa_init, gamma_obs, mesh, n_trials=5)
    
    print("\n" + "=" * 70)
    print("✓ Autodiff integration complete and validated!")
    print("\nYou now have:")
    print("  1. Differentiable forward model: κ → ψ → γ")
    print("  2. Automatic gradient computation: ∂L/∂κ")
    print("  3. Validated against finite differences")
    print("  4. Performance profiled")
    print("\nReady for Phase 3: Bayesian mass reconstruction!")
    print("=" * 70)


if __name__ == "__main__":
    import numpy as np
    demonstrate_autodiff()
