"""
inverse.py
==========
MAP mass reconstruction: γ_obs → κ_MAP.

    κ_MAP = argmin  ‖γ_pred(κ) − γ_obs‖²  +  λ ‖∇κ‖²

L-BFGS with a PURE NUMPY adjoint gradient.

Why not JAX autodiff in the optimizer?
---------------------------------------
The original _make_obj_and_grad did:

    @jax.jit
    def val_grad(kappa):
        return self.fwd.grad_fn(kappa, g1, g2)

forward.py uses custom_vjp + jax.pure_callback to escape JAX tracing for
SuperLU solves. Under jax.jit, JAX traces through the custom_vjp but
pure_callback gradients are silently zeroed out — the compiled gradient
returns ~0 everywhere. Test 3 passes because validate_gradients calls
grad_fn in EAGER mode (no jit); the optimizer runs in compiled mode where
the bug surfaces.

Fix: write the adjoint explicitly in numpy. For our linear forward model
it's closed-form and equally cheap (2 sparse solves per iteration):

    Forward:  ψ = K⁻¹(−2Mκ),   γ = Sψ
    Residual: r = γ_pred − γ_obs
    ∂L/∂κ   = −4 Mᵀ K⁻¹(S1ᵀr1 + S2ᵀr2)  +  2λ K κ

where K is symmetric so K⁻¹ adjoint = K⁻¹, and Kκ is the H¹ regularizer.
"""

import sys, os
import numpy as np
import scipy.optimize as sopt
import scipy.fft as sfft
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Support both package import (from .fem) and direct execution (python src/inverse.py)
try:
    from .fem     import FEMOperators, build_operators
    from .forward import DifferentiableForward
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.fem     import FEMOperators, build_operators
    from src.forward import DifferentiableForward


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReconstructionResult:
    kappa_map   : np.ndarray   # (n_nodes,) reconstructed convergence
    psi_map     : np.ndarray   # (n_nodes,) reconstructed potential
    gamma1_pred : np.ndarray   # (n_nodes,) predicted shear (at optimum)
    gamma2_pred : np.ndarray   # (n_nodes,) predicted shear (at optimum)
    loss_history: list         # loss at each obj_grad call
    n_iter      : int          # L-BFGS iterations
    converged   : bool
    time_s      : float


# ══════════════════════════════════════════════════════════════════════════════
# MAP Reconstructor
# ══════════════════════════════════════════════════════════════════════════════

class MAPReconstructor:
    """
    MAP mass reconstruction: γ_obs → κ_MAP via L-BFGS.

    Uses an explicit numpy adjoint — no JAX/JIT in the optimization loop.
    JAX is still used in forward.py for Test 3 gradient validation (eager).

    Parameters
    ----------
    fwd           : DifferentiableForward  (carries ops and lam_reg)
    maxiter       : int    max L-BFGS iterations (default 500)
    gtol          : float  gradient-norm convergence tolerance
    callback_every: int    print every N calls (0 = silent)
    """

    def __init__(self, fwd: DifferentiableForward,
                 maxiter: int = 500,
                 gtol: float = 1e-8,
                 callback_every: int = 20):
        self.fwd            = fwd
        self.maxiter        = maxiter
        self.gtol           = gtol
        self.callback_every = callback_every
        self.ops            = fwd.ops

    # ── Closed-form numpy adjoint ──────────────────────────────────────────────

    def _make_obj_and_grad(self,
                           gamma1_obs: np.ndarray,
                           gamma2_obs: np.ndarray):
        """
        Build (loss, grad) callable for scipy.optimize.minimize (jac=True).

        Forward model is identical to ops.forward():
            rhs = -2 M κ  (zero at boundary)
            ψ   = K⁻¹ rhs
            γ   = S ψ

        Adjoint gradient (derived by chain rule):
            r   = γ_pred - γ_obs
            adj = K⁻¹ (S1ᵀ r1 + S2ᵀ r2)   [K symmetric → K⁻ᵀ = K⁻¹]
            ∂L/∂κ|data = -4 Mᵀ adj
            ∂L/∂κ|reg  = +2λ K κ            [K = FEM stiffness = ‖∇κ‖² kernel]
        """
        ops   = self.ops
        M, S1, S2 = ops.M, ops.S1, ops.S2
        K_lu  = ops.K_lu
        K_mat = ops.K          # sparse stiffness; used for regularizer Kκ
        bnd   = ops.boundary
        lam   = self.fwd.lam_reg

        g1o = np.asarray(gamma1_obs, dtype=np.float64)
        g2o = np.asarray(gamma2_obs, dtype=np.float64)

        loss_history = []
        call_count   = [0]

        def obj_grad(kappa_np: np.ndarray):
            kappa = np.asarray(kappa_np, dtype=np.float64)

            # ── Forward ───────────────────────────────────────────────────────
            rhs = -2.0 * (M @ kappa)
            rhs[bnd] = 0.0
            psi  = K_lu.solve(rhs)
            g1p  = S1 @ psi
            g2p  = S2 @ psi

            # ── Loss ──────────────────────────────────────────────────────────
            r1 = g1p - g1o
            r2 = g2p - g2o
            data_loss = float(np.dot(r1, r1) + np.dot(r2, r2))

            Kk       = K_mat @ kappa
            reg_loss = float(np.dot(kappa, Kk))
            loss     = data_loss + lam * reg_loss

            loss_history.append(loss)
            call_count[0] += 1
            if self.callback_every > 0 and call_count[0] % self.callback_every == 0:
                print(f"  call {call_count[0]:4d}  loss = {loss:.6e}")

            # ── Adjoint ───────────────────────────────────────────────────────
            # ∂(‖r‖²)/∂κ  via  A = S K⁻¹(−2M),  Aᵀ = −2Mᵀ K⁻¹ Sᵀ
            rhs_adj = S1.T @ r1 + S2.T @ r2
            rhs_adj[bnd] = 0.0
            adj  = K_lu.solve(rhs_adj)
            grad = -4.0 * (M.T @ adj) + 2.0 * lam * Kk

            return loss, grad.astype(np.float64)

        return obj_grad, loss_history

    # ── Reconstruction ─────────────────────────────────────────────────────────

    def reconstruct(self,
                    gamma1_obs: np.ndarray,
                    gamma2_obs: np.ndarray,
                    kappa_init: Optional[np.ndarray] = None,
                    verbose: bool = True) -> Tuple[np.ndarray, ReconstructionResult]:

        n = self.ops.n_nodes
        if kappa_init is None:
            kappa_init = np.zeros(n)

        obj_grad, loss_history = self._make_obj_and_grad(gamma1_obs, gamma2_obs)

        # Sanity-check gradient before handing to L-BFGS
        loss0, grad0 = obj_grad(kappa_init.copy())
        grad_norm0   = np.linalg.norm(grad0)
        loss_history.clear()

        if verbose:
            print("=" * 60)
            print("MAP Reconstruction  (L-BFGS, numpy adjoint)")
            print(f"  n_nodes   = {n}")
            print(f"  λ_reg     = {self.fwd.lam_reg:.2e}")
            print(f"  maxiter   = {self.maxiter}  |  gtol = {self.gtol:.0e}")
            print(f"  loss(κ=0) = {loss0:.4e}  |  ‖∇L‖(κ=0) = {grad_norm0:.4e}")
            print("=" * 60)

        t0 = time.perf_counter()

        result_opt = sopt.minimize(
            fun     = obj_grad,
            x0      = kappa_init.copy(),
            method  = 'L-BFGS-B',
            jac     = True,
            options = {
                'maxiter': self.maxiter,
                'gtol'   : self.gtol,
                'ftol'   : 1e-30,   # never stop on f-reduction; gtol drives convergence
                'maxls'  : 100,
                'maxcor' : 20,
                'iprint' : -1,
            },
        )

        elapsed   = time.perf_counter() - t0
        kappa_map = result_opt.x

        # Recompute final fields
        rhs          = -2.0 * (self.ops.M @ kappa_map)
        rhs[self.ops.boundary] = 0.0
        psi_map      = self.ops.K_lu.solve(rhs)
        g1p          = np.array(self.ops.S1 @ psi_map)
        g2p          = np.array(self.ops.S2 @ psi_map)

        if verbose:
            print(f"\n  Converged : {result_opt.success}")
            print(f"  Message   : {result_opt.message}")
            print(f"  Iterations: {result_opt.nit}")
            print(f"  Fcn calls : {result_opt.nfev}")
            print(f"  Final loss: {result_opt.fun:.6e}")
            print(f"  Wall time : {elapsed:.2f} s")
            print(f"  max|κ_MAP|: {np.abs(kappa_map).max():.4f}")
            print("=" * 60)

        return kappa_map, ReconstructionResult(
            kappa_map    = kappa_map,
            psi_map      = psi_map,
            gamma1_pred  = g1p,
            gamma2_pred  = g2p,
            loss_history = loss_history,
            n_iter       = result_opt.nit,
            converged    = result_opt.success,
            time_s       = elapsed,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Kaiser-Squires (FFT) reference method
# ══════════════════════════════════════════════════════════════════════════════

def kaiser_squires(gamma1_grid: np.ndarray,
                   gamma2_grid: np.ndarray) -> np.ndarray:
    """
    Kaiser-Squires mass reconstruction on a regular periodic grid.

        κ̂(k) = [(k₁²−k₂²) γ̂₁ + 2k₁k₂ γ̂₂] / |k|²
    """
    ny, nx = gamma1_grid.shape
    kx = sfft.fftfreq(nx) * 2 * np.pi
    ky = sfft.fftfreq(ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2;  K2[0, 0] = 1.0

    kappa_hat = ((KX**2 - KY**2) * sfft.fft2(gamma1_grid)
                 + 2 * KX * KY * sfft.fft2(gamma2_grid)) / K2
    kappa_hat[0, 0] = 0.0
    return np.real(sfft.ifft2(kappa_hat))


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic benchmark
# ══════════════════════════════════════════════════════════════════════════════

def _gaussian_kappa(x, y, A=1.0, sigma=0.5, cx=0., cy=0.):
    return A * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))


def _gaussian_shear_exact(x, y, A=1.0, sigma=0.5, cx=0., cy=0.):
    """Analytic shear for a Gaussian convergence."""
    dx, dy = x-cx, y-cy
    k  = A * np.exp(-(dx**2+dy**2)/(2*sigma**2))
    g1 = k * (dy**2 - dx**2) / (2*sigma**2)
    g2 = -k * dx * dy / sigma**2
    return g1, g2


def run_comparison(nx: int = 20,
                   noise_level: float = 0.05,
                   lam_reg: float = 1e-2,
                   domain: tuple = (-2.5, 2.5, -2.5, 2.5),
                   A: float = 1.0, sigma: float = 0.5,
                   apply_mask: bool = False,
                   mask_center: tuple = (0.8, 0.8),
                   mask_radius: float = 0.4,
                   verbose: bool = True) -> dict:

    xmin, xmax, ymin, ymax = domain

    if verbose:
        print("\n" + "═"*60)
        print(f"FEM-MAP vs Kaiser-Squires Benchmark")
        print(f"  {nx}×{nx} P3 mesh  |  noise={noise_level:.0%}  |  λ={lam_reg:.0e}")
        print("═"*60)

    ops = build_operators(nx, nx, xmin, xmax, ymin, ymax, verbose=verbose)
    fwd = DifferentiableForward(ops, lam_reg=lam_reg)
    rec = MAPReconstructor(fwd, maxiter=500, gtol=1e-9,
                           callback_every=50 if verbose else 0)

    nodes = np.array(ops.mesh.nodes)
    x, y  = nodes[:, 0], nodes[:, 1]

    kappa_true           = _gaussian_kappa(x, y, A, sigma)
    #gamma1_true, gamma2_true = _gaussian_shear_exact(x, y, A, sigma)
    gamma1_true, gamma2_true = ops.forward(kappa_true)

    np.random.seed(42)
    ns = noise_level * np.max(np.abs(gamma1_true))
    g1_obs = gamma1_true + np.random.randn(len(x)) * ns
    g2_obs = gamma2_true + np.random.randn(len(x)) * ns

    mask_nodes = np.zeros(len(x), dtype=bool)
    if apply_mask:
        mcx, mcy = mask_center
        mask_nodes = (x-mcx)**2 + (y-mcy)**2 < mask_radius**2
        g1_obs[mask_nodes] = 0.0
        g2_obs[mask_nodes] = 0.0
        if verbose:
            print(f"  Masked {mask_nodes.sum()} nodes "
                  f"({100*mask_nodes.mean():.1f}%)\n")

    kappa_map, result = rec.reconstruct(g1_obs, g2_obs, verbose=verbose)

    # Kaiser-Squires on regular grid
    from scipy.interpolate import griddata
    n_ks = nx + 1
    xi   = np.linspace(xmin, xmax, n_ks)
    yi   = np.linspace(ymin, ymax, n_ks)
    XI, YI = np.meshgrid(xi, yi)
    pts  = np.column_stack([x, y])
    g1g  = griddata(pts, g1_obs, (XI,YI), method='linear', fill_value=0.)
    g2g  = griddata(pts, g2_obs, (XI,YI), method='linear', fill_value=0.)
    ks_g = kaiser_squires(g1g, g2g)
    kappa_ks = griddata(np.column_stack([XI.ravel(), YI.ravel()]),
                        ks_g.ravel(), pts, method='linear', fill_value=0.)

    l2_map = np.sqrt(np.mean((kappa_map - kappa_true)**2))
    l2_ks  = np.sqrt(np.mean((kappa_ks  - kappa_true)**2))

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Reconstruction quality  (L2 error vs truth)")
        print(f"  FEM-MAP  : {l2_map:.4f}")
        print(f"  K-S      : {l2_ks:.4f}")
        impr = (l2_ks - l2_map) / l2_ks * 100
        print(f"  Improvement: {impr:+.1f}%  (+ = FEM-MAP better)")
        print(f"{'─'*60}\n")

    _plot_comparison(ops, nodes, kappa_true, kappa_map, kappa_ks,
                     result, l2_map, l2_ks, noise_level, apply_mask,
                     mask_nodes, fname="map_reconstruction.png")

    return dict(kappa_true=kappa_true, kappa_map=kappa_map, kappa_ks=kappa_ks,
                l2_map=l2_map, l2_ks=l2_ks, ops=ops, result=result)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _plot_comparison(ops, nodes, kappa_true, kappa_map, kappa_ks,
                     result, l2_map, l2_ks, noise_level, apply_mask,
                     mask_nodes, fname="map_reconstruction.png"):

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor('#0e0e0e')

    el  = np.array(ops.mesh.elements)[:, :3]
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=el)
    vmax = np.percentile(kappa_true, 99)

    def panel(ax, data, title, cmap='hot', sym=False, vmax_=None):
        ax.set_facecolor('#1a1a1a')
        v    = vmax_ if vmax_ is not None else (
               np.percentile(np.abs(data[np.isfinite(data)]), 98) if sym
               else np.percentile(data[np.isfinite(data)], 99))
        vmin = -v if sym else 0
        tcf  = ax.tricontourf(tri, data, levels=40, cmap=cmap,
                              vmin=vmin, vmax=v, extend='both')
        fig.colorbar(tcf, ax=ax, fraction=0.04, pad=0.02
                     ).ax.tick_params(labelsize=7, colors='#aaa')
        ax.set_title(title, fontsize=10, color='#ddd', pad=5)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7, colors='#888')

    panel(axes[0], kappa_true,              'κ  truth',                  vmax_=vmax)
    panel(axes[1], kappa_map,               f'κ  FEM-MAP\nL2={l2_map:.3f}',  vmax_=vmax)
    panel(axes[2], kappa_ks,                f'κ  Kaiser-Squires\nL2={l2_ks:.3f}', vmax_=vmax)
    panel(axes[3], kappa_map - kappa_true,  'MAP residual',  'RdBu_r', sym=True)

    if len(result.loss_history) > 5:
        ax5 = axes[4]
        ax5.set_facecolor('#1a1a1a')
        ax5.semilogy(result.loss_history, color='#00ff80', lw=1.5)
        ax5.set_xlabel('L-BFGS call', fontsize=9, color='#aaa')
        ax5.set_ylabel('Loss', fontsize=9, color='#aaa')
        ax5.set_title('Convergence', fontsize=10, color='#ddd')
        ax5.tick_params(labelsize=7, colors='#888')
        ax5.grid(True, alpha=0.25)
    else:
        panel(axes[4], kappa_ks - kappa_true, 'KS residual', 'RdBu_r', sym=True)

    fig.suptitle(f"MAP Mass Reconstruction  |  noise={noise_level:.0%}"
                 f"{'  + mask' if apply_mask else ''}",
                 fontsize=13, color='#eee', y=1.01)
    plt.tight_layout()
    plt.savefig(fname, dpi=160, facecolor='#0e0e0e', bbox_inches='tight')
    print(f"✅  Saved: {fname}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "★"*55)
    print("  DEMO 1: Noiseless reconstruction (sanity check)")
    print("★"*55)
    out1 = run_comparison(nx=20, noise_level=0.0,  lam_reg=1e-5)

    print("\n" + "★"*55)
    print("  DEMO 2: 10% noise  (FEM-MAP vs Kaiser-Squires)")
    print("★"*55)
    out2 = run_comparison(nx=20, noise_level=0.10, lam_reg=1e-2)

    print("\n" + "★"*55)
    print("  DEMO 3: 10% noise + star mask")
    print("★"*55)
    out3 = run_comparison(nx=20, noise_level=0.10, lam_reg=1e-2,
                          apply_mask=True, mask_center=(0.6, 0.6), mask_radius=0.5)

    print("\n" + "═"*55)
    print("Summary")
    print("═"*55)
    for lbl, out in [("Noiseless", out1), ("10% noise", out2), ("10%+mask", out3)]:
        print(f"  {lbl:20s}  MAP={out['l2_map']:.4f}  KS={out['l2_ks']:.4f}")
    print("═"*55)