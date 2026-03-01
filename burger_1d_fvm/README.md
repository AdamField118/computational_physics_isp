# 1D Burgers Equation - Comprehensive Project Plan
**Finite Volume Method Learning Project**

---

## Executive Summary

**Objective**: Implement a production-quality 1D Burgers equation solver using finite volume methods to master shock capturing, Riemann solvers, flux limiters, and the foundations of computational fluid dynamics.

**Timeline**: 2-3 weeks

**Languages**: Fortran (computational kernel) + Python (driver/visualization)

**Deliverables**:
1. FVM solver with multiple Riemann solvers
2. MUSCL reconstruction with TVD limiters
3. Comprehensive validation suite
4. Interactive web visualization
5. Technical documentation

---

## Mathematical Foundation

### The Burgers Equation

**Inviscid (hyperbolic)**:
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0, \quad x \in [a, b], \quad t > 0$$

**Viscous (hyperbolic-parabolic)**:
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

**Conservative form**:
$$\frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\left(\frac{u^2}{2}\right) = \nu \frac{\partial^2 u}{\partial x^2}$$

### Physical Interpretation

| Context | $u$ represents | Physical meaning |
|---------|---------------|------------------|
| Traffic flow | Vehicle density | Bunching → shocks |
| Gas dynamics | Velocity | Simplified Euler |
| Acoustics | Pressure | Nonlinear waves |

### Key Properties

1. **Nonlinearity**: $u u_x$ causes wave steepening
2. **Shock formation**: Smooth IC → discontinuity in finite time
3. **Entropy condition**: Physically correct shock selection
4. **Viscous regularization**: $\nu > 0$ smooths shocks
5. **Conservation**: $\int u \, dx$ conserved (inviscid)

### Exact Solutions (for validation)

**Shock wave (Rankine-Hugoniot)**:
$$u(x, t) = \begin{cases} u_L & x < st \\ u_R & x > st \end{cases}, \quad s = \frac{u_L + u_R}{2}$$

**Rarefaction wave**:
$$u(x, t) = \begin{cases} u_L & x < u_L t \\ \frac{x}{t} & u_L t < x < u_R t \\ u_R & x > u_R t \end{cases}$$

**Viscous shock profile**:
$$u(x, t) = \frac{1}{2}(u_L + u_R) - \frac{1}{2}(u_R - u_L) \tanh\left(\frac{(u_R - u_L)(x - st)}{4\nu}\right)$$

**N-wave** (shock + rarefaction):
$$u(x, 0) = \begin{cases} A & |x| < 1 \\ 0 & |x| > 1 \end{cases}$$

---

## Numerical Methods

### Finite Volume Discretization

**Domain discretization**:
- Cells: $[x_{i-1/2}, x_{i+1/2}]$ with center $x_i$
- Cell average: $u_i(t) = \frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} u(x, t) \, dx$

**Semi-discrete form**:
$$\frac{du_i}{dt} = -\frac{1}{\Delta x}\left(F_{i+1/2} - F_{i-1/2}\right) + \nu \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$$

where $F_{i+1/2} = F(u_i, u_{i+1})$ is the **numerical flux**.

### Riemann Solvers (for $F_{i+1/2}$)

#### 1. Godunov (First-Order Upwind)

**Physical flux**: $f(u) = \frac{u^2}{2}$

**Godunov flux**: Exact solution of local Riemann problem
$$F^{God}(u_L, u_R) = \begin{cases}
\frac{u_L^2}{2} & u_L > 0, u_R > 0 \\
\frac{u_R^2}{2} & u_L < 0, u_R < 0 \\
0 & u_L > 0 > u_R \\
\max\left(\frac{u_L^2}{2}, \frac{u_R^2}{2}\right) & \text{transonic}
\end{cases}$$

**Properties**:
- Monotone (no oscillations)
- Entropy satisfying
- First-order accurate: $O(\Delta x)$
- Very diffusive

#### 2. Lax-Friedrichs

$$F^{LF}(u_L, u_R) = \frac{1}{2}(f(u_L) + f(u_R)) - \frac{\alpha}{2}(u_R - u_L)$$

where $\alpha = \max(|u_L|, |u_R|)$ (maximum wave speed).

**Properties**:
- Simple, symmetric
- Very stable
- Most diffusive
- Good for debugging

#### 3. Roe Flux

**Linearized Riemann solver**: $f(u_R) - f(u_L) = \bar{a} (u_R - u_L)$

where $\bar{a} = \frac{u_L + u_R}{2}$ (Roe average).

$$F^{Roe}(u_L, u_R) = \frac{1}{2}(f(u_L) + f(u_R)) - \frac{1}{2}|\bar{a}|(u_R - u_L)$$

**Properties**:
- Exact for linear problems
- Less diffusive than LF
- Entropy fix needed for sonic points

### High-Resolution Methods (MUSCL)

#### Piecewise Linear Reconstruction

Instead of constant $u_i$, use **piecewise linear** in each cell:
$$u_i(x) = u_i + \sigma_i \frac{x - x_i}{\Delta x}, \quad x \in [x_{i-1/2}, x_{i+1/2}]$$

**Interface extrapolation**:
$$u_{i+1/2}^- = u_i + \frac{\sigma_i}{2}$$
$$u_{i+1/2}^+ = u_{i+1} - \frac{\sigma_{i+1}}{2}$$

**Then**: Apply Riemann solver to $(u_{i+1/2}^-, u_{i+1/2}^+)$.

#### TVD Limiters (prevent oscillations)

**Slope $\sigma_i$ must be limited** to preserve TVD property.

**Minmod** (most dissipative):
$$\sigma_i = \text{minmod}\left(u_i - u_{i-1}, u_{i+1} - u_i\right)$$
$$\text{minmod}(a, b) = \begin{cases} \min(a, b) & a, b > 0 \\ \max(a, b) & a, b < 0 \\ 0 & \text{otherwise} \end{cases}$$

**Van Leer**:
$$\sigma_i = \frac{2(u_i - u_{i-1})(u_{i+1} - u_i)}{u_{i+1} - u_{i-1}} \quad \text{if } (u_i - u_{i-1})(u_{i+1} - u_i) > 0, \text{ else } 0$$

**Superbee** (least dissipative):
$$\sigma_i = \text{maxmod}\left(\text{minmod}(2(u_i - u_{i-1}), u_{i+1} - u_i), \text{minmod}(u_i - u_{i-1}, 2(u_{i+1} - u_i))\right)$$

**Properties**:
- TVD: Total Variation Diminishing
- Second-order accuracy in smooth regions
- First-order near extrema (natural limiter action)
- No spurious oscillations

### Time Stepping

#### Explicit Methods

**Forward Euler** (first-order):
$$u_i^{n+1} = u_i^n - \frac{\Delta t}{\Delta x}\left(F_{i+1/2}^n - F_{i-1/2}^n\right)$$

**Heun (RK2)** (second-order):
$$u^* = u^n - \Delta t \mathcal{L}(u^n)$$
$$u^{n+1} = \frac{1}{2}u^n + \frac{1}{2}u^* - \frac{\Delta t}{2}\mathcal{L}(u^*)$$

**SSP-RK3** (third-order, strong stability preserving):
$$u^{(1)} = u^n + \Delta t \mathcal{L}(u^n)$$
$$u^{(2)} = \frac{3}{4}u^n + \frac{1}{4}u^{(1)} + \frac{\Delta t}{4}\mathcal{L}(u^{(1)})$$
$$u^{n+1} = \frac{1}{3}u^n + \frac{2}{3}u^{(2)} + \frac{2\Delta t}{3}\mathcal{L}(u^{(2)})$$

#### Implicit Methods (for viscous terms)

**Crank-Nicolson** (diffusion):
$$\frac{u_i^{n+1} - u_i^n}{\Delta t} = \nu \frac{u_{i+1}^{n+1} - 2u_i^{n+1} + u_{i-1}^{n+1}}{2\Delta x^2} + \nu \frac{u_{i+1}^{n} - 2u_i^{n} + u_{i-1}^{n}}{2\Delta x^2}$$

**Operator splitting** (for viscous Burgers):
1. **Convection step** (explicit): $u^* = u^n - \Delta t \frac{\partial}{\partial x}\left(\frac{(u^n)^2}{2}\right)$
2. **Diffusion step** (implicit): $u^{n+1} = u^* + \nu \Delta t \frac{\partial^2 u^{n+1}}{\partial x^2}$

### CFL Condition

**Hyperbolic**:
$$\Delta t \leq \frac{\Delta x}{\max_i |u_i|}$$

**Parabolic** (if $\nu > 0$):
$$\Delta t \leq \frac{\Delta x^2}{2\nu}$$

**Combined** (viscous Burgers):
$$\Delta t = \text{CFL} \cdot \min\left(\frac{\Delta x}{\max_i |u_i|}, \frac{\Delta x^2}{2\nu}\right)$$

Typical: $\text{CFL} \in [0.5, 0.9]$ for stability.

---

## Implementation Plan

### Phase 1: Infrastructure (Days 1-2)

#### Fortran Modules

**`burgers_types.f90`** - Data structures
```fortran
module burgers_types
    implicit none
    integer, parameter :: dp = selected_real_kind(15, 307)
    
    type :: grid_t
        integer :: nx                    ! Number of cells
        real(dp) :: x_min, x_max        ! Domain bounds
        real(dp) :: dx                   ! Cell size
        real(dp), allocatable :: x(:)    ! Cell centers
        real(dp), allocatable :: x_face(:) ! Cell faces
    end type
    
    type :: config_t
        real(dp) :: nu                   ! Viscosity
        real(dp) :: cfl                  ! CFL number
        character(len=20) :: flux_type   ! 'godunov', 'lf', 'roe'
        character(len=20) :: limiter     ! 'none', 'minmod', 'vanleer', 'superbee'
        integer :: time_order            ! 1, 2, or 3 (RK order)
    end type
end module
```

**`burgers_grid.f90`** - Grid initialization
```fortran
module burgers_grid
    use burgers_types
    contains
    
    subroutine initialize_grid(grid, nx, x_min, x_max)
        type(grid_t), intent(out) :: grid
        integer, intent(in) :: nx
        real(dp), intent(in) :: x_min, x_max
        integer :: i
        
        grid%nx = nx
        grid%x_min = x_min
        grid%x_max = x_max
        grid%dx = (x_max - x_min) / nx
        
        allocate(grid%x(nx), grid%x_face(nx+1))
        
        do i = 1, nx+1
            grid%x_face(i) = x_min + (i-1) * grid%dx
        end do
        
        do i = 1, nx
            grid%x(i) = x_min + (i - 0.5_dp) * grid%dx
        end do
    end subroutine
end module
```

#### Python Infrastructure

**`python/solver.py`** - Main solver class
```python
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

@dataclass
class BurgersConfig:
    """Configuration for Burgers solver"""
    nu: float = 0.0              # Viscosity
    cfl: float = 0.5             # CFL number
    flux_type: str = 'godunov'   # Riemann solver
    limiter: str = 'minmod'      # TVD limiter
    time_order: int = 2          # Time integration order
    
class BurgersSolver1D:
    """1D Burgers equation solver"""
    
    def __init__(self, nx: int, domain: Tuple[float, float], 
                 config: BurgersConfig):
        self.nx = nx
        self.x_min, self.x_max = domain
        self.dx = (self.x_max - self.x_min) / nx
        self.config = config
        
        # Grid
        self.x = np.linspace(self.x_min + self.dx/2, 
                            self.x_max - self.dx/2, nx)
        self.x_face = np.linspace(self.x_min, self.x_max, nx+1)
        
    def solve(self, u0: np.ndarray, t_final: float, 
              save_interval: Optional[float] = None):
        """Main solve routine - calls Fortran backend"""
        pass
```

### Phase 2: Riemann Solvers (Days 3-4)

**`riemann_solvers.f90`**
```fortran
module riemann_solvers
    use burgers_types
    implicit none
    
contains

    function godunov_flux(u_L, u_R) result(F)
        real(dp), intent(in) :: u_L, u_R
        real(dp) :: F
        
        if (u_L >= 0.0_dp .and. u_R >= 0.0_dp) then
            F = 0.5_dp * u_L**2
        else if (u_L <= 0.0_dp .and. u_R <= 0.0_dp) then
            F = 0.5_dp * u_R**2
        else if (u_L > 0.0_dp .and. u_R < 0.0_dp) then
            F = 0.0_dp  ! Shock
        else
            ! Transonic rarefaction
            F = max(0.5_dp * u_L**2, 0.5_dp * u_R**2)
        end if
    end function
    
    function lax_friedrichs_flux(u_L, u_R) result(F)
        real(dp), intent(in) :: u_L, u_R
        real(dp) :: F, alpha
        
        alpha = max(abs(u_L), abs(u_R))
        F = 0.5_dp * (0.5_dp * u_L**2 + 0.5_dp * u_R**2) - &
            0.5_dp * alpha * (u_R - u_L)
    end function
    
    function roe_flux(u_L, u_R) result(F)
        real(dp), intent(in) :: u_L, u_R
        real(dp) :: F, u_bar
        
        u_bar = 0.5_dp * (u_L + u_R)
        F = 0.5_dp * (0.5_dp * u_L**2 + 0.5_dp * u_R**2) - &
            0.5_dp * abs(u_bar) * (u_R - u_L)
    end function
    
end module
```

**Testing**: Compare all three solvers on Riemann problem.

### Phase 3: MUSCL Reconstruction (Days 5-6)

**`reconstruction.f90`**
```fortran
module reconstruction
    use burgers_types
    implicit none
    
contains

    function minmod(a, b) result(limited)
        real(dp), intent(in) :: a, b
        real(dp) :: limited
        
        if (a > 0.0_dp .and. b > 0.0_dp) then
            limited = min(a, b)
        else if (a < 0.0_dp .and. b < 0.0_dp) then
            limited = max(a, b)
        else
            limited = 0.0_dp
        end if
    end function
    
    function van_leer(a, b) result(limited)
        real(dp), intent(in) :: a, b
        real(dp) :: limited
        
        if (a * b > 0.0_dp) then
            limited = 2.0_dp * a * b / (a + b)
        else
            limited = 0.0_dp
        end if
    end function
    
    subroutine muscl_reconstruct(u, sigma, limiter_type)
        real(dp), intent(in) :: u(:)
        real(dp), intent(out) :: sigma(:)
        character(len=*), intent(in) :: limiter_type
        integer :: i, n
        real(dp) :: slope_left, slope_right
        
        n = size(u)
        sigma = 0.0_dp
        
        do i = 2, n-1
            slope_left = u(i) - u(i-1)
            slope_right = u(i+1) - u(i)
            
            select case(trim(limiter_type))
            case('minmod')
                sigma(i) = minmod(slope_left, slope_right)
            case('vanleer')
                sigma(i) = van_leer(slope_left, slope_right)
            case('none')
                sigma(i) = 0.5_dp * (slope_left + slope_right)
            end select
        end do
    end subroutine
    
end module
```

**Testing**: Verify second-order convergence on smooth problems.

### Phase 4: Time Integration (Days 7-8)

**`time_stepping.f90`**
```fortran
module time_stepping
    use burgers_types
    use riemann_solvers
    use reconstruction
    implicit none
    
contains

    subroutine compute_rhs(u, rhs, grid, config)
        real(dp), intent(in) :: u(:)
        real(dp), intent(out) :: rhs(:)
        type(grid_t), intent(in) :: grid
        type(config_t), intent(in) :: config
        
        real(dp), allocatable :: sigma(:), u_L(:), u_R(:), flux(:)
        integer :: i, nx
        
        nx = grid%nx
        allocate(sigma(nx), u_L(nx+1), u_R(nx+1), flux(nx+1))
        
        ! MUSCL reconstruction
        call muscl_reconstruct(u, sigma, config%limiter)
        
        ! Extrapolate to interfaces
        do i = 1, nx
            u_R(i) = u(i) - 0.5_dp * sigma(i)
        end do
        do i = 2, nx+1
            u_L(i) = u(i-1) + 0.5_dp * sigma(i-1)
        end do
        
        ! Boundary conditions (periodic for now)
        u_L(1) = u_R(nx)
        u_R(nx+1) = u_L(1)
        
        ! Compute fluxes
        do i = 1, nx+1
            select case(trim(config%flux_type))
            case('godunov')
                flux(i) = godunov_flux(u_L(i), u_R(i))
            case('lf')
                flux(i) = lax_friedrichs_flux(u_L(i), u_R(i))
            case('roe')
                flux(i) = roe_flux(u_L(i), u_R(i))
            end select
        end do
        
        ! Compute RHS
        do i = 1, nx
            rhs(i) = -(flux(i+1) - flux(i)) / grid%dx
        end do
        
        ! Add viscous term if nu > 0
        if (config%nu > 0.0_dp) then
            do i = 2, nx-1
                rhs(i) = rhs(i) + config%nu * &
                    (u(i+1) - 2.0_dp*u(i) + u(i-1)) / grid%dx**2
            end do
        end if
    end subroutine
    
    subroutine rk2_step(u, dt, grid, config)
        real(dp), intent(inout) :: u(:)
        real(dp), intent(in) :: dt
        type(grid_t), intent(in) :: grid
        type(config_t), intent(in) :: config
        
        real(dp), allocatable :: u_star(:), rhs(:)
        integer :: nx
        
        nx = grid%nx
        allocate(u_star(nx), rhs(nx))
        
        ! Stage 1
        call compute_rhs(u, rhs, grid, config)
        u_star = u + dt * rhs
        
        ! Stage 2
        call compute_rhs(u_star, rhs, grid, config)
        u = 0.5_dp * u + 0.5_dp * u_star + 0.5_dp * dt * rhs
    end subroutine
    
end module
```

### Phase 5: Python Interface (Day 9)

**`python_interface.f90`**
```fortran
module python_interface
    use burgers_types
    use burgers_grid
    use time_stepping
    implicit none
    
contains

    subroutine solve_burgers(u_init, t_final, nx, x_min, x_max, &
                             nu, cfl, flux_type, limiter, &
                             u_final, n_steps)
        !f2py intent(in) :: u_init, t_final, nx, x_min, x_max
        !f2py intent(in) :: nu, cfl, flux_type, limiter
        !f2py intent(out) :: u_final, n_steps
        
        real(dp), intent(in) :: u_init(:)
        real(dp), intent(in) :: t_final, x_min, x_max, nu, cfl
        integer, intent(in) :: nx
        character(len=*), intent(in) :: flux_type, limiter
        real(dp), intent(out) :: u_final(nx)
        integer, intent(out) :: n_steps
        
        type(grid_t) :: grid
        type(config_t) :: config
        real(dp), allocatable :: u(:)
        real(dp) :: t, dt, u_max
        integer :: step
        
        ! Initialize
        call initialize_grid(grid, nx, x_min, x_max)
        
        config%nu = nu
        config%cfl = cfl
        config%flux_type = flux_type
        config%limiter = limiter
        config%time_order = 2
        
        allocate(u(nx))
        u = u_init
        
        ! Time stepping
        t = 0.0_dp
        step = 0
        
        do while (t < t_final)
            ! Compute dt
            u_max = maxval(abs(u))
            dt = cfl * grid%dx / max(u_max, 1.0e-10_dp)
            
            if (nu > 0.0_dp) then
                dt = min(dt, cfl * grid%dx**2 / (2.0_dp * nu))
            end if
            
            if (t + dt > t_final) dt = t_final - t
            
            ! Take step
            call rk2_step(u, dt, grid, config)
            
            t = t + dt
            step = step + 1
        end do
        
        u_final = u
        n_steps = step
    end subroutine
    
end module
```

### Phase 6: Validation Suite (Days 10-12)

**Test cases to implement**:

#### Test 1: Shock Formation
```python
def test_shock_formation():
    """Sine wave develops shock at t ≈ 1"""
    solver = BurgersSolver1D(nx=400, domain=(0, 2*np.pi), 
                             config=BurgersConfig(nu=0.0))
    
    u0 = -np.sin(solver.x)
    
    # Should form shock
    u_final = solver.solve(u0, t_final=1.5)
    
    # Check for discontinuity
    max_gradient = np.max(np.abs(np.diff(u_final)))
    assert max_gradient > 10.0  # Steep gradient indicates shock
```

#### Test 2: Riemann Problem
```python
def test_riemann_shock():
    """Validate against exact Riemann solution"""
    solver = BurgersSolver1D(nx=200, domain=(-1, 1))
    
    # Left/right states
    u_L, u_R = 1.0, 0.0
    shock_speed = 0.5 * (u_L + u_R)
    
    u0 = np.where(solver.x < 0, u_L, u_R)
    t_final = 0.5
    
    u_num = solver.solve(u0, t_final)
    
    # Exact solution
    shock_pos = shock_speed * t_final
    u_exact = np.where(solver.x < shock_pos, u_L, u_R)
    
    # Compare (away from shock)
    mask = np.abs(solver.x - shock_pos) > 2*solver.dx
    error = np.linalg.norm(u_num[mask] - u_exact[mask], 1) * solver.dx
    
    assert error < 0.01
```

#### Test 3: Convergence Study
```python
def test_convergence_smooth():
    """Second-order convergence on smooth solution"""
    errors = []
    dx_values = []
    
    for nx in [50, 100, 200, 400]:
        solver = BurgersSolver1D(nx=nx, domain=(0, 1),
                                config=BurgersConfig(limiter='minmod'))
        
        u0 = np.exp(-20 * (solver.x - 0.5)**2)
        u_final = solver.solve(u0, t_final=0.1)
        
        # Reference solution (very fine grid)
        solver_ref = BurgersSolver1D(nx=1600, domain=(0, 1))
        u_ref = solver_ref.solve(u0_fine, t_final=0.1)
        
        # Interpolate and compare
        u_interp = interp1d(solver_ref.x, u_ref)(solver.x)
        error = np.linalg.norm(u_final - u_interp, 2) * np.sqrt(solver.dx)
        
        errors.append(error)
        dx_values.append(solver.dx)
    
    # Check convergence rate
    rate = np.polyfit(np.log(dx_values), np.log(errors), 1)[0]
    assert rate > 1.8  # Should be ~2.0
```

#### Test 4: Conservation
```python
def test_conservation():
    """Mass should be conserved exactly"""
    solver = BurgersSolver1D(nx=200, domain=(0, 1),
                            config=BurgersConfig(nu=0.0))
    
    u0 = np.sin(2*np.pi * solver.x)
    mass_init = np.sum(u0) * solver.dx
    
    u_final = solver.solve(u0, t_final=1.0)
    mass_final = np.sum(u_final) * solver.dx
    
    assert abs(mass_final - mass_init) < 1e-12
```

### Phase 7: Visualization (Days 13-14)

**`python/visualization.py`**
```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

def plot_solution(x, u, t=None, exact=None, title='Burgers Solution'):
    """Plot solution at single time"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x, u, 'b-', linewidth=2, label='Numerical')
    
    if exact is not None:
        ax.plot(x, exact, 'r--', linewidth=2, label='Exact')
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('u', fontsize=14)
    
    if t is not None:
        title += f' (t = {t:.3f})'
    ax.set_title(title, fontsize=16)
    
    if exact is not None:
        ax.legend(fontsize=12)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax

def plot_spacetime_diagram(x, times, u_history, title='Space-Time Diagram'):
    """2D plot showing solution evolution"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    X, T = np.meshgrid(x, times)
    
    contour = ax.contourf(X, T, u_history, levels=30, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, label='u(x,t)')
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('t', fontsize=14)
    ax.set_title(title, fontsize=16)
    
    plt.tight_layout()
    return fig, ax

def animate_solution(x, times, u_history, save_path=None):
    """Create animation of solution evolution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line, = ax.plot([], [], 'b-', linewidth=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14)
    
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(u_history.min() - 0.1, u_history.max() + 0.1)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('u', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(frame):
        line.set_data(x, u_history[frame])
        time_text.set_text(f't = {times[frame]:.3f}')
        return line, time_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(times), interval=50,
                                  blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
    
    return anim

def plot_convergence_study(dx_values, errors, expected_order=2):
    """Plot convergence rates"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.loglog(dx_values, errors, 'o-', linewidth=2, markersize=8,
             label='Numerical error')
    
    # Reference slope
    dx_ref = np.array([dx_values[0], dx_values[-1]])
    error_ref = errors[0] * (dx_ref / dx_values[0])**expected_order
    ax.loglog(dx_ref, error_ref, 'k--', linewidth=1.5,
             label=f'O(Δx^{expected_order})')
    
    # Compute actual rate
    rate = np.polyfit(np.log(dx_values), np.log(errors), 1)[0]
    
    ax.set_xlabel('Δx', fontsize=14)
    ax.set_ylabel('L² Error', fontsize=14)
    ax.set_title(f'Convergence Study (rate = {rate:.2f})', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig, ax
```

**Interactive web visualization**:
```javascript
// web/burgers_viz.js
function createBurgersVisualization(container, data) {
    // Use Chart.js for interactive plots
    // Allow user to:
    // - Select flux type
    // - Adjust viscosity
    // - Choose initial condition
    // - Animate solution
}
```

---

## Validation Benchmarks

### Benchmark 1: Shock Formation Time

**Problem**: Smooth sine wave → shock

**Initial condition**: $u(x, 0) = -\sin(x)$, $x \in [0, 2\pi]$

**Theory**: Shock forms when $u_x \to -\infty$, which occurs at:
$$t_s = \frac{1}{\max|u_x(x, 0)|} = \frac{1}{1} = 1$$

**Validation**: Measure when $\max|u_x|$ exceeds threshold (e.g., 100).

**Expected**: $t_s^{num} \approx 1.0$ for well-resolved simulations.

### Benchmark 2: Shock Speed Accuracy

**Problem**: Constant left/right states

**Initial condition**: $u(x, 0) = \begin{cases} 1 & x < 0 \\ 0 & x > 0 \end{cases}$

**Exact shock speed**: $s = \frac{u_L + u_R}{2} = 0.5$

**Validation**: Track shock position $x_s(t)$ (where $|u_x|$ is maximum).

**Expected**: $x_s(t) = 0.5t$ with error $< 1\%$.

### Benchmark 3: Viscous Shock Profile

**Problem**: Steady traveling wave

**Viscous Burgers**: $u_t + u u_x = \nu u_{xx}$

**Exact solution**:
$$u(x, t) = \frac{1}{2}(u_L + u_R) - \frac{1}{2}(u_R - u_L) \tanh\left(\frac{(u_R - u_L)(x - st)}{4\nu}\right)$$

**Parameters**: $u_L = 1$, $u_R = 0$, $\nu = 0.01$, $s = 0.5$

**Validation**: Compare numerical vs exact at $t = 1.0$.

**Expected**: $L^2$ error $< 10^{-3}$ for $\Delta x = 0.01$.

### Benchmark 4: Limiter Comparison

**Problem**: Same IC, compare limiters

**Initial condition**: Square wave (sharp gradients)

**Limiters tested**: None, Minmod, Van Leer, Superbee

**Metrics**:
- Solution smoothness (no oscillations?)
- Shock resolution (width)
- CPU time

**Expected**:
- No limiter → Oscillations (Gibbs phenomenon)
- Minmod → Most diffusive, widest shock
- Van Leer → Moderate
- Superbee → Sharpest, but may overshoot slightly

---

## Expected Results

### Convergence Rates

| Region | Godunov | MUSCL-Minmod | MUSCL-Van Leer |
|--------|---------|--------------|----------------|
| Smooth | $O(\Delta x)$ | $O(\Delta x^2)$ | $O(\Delta x^2)$ |
| Shock | $O(\Delta x^{1/2})$ | $O(\Delta x)$ | $O(\Delta x)$ |

### Performance Targets

| Grid size | Time steps | Wall time (Fortran) | Speedup vs Python |
|-----------|------------|---------------------|-------------------|
| 100 | ~500 | < 0.01 s | ~50× |
| 1,000 | ~5,000 | < 0.1 s | ~100× |
| 10,000 | ~50,000 | < 5 s | ~200× |

### Visual Outputs

1. **Shock formation animation** (sine wave → shock)
2. **Space-time diagram** (characteristic lines, shock path)
3. **Limiter comparison** (side-by-side plots)
4. **Convergence plots** (log-log error vs Δx)
5. **Phase space** (u vs u_x showing shock steepening)

---

## Deliverables

### Code
- [ ] Complete Fortran FVM solver (6 modules, ~800 lines)
- [ ] Python driver and analysis tools (~400 lines)
- [ ] f2py interface
- [ ] Comprehensive test suite (>90% coverage)

### Documentation
- [ ] Mathematical derivation document
- [ ] Code documentation (docstrings, comments)
- [ ] User guide with examples
- [ ] Validation report

### Visualizations
- [ ] 4+ benchmark result plots
- [ ] Convergence study graphs
- [ ] Interactive web demo (optional)
- [ ] Animation: shock formation

### Blog Post
- [ ] Project overview
- [ ] Mathematical background
- [ ] Implementation highlights
- [ ] Results and validation
- [ ] Comparison to FEM approach

---

## Learning Objectives

By completing this project, you will master:

### Numerical Methods
- [x] Conservative finite volume discretization
- [x] Riemann solver theory and implementation
- [x] High-resolution schemes (MUSCL)
- [x] TVD limiters (minmod, van Leer, superbee)
- [x] Time integration (RK2, RK3, SSP methods)
- [x] CFL condition and stability

### Computational Physics
- [x] Shock capturing without tracking
- [x] Entropy conditions and uniqueness
- [x] Nonlinear wave propagation
- [x] Viscous vs inviscid behavior
- [x] Conservation properties

### Software Engineering
- [x] Fortran-Python hybrid architecture
- [x] f2py interfacing
- [x] Modular code design
- [x] Comprehensive testing
- [x] Scientific visualization
- [x] Performance optimization

### FVM Fundamentals
- [x] Cell-centered vs node-centered
- [x] Flux functions
- [x] Upwind methods
- [x] Limiters and TVD property
- [x] Convergence theory

---

## Extensions & Next Steps

### Immediate Extensions
- [ ] Adaptive time stepping (error-based CFL)
- [ ] Different boundary conditions (inflow, outflow, reflective)
- [ ] Entropy fix for sonic points
- [ ] Compare RK2 vs RK3 vs RK4

### Advanced FVM
- [ ] 2D Burgers equation
- [ ] Discontinuous Galerkin methods
- [ ] WENO reconstruction (5th order)
- [ ] Adaptive mesh refinement

### Applications
- [ ] Traffic flow modeling
- [ ] Gas dynamics (scalar Euler)
- [ ] Reaction-advection equations

### Bridge to CFD
- [ ] Vector Burgers → Euler equations
- [ ] Shallow water equations (next project!)
- [ ] Add source terms
- [ ] Moving to systems

---

## Timeline

### Week 1: Foundation
- **Day 1-2**: Infrastructure (types, grid, Python class)
- **Day 3-4**: Riemann solvers (Godunov, LF, Roe)
- **Day 5-6**: MUSCL reconstruction + limiters
- **Day 7**: Integration

### Week 2: Validation & Polish
- **Day 8-9**: Time stepping + f2py interface
- **Day 10-11**: Test suite (4 benchmarks)
- **Day 12-13**: Visualization tools
- **Day 14**: Documentation + blog post

### Week 3 (Optional): Advanced Topics
- **Day 15-16**: WENO reconstruction
- **Day 17-18**: 2D extension
- **Day 19-21**: Web interface + interactive demos

---

## Success Criteria

### Must Have
- [x] All 4 benchmarks pass
- [x] Second-order convergence demonstrated
- [x] Conservation to machine precision
- [x] No spurious oscillations with limiters
- [x] Fortran >50× faster than pure Python

### Should Have
- [ ] Interactive visualizations
- [ ] Complete documentation
- [ ] Web demo deployed
- [ ] Blog post written

### Nice to Have
- [ ] WENO implementation
- [ ] 2D extension
- [ ] Published on GitHub
- [ ] Video explanation

---

## Resources

### Textbooks
- LeVeque, *Finite Volume Methods for Hyperbolic Problems* (2002) - **Chapter 4**
- Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics* (2009)
- Laney, *Computational Gasdynamics* (1998)

### Papers
- Godunov (1959) - Original upwind scheme
- van Leer (1979) - MUSCL reconstruction
- Harten (1983) - TVD schemes and limiters
- Shu & Osher (1988) - ENO schemes

### Exact Solutions
- Whitham, *Linear and Nonlinear Waves* (1974)
- Cole (1951) - Hopf-Cole transformation

### Software References
- Clawpack (Python/Fortran) - https://www.clawpack.org
- SU2 (C++) - Open-source CFD

---

## Notes

This project is the **essential foundation** for all FVM work. Burgers contains all the key concepts (shocks, Riemann solvers, limiters) without the complexity of systems. Master this before moving to shallow water or Navier-Stokes.

**Key insight**: The numerical flux $F_{i+1/2}$ is the heart of FVM. Everything else (reconstruction, time stepping) supports computing this flux accurately and efficiently.