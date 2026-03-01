# 2D Shallow Water Equations - Comprehensive Project Plan
**Production-Quality FVM for Geophysical Flows**

---

## Executive Summary

**Objective**: Implement a robust, well-balanced finite volume solver for the 2D shallow water equations, demonstrating HLL/HLLC Riemann solvers, source term balancing, and wet/dry front treatment.

**Timeline**: 3-4 weeks

**Prerequisites**: Completed 1D Burgers project (Riemann solvers, MUSCL, limiters)

**Languages**: Fortran (computational engine) + Python (driver/analysis) + JavaScript (web viz)

**Deliverables**:
1. 2D FVM solver with well-balanced schemes
2. HLL and HLLC Riemann solvers for systems
3. Hydrostatic reconstruction for wet/dry
4. Benchmark validation suite (4 test cases)
5. Interactive 2D visualization
6. Technical documentation

---

## Mathematical Foundation

### Shallow Water Equations (Conservative Form)

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}}{\partial x} + \frac{\partial \mathbf{G}}{\partial y} = \mathbf{S}$$

where:

**State vector**:
$$\mathbf{U} = \begin{pmatrix} h \\ hu \\ hv \end{pmatrix}$$

**x-direction flux**:
$$\mathbf{F}(\mathbf{U}) = \begin{pmatrix} hu \\ hu^2 + \frac{1}{2}gh^2 \\ huv \end{pmatrix}$$

**y-direction flux**:
$$\mathbf{G}(\mathbf{U}) = \begin{pmatrix} hv \\ huv \\ hv^2 + \frac{1}{2}gh^2 \end{pmatrix}$$

**Source term** (bathymetry):
$$\mathbf{S} = \begin{pmatrix} 0 \\ -gh\frac{\partial b}{\partial x} \\ -gh\frac{\partial b}{\partial y} \end{pmatrix}$$

**Variables**:
- $h(x, y, t)$ - water depth [m]
- $(u, v)$ - depth-averaged velocity [m/s]
- $b(x, y)$ - bottom elevation (bathymetry) [m]
- $g = 9.81$ m/s² - gravitational acceleration
- $\eta = h + b$ - surface elevation [m]

### Physical Interpretation

**Mass conservation**:
$$\frac{\partial h}{\partial t} + \nabla \cdot (h\mathbf{u}) = 0$$
*Water is incompressible in vertical.*

**Momentum conservation**:
$$\frac{\partial (h\mathbf{u})}{\partial t} + \nabla \cdot (h\mathbf{u} \otimes \mathbf{u}) + \frac{1}{2}g\nabla h^2 = -gh\nabla b$$
*Pressure gradient balanced by gravity.*

### Hyperbolic Structure

**Eigenvalues** (wave speeds in x-direction):
$$\lambda_1 = u - c, \quad \lambda_2 = u, \quad \lambda_3 = u + c$$

where $c = \sqrt{gh}$ is the **gravity wave speed**.

**Physical meaning**:
- $\lambda_1, \lambda_3$: Gravity waves propagating left/right
- $\lambda_2$: Material (contact) wave (carries $v$ discontinuity)

**Eigenvectors** (characteristic variables):
$$\mathbf{r}_1 = \begin{pmatrix} 1 \\ u - c \\ v \end{pmatrix}, \quad
\mathbf{r}_2 = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}, \quad
\mathbf{r}_3 = \begin{pmatrix} 1 \\ u + c \\ v \end{pmatrix}$$

### Froude Number

$$Fr = \frac{|\mathbf{u}|}{\sqrt{gh}}$$

**Regimes**:
- $Fr < 1$: **Subcritical** (slow flow, gravity dominates)
- $Fr = 1$: **Critical** (hydraulic jump)
- $Fr > 1$: **Supercritical** (fast flow, inertia dominates)

### Steady States (Lake at Rest)

**Lake at rest**: $\mathbf{u} = 0$, $\eta = h + b = \text{const}$

**Key property**: Fluxes must **exactly balance** source terms.

This is NON-TRIVIAL: standard schemes generate **spurious currents** that dominate the solution!

---

## Numerical Challenges

### Challenge 1: Well-Balancing

**Problem**: Standard FVM doesn't preserve lake at rest.

**Example**: Still water over a bump.
- Exact: $u = 0$ forever
- Standard FVM: Generates artificial currents $u \sim O(1)$ immediately!

**Solution**: **Well-balanced schemes** that discretize fluxes and sources together.

### Challenge 2: Wet/Dry Fronts

**Problem**: $h \to 0$ at shorelines (moving boundaries).

**Issues**:
- Division by zero: $u = \frac{hu}{h}$ undefined
- Negative depth: $h < 0$ non-physical
- Wave speeds: $c = \sqrt{gh}$ singular

**Solution**: **Hydrostatic reconstruction** + **positivity-preserving limiters**.

### Challenge 3: Geometric Source Terms

**Problem**: Source depends on $\nabla b$, which is spatially varying.

**Issue**: Must compute $b_{i+1/2}$ at cell interfaces (half-integer indices).

**Solution**: Reconstruct $\eta = h + b$ instead of $h$ directly.

### Challenge 4: System Riemann Problem

**Problem**: 3×3 system (not scalar like Burgers).

**Complexity**:
- 3 waves (not 1)
- Nonlinear wave speeds
- Contact discontinuities
- Transonic rarefactions

**Solution**: Approximate Riemann solvers (HLL, HLLC, Roe).

---

## Numerical Methods

### Finite Volume Discretization

**Cell $(i, j)$**: $\Omega_{ij} = [x_{i-1/2}, x_{i+1/2}] \times [y_{j-1/2}, y_{j+1/2}]$

**Cell average**:
$$\mathbf{U}_{ij}(t) = \frac{1}{\Delta x \Delta y} \iint_{\Omega_{ij}} \mathbf{U}(x, y, t) \, dA$$

**Semi-discrete form**:
$$\frac{d\mathbf{U}_{ij}}{dt} = -\frac{1}{\Delta x}\left(\mathbf{F}_{i+1/2,j} - \mathbf{F}_{i-1/2,j}\right) - \frac{1}{\Delta y}\left(\mathbf{G}_{i,j+1/2} - \mathbf{G}_{i,j-1/2}\right) + \mathbf{S}_{ij}$$

### HLL Riemann Solver

**Input**: Left/right states $\mathbf{U}_L$, $\mathbf{U}_R$ at interface (rotated to normal direction).

**Wave speed estimates**:
$$S_L = \min(u_L - c_L, u_R - c_R)$$
$$S_R = \max(u_L + c_L, u_R + c_R)$$

**HLL flux**:
$$\mathbf{F}_{HLL} = \begin{cases}
\mathbf{F}_L & S_L \geq 0 \\
\frac{S_R \mathbf{F}_L - S_L \mathbf{F}_R + S_L S_R (\mathbf{U}_R - \mathbf{U}_L)}{S_R - S_L} & S_L < 0 < S_R \\
\mathbf{F}_R & S_R \leq 0
\end{cases}$$

**Properties**:
- **Positive depth**: Guarantees $h \geq 0$
- **Entropy satisfying**: Dissipates entropy
- **Robust**: Works even when $h_L = 0$ or $h_R = 0$

**Drawback**: Smears contact discontinuities (linearly degenerate field).

### HLLC Riemann Solver

**Improvement**: Resolve middle wave (contact).

**Middle wave speed**:
$$S_M = \frac{S_L h_R (u_R - S_R) - S_R h_L (u_L - S_L)}{h_R(u_R - S_R) - h_L(u_L - S_L)}$$

**Star states** (intermediate):
$$\mathbf{U}_L^* = h_L \frac{S_L - u_L}{S_L - S_M} \begin{pmatrix} 1 \\ S_M \\ v_L \end{pmatrix}, \quad
\mathbf{U}_R^* = h_R \frac{S_R - u_R}{S_R - S_M} \begin{pmatrix} 1 \\ S_M \\ v_R \end{pmatrix}$$

**HLLC flux**:
$$\mathbf{F}_{HLLC} = \begin{cases}
\mathbf{F}_L & S_L \geq 0 \\
\mathbf{F}_L + S_L(\mathbf{U}_L^* - \mathbf{U}_L) & S_L < 0 \leq S_M \\
\mathbf{F}_R + S_R(\mathbf{U}_R^* - \mathbf{U}_R) & S_M < 0 < S_R \\
\mathbf{F}_R & S_R \leq 0
\end{cases}$$

**Properties**:
- Resolves contact (sharp $v$ jumps)
- Still positive depth
- Slightly more expensive than HLL

### Well-Balanced Scheme (Surface Gradient Method)

**Key idea**: Reconstruct $\eta = h + b$ instead of $h$.

**At interface** $(i+1/2, j)$:

$$\eta_{i+1/2,j}^L = \eta_{ij} + \frac{\Delta x}{2} (\nabla \eta)_{ij}$$
$$\eta_{i+1/2,j}^R = \eta_{i+1,j} - \frac{\Delta x}{2} (\nabla \eta)_{i+1,j}$$

**Reconstruct depth**:
$$h_{i+1/2,j}^L = \max(0, \eta_{i+1/2,j}^L - b_{i+1/2,j})$$
$$h_{i+1/2,j}^R = \max(0, \eta_{i+1/2,j}^R - b_{i+1/2,j})$$

where $b_{i+1/2,j} = \max(b_{ij}, b_{i+1,j})$ (ensures positivity).

**Source term balancing**:
$$\mathbf{S}_{ij}^x = -\frac{g}{\Delta x}\left[ \frac{h_{i+1/2,j}^L + h_{i+1/2,j}^R}{2} (b_{i+1,j} - b_{ij}) \right]$$

**Result**: Lake at rest is preserved to **machine precision**.

### Hydrostatic Reconstruction (Wet/Dry)

**Problem**: Standard reconstruction gives $h < 0$ near dry cells.

**Hydrostatic method** (Audusse et al. 2004):

At interface $(i+1/2, j)$:

$$b_{i+1/2,j} = \max(b_{ij}, b_{i+1,j})$$

$$h_{i+1/2,j}^L = \max(0, h_{ij} + b_{ij} - b_{i+1/2,j})$$
$$h_{i+1/2,j}^R = \max(0, h_{i+1,j} + b_{i+1,j} - b_{i+1/2,j})$$

**Effect**: Automatically handles:
- Dry cells ($h = 0$)
- Wetting (transition $h: 0 \to h_0$)
- Drying (transition $h: h_0 \to 0$)

**Property**: **Positivity-preserving** under CFL condition.

### MUSCL Reconstruction (2D)

**Gradient reconstruction** (Green-Gauss):
$$(\nabla \mathbf{U})_{ij} = \frac{1}{\Delta x \Delta y} \oint_{\partial \Omega_{ij}} \mathbf{U} \, \mathbf{n} \, ds$$

**Approximate**:
$$\left(\frac{\partial \mathbf{U}}{\partial x}\right)_{ij} = \frac{\mathbf{U}_{i+1,j} - \mathbf{U}_{i-1,j}}{2\Delta x}$$
$$\left(\frac{\partial \mathbf{U}}{\partial y}\right)_{ij} = \frac{\mathbf{U}_{i,j+1} - \mathbf{U}_{i,j-1}}{2\Delta y}$$

**Apply limiters** (component-by-component):
$$\left(\frac{\partial \mathbf{U}}{\partial x}\right)_{ij}^{lim} = \text{minmod}\left(\frac{\mathbf{U}_{ij} - \mathbf{U}_{i-1,j}}{\Delta x}, \frac{\mathbf{U}_{i+1,j} - \mathbf{U}_{ij}}{\Delta x}\right)$$

**Extrapolate to interfaces**:
$$\mathbf{U}_{i+1/2,j}^L = \mathbf{U}_{ij} + \frac{\Delta x}{2}\left(\frac{\partial \mathbf{U}}{\partial x}\right)_{ij}^{lim}$$

### Time Stepping: SSP-RK3

**Strong Stability Preserving** Runge-Kutta (3rd order):

$$\mathbf{U}^{(1)} = \mathbf{U}^n + \Delta t \mathcal{L}(\mathbf{U}^n)$$
$$\mathbf{U}^{(2)} = \frac{3}{4}\mathbf{U}^n + \frac{1}{4}\mathbf{U}^{(1)} + \frac{\Delta t}{4}\mathcal{L}(\mathbf{U}^{(1)})$$
$$\mathbf{U}^{n+1} = \frac{1}{3}\mathbf{U}^n + \frac{2}{3}\mathbf{U}^{(2)} + \frac{2\Delta t}{3}\mathcal{L}(\mathbf{U}^{(2)})$$

where $\mathcal{L}(\mathbf{U})$ is the RHS (fluxes + source).

**Properties**:
- TVD (Total Variation Diminishing)
- 3rd order accurate in time
- No spurious oscillations

### CFL Condition

$$\Delta t = \text{CFL} \cdot \min_{i,j} \left( \frac{\Delta x}{|u_{ij}| + c_{ij}}, \frac{\Delta y}{|v_{ij}| + c_{ij}} \right)$$

where $c_{ij} = \sqrt{g h_{ij}}$.

Typical: $\text{CFL} = 0.5$ for stability.

---

## Implementation Plan

### Phase 1: Infrastructure (Days 1-3)

#### Fortran Modules

**`shallow_water_types.f90`**
```fortran
module shallow_water_types
    implicit none
    integer, parameter :: dp = selected_real_kind(15, 307)
    
    type :: grid_t
        integer :: nx, ny               ! Grid size
        real(dp) :: x_min, x_max        ! Domain bounds
        real(dp) :: y_min, y_max
        real(dp) :: dx, dy              ! Cell size
        real(dp), allocatable :: x(:)   ! Cell centers
        real(dp), allocatable :: y(:)
        real(dp), allocatable :: b(:,:) ! Bathymetry
    end type
    
    type :: state_t
        real(dp), allocatable :: h(:,:)  ! Depth
        real(dp), allocatable :: hu(:,:) ! x-momentum
        real(dp), allocatable :: hv(:,:) ! y-momentum
    end type
    
    type :: config_t
        real(dp) :: g                    ! Gravity
        real(dp) :: cfl                  ! CFL number
        character(len=20) :: riemann     ! 'hll' or 'hllc'
        character(len=20) :: limiter     ! 'minmod', 'vanleer'
        logical :: well_balanced         ! Use surface gradient?
    end type
    
    ! Conservative variable vector
    type :: conserved_t
        real(dp) :: h, hu, hv
    end type
end module
```

**`grid_utils.f90`**
```fortran
module grid_utils
    use shallow_water_types
    contains
    
    subroutine initialize_grid(grid, nx, ny, domain)
        type(grid_t), intent(out) :: grid
        integer, intent(in) :: nx, ny
        real(dp), intent(in) :: domain(4)  ! [x_min, x_max, y_min, y_max]
        integer :: i, j
        
        grid%nx = nx
        grid%ny = ny
        grid%x_min = domain(1)
        grid%x_max = domain(2)
        grid%y_min = domain(3)
        grid%y_max = domain(4)
        
        grid%dx = (grid%x_max - grid%x_min) / nx
        grid%dy = (grid%y_max - grid%y_min) / ny
        
        allocate(grid%x(nx), grid%y(ny), grid%b(nx, ny))
        
        do i = 1, nx
            grid%x(i) = grid%x_min + (i - 0.5_dp) * grid%dx
        end do
        
        do j = 1, ny
            grid%y(j) = grid%y_min + (j - 0.5_dp) * grid%dy
        end do
        
        grid%b = 0.0_dp  ! Flat bottom default
    end subroutine
    
    subroutine set_bathymetry(grid, b_func)
        type(grid_t), intent(inout) :: grid
        interface
            function b_func(x, y) result(b)
                import :: dp
                real(dp), intent(in) :: x, y
                real(dp) :: b
            end function
        end interface
        integer :: i, j
        
        do j = 1, grid%ny
            do i = 1, grid%nx
                grid%b(i,j) = b_func(grid%x(i), grid%y(j))
            end do
        end do
    end subroutine
end module
```

#### Python Infrastructure

**`python/solver.py`**
```python
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, List

@dataclass
class ShallowWaterConfig:
    """Configuration for shallow water solver"""
    g: float = 9.81              # Gravity [m/s²]
    cfl: float = 0.5             # CFL number
    riemann: str = 'hll'         # Riemann solver
    limiter: str = 'minmod'      # Slope limiter
    well_balanced: bool = True   # Use surface gradient?
    time_order: int = 3          # RK order (1, 2, or 3)

class ShallowWaterSolver2D:
    """2D Shallow Water Equation Solver"""
    
    def __init__(self, nx: int, ny: int, 
                 domain: Tuple[Tuple[float, float], Tuple[float, float]],
                 config: ShallowWaterConfig = ShallowWaterConfig()):
        """
        Parameters:
        -----------
        nx, ny : int
            Grid size
        domain : ((x_min, x_max), (y_min, y_max))
            Physical domain bounds
        config : ShallowWaterConfig
            Solver configuration
        """
        self.nx = nx
        self.ny = ny
        self.domain = domain
        self.config = config
        
        # Grid
        x_bounds, y_bounds = domain
        self.dx = (x_bounds[1] - x_bounds[0]) / nx
        self.dy = (y_bounds[1] - y_bounds[0]) / ny
        
        self.x = np.linspace(x_bounds[0] + self.dx/2, 
                            x_bounds[1] - self.dx/2, nx)
        self.y = np.linspace(y_bounds[0] + self.dy/2, 
                            y_bounds[1] - self.dy/2, ny)
        
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Bathymetry (flat by default)
        self.bathymetry = np.zeros((nx, ny))
    
    def set_bathymetry(self, b: np.ndarray):
        """Set bottom topography"""
        assert b.shape == (self.nx, self.ny)
        self.bathymetry = b.copy()
    
    def solve(self, h0: np.ndarray, hu0: np.ndarray, hv0: np.ndarray,
              t_final: float, output_interval: Optional[float] = None):
        """
        Solve shallow water equations
        
        Parameters:
        -----------
        h0, hu0, hv0 : ndarray, shape (nx, ny)
            Initial conditions
        t_final : float
            Final time
        output_interval : float, optional
            Time interval for saving snapshots
        
        Returns:
        --------
        result : dict
            Contains 'times', 'h', 'hu', 'hv' arrays
        """
        # Call Fortran backend
        pass
```

**`python/initial_conditions.py`**
```python
import numpy as np

def dam_break(grid, h_left: float, h_right: float, 
              dam_position: float, orientation: str = 'x'):
    """
    Classical dam break problem
    
    Parameters:
    -----------
    grid : ShallowWaterSolver2D
        Solver instance
    h_left : float
        Depth on left side of dam
    h_right : float
        Depth on right side of dam
    dam_position : float
        Location of dam
    orientation : 'x' or 'y'
        Direction of dam
    
    Returns:
    --------
    h0, hu0, hv0 : ndarray
        Initial conditions
    """
    h0 = np.zeros((grid.nx, grid.ny))
    
    if orientation == 'x':
        mask = grid.X < dam_position
        h0[mask] = h_left
        h0[~mask] = h_right
    else:
        mask = grid.Y < dam_position
        h0[mask] = h_left
        h0[~mask] = h_right
    
    hu0 = np.zeros_like(h0)
    hv0 = np.zeros_like(h0)
    
    return h0, hu0, hv0

def circular_dam_break(grid, h_inner: float, h_outer: float,
                       center: Tuple[float, float], radius: float):
    """
    Circular dam break (2D radial problem)
    """
    cx, cy = center
    r = np.sqrt((grid.X - cx)**2 + (grid.Y - cy)**2)
    
    h0 = np.where(r < radius, h_inner, h_outer)
    hu0 = np.zeros_like(h0)
    hv0 = np.zeros_like(h0)
    
    return h0, hu0, hv0
```

### Phase 2: Riemann Solvers (Days 4-6)

**`riemann_solvers.f90`**
```fortran
module riemann_solvers
    use shallow_water_types
    implicit none
    
contains

    subroutine rotate_to_normal(U, nx, ny, U_rot)
        ! Rotate state vector to interface-normal coordinate system
        type(conserved_t), intent(in) :: U
        real(dp), intent(in) :: nx, ny
        type(conserved_t), intent(out) :: U_rot
        
        U_rot%h = U%h
        U_rot%hu = U%hu * nx + U%hv * ny     ! Normal momentum
        U_rot%hv = -U%hu * ny + U%hv * nx    ! Tangential momentum
    end subroutine
    
    subroutine rotate_from_normal(U_rot, nx, ny, U)
        ! Inverse rotation
        type(conserved_t), intent(in) :: U_rot
        real(dp), intent(in) :: nx, ny
        type(conserved_t), intent(out) :: U
        
        U%h = U_rot%h
        U%hu = U_rot%hu * nx - U_rot%hv * ny
        U%hv = U_rot%hu * ny + U_rot%hv * nx
    end subroutine
    
    subroutine physical_flux(U, g, F)
        ! Compute physical flux F(U)
        type(conserved_t), intent(in) :: U
        real(dp), intent(in) :: g
        type(conserved_t), intent(out) :: F
        real(dp) :: u, v
        
        if (U%h > 1.0e-10_dp) then
            u = U%hu / U%h
            v = U%hv / U%h
        else
            u = 0.0_dp
            v = 0.0_dp
        end if
        
        F%h = U%hu
        F%hu = U%hu * u + 0.5_dp * g * U%h**2
        F%hv = U%hu * v
    end subroutine
    
    subroutine hll_flux(U_L, U_R, g, F_HLL)
        ! HLL approximate Riemann solver
        type(conserved_t), intent(in) :: U_L, U_R
        real(dp), intent(in) :: g
        type(conserved_t), intent(out) :: F_HLL
        
        real(dp) :: h_L, h_R, u_L, u_R, c_L, c_R
        real(dp) :: S_L, S_R
        type(conserved_t) :: F_L, F_R
        
        ! Extract primitive variables
        h_L = U_L%h
        h_R = U_R%h
        
        if (h_L > 1.0e-10_dp) then
            u_L = U_L%hu / h_L
            c_L = sqrt(g * h_L)
        else
            u_L = 0.0_dp
            c_L = 0.0_dp
        end if
        
        if (h_R > 1.0e-10_dp) then
            u_R = U_R%hu / h_R
            c_R = sqrt(g * h_R)
        else
            u_R = 0.0_dp
            c_R = 0.0_dp
        end if
        
        ! Wave speed estimates
        S_L = min(u_L - c_L, u_R - c_R)
        S_R = max(u_L + c_L, u_R + c_R)
        
        ! Physical fluxes
        call physical_flux(U_L, g, F_L)
        call physical_flux(U_R, g, F_R)
        
        ! HLL flux
        if (S_L >= 0.0_dp) then
            F_HLL = F_L
        else if (S_R <= 0.0_dp) then
            F_HLL = F_R
        else
            ! Intermediate state
            F_HLL%h = (S_R * F_L%h - S_L * F_R%h + S_L * S_R * (U_R%h - U_L%h)) / (S_R - S_L)
            F_HLL%hu = (S_R * F_L%hu - S_L * F_R%hu + S_L * S_R * (U_R%hu - U_L%hu)) / (S_R - S_L)
            F_HLL%hv = (S_R * F_L%hv - S_L * F_R%hv + S_L * S_R * (U_R%hv - U_L%hv)) / (S_R - S_L)
        end if
    end subroutine
    
    subroutine hllc_flux(U_L, U_R, g, F_HLLC)
        ! HLLC Riemann solver (resolves contact)
        type(conserved_t), intent(in) :: U_L, U_R
        real(dp), intent(in) :: g
        type(conserved_t), intent(out) :: F_HLLC
        
        real(dp) :: h_L, h_R, u_L, u_R, v_L, v_R, c_L, c_R
        real(dp) :: S_L, S_R, S_M
        type(conserved_t) :: F_L, F_R, U_L_star, U_R_star
        
        ! Extract primitives
        h_L = U_L%h
        h_R = U_R%h
        
        if (h_L > 1.0e-10_dp) then
            u_L = U_L%hu / h_L
            v_L = U_L%hv / h_L
            c_L = sqrt(g * h_L)
        else
            u_L = 0.0_dp
            v_L = 0.0_dp
            c_L = 0.0_dp
        end if
        
        if (h_R > 1.0e-10_dp) then
            u_R = U_R%hu / h_R
            v_R = U_R%hv / h_R
            c_R = sqrt(g * h_R)
        else
            u_R = 0.0_dp
            v_R = 0.0_dp
            c_R = 0.0_dp
        end if
        
        ! Wave speeds
        S_L = min(u_L - c_L, u_R - c_R)
        S_R = max(u_L + c_L, u_R + c_R)
        
        ! Middle wave speed
        S_M = (S_L * h_R * (u_R - S_R) - S_R * h_L * (u_L - S_L)) / &
              (h_R * (u_R - S_R) - h_L * (u_L - S_L))
        
        ! Star states
        U_L_star%h = h_L * (S_L - u_L) / (S_L - S_M)
        U_L_star%hu = U_L_star%h * S_M
        U_L_star%hv = U_L_star%h * v_L
        
        U_R_star%h = h_R * (S_R - u_R) / (S_R - S_M)
        U_R_star%hu = U_R_star%h * S_M
        U_R_star%hv = U_R_star%h * v_R
        
        ! Physical fluxes
        call physical_flux(U_L, g, F_L)
        call physical_flux(U_R, g, F_R)
        
        ! HLLC flux
        if (S_L >= 0.0_dp) then
            F_HLLC = F_L
        else if (S_M >= 0.0_dp) then
            F_HLLC%h = F_L%h + S_L * (U_L_star%h - U_L%h)
            F_HLLC%hu = F_L%hu + S_L * (U_L_star%hu - U_L%hu)
            F_HLLC%hv = F_L%hv + S_L * (U_L_star%hv - U_L%hv)
        else if (S_R >= 0.0_dp) then
            F_HLLC%h = F_R%h + S_R * (U_R_star%h - U_R%h)
            F_HLLC%hu = F_R%hu + S_R * (U_R_star%hu - U_R%hu)
            F_HLLC%hv = F_R%hv + S_R * (U_R_star%hv - U_R%hv)
        else
            F_HLLC = F_R
        end if
    end subroutine
    
end module
```

### Phase 3: Well-Balanced Scheme (Days 7-9)

**`well_balanced.f90`**
```fortran
module well_balanced
    use shallow_water_types
    implicit none
    
contains

    subroutine surface_gradient_reconstruction(h, b, dx, dy, eta, deta_dx, deta_dy)
        ! Reconstruct surface elevation and gradients
        real(dp), intent(in) :: h(:,:), b(:,:)
        real(dp), intent(in) :: dx, dy
        real(dp), intent(out) :: eta(:,:)
        real(dp), intent(out) :: deta_dx(:,:), deta_dy(:,:)
        integer :: i, j, nx, ny
        
        nx = size(h, 1)
        ny = size(h, 2)
        
        ! Surface elevation
        eta = h + b
        
        ! Gradients (central difference with minmod limiter)
        do j = 1, ny
            do i = 2, nx-1
                deta_dx(i,j) = minmod((eta(i,j) - eta(i-1,j))/dx, &
                                     (eta(i+1,j) - eta(i,j))/dx)
            end do
            deta_dx(1,j) = 0.0_dp
            deta_dx(nx,j) = 0.0_dp
        end do
        
        do i = 1, nx
            do j = 2, ny-1
                deta_dy(i,j) = minmod((eta(i,j) - eta(i,j-1))/dy, &
                                     (eta(i,j+1) - eta(i,j))/dy)
            end do
            deta_dy(i,1) = 0.0_dp
            deta_dy(i,ny) = 0.0_dp
        end do
    end subroutine
    
    subroutine hydrostatic_reconstruct(h_L, h_R, b_L, b_R, h_L_star, h_R_star)
        ! Hydrostatic reconstruction for positivity
        real(dp), intent(in) :: h_L, h_R, b_L, b_R
        real(dp), intent(out) :: h_L_star, h_R_star
        real(dp) :: b_interface
        
        b_interface = max(b_L, b_R)
        
        h_L_star = max(0.0_dp, h_L + b_L - b_interface)
        h_R_star = max(0.0_dp, h_R + b_R - b_interface)
    end subroutine
    
    subroutine compute_source_term_balanced(h, b, dx, dy, g, S_x, S_y)
        ! Well-balanced source term discretization
        real(dp), intent(in) :: h(:,:), b(:,:)
        real(dp), intent(in) :: dx, dy, g
        real(dp), intent(out) :: S_x(:,:), S_y(:,:)
        integer :: i, j, nx, ny
        real(dp) :: b_L, b_R, h_L, h_R, h_avg
        
        nx = size(h, 1)
        ny = size(h, 2)
        
        ! x-direction source
        do j = 1, ny
            do i = 2, nx
                b_L = b(i-1,j)
                b_R = b(i,j)
                h_L = h(i-1,j)
                h_R = h(i,j)
                
                h_avg = 0.5_dp * (h_L + h_R)
                S_x(i,j) = -g * h_avg * (b_R - b_L) / dx
            end do
            S_x(1,j) = 0.0_dp
        end do
        
        ! y-direction source
        do i = 1, nx
            do j = 2, ny
                b_L = b(i,j-1)
                b_R = b(i,j)
                h_L = h(i,j-1)
                h_R = h(i,j)
                
                h_avg = 0.5_dp * (h_L + h_R)
                S_y(i,j) = -g * h_avg * (b_R - b_L) / dy
            end do
            S_y(i,1) = 0.0_dp
        end do
    end subroutine
    
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
    
end module
```

### Phase 4: Time Stepping (Days 10-12)

**`time_stepping.f90`**
```fortran
module time_stepping
    use shallow_water_types
    use riemann_solvers
    use well_balanced
    implicit none
    
contains

    subroutine compute_rhs(state, grid, config, rhs_h, rhs_hu, rhs_hv)
        ! Compute right-hand side: d/dt U = RHS(U)
        type(state_t), intent(in) :: state
        type(grid_t), intent(in) :: grid
        type(config_t), intent(in) :: config
        real(dp), intent(out) :: rhs_h(:,:), rhs_hu(:,:), rhs_hv(:,:)
        
        integer :: i, j
        real(dp), allocatable :: flux_x_h(:,:), flux_x_hu(:,:), flux_x_hv(:,:)
        real(dp), allocatable :: flux_y_h(:,:), flux_y_hu(:,:), flux_y_hv(:,:)
        real(dp), allocatable :: S_x(:,:), S_y(:,:)
        type(conserved_t) :: U_L, U_R, F
        
        allocate(flux_x_h(grid%nx+1, grid%ny))
        allocate(flux_x_hu(grid%nx+1, grid%ny))
        allocate(flux_x_hv(grid%nx+1, grid%ny))
        allocate(flux_y_h(grid%nx, grid%ny+1))
        allocate(flux_y_hu(grid%nx, grid%ny+1))
        allocate(flux_y_hv(grid%nx, grid%ny+1))
        allocate(S_x(grid%nx, grid%ny))
        allocate(S_y(grid%nx, grid%ny))
        
        ! Compute x-direction fluxes
        do j = 1, grid%ny
            do i = 1, grid%nx+1
                if (i == 1) then
                    ! Left boundary (transmissive)
                    U_L%h = state%h(1,j)
                    U_L%hu = state%hu(1,j)
                    U_L%hv = state%hv(1,j)
                    U_R = U_L
                else if (i == grid%nx+1) then
                    ! Right boundary (transmissive)
                    U_L%h = state%h(grid%nx,j)
                    U_L%hu = state%hu(grid%nx,j)
                    U_L%hv = state%hv(grid%nx,j)
                    U_R = U_L
                else
                    ! Interior interface
                    U_L%h = state%h(i-1,j)
                    U_L%hu = state%hu(i-1,j)
                    U_L%hv = state%hv(i-1,j)
                    
                    U_R%h = state%h(i,j)
                    U_R%hu = state%hu(i,j)
                    U_R%hv = state%hv(i,j)
                end if
                
                ! Solve Riemann problem
                if (trim(config%riemann) == 'hll') then
                    call hll_flux(U_L, U_R, config%g, F)
                else
                    call hllc_flux(U_L, U_R, config%g, F)
                end if
                
                flux_x_h(i,j) = F%h
                flux_x_hu(i,j) = F%hu
                flux_x_hv(i,j) = F%hv
            end do
        end do
        
        ! Compute y-direction fluxes (similar)
        ! ... [similar code for y-direction]
        
        ! Compute source terms
        if (config%well_balanced) then
            call compute_source_term_balanced(state%h, grid%b, &
                grid%dx, grid%dy, config%g, S_x, S_y)
        else
            S_x = 0.0_dp
            S_y = 0.0_dp
        end if
        
        ! Assemble RHS
        do j = 1, grid%ny
            do i = 1, grid%nx
                rhs_h(i,j) = -(flux_x_h(i+1,j) - flux_x_h(i,j))/grid%dx &
                            -(flux_y_h(i,j+1) - flux_y_h(i,j))/grid%dy
                
                rhs_hu(i,j) = -(flux_x_hu(i+1,j) - flux_x_hu(i,j))/grid%dx &
                             -(flux_y_hu(i,j+1) - flux_y_hu(i,j))/grid%dy &
                             + S_x(i,j)
                
                rhs_hv(i,j) = -(flux_x_hv(i+1,j) - flux_x_hv(i,j))/grid%dx &
                             -(flux_y_hv(i,j+1) - flux_y_hv(i,j))/grid%dy &
                             + S_y(i,j)
            end do
        end do
    end subroutine
    
    subroutine ssp_rk3_step(state, grid, config, dt)
        ! SSP-RK3 time step
        type(state_t), intent(inout) :: state
        type(grid_t), intent(in) :: grid
        type(config_t), intent(in) :: config
        real(dp), intent(in) :: dt
        
        type(state_t) :: state1, state2
        real(dp), allocatable :: rhs_h(:,:), rhs_hu(:,:), rhs_hv(:,:)
        
        allocate(state1%h(grid%nx, grid%ny))
        allocate(state1%hu(grid%nx, grid%ny))
        allocate(state1%hv(grid%nx, grid%ny))
        allocate(state2%h(grid%nx, grid%ny))
        allocate(state2%hu(grid%nx, grid%ny))
        allocate(state2%hv(grid%nx, grid%ny))
        allocate(rhs_h(grid%nx, grid%ny))
        allocate(rhs_hu(grid%nx, grid%ny))
        allocate(rhs_hv(grid%nx, grid%ny))
        
        ! Stage 1
        call compute_rhs(state, grid, config, rhs_h, rhs_hu, rhs_hv)
        state1%h = state%h + dt * rhs_h
        state1%hu = state%hu + dt * rhs_hu
        state1%hv = state%hv + dt * rhs_hv
        
        ! Stage 2
        call compute_rhs(state1, grid, config, rhs_h, rhs_hu, rhs_hv)
        state2%h = 0.75_dp * state%h + 0.25_dp * state1%h + 0.25_dp * dt * rhs_h
        state2%hu = 0.75_dp * state%hu + 0.25_dp * state1%hu + 0.25_dp * dt * rhs_hu
        state2%hv = 0.75_dp * state%hv + 0.25_dp * state1%hv + 0.25_dp * dt * rhs_hv
        
        ! Stage 3
        call compute_rhs(state2, grid, config, rhs_h, rhs_hu, rhs_hv)
        state%h = (1.0_dp/3.0_dp) * state%h + (2.0_dp/3.0_dp) * state2%h + &
                 (2.0_dp/3.0_dp) * dt * rhs_h
        state%hu = (1.0_dp/3.0_dp) * state%hu + (2.0_dp/3.0_dp) * state2%hu + &
                  (2.0_dp/3.0_dp) * dt * rhs_hu
        state%hv = (1.0_dp/3.0_dp) * state%hv + (2.0_dp/3.0_dp) * state2%hv + &
                  (2.0_dp/3.0_dp) * dt * rhs_hv
    end subroutine
    
end module
```

### Phase 5: Validation Suite (Days 13-16)

#### Test 1: Lake at Rest
```python
def test_lake_at_rest():
    """Well-balanced test: should preserve u=v=0 to machine precision"""
    solver = ShallowWaterSolver2D(nx=50, ny=50, 
                                  domain=((0, 100), (0, 100)))
    
    # Gaussian bump bathymetry
    b = 0.8 * np.exp(-((solver.X - 50)**2 + (solver.Y - 50)**2) / 10**2)
    solver.set_bathymetry(b)
    
    # Still water
    eta0 = 1.0  # Constant surface
    h0 = np.maximum(0.0, eta0 - b)
    hu0 = np.zeros_like(h0)
    hv0 = np.zeros_like(h0)
    
    result = solver.solve(h0, hu0, hv0, t_final=10.0)
    
    # Check velocity remains zero
    u_max = np.max(np.abs(result['hu'][-1] / (result['h'][-1] + 1e-10)))
    v_max = np.max(np.abs(result['hv'][-1] / (result['h'][-1] + 1e-10)))
    
    assert u_max < 1e-12
    assert v_max < 1e-12
    print("✓ Lake at rest preserved to machine precision")
```

#### Test 2: 1D Dam Break (Ritter)
```python
def test_dam_break_1d():
    """Compare to exact Ritter solution"""
    from validation import ritter_solution
    
    solver = ShallowWaterSolver2D(nx=200, ny=1, 
                                  domain=((0, 200), (0, 1)))
    
    h_L, h_R = 10.0, 1.0
    h0, hu0, hv0 = dam_break(solver, h_L, h_R, 100.0, 'x')
    
    t_final = 5.0
    result = solver.solve(h0, hu0, hv0, t_final)
    
    # Exact solution
    h_exact, u_exact = ritter_solution(solver.x, t_final, h_L, h_R, 100.0)
    
    # Compare (away from shock/rarefaction)
    mask = (solver.x > 50) & (solver.x < 150)
    L1_error = np.sum(np.abs(result['h'][-1][:,0][mask] - h_exact[mask])) * solver.dx
    
    assert L1_error < 0.5
    print(f"✓ Dam break L¹ error: {L1_error:.4f}")
```

#### Test 3: Circular Dam Break
```python
def test_circular_dam_break():
    """2D radial symmetry preservation"""
    solver = ShallowWaterSolver2D(nx=100, ny=100,
                                  domain=((0, 100), (0, 100)))
    
    h0, hu0, hv0 = circular_dam_break(solver, h_inner=10.0, h_outer=1.0,
                                      center=(50, 50), radius=20)
    
    result = solver.solve(h0, hu0, hv0, t_final=5.0)
    
    # Check radial symmetry
    r = np.sqrt((solver.X - 50)**2 + (solver.Y - 50)**2)
    h_final = result['h'][-1]
    
    # Bin by radius and check variance
    r_bins = np.linspace(0, 50, 20)
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.sum(mask) > 10:
            h_std = np.std(h_final[mask])
            assert h_std < 0.1  # Small variation at fixed radius
    
    print("✓ Radial symmetry preserved")
```

#### Test 4: Thacker's Oscillating Basin
```python
def test_thacker_basin():
    """Exact oscillating solution"""
    from validation import thacker_solution
    
    solver = ShallowWaterSolver2D(nx=50, ny=50,
                                  domain=((-50, 50), (-50, 50)))
    
    # Parabolic bathymetry
    L = 40.0
    h0_param = 1.0
    b = h0_param * (1 - (solver.X**2 + solver.Y**2) / L**2)
    solver.set_bathymetry(b)
    
    # Exact initial condition
    omega = 2.0  # Angular frequency
    h0, hu0, hv0 = thacker_solution(solver, omega, t=0.0)
    
    # Evolve one period
    T = 2 * np.pi / omega
    result = solver.solve(h0, hu0, hv0, t_final=T)
    
    # Should return to IC
    h_exact, hu_exact, hv_exact = thacker_solution(solver, omega, t=T)
    
    L2_error = np.sqrt(np.sum((result['h'][-1] - h_exact)**2) * solver.dx * solver.dy)
    
    assert L2_error < 0.01
    print(f"✓ Thacker basin L² error: {L2_error:.6e}")
```

### Phase 6: Visualization (Days 17-19)

**`python/visualization.py`**
```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import numpy as np

def plot_2d_solution(solver, h, title='Depth', save_path=None):
    """Plot 2D depth field"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.contourf(solver.X, solver.Y, h, levels=20, cmap='viridis')
    plt.colorbar(im, ax=ax, label='h [m]')
    
    ax.set_xlabel('x [m]', fontsize=14)
    ax.set_ylabel('y [m]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_velocity_field(solver, h, hu, hv, title='Velocity Field'):
    """Plot velocity vectors"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    u = np.where(h > 1e-3, hu / h, 0.0)
    v = np.where(h > 1e-3, hv / h, 0.0)
    
    # Depth as background
    im = ax.contourf(solver.X, solver.Y, h, levels=20, cmap='viridis', alpha=0.7)
    
    # Velocity vectors (subsample for clarity)
    step = max(1, solver.nx // 20)
    ax.quiver(solver.X[::step, ::step], solver.Y[::step, ::step],
             u[::step, ::step], v[::step, ::step],
             color='white', alpha=0.8)
    
    plt.colorbar(im, ax=ax, label='Depth [m]')
    ax.set_xlabel('x [m]', fontsize=14)
    ax.set_ylabel('y [m]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')
    
    return fig, ax

def animate_solution(solver, times, h_history, save_path=None):
    """Create animation of depth evolution"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmin, vmax = h_history.min(), h_history.max()
    
    im = ax.contourf(solver.X, solver.Y, h_history[0], 
                     levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Depth [m]')
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                       fontsize=14, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    ax.set_xlabel('x [m]', fontsize=14)
    ax.set_ylabel('y [m]', fontsize=14)
    ax.set_aspect('equal')
    
    def animate(frame):
        ax.clear()
        im = ax.contourf(solver.X, solver.Y, h_history[frame],
                        levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        time_text.set_text(f't = {times[frame]:.2f} s')
        ax.set_xlabel('x [m]', fontsize=14)
        ax.set_ylabel('y [m]', fontsize=14)
        ax.set_aspect('equal')
        return im, time_text
    
    anim = animation.FuncAnimation(fig, animate, frames=len(times),
                                  interval=50, blit=False)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
    
    return anim

def plot_cross_section(solver, h, y_slice, title='Cross Section'):
    """Plot 1D slice through 2D domain"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    j = np.argmin(np.abs(solver.y - y_slice))
    
    ax.plot(solver.x, h[:, j], 'b-', linewidth=2)
    ax.fill_between(solver.x, 0, h[:, j], alpha=0.3)
    
    if solver.bathymetry is not None:
        ax.fill_between(solver.x, 0, solver.bathymetry[:, j], 
                       color='brown', alpha=0.5, label='Bottom')
    
    ax.set_xlabel('x [m]', fontsize=14)
    ax.set_ylabel('h [m]', fontsize=14)
    ax.set_title(f'{title} (y = {y_slice:.1f})', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    return fig, ax
```

### Phase 7: Web Visualization (Days 20-21)

**Interactive 2D visualization using Three.js**:

```javascript
// web/shallow_water_viz.js
class ShallowWaterViz {
    constructor(containerId, data) {
        this.container = document.getElementById(containerId);
        this.data = data;  // {times, h, hu, hv, X, Y, bathymetry}
        
        this.setupScene();
        this.createMesh();
        this.setupControls();
        this.animate();
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75, this.container.offsetWidth / this.container.offsetHeight, 0.1, 1000
        );
        this.renderer = new THREE.WebGLRenderer({antialias: true});
        this.renderer.setSize(this.container.offsetWidth, this.container.offsetHeight);
        this.container.appendChild(this.renderer.domElement);
        
        this.camera.position.set(50, 50, 100);
        this.camera.lookAt(0, 0, 0);
        
        // Lighting
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(50, 100, 50);
        this.scene.add(light);
    }
    
    createMesh() {
        const nx = this.data.X.length;
        const ny = this.data.X[0].length;
        
        const geometry = new THREE.PlaneGeometry(nx, ny, nx-1, ny-1);
        const material = new THREE.MeshPhongMaterial({
            color: 0x00aaff,
            side: THREE.DoubleSide,
            wireframe: false
        });
        
        this.waterMesh = new THREE.Mesh(geometry, material);
        this.scene.add(this.waterMesh);
        
        // Update vertices for initial state
        this.updateMesh(0);
    }
    
    updateMesh(timeIndex) {
        const h = this.data.h[timeIndex];
        const vertices = this.waterMesh.geometry.attributes.position.array;
        
        for (let i = 0; i < h.length; i++) {
            for (let j = 0; j < h[0].length; j++) {
                const index = i * h[0].length + j;
                vertices[index * 3 + 2] = h[i][j];  // Set z to depth
            }
        }
        
        this.waterMesh.geometry.attributes.position.needsUpdate = true;
        this.waterMesh.geometry.computeVertexNormals();
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }
}
```

---

## Expected Results

### Well-Balanced Property

| Test | Standard FVM | Well-Balanced |
|------|--------------|---------------|
| Lake at rest | Spurious currents $u \sim O(1)$ | $u < 10^{-12}$ |
| Smooth bump | False drainage | Exact preservation |

### Convergence Rates

| Region | HLL | HLLC |
|--------|-----|------|
| Smooth | $O(\Delta x^2)$ | $O(\Delta x^2)$ |
| Shock | $O(\Delta x)$ | $O(\Delta x)$ |
| Contact | $O(\Delta x^{1/2})$ | $O(\Delta x)$ |

### Performance

| Grid | Time steps | Wall time | Speedup vs Python |
|------|------------|-----------|-------------------|
| 50×50 | ~1,000 | 1 s | ~100× |
| 100×100 | ~4,000 | 10 s | ~150× |
| 200×200 | ~16,000 | 2 min | ~200× |

---

## Deliverables

### Code
- [ ] Fortran solver (8 modules, ~1500 lines)
- [ ] Python interface (~600 lines)
- [ ] Test suite (4 benchmarks)
- [ ] Visualization tools

### Documentation
- [ ] Mathematical derivation
- [ ] Implementation guide
- [ ] User manual
- [ ] API reference

### Visualizations
- [ ] 4 benchmark animations
- [ ] Interactive 3D demo
- [ ] Comparison plots (HLL vs HLLC)

### Blog Post
- [ ] Project overview
- [ ] Well-balanced schemes explained
- [ ] Validation results
- [ ] Connection to 1D Burgers

---

## Learning Objectives

By completing this project, you will master:

- [x] **Hyperbolic systems** (eigenvalues, characteristics)
- [x] **HLL/HLLC Riemann solvers**
- [x] **Well-balanced schemes** (critical for geophysical flows)
- [x] **Source term discretization**
- [x] **Wet/dry treatment** (hydrostatic reconstruction)
- [x] **2D finite volume assembly**
- [x] **SSP time stepping**
- [x] **Conservation properties**
- [x] **Production-quality FVM**

---

## Extensions

### Immediate
- [ ] Friction terms (Manning's law)
- [ ] Coriolis force
- [ ] Rainfall/inflow sources
- [ ] Non-reflecting boundary conditions

### Advanced
- [ ] WENO reconstruction (5th order)
- [ ] Adaptive mesh refinement
- [ ] Sediment transport
- [ ] Real bathymetry data

### Production
- [ ] MPI parallelization
- [ ] GPU acceleration
- [ ] NetCDF I/O
- [ ] NOAA benchmark validation

---

## Success Criteria

### Must Have
- [x] All 4 benchmarks pass
- [x] Lake at rest to machine precision
- [x] Positive depth always
- [x] Mass conservation exact
- [x] No spurious oscillations

### Should Have
- [ ] Interactive 3D visualization
- [ ] Complete documentation
- [ ] Blog post published
- [ ] Web demo deployed

### Nice to Have
- [ ] AMR implementation
- [ ] Real tsunami simulation
- [ ] Video presentation
- [ ] GitHub release

---

## Timeline

**Week 1**: Infrastructure + Riemann solvers
**Week 2**: Well-balanced schemes + time stepping
**Week 3**: Validation + visualization
**Week 4**: Polish + documentation + extensions

---

## Resources

### Textbooks
- Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows* (2001)
- LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Ch. 13
- Guinot, *Wave Propagation in Fluids* (2nd ed.)

### Papers
- Audusse et al. (2004) - Hydrostatic reconstruction
- Kurganov & Petrova (2007) - Central-upwind schemes
- Toro (1999) - HLLC solver

### Software
- GeoClaw (production shallow water solver)
- ANUGA (tsunami modeling)

This project demonstrates **production-quality FVM** with all the essential techniques for real geophysical modeling. It builds directly on 1D Burgers while introducing system solvers and well-balanced schemes—the foundations of modern CFD.