# N-Body Gravitational Simulation: Theory & Methods

**Adam Field - Computational Physics ISP**  
**Worcester Polytechnic Institute**

---

## Table of Contents

1. [The N-Body Problem](#the-n-body-problem)
2. [Gravitational Physics](#gravitational-physics)
3. [Numerical Integration Methods](#numerical-integration-methods)
4. [Computational Complexity](#computational-complexity)
5. [Numerical Stability & Softening](#numerical-stability--softening)
6. [Energy Conservation](#energy-conservation)
7. [Implementation Considerations](#implementation-considerations)

---

## The N-Body Problem

The **gravitational N-body problem** is one of the most fundamental problems in computational physics and astrophysics. It involves computing the trajectories of N particles (stars, planets, galaxies, etc.) under their mutual gravitational attraction.

### Historical Context

- **Two-body problem**: Solved analytically by Newton (1687) - bodies orbit their center of mass in conic sections
- **Three-body problem**: No general closed-form solution exists (Poincaré, 1890s)
- **N-body problem (N ≥ 3)**: Must be solved numerically for most initial conditions

### Applications

- **Astrophysics**: Galaxy formation, stellar dynamics, planetary systems
- **Cosmology**: Dark matter simulations, large-scale structure formation
- **Space missions**: Trajectory planning (e.g., Lagrange points, gravitational slingshots)
- **Molecular dynamics**: Similar equations with different force laws

---

## Gravitational Physics

### Newton's Law of Universal Gravitation

The force between two point masses $m_i$ and $m_j$ separated by distance $r_{ij}$ is:

$$
\vec{F}_{ij} = -G \frac{m_i m_j}{r_{ij}^2} \hat{r}_{ij}
$$

where:
- $G$ = gravitational constant (6.674 × 10⁻¹¹ m³ kg⁻¹ s⁻²)
- $\hat{r}_{ij}$ = unit vector from particle $i$ to particle $j$
- The negative sign indicates attraction

### Force on Particle i

For N particles, the total gravitational force on particle $i$ is:

$$
\vec{F}_i = G m_i \sum_{j \neq i} \frac{m_j (\vec{r}_j - \vec{r}_i)}{|\vec{r}_j - \vec{r}_i|^3}
$$

Note: We can factor out $m_i$ and work with acceleration directly:

$$
\vec{a}_i = \frac{\vec{F}_i}{m_i} = G \sum_{j \neq i} \frac{m_j (\vec{r}_j - \vec{r}_i)}{|\vec{r}_j - \vec{r}_i|^3}
$$

### Equations of Motion

The system is governed by Newton's second law:

$$
\frac{d\vec{v}_i}{dt} = \vec{a}_i(\vec{r}_1, \vec{r}_2, \ldots, \vec{r}_N)
$$

$$
\frac{d\vec{r}_i}{dt} = \vec{v}_i
$$

This is a system of **6N coupled ordinary differential equations** (3N positions + 3N velocities).

---

## Numerical Integration Methods

Since the N-body problem has no general analytical solution, we must integrate the equations of motion numerically.

### Time-Stepping Schemes

#### 1. Euler Method (First-Order)

The simplest approach:

$$
\vec{r}_i(t + \Delta t) = \vec{r}_i(t) + \vec{v}_i(t) \Delta t
$$

$$
\vec{v}_i(t + \Delta t) = \vec{v}_i(t) + \vec{a}_i(t) \Delta t
$$

**Pros**: Simple to implement  
**Cons**: 
- Only first-order accurate: $O(\Delta t)$ local error, $O(\Delta t^0)$ global error
- **Not symplectic**: Energy drifts systematically over time
- Unstable for oscillatory systems

**Verdict**: ❌ Not suitable for N-body simulations

---

#### 2. Leapfrog Method (Second-Order)

Alternates position and velocity updates:

$$
\vec{r}_i(t + \Delta t) = \vec{r}_i(t) + \vec{v}_i(t + \Delta t/2) \Delta t
$$

$$
\vec{v}_i(t + \Delta t/2) = \vec{v}_i(t - \Delta t/2) + \vec{a}_i(t) \Delta t
$$

**Pros**: Second-order accurate, symplectic  
**Cons**: Needs special handling for initial conditions

---

#### 3. Velocity Verlet (Second-Order) ✅

This is the method we use in all implementations:

$$
\vec{r}_i(t + \Delta t) = \vec{r}_i(t) + \vec{v}_i(t) \Delta t + \frac{1}{2} \vec{a}_i(t) \Delta t^2
$$

$$
\vec{a}_i(t + \Delta t) = \text{compute\_acceleration}(\vec{r}(t + \Delta t))
$$

$$
\vec{v}_i(t + \Delta t) = \vec{v}_i(t) + \frac{1}{2} [\vec{a}_i(t) + \vec{a}_i(t + \Delta t)] \Delta t
$$

**Pros**:
- **Second-order accurate**: $O(\Delta t^2)$ local error, $O(\Delta t^2)$ global error
- **Symplectic**: Conserves energy to high precision over long times
- **Time-reversible**: Running backwards recovers initial conditions
- **Self-starting**: Doesn't require special initialization

**Cons**:
- Requires two force evaluations per step (but second evaluation is needed for next step)

**Why Velocity Verlet?**

Velocity Verlet is a **symplectic integrator**, meaning it preserves the structure of Hamiltonian systems. For conservative systems like gravity:

- Energy is conserved to within roundoff error (no systematic drift)
- Phase space volume is preserved (Liouville's theorem)
- Long-term stability for orbital dynamics

This is crucial for astrophysical simulations where we need to track systems for millions of years.

---

#### 4. Runge-Kutta Methods (RK4)

Fourth-order Runge-Kutta:

$$
\vec{r}(t + \Delta t) = \vec{r}(t) + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

where $k_1, k_2, k_3, k_4$ are intermediate evaluations.

**Pros**: Fourth-order accurate $O(\Delta t^4)$  
**Cons**: 
- Not symplectic (energy drifts)
- 4× more force evaluations per step
- Not ideal for Hamiltonian systems

**Verdict**: Better for dissipative systems, not gravitational N-body

---

### Timestep Selection

The timestep $\Delta t$ must be chosen carefully:

**Too large**: 
- Integration becomes inaccurate
- Energy drifts significantly
- Close encounters cause instability

**Too small**:
- Computation becomes unnecessarily slow
- Roundoff error accumulates

**Rule of thumb**: 

$$
\Delta t \sim 0.01 \times T_{\text{orbital}}
$$

where $T_{\text{orbital}}$ is the typical orbital period of the system.

For our simulations with $G=1$, $R \sim 10$, and $M \sim 1$:

$$
T \sim 2\pi \sqrt{\frac{R^3}{GM}} \sim 2\pi \sqrt{\frac{10^3}{1}} \sim 200
$$

So $\Delta t \sim 2$ is reasonable, but we use $\Delta t = 0.01$ to be conservative.

---

## Computational Complexity

### Direct Summation (Brute Force)

Our implementation computes all pairwise forces:

```python
for i in range(N):
    for j in range(N):
        if i != j:
            force[i] += compute_force(i, j)
```

**Complexity**: $O(N^2)$ per timestep

For $N$ particles over $T$ timesteps:
- **Total force evaluations**: $N(N-1) \times T \approx N^2 T$
- **Memory**: $O(N)$ (only current state needed)

### Scaling Analysis

| N     | Force Evaluations | Time per Step (JAX GPU) |
|-------|-------------------|-------------------------|
| 10    | 90                | 0.085 ms                |
| 50    | 2,450             | 0.096 ms                |
| 100   | 9,900             | 0.125 ms                |
| 500   | 249,500           | 0.088 ms                |
| 1000  | 999,000           | 0.080 ms                |

Notice: GPU performance stays roughly constant! This is because:
1. GPU has massive parallelism (thousands of cores)
2. For small N, GPU is underutilized
3. For large N, all cores are busy but computation is still fast

CPU implementations show clear $O(N^2)$ scaling:

| N     | Fortran OpenMP | C     | Python |
|-------|----------------|-------|--------|
| 10    | 0.015 ms       | 0.0006 ms | 0.035 ms |
| 100   | 0.034 ms       | 0.085 ms | 1.26 ms |
| 1000  | 1.37 ms        | 5.5 ms | 110.8 ms |

Ratio (1000/10): ~90× slower, which is roughly $(1000/10)^2 = 10000/100 = 100$ as expected for $O(N^2)$.

---

### Tree Methods (Advanced)

For very large N (> 10,000), direct summation becomes prohibitively slow. Hierarchical tree methods can reduce complexity:

#### Barnes-Hut Tree (1986)

**Idea**: Group distant particles into "super-particles"

**Algorithm**:
1. Build octree (3D) or quadtree (2D) of particle positions
2. For each particle, traverse tree:
   - If node is far enough, use node's center of mass
   - Otherwise, recurse to children
3. Threshold: $\theta = s/d$ (node size / distance)

**Complexity**: $O(N \log N)$ per timestep

**Accuracy**: Controlled by $\theta$ parameter
- $\theta \to 0$: Exact (but slow)
- $\theta \sim 0.5$: Good balance (~1% error)

#### Fast Multipole Method (FMM)

**Complexity**: $O(N)$ per timestep  
**Used in**: Million-particle galaxy simulations

We do not implement these in this project, but they're important for production astrophysical codes.

---

## Numerical Stability & Softening

### The Singularity Problem

Newton's law has a singularity at $r = 0$:

$$
F \propto \frac{1}{r^2} \to \infty \text{ as } r \to 0
$$

In practice, this causes problems when:
1. Two particles pass very close to each other
2. Numerical roundoff causes $r \approx 0$

### Softening Parameter

We modify the force law to avoid the singularity:

$$
\vec{F}_{ij} = -G \frac{m_i m_j (\vec{r}_j - \vec{r}_i)}{(|\vec{r}_j - \vec{r}_i|^2 + \epsilon^2)^{3/2}}
$$

where $\epsilon$ is the **softening length**.

**Effect**:
- For $r \gg \epsilon$: Force is unaffected
- For $r \lesssim \epsilon$: Force is softened (reduced)
- At $r = 0$: Force is finite

**Physical interpretation**: Treats particles as having a finite size $\sim \epsilon$ instead of being point masses.

**Choosing $\epsilon$**:
- Too large: Dynamics are unphysical
- Too small: Doesn't prevent instability
- **Rule of thumb**: $\epsilon \sim 0.01 \times R_{\text{system}}$

In our simulations: $\epsilon = 0.1$ with $R \sim 10$ is reasonable.

---

## Energy Conservation

### Conserved Quantities

For an isolated N-body system, the following are conserved:

1. **Total Energy**: $E = KE + PE$
2. **Total Momentum**: $\vec{P} = \sum_i m_i \vec{v}_i$
3. **Total Angular Momentum**: $\vec{L} = \sum_i \vec{r}_i \times m_i \vec{v}_i$

### Energy Components

**Kinetic Energy**:

$$
KE = \sum_{i=1}^N \frac{1}{2} m_i v_i^2
$$

**Potential Energy**:

$$
PE = -G \sum_{i=1}^N \sum_{j>i} \frac{m_i m_j}{\sqrt{r_{ij}^2 + \epsilon^2}}
$$

Note: Sum only over $j > i$ to avoid double-counting pairs.

**Total Energy**:

$$
E = KE + PE
$$

### Energy Drift

Even with symplectic integrators, energy is not **exactly** conserved due to:
1. Finite timestep ($\Delta t > 0$)
2. Floating-point roundoff errors

**Expected drift** for Velocity Verlet:

$$
\Delta E \sim O(\Delta t^2) \times T_{\text{total}}
$$

For our simulations:
- $\Delta t = 0.01$
- $T_{\text{total}} = 10$ (1000 steps)
- Expected: $\Delta E / E \sim 10^{-4}$ to $10^{-6}$

**Observed** (from benchmarks):
- N=100: Energy drift ~ 0.022% ✅
- N=1000: Energy drift ~ 3-4% (larger due to chaotic dynamics)

The larger drift at N=1000 is expected for **chaotic systems** where small perturbations grow exponentially.

### Monitoring Energy

Energy drift is the best diagnostic for numerical accuracy:

```python
E_drift = abs(E_final - E_initial) / abs(E_initial) * 100
```

**Acceptable thresholds**:
- < 0.1%: Excellent
- 0.1% - 1%: Good
- 1% - 10%: Acceptable for chaotic systems
- > 10%: Check timestep and integrator

---

## Implementation Considerations

### Vectorization

**Naive approach** (nested loops):
```python
for i in range(N):
    for j in range(N):
        if i != j:
            dx = x[j] - x[i]
            dist = sqrt(dx**2 + dy**2 + dz**2)
            force[i] += G * m[j] * dx / dist**3
```

**Vectorized approach** (NumPy/JAX):
```python
# Compute all pairwise displacements at once
displacements = positions[None, :, :] - positions[:, None, :]  # (N, N, 3)
distances = sqrt(sum(displacements**2, axis=2) + epsilon**2)   # (N, N)
forces = G * displacements / distances[:,:,None]**3             # (N, N, 3)
accelerations = sum(forces * masses[None, :, None], axis=1)    # (N, 3)
```

**Speedup**: ~10-100× on CPU, ~1000× on GPU!

### GPU Acceleration

Modern GPUs excel at N-body simulations because:

1. **Massive parallelism**: 1000s of cores compute forces simultaneously
2. **High memory bandwidth**: Can load all particle data quickly
3. **SIMD operations**: Same operation on many particles

**When GPU wins**:
- Large N (> 500): More work to parallelize
- Many timesteps: Amortizes compilation overhead
- Same simulation repeated: JIT compilation cost paid once

**When CPU might be competitive**:
- Small N (< 100): GPU underutilized
- Few timesteps: JIT overhead dominates
- One-off calculations: Compilation not worth it

### Memory Considerations

**Storage requirements**:
- Full trajectory: $(N \times 3 \times T) \times 8$ bytes (double precision)
- Example: N=1000, T=10000 → 240 MB

**Strategies**:
1. Save every $k$-th frame (`save_every` parameter)
2. Only save final state
3. Stream to disk during simulation
4. Use single precision (4 bytes) if acceptable

---

## Summary

The N-body problem demonstrates the interplay between:

- **Physics**: Newton's laws, energy conservation, chaos
- **Mathematics**: Symplectic geometry, numerical analysis
- **Computer Science**: Algorithms, parallelization, GPU computing

Our implementation using **Velocity Verlet** provides:
- ✅ Second-order accuracy
- ✅ Energy conservation
- ✅ Stability for long simulations
- ✅ Efficiency across multiple platforms

The **direct summation** approach is:
- ✅ Exact (within floating-point precision)
- ✅ Simple to implement
- ❌ $O(N^2)$ scaling (slow for large N)

For production astrophysical simulations with $N > 10^6$, tree methods or FMM are necessary, but our direct approach is perfect for learning and moderate-scale problems.

---

## References

1. **Press, W. H., et al.** *Numerical Recipes* (Cambridge, 2007)
   - Chapter 17: Integration of ODEs
   
2. **Barnes, J. & Hut, P.** "A hierarchical O(N log N) force-calculation algorithm" *Nature* **324**, 446-449 (1986)

3. **Aarseth, S.J.** *Gravitational N-Body Simulations* (Cambridge, 2003)
   - The definitive textbook on N-body methods

4. **Hairer, E., Lubich, C., Wanner, G.** *Geometric Numerical Integration* (Springer, 2006)
   - Theory of symplectic integrators

5. **Springel, V.** "The cosmological simulation code GADGET-2" *MNRAS* **364**, 1105-1134 (2005)
   - Modern production N-body code

6. JAX Documentation: https://jax.readthedocs.io/
7. NumPy/SciPy Documentation: https://numpy.org/doc/

---

*This document is part of Adam Field's Computational Physics Independent Study Project at Worcester Polytechnic Institute (2026).*