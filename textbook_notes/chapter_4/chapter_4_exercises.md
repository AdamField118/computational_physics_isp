---
title: "Chapter 4 Exercises: Transformation Theory and Implementation"
date: "2026-01-30"
tags: "Exercises"
snippet: "Essential exercises on affine transformations, reference elements, mesh quality, and numerical conditioning for finite element implementation."
---

# Chapter 4: Finite Element Spaces - Exercises

## Exercise 4.x.11: Reference Triangle Interpolation

**Problem**: Let $K$ be the reference triangle with vertices $(0,0), (1,0)$ and $(0,1)$, and $\zeta \in W_p^m(K)$ for $m \geq 2$ and $1 \leq p \leq \infty$. Show that:

$$\left\|\frac{\partial}{\partial x_i}(\zeta - I\zeta)\right\|_{L^p(K)} \leq C\left|\frac{\partial K}{\partial x_i}\right|_{W_p^m(K)}$$

where $I$ is the Lagrange finite element interpolant.

### Solution

This exercise establishes the fundamental interpolation error estimate on the reference element.

**Part 1: Setup**

For $P_1$ elements, the interpolant $I\zeta$ matches $\zeta$ at the three vertices:
- $v_1 = (0,0)$
- $v_2 = (1,0)$  
- $v_3 = (0,1)$

The interpolant is:
$$I\zeta(x,y) = \zeta(0,0)(1-x-y) + \zeta(1,0)x + \zeta(0,1)y$$

**Part 2: Key Observation**

The error $e = \zeta - I\zeta$ satisfies:
- $e(v_i) = 0$ for $i = 1,2,3$ (vanishes at vertices)
- $e \in W_p^m(K)$ (same regularity as $\zeta$)

**Part 3: Bramble-Hilbert Lemma Application**

Since $I$ reproduces linear functions exactly, for any linear function $\ell$:
$$I\ell = \ell$$

Therefore:
$$\zeta - I\zeta = (\zeta - \ell) - I(\zeta - \ell)$$

By the Bramble-Hilbert lemma, for $m \geq 2$:

$$\|e\|_{W_1^p(K)} \leq C |e|_{W_m^p(K)} = C |\zeta|_{W_m^p(K)}$$

since $I$ reproduces polynomials up to degree $1$.

**Part 4: Derivative Estimate**

For the derivative in the $x_i$ direction:

$$\left\|\frac{\partial e}{\partial x_i}\right\|_{L^p(K)} = \left\|\frac{\partial}{\partial x_i}(\zeta - I\zeta)\right\|_{L^p(K)}$$

Since $I\zeta$ is linear, $\frac{\partial I\zeta}{\partial x_i}$ is constant on $K$.

By Poincaré-type inequalities for functions vanishing at vertices:

$$\left\|\frac{\partial e}{\partial x_i}\right\|_{L^p(K)} \leq C \sum_{|\alpha|=m} \left\|\frac{\partial^\alpha \zeta}{\partial x^\alpha}\right\|_{L^p(K)}$$

**Part 5: Explicit Computation for $P_1$**

For the reference triangle, the gradients of basis functions are:

$$\nabla \phi_1 = \begin{pmatrix} -1 \\ -1 \end{pmatrix}, \quad \nabla \phi_2 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \nabla \phi_3 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

The interpolation error derivative:

$$\frac{\partial e}{\partial x} = \frac{\partial \zeta}{\partial x} - \left[-\zeta(0,0) + \zeta(1,0)\right]$$

$$\frac{\partial e}{\partial y} = \frac{\partial \zeta}{\partial y} - \left[-\zeta(0,0) + \zeta(0,1)\right]$$

**Part 6: Reference Stiffness Matrix**

The reference stiffness matrix for $P_1$ elements is:

$$\widehat{K} = \frac{1}{2}\begin{pmatrix}
\nabla\phi_1 \cdot \nabla\phi_1 & \nabla\phi_1 \cdot \nabla\phi_2 & \nabla\phi_1 \cdot \nabla\phi_3 \\
\nabla\phi_2 \cdot \nabla\phi_1 & \nabla\phi_2 \cdot \nabla\phi_2 & \nabla\phi_2 \cdot \nabla\phi_3 \\
\nabla\phi_3 \cdot \nabla\phi_1 & \nabla\phi_3 \cdot \nabla\phi_2 & \nabla\phi_3 \cdot \nabla\phi_3
\end{pmatrix}$$

Computing:

$$\widehat{K} = \frac{1}{2}\begin{pmatrix}
2 & -1 & -1 \\
-1 & 1 & 0 \\
-1 & 0 & 1
\end{pmatrix}$$

**Verification**:
- $\nabla\phi_1 \cdot \nabla\phi_1 = (-1)^2 + (-1)^2 = 2$
- $\nabla\phi_1 \cdot \nabla\phi_2 = (-1)(1) + (-1)(0) = -1$
- $\nabla\phi_2 \cdot \nabla\phi_2 = 1^2 + 0^2 = 1$

This is the **fundamental building block** for all 2D FEM assembly!

### Interactive Visualization: Reference Triangle Computation

[codeContainer](./textbook_notes/chapter_4/ex_11_reference_triangle.js)

## Exercise 4.x.10: Minimum Angle Condition

**Problem**: Let $\{T^h\}$ be a family of triangulations of $\Omega \subset \mathbb{R}^2$. Show that $\{T^h\}$ is non-degenerate if and only if all the angles of the triangles in $\{T^h\}$ are bounded below by a positive constant.

### Solution

This establishes the geometric condition for shape regularity.

**Part 1: Forward Direction** (Non-degenerate ⇒ Bounded angles)

Assume $\{T^h\}$ is non-degenerate, meaning there exists $\gamma > 0$ such that:

$$\frac{h_K}{\rho_K} \leq \gamma$$

for all $K \in T^h$ and all $h$, where $h_K = \text{diam}(K)$ and $\rho_K$ is the inradius.

**Key geometric fact**: For a triangle with inradius $\rho$, circumradius $R$, and minimum angle $\theta_{min}$:

$$\rho = R \sin(\theta_1) \sin(\theta_2) \sin(\theta_3)$$

$$\sin(\theta_{min}) \geq \frac{\rho}{R}$$

Since $h_K \leq 2R$ (diameter bounded by twice circumradius):

$$\frac{h_K}{\rho_K} \geq \frac{2R}{\rho} \geq \frac{2}{\sin(\theta_{min})}$$

If $h_K/\rho_K \leq \gamma$, then:

$$\sin(\theta_{min}) \geq \frac{2}{\gamma}$$

This bounds $\theta_{min}$ from below.

**Part 2: Reverse Direction** (Bounded angles ⇒ Non-degenerate)

Assume all angles $\theta$ in all triangles satisfy $\theta \geq \theta_0 > 0$.

For a triangle, the inradius and area are related by:

$$\rho = \frac{2A}{p}$$

where $A$ is area and $p$ is perimeter.

The area in terms of sides $a, b, c$ and angle $C$:

$$A = \frac{1}{2}ab\sin(C) \geq \frac{1}{2}ab\sin(\theta_0)$$

Since $h_K = \max(a,b,c)$, we have $a, b, c \leq h_K$.

The perimeter $p = a + b + c \leq 3h_K$.

Therefore:

$$\rho \geq \frac{2 \cdot \frac{1}{2}ab\sin(\theta_0)}{3h_K} \geq \frac{h_K^2 \sin(\theta_0)}{3h_K} = \frac{h_K \sin(\theta_0)}{3}$$

Thus:

$$\frac{h_K}{\rho_K} \leq \frac{3}{\sin(\theta_0)}$$

which proves non-degeneracy with $\gamma = 3/\sin(\theta_0)$.

**Part 3: Practical Implications**

| Minimum Angle | Shape Regularity Constant | Quality |
|---------------|---------------------------|---------|
| $30°$ | $\gamma = 6$ | Good |
| $15°$ | $\gamma \approx 12$ | Acceptable |
| $5°$ | $\gamma \approx 35$ | Poor (skinny) |
| $1°$ | $\gamma \approx 170$ | Very poor |

**Mesh generation rule**: Avoid angles less than $20°$ in practice.

### Interactive Visualization: Angle Checker

[codeContainer](./textbook_notes/chapter_4/ex_10_angle_checker.js)

## Exercise 4.x.5: Homogeneity Argument

**Problem**: Complete the proof of the general case of Theorem 4.4.4 by using a homogeneity argument.

### Solution

This demonstrates the scaling technique that appears throughout FEM analysis.

**Theorem 4.4.4**: For $v \in P_k(K)$ and $0 \leq s \leq m \leq k+1$:

$$|v|_{H^m(K)} \leq C h_K^{s-m} |v|_{H^s(K)}$$

**Part 1: Reference Element Estimate**

First, prove the result on the reference element $\widehat{K}$. For $\widehat{\,v\,} \in P_k(\widehat{K})$:

$$|\widehat{\,v\,}|_{H^m(\widehat{K})} \leq \widehat{C} |\widehat{\,v\,}|_{H^s(\widehat{K})}$$

This holds because:
- $\dim(P_k(\widehat{K}))$ is finite
- All norms are equivalent on finite-dimensional spaces
- $|\cdot|_{H^m}$ and $|\cdot|_{H^s}$ are both seminorms on $P_k$

**Part 2: Scaling Under Affine Map**

Let $F_K(\widehat{x}) = B_K \widehat{x} + b_K$ map $\widehat{K} \to K$.

For $v: K \to \mathbb{R}$, define $\widehat{\,v\,} = v \circ F_K$.

**Key scaling relations**:

For $L^2$ norm:
$$\|v\|_{L^2(K)}^2 = \int_K v^2 \, dx = \int_{\widehat{K}} \widehat{\,v\,}^2 |J_K| \, d\widehat{x} = |J_K| \|\widehat{\,v\,}\|_{L^2(\widehat{K})}^2$$

For $H^1$ seminorm:
$$|v|_{H^1(K)}^2 = \int_K |\nabla v|^2 \, dx = \int_{\widehat{K}} |B_K^{-T} \nabla \widehat{\,v\,}|^2 |J_K| \, d\widehat{x}$$

$$= |J_K| \int_{\widehat{K}} \nabla \widehat{\,v\,}^T (B_K^{-1} B_K^{-T}) \nabla \widehat{\,v\,} \, d\widehat{x}$$

**Bound the eigenvalues**: If $\lambda_{min}$ and $\lambda_{max}$ are the minimum and maximum eigenvalues of $B_K^{-1} B_K^{-T}$:

$$\lambda_{min} |\nabla \widehat{\,v\,}|^2 \leq \nabla \widehat{\,v\,}^T (B_K^{-1} B_K^{-T}) \nabla \widehat{\,v\,} \leq \lambda_{max} |\nabla \widehat{\,v\,}|^2$$

Therefore:

$$|v|_{H^1(K)}^2 \sim |J_K| \|B_K^{-1}\|^2 |\widehat{\,v\,}|_{H^1(\widehat{K})}^2$$

**Part 3: Relate to $h_K$**

For a shape-regular family, $\|B_K\| \sim h_K$ and $\|B_K^{-1}\| \sim h_K^{-1}$.

Thus:
- $|J_K| \sim h_K^d$ (volume scaling)
- $\|B_K^{-T}\| \sim h_K^{-1}$ (inverse scaling)

For $m$-th derivatives:

$$|v|_{H^m(K)} \sim h_K^{d/2 - m} |\widehat{\,v\,}|_{H^m(\widehat{K})}$$

**Part 4: Combine Estimates**

On reference element: $|\widehat{\,v\,}|_{H^m(\widehat{K})} \leq \widehat{C} |\widehat{\,v\,}|_{H^s(\widehat{K})}$

Scale to physical element:

$$h_K^{m - d/2} |v|_{H^m(K)} \leq \widehat{C} h_K^{s - d/2} |v|_{H^s(K)}$$

Therefore:

$$|v|_{H^m(K)} \leq \widehat{C} h_K^{s-m} |v|_{H^s(K)}$$

**Key insight**: The $h_K^{s-m}$ factor with $s < m$ gives $h_K^{-(\text{positive})}$, which blows up as $h \to 0$. This is why it's called an "inverse" inequality.

### Interactive Visualization: Homogeneity Scaling

[codeContainer](./textbook_notes/chapter_4/ex_5_homogeneity.js)

## Exercise 4.x.17: Condition Number of Stiffness Matrix

**Problem**: Consider the stiffness matrix $K$ for piecewise linear functions on a quasi-uniform mesh in one dimension as in Sect. 0.5. Prove that the condition number (Isaacson & Keller 1966) of $K$ is bounded by $O(h^{-2})$.

### Solution

This fundamental result explains why fine meshes lead to ill-conditioned systems.

**Part 1: 1D Stiffness Matrix**

For a uniform mesh with $n$ interior nodes and spacing $h = 1/(n+1)$, the stiffness matrix is:

$$K = \frac{1}{h}\begin{pmatrix}
2 & -1 & & & \\
-1 & 2 & -1 & & \\
& -1 & 2 & -1 & \\
& & \ddots & \ddots & \ddots \\
& & & -1 & 2
\end{pmatrix}$$

**Part 2: Eigenvalue Analysis**

The eigenvalues of this tridiagonal matrix are known:

$$\lambda_k = \frac{4}{h}\sin^2\left(\frac{k\pi}{2(n+1)}\right), \quad k = 1, 2, \ldots, n$$

**Smallest eigenvalue**:
$$\lambda_1 = \frac{4}{h}\sin^2\left(\frac{\pi}{2(n+1)}\right) \approx \frac{4}{h} \cdot \frac{\pi^2}{4(n+1)^2} = \frac{\pi^2}{h(n+1)^2}$$

Since $h = 1/(n+1)$:
$$\lambda_1 \approx \pi^2 h$$

**Largest eigenvalue**:
$$\lambda_n = \frac{4}{h}\sin^2\left(\frac{n\pi}{2(n+1)}\right) \approx \frac{4}{h}$$

(as $n \to \infty$, $\sin^2(n\pi/(2(n+1))) \to 1$)

**Part 3: Condition Number**

The condition number is:

$$\kappa(K) = \frac{\lambda_{max}}{\lambda_{min}} = \frac{4/h}{\pi^2 h} = \frac{4}{\pi^2 h^2} = O(h^{-2})$$

**Part 4: 2D Extension**

For 2D problems on a quasi-uniform mesh:
- Each node couples to $O(1)$ neighbors
- Eigenvalue bounds scale similarly
- Condition number: $\kappa(K) = O(h^{-2})$

**Part 5: Practical Implications**

| Mesh size $h$ | DOFs $n$ | $\kappa(K)$ | Direct solve time |
|---------------|----------|-------------|-------------------|
| $h = 0.1$ | $\sim 100$ | $\sim 100$ | Fast |
| $h = 0.01$ | $\sim 10^4$ | $\sim 10^4$ | Slow |
| $h = 0.001$ | $\sim 10^6$ | $\sim 10^6$ | Very slow |

**Why this matters**:
- Iterative solvers (CG, GMRES) converge in $O(\sqrt{\kappa})$ iterations
- Halving $h$: $4\times$ worse conditioning, $2\times$ more CG iterations
- Preconditioning is **essential** for fine meshes

### Interactive Visualization: Condition Number Demo

[codeContainer](./textbook_notes/chapter_4/ex_17_condition_number.js)

## Exercise 4.x.21: Bilinear Quadrilateral Map

**Problem**: Let $\widehat{K}$ be the unit square and $K$ be a convex quadrilateral. Show that there exists a diffeomorphism $F: K \to \widehat{K}$ such that the components belong to $Q_1$.

### Solution

This shows how to use quadrilateral elements instead of triangles.

**Part 1: Bilinear Map Construction**

Let $K$ have vertices $v_1, v_2, v_3, v_4$ (ordered counterclockwise).

Define the bilinear map $F_K: [-1,1]^2 \to K$ by:

$$F_K(\xi, \eta) = \sum_{i=1}^4 N_i(\xi, \eta) v_i$$

where $N_i$ are the bilinear shape functions:

$$N_1(\xi,\eta) = \frac{(1-\xi)(1-\eta)}{4}$$
$$N_2(\xi,\eta) = \frac{(1+\xi)(1-\eta)}{4}$$
$$N_3(\xi,\eta) = \frac{(1+\xi)(1+\eta)}{4}$$
$$N_4(\xi,\eta) = \frac{(1-\xi)(1+\eta)}{4}$$

**Verification**:
- $F_K(-1,-1) = N_1(-1,-1)v_1 = v_1$ 
- $F_K(1,-1) = N_2(1,-1)v_2 = v_2$ 
- $F_K(1,1) = N_3(1,1)v_3 = v_3$ 
- $F_K(-1,1) = N_4(-1,1)v_4 = v_4$ 

**Part 2: Jacobian Computation**

$$\frac{\partial F_K}{\partial \xi} = \sum_{i=1}^4 \frac{\partial N_i}{\partial \xi} v_i$$

$$\frac{\partial F_K}{\partial \eta} = \sum_{i=1}^4 \frac{\partial N_i}{\partial \eta} v_i$$

The Jacobian matrix:

$$J = \begin{pmatrix}
\frac{\partial x}{\partial \xi} & \frac{\partial x}{\partial \eta} \\
\frac{\partial y}{\partial \xi} & \frac{\partial y}{\partial \eta}
\end{pmatrix}$$

**Part 3: Bilinearity**

Note that $F_K$ is NOT affine (unless $K$ is a parallelogram):

$$F_K(\xi, \eta) = \underbrace{a_0 + a_1\xi + a_2\eta}_{\text{affine}} + \underbrace{a_3\xi\eta}_{\text{bilinear term}}$$

The Jacobian $\det(J)$ varies across the element!

**Part 4: Convexity Ensures Invertibility**

For $K$ convex, $\det(J) > 0$ everywhere in $[-1,1]^2$, so $F_K$ is a diffeomorphism.

**Non-convex warning**: If $K$ is not convex, the map can fold back on itself (negative Jacobian).

**Part 5: Integration on Quadrilaterals**

$$\int_K f(x,y) \, dA = \int_{-1}^1 \int_{-1}^1 f(F_K(\xi,\eta)) |\det(J(\xi,\eta))| \, d\xi \, d\eta$$

Unlike triangles, we need **numerical quadrature** because $\det(J)$ is not constant.

### Interactive Visualization: Quadrilateral Mapping

[codeContainer](./textbook_notes/chapter_4/ex_21_quad_mapping.js)

## Exercise 4.x.1: Taylor's Theorem Application

**Problem**: Show that for $|x| \leq m-1$, $D_x^m p_x(x) = I_0^m \frac{|D_x^m u(x)|}{|x|!}$ (cf. the proof of Proposition 4.1.1).

### Solution

This establishes the foundation for polynomial approximation theory.

**Part 1: Taylor Expansion**

For $u \in C^m([a,b])$, Taylor's theorem gives:

$$u(x) = \sum_{k=0}^{m-1} \frac{u^{(k)}(x_0)}{k!}(x-x_0)^k + R_m(x)$$

where the remainder is:

$$R_m(x) = \frac{1}{(m-1)!}\int_{x_0}^x u^{(m)}(t)(x-t)^{m-1} \, dt$$

**Part 2: Interpolation Error**

Let $I_h u$ be the polynomial interpolant of degree $m-1$ matching $u$ at $m$ points.

The error satisfies:

$$u(x) - I_h u(x) = \frac{u^{(m)}(\xi)}{m!}\prod_{i=1}^m (x-x_i)$$

for some $\xi \in (a,b)$.

**Part 3: Norm Estimate**

$$\|u - I_h u\|_{L^\infty} \leq \frac{h^m}{m!}\|u^{(m)}\|_{L^\infty}$$

where $h = \max_i |x_{i+1} - x_i|$.

**Key insight**: The error is $O(h^m)$, explaining why higher-degree polynomials give better approximation.

## Exercise 4.x.6: Quasi-Uniform Meshes

**Problem**: Prove that a family $\{T^h\}$ of triangulations is quasi-uniform if and only if it is non-degenerate and there exist positive constants $c$ and $C$, independent of $h$, such that $c h \leq h_K \leq C h$ for all $K \in T^h$.

### Solution

This defines the "gold standard" for mesh families.

**Part 1: Definition Check**

A family is **quasi-uniform** if:
1. Shape regular: $h_K/\rho_K \leq \gamma$
2. Size bounded: $c h \leq h_K \leq C h$ where $h = \max_K h_K$

**Part 2: Forward Direction** (Quasi-uniform ⇒ Conditions)

If quasi-uniform, then by definition non-degenerate and $ch \leq h_K \leq h$ (with $C=1$).

**Part 3: Reverse Direction** (Conditions ⇒ Quasi-uniform)

Assume:
- Non-degenerate: $h_K/\rho_K \leq \gamma$
- Size bounded: $ch \leq h_K \leq Ch$

Then all elements have comparable size, and shape regularity ensures no degeneracy.

**Part 4: Examples**

| Mesh Type | Quasi-Uniform? | Why |
|-----------|----------------|-----|
| Uniform square grid |  | All $h_K = h$ |
| Locally refined (adaptive) | ✗ | Violates $h_K \geq ch$ |
| Graded mesh ($h_K \sim x^\alpha$) | ✗ | Elements vary by $O(h)$ |
| Random Delaunay | ? | Depends on point distribution |

**Importance**: Error estimates are simplest on quasi-uniform meshes, but adaptive refinement (non-quasi-uniform) is often more efficient.

## Key Takeaways

1. **Reference element computation** (4.x.11) is the assembly kernel
2. **Mesh quality** (4.x.10) prevents numerical disasters
3. **Scaling arguments** (4.x.5) connect reference ↔ physical estimates
4. **Condition number** (4.x.17) grows as $O(h^{-2})$ - need preconditioners
5. **Quadrilaterals** (4.x.21) use bilinear maps (not affine!)
6. **Taylor expansion** (4.x.1) explains why polynomials work
7. **Quasi-uniformity** (4.x.6) simplifies theory but limits adaptivity