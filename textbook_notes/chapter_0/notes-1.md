---
title: "Basic Concepts: Chapter 0"
date: "2026-01-21"
tags: "Notes"
snippet: "A one-dimensional introduction to the finite element method."
---

# Introduction

Chapter 0 provides a one-dimensional introduction to the finite element method, serving as a microcosm of the entire book. The chapter develops the fundamental concepts through the lens of two-point boundary value problems, leaving some theoretical loose ends that motivate the study of Sobolev spaces in Chapter 1.

## 0.1 Weak Formulation of Boundary Value Problems

### Model Problem

We consider the two-point boundary value problem:

$$-\frac{d^2 u}{dx^2} = f \text{ in } (0,1)$$
$$u(0) = 0, \quad u'(1) = 0$$

### Derivation of Weak Form

For any sufficiently regular test function $v$ with $v(0) = 0$, integration by parts yields:

$$(f, v) = \int_0^1 f(x)v(x)dx = \int_0^1 -u''(x)v(x)dx = \int_0^1 u'(x)v'(x)dx =: a(u,v)$$

**Key Definitions:**
- **Energy inner product**: $a(v,w) = \int_0^1 v'(x)w'(x)dx$
- **L² inner product**: $(f,v) = \int_0^1 f(x)v(x)dx$
- **Function space**: $V = \{v \in L^2(0,1) : a(v,v) < \infty \text{ and } v(0) = 0\}$

### Variational Formulation

Find $u \in V$ such that:
$$a(u,v) = (f,v) \quad \forall v \in V$$

This is called the **variational** or **weak formulation** because:
1. It allows the test function $v$ to vary arbitrarily
2. It requires less regularity than the classical formulation
3. It naturally incorporates boundary conditions

### Boundary Conditions

| Boundary Condition | Variational Name | Proper Name |
|-------------------|------------------|-------------|
| $u(x) = 0$ | Essential | Dirichlet |
| $u'(x) = 0$ | Natural | Neumann |

- **Essential (Dirichlet)**: Explicitly enforced in the definition of $V$
- **Natural (Neumann)**: Implicitly incorporated through integration by parts

**Theorem 0.1.4**: If $f \in C^0([0,1])$ and $u \in C^2([0,1])$ satisfy the weak formulation, then $u$ solves the original BVP.

*Proof sketch*: Integration by parts shows that $(f - (-u''), v) = 0$ for all $v \in V \cap C^1([0,1])$ with $v(1) = 0$. By choosing appropriate test functions, we can show $-u'' = f$ and $u'(1) = 0$.

## 0.2 Ritz-Galerkin Approximation

### Discrete Problem

Let $S \subset V$ be a finite-dimensional subspace. The Ritz-Galerkin approximation is:

Find $u_S \in S$ such that:
$$a(u_S, v) = (f,v) \quad \forall v \in S$$

### Matrix Formulation

Given a basis $\{\phi_i : 1 \leq i \leq n\}$ of $S$, write $u_S = \sum_{j=1}^n U_j \phi_j$ and define:
- **Stiffness matrix**: $K_{ij} = a(\phi_j, \phi_i)$
- **Load vector**: $F_i = (f, \phi_i)$
- **Solution vector**: $U = (U_j)$

The discrete problem becomes:
$$KU = F$$

**Theorem 0.2.2**: Given $f \in L^2(0,1)$, the discrete problem has a unique solution.

*Proof*: For finite-dimensional systems, uniqueness implies existence. Suppose $KV = 0$ for some $V \neq 0$. Then $v = \sum V_j \phi_j$ satisfies $a(v, \phi_j) = 0$ for all $j$, which implies $a(v,v) = 0$, hence $v' \equiv 0$. Since $v(0) = 0$ (from $v \in V$), we have $v \equiv 0$, contradicting $V \neq 0$.

### Properties of Stiffness Matrix

The stiffness matrix $K$ is:
1. **Symmetric**: $K_{ij} = K_{ji}$ (since $a(\cdot, \cdot)$ is symmetric)
2. **Positive definite**: $V^T K V = a(v,v) \geq 0$ where $v = \sum v_j \phi_j$

## 0.3 Error Estimates

### Energy Norm

Define the **energy norm**: $\|v\|_E = \sqrt{a(v,v)} = \left(\int_0^1 (v')^2 dx\right)^{1/2}$

### Fundamental Orthogonality

Subtracting the discrete problem from the continuous problem:
$$a(u - u_S, w) = 0 \quad \forall w \in S$$

This is the **Galerkin orthogonality** - the error is orthogonal to the approximation space in the energy inner product.

### Best Approximation Property

**Theorem 0.3.3**: $\|u - u_S\|_E = \min_{v \in S} \|u - v\|_E$

*Proof*: For any $v \in S$:
$$\|u - u_S\|_E^2 = a(u - u_S, u - v) + a(u - u_S, v - u_S) = a(u - u_S, u - v)$$
(using orthogonality). By Schwarz inequality: $|a(v,w)| \leq \|v\|_E \|w\|_E$, so:
$$\|u - u_S\|_E^2 \leq \|u - u_S\|_E \|u - v\|_E$$

### Duality Argument (Aubin-Nitsche Trick)

To estimate the $L^2$ error, solve the **dual problem**:
$$-w'' = u - u_S \text{ on } [0,1], \quad w(0) = w'(1) = 0$$

Integration by parts gives:
$$\|u - u_S\|^2 = (u - u_S, u - u_S) = (u - u_S, -w'') = a(u - u_S, w)$$

Using orthogonality and Schwarz inequality:
$$\|u - u_S\|^2 = a(u - u_S, w - v) \leq \|u - u_S\|_E \|w - v\|_E$$

for any $v \in S$.

**Theorem 0.3.5** (with approximation assumption): If $\inf_{v \in S} \|w - v\|_E \leq \epsilon \|w''\|$ for solutions $w$ of the dual problem, then:
$$\|u - u_S\| \leq \epsilon \|u - u_S\|_E \leq \epsilon^2 \|u''\| = \epsilon^2 \|f\|$$

This shows **superconvergence**: the $L^2$ error converges at order $\epsilon^2$ while the energy error converges at order $\epsilon$.

## 0.4 Piecewise Polynomial Spaces - The Finite Element Method

### Mesh Definition

Let $0 = x_0 < x_1 < \cdots < x_n = 1$ be a partition of $[0,1]$ with mesh size $h = \max_{1 \leq i \leq n} (x_i - x_{i-1})$.

### Finite Element Space

Define $S$ as the space of functions $v$ such that:
1. $v \in C^0([0,1])$ (continuous)
2. $v|_{[x_{i-1}, x_i]}$ is linear for $i = 1, \ldots, n$ (piecewise linear)
3. $v(0) = 0$ (boundary condition)

### Nodal Basis Functions

For $i = 1, \ldots, n$, define the **hat function** $\phi_i$ by:
$$\phi_i(x_j) = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

**Lemma 0.4.1**: $\{\phi_i : 1 \leq i \leq n\}$ is a basis for $S$.

### Interpolation Operator

For $v \in C^0([0,1])$, define the **interpolant**:
$$v_I = \sum_{i=1}^n v(x_i) \phi_i$$

**Lemma 0.4.4**: If $v \in S$, then $v = v_I$ (the interpolant reproduces functions in $S$).

### Interpolation Error Estimate

**Theorem 0.4.5**: $\|u - u_I\|_E \leq Ch \|u''\|$ where $C$ is independent of $h$ and $u$.

*Proof sketch*: 
1. Reduce to a mesh-independent estimate on the reference interval $[0,1]$ using an **affine transformation** (homogeneity/scaling argument)
2. Let $w = \tilde{e}$ be the interpolation error on $[0,1]$ with $w(0) = w(1) = 0$
3. By Rolle's theorem, $w'(\xi) = 0$ for some $\xi \in (0,1)$
4. For any $y \in [0,1]$:
$$w'(y) = \int_\xi^y w''(x)dx$$
5. Apply Schwarz inequality:
$$|w'(y)| \leq |y - \xi|^{1/2} \left(\int_\xi^y |w''(x)|^2 dx\right)^{1/2} \leq |y - \xi|^{1/2} \|w''\|$$
6. Square and integrate to get:
$$\int_0^1 (w')^2 dx \leq c \int_0^1 |w''|^2 dx$$
with $c = \sup_{0 < \xi < 1} \int_0^1 |y - \xi| dy = 1/2$

**Corollary 0.4.7**: $\|u - u_S\| + Ch\|u - u_S\|_E \leq 2(Ch)^2 \|u''\|$

This gives:
- **Energy norm convergence**: $O(h)$  
- **L² norm convergence**: $O(h^2)$ (superconvergence!)

## 0.5 Relationship to Difference Methods

### Stiffness Matrix Entries

For the piecewise linear basis with $h_i = x_i - x_{i-1}$:
$$K_{ii} = h_i^{-1} + h_{i+1}^{-1}, \quad K_{i,i+1} = K_{i+1,i} = -h_{i+1}^{-1}$$

### Load Vector

For smooth $f$:
$$(f, \phi_i) = \frac{1}{2}(h_i + h_{i+1})(f(x_i) + O(h))$$

### Difference Equation Form

The $i$-th equation of $KU = F$ becomes:
$$-2\left(\frac{U_{i+1} - U_i}{h_i + h_{i+1}} - \frac{U_i - U_{i-1}}{h_{i+1}h_i}\right) = f(x_i) + O(h)$$

For a **uniform mesh** ($h_i = h$ for all $i$):
$$-\frac{U_{i+1} - 2U_i + U_{i-1}}{h^2} = f(x_i) + O(h^2)$$

This is the standard second-order accurate finite difference approximation.

### Key Observation

Even on non-uniform meshes, the finite element method produces second-order accurate solutions (in $L^2$), even though the difference equations may only be first-order consistent. This demonstrates the robustness of the variational formulation.

## 0.6 Computer Implementation

### Global-to-Local Index

For element $e$ based on interval $I_e = [x_{e-1}, x_e]$, define:
$$i(e, j) = e + j - 1 \quad \text{for } e = 1, \ldots, n \text{ and } j = 0, 1$$

This maps local node numbers ($j = 0$ for left, $j = 1$ for right) to global node numbers.

### Local Basis Functions

Define reference basis functions on $[0,1]$:
$$\phi_0(x) = 1 - x, \quad \phi_1(x) = x$$

For element $e$, the local basis functions are:
$$\phi_e^j(x) = \phi_j\left(\frac{x - x_{e-1}}{x_e - x_{e-1}}\right)$$

### Assembly Process

Any piecewise linear function can be written:
$$f = \sum_e \sum_{j=0}^1 f_{i(e,j)} \phi_e^j$$

The bilinear form is assembled:
$$a(v, w) = \sum_e a_e(v, w)$$

where the **local bilinear form** is:
$$a_e(v, w) = (x_e - x_{e-1})^{-1} \begin{pmatrix} v_{i(e,0)} \\ v_{i(e,1)} \end{pmatrix}^T K \begin{pmatrix} w_{i(e,0)} \\ w_{i(e,1)} \end{pmatrix}$$

with **local stiffness matrix**:
$$K_{i,j} = \int_0^1 \phi'_{i-1} \phi'_{j-1} dx$$

## 0.7 Local Estimates

### Green's Function

For any point $x \in [0,1]$, define:
$$g_x(t) = \begin{cases} t & t < x \\ x & \text{otherwise} \end{cases}$$

This satisfies $g_x'' = 0$ except at $x$, and integration by parts shows:
$$v(x) = a(v, g_x) \quad \forall v \in V$$

### Nodal Exactness

**Key observation**: For piecewise linear elements, $g_{x_i} \in S$, so:
$$(u - u_S)(x_i) = a(u - u_S, g_{x_i}) = a(u - u_S, g_{x_i} - v) = 0$$

for any $v \in S$. Therefore, **$u_S = u_I$** (the Galerkin solution equals the interpolant).

**Theorem 0.7.2**: $\|u - u_S\|_{\max} \leq Ch^2 \|u''\|_{\max}$

This gives **pointwise second-order convergence**.

## 0.8 Adaptive Approximation

### Uniform vs. Adaptive Meshes

Consider approximating functions with $\int_0^1 |u'(x)| dx = 1$ using piecewise constants on a partition of size $n$.

**Fixed mesh result**: For any fixed partition $\Delta$:
$$\inf_{v \in S_\Delta} \|u - v\|_{\max} \leq C$$
with $C$ independent of $n$ (best possible is $p = 0$ in $O(n^{-p})$).

**Adaptive mesh result**: For each $u$, we can construct a partition $\Delta$ with size $n$ such that:
$$\inf_{v \in S_\Delta} \|u - v\|_{\max} \leq \frac{1}{n}$$

### Adaptive Mesh Construction

Define the cumulative derivative function:
$$\phi(x) = \int_0^x |u'(t)| dt$$

Choose partition points where $\phi(x_i) = i/n$. This ensures:
$$\int_{x_{i-1}}^{x_i} |u'(t)| dt = \frac{1}{n}$$

Approximating $u$ by the constant $c_i = u(x_{i-1})$ on $[x_{i-1}, x_i]$:
$$|u(x) - c_i| = \left|\int_{x_{i-1}}^x u'(t) dt\right| \leq \frac{1}{n}$$

This demonstrates **order of magnitude improvement** ($n^{-1}$ vs. $n^0$) when meshes are adapted to the solution.

## 0.9 Weighted Norm Estimates

### Mesh Function

Define $h(x)$ as a piecewise linear function measuring local mesh size:
$$h(x_j) = h_j + h_{j+1}$$

where $h_j = x_j - x_{j-1}$. On each interval $[x_{j-1}, x_j]$, we have $h(x) \geq h_j$.

### Weighted Energy Estimate

Without mesh restrictions:
$$\|u - u_S\|_E \leq \frac{1}{\sqrt{2}} \|hu''\|$$

where $\|f\| = \left(\int_0^1 f^2 dx\right)^{1/2}$ is the $L^2$ norm.

### Weighted L² Estimate

$$\|u - u_S\| \leq \frac{1}{\sqrt{2}} \|h(u - u_S)'\|$$

### Full L² Estimate with Mesh Variation Constraint

Let $M = \|h'\|_{\max}$ measure the **mesh variation**. If $M$ is sufficiently small (roughly $M < 1 + 1/\sqrt{14} \approx 1.27$), then:
$$\|u - u_S\| \leq C(M) \|h^2 u''\|$$

where $C(M) \to 7/4$ as $M \to 0$.

### Mesh Variation Interpretation

Since $h'|_{(x_{i-1}, x_i)} = (h_{i+1} - h_{i-1})/h_i$, the condition $|h'|$ small means:
$$r_{i+1} - \frac{1}{r_i} \text{ is small}$$

where $r_i = h_i/h_{i-1}$ is the ratio of adjacent mesh intervals.

**Important**: This allows **geometric mesh grading** ($x_i = e^{\delta(i-n)}$ for small $\delta$) while maintaining second-order convergence.

**Theorem 0.9.7** (Summary):
- Without restrictions: $\|u - u_S\|_E \leq \frac{1}{\sqrt{2}} \|hu''\|$ and $\|u - u_S\| \leq \frac{1}{\sqrt{2}} \|h(u - u_S)'\|$
- With small mesh variation $M$: $\|u - u_S\| \leq C(M) \|h^2 u''\|$

## Key Takeaways

1. **Weak formulation** provides a systematic framework for deriving discrete schemes
2. **Galerkin orthogonality** is fundamental to all error analysis
3. **Best approximation** in energy norm follows immediately from orthogonality
4. **Duality arguments** (Aubin-Nitsche) give improved convergence in weaker norms
5. **Homogeneity arguments** (scaling) reduce mesh-dependent estimates to reference element estimates
6. **Finite element methods** are equivalent to finite difference methods on structured meshes but more flexible
7. **Adaptive meshes** can dramatically improve approximation efficiency
8. **Weighted norms** allow analysis on non-uniform meshes with controlled variation
9. The **interpolant equals the Galerkin solution** for piecewise linear elements (nodal exactness)
10. **Second-order convergence** is achieved in $L^2$ norm even with first-order consistency in difference form

## Important Inequalities

- **Schwarz**: $|a(v,w)| \leq \|v\|_E \|w\|_E$
- **Sobolev (1D)**: $|w'(y)| \leq |y - \xi|^{1/2} \|w''\|$ when $w'(\xi) = 0$
- **Arithmetic-geometric mean**: $ab \leq \frac{1}{2}(a^2 + b^2)$ or $ab \leq \frac{\delta}{2}a^2 + \frac{1}{2\delta}b^2$

## Notation Summary

- $a(u,v) = \int_0^1 u'v' dx$ - energy inner product
- $(f,v) = \int_0^1 fv dx$ - $L^2$ inner product  
- $\|v\|_E = \sqrt{a(v,v)}$ - energy norm
- $\|v\| = \sqrt{(v,v)}$ - $L^2$ norm
- $\|v\|_{\max} = \max_{x \in [0,1]} |v(x)|$ - maximum norm
- $h = \max_i (x_i - x_{i-1})$ - mesh size
- $V$ - infinite-dimensional solution space
- $S$ - finite-dimensional approximation space
- $u$ - exact solution
- $u_S$ - Galerkin approximation
- $u_I$ - interpolant