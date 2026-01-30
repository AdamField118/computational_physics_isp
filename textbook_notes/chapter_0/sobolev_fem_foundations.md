---
title: "FEM Foundations: Sobolev Spaces to Error Analysis"
date: "2026-01-29"
tags: "Notes"
snippet: "Comprehensive mathematical foundations of the finite element method, from weak derivatives and Sobolev spaces through Galerkin approximation and error analysis."
---

# Finite Element Method: Mathematical Foundations

These notes cover the rigorous mathematical framework underlying the finite element method, based on Brenner & Scott's *The Mathematical Theory of Finite Element Methods* (3rd edition). We build from Lebesgue spaces through Sobolev spaces to the complete variational framework.

## 1. The Model Problem

Consider the one-dimensional boundary value problem:

$$-u''(x) = f(x), \quad x \in (0,1)$$

with boundary conditions $u(0) = u(1) = 0$.

**Why can't we just solve this directly?**

Classical solutions require $u \in C^2[0,1]$, but:
- Many physical problems have discontinuous coefficients
- Numerical methods work with piecewise polynomials (not smooth)
- We need a framework that accommodates "rough" functions

This motivates **weak formulations** and **Sobolev spaces**.

## 2. Lebesgue Spaces $L^p(\Omega)$

### Definition

For $1 \leq p < \infty$, the space $L^p(\Omega)$ consists of measurable functions with finite norm:

$$\|v\|_{L^p} = \left( \int_\Omega |v|^p \, dx \right)^{1/p} < \infty$$

For $p = \infty$:

$$\|v\|_{L^\infty} = \text{ess sup}_{x \in \Omega} |v(x)|$$

### Key Inequalities

| Inequality | Statement | Use Case |
|------------|-----------|----------|
| Hölder | $\|fg\|_{L^1} \leq \|f\|_{L^p} \|g\|_{L^q}$ where $\frac{1}{p} + \frac{1}{q} = 1$ | Bounding products |
| Cauchy-Schwarz | $\|(f,g)\| \leq \|f\|_{L^2} \|g\|_{L^2}$ | Inner product bounds |
| Minkowski | $\|f + g\|_{L^p} \leq \|f\|_{L^p} + \|g\|_{L^p}$ | Triangle inequality |

### The Special Role of $L^2$

$L^2(\Omega)$ is a **Hilbert space** with inner product:

$$(f, g) = \int_\Omega f(x) g(x) \, dx$$

This inner product structure is essential for:
- Orthogonal projections (Galerkin method)
- Riesz representation theorem
- Spectral theory

## 3. Weak Derivatives

### Motivation

Consider $u(x) = |x|$ on $(-1, 1)$. Classically, $u'(0)$ doesn't exist. But we can define a **weak derivative** that captures the essential behavior.

### Definition

A function $v \in L^1_{loc}(\Omega)$ is the **weak derivative** of $u$ (written $v = D^\alpha u$) if:

$$\int_\Omega u \, D^\alpha \phi \, dx = (-1)^{|\alpha|} \int_\Omega v \, \phi \, dx \quad \forall \phi \in C_0^\infty(\Omega)$$

This comes from integration by parts with the boundary terms vanishing.

### Example: $u(x) = |x|$

For $\phi \in C_0^\infty(-1,1)$:

$$\int_{-1}^1 |x| \phi'(x) \, dx = \int_{-1}^0 (-x) \phi'(x) \, dx + \int_0^1 x \phi'(x) \, dx$$

Integrating by parts (boundary terms vanish):

$$= \int_{-1}^0 \phi(x) \, dx - \int_0^1 \phi(x) \, dx = -\int_{-1}^1 \text{sgn}(x) \phi(x) \, dx$$

Therefore, the weak derivative of $|x|$ is $\text{sgn}(x)$.

### Non-Example: Heaviside Function

The Heaviside function $H(x)$ has no weak derivative in $L^1_{loc}$. If it did, integration by parts would require:

$$\int_{-1}^1 H(x) \phi'(x) \, dx = -\int_0^1 \phi'(x) \, dx = \phi(0)$$

But no $L^1_{loc}$ function $v$ satisfies $\int v \phi \, dx = \phi(0)$ for all test functions.

## 4. Sobolev Spaces

### Definition

The Sobolev space $W_k^p(\Omega)$ consists of functions whose weak derivatives up to order $k$ are in $L^p$:

$$W_k^p(\Omega) = \{ v \in L^p(\Omega) : D^\alpha v \in L^p(\Omega) \text{ for } |\alpha| \leq k \}$$

With norm:

$$\|v\|_{W_k^p} = \left( \sum_{|\alpha| \leq k} \|D^\alpha v\|_{L^p}^p \right)^{1/p}$$

### Hilbert-Sobolev Spaces

When $p = 2$, we write $H^k(\Omega) = W_k^2(\Omega)$. These are Hilbert spaces with inner product:

$$(u, v)_{H^k} = \sum_{|\alpha| \leq k} (D^\alpha u, D^\alpha v)_{L^2}$$

### The Space $H_0^1(\Omega)$

$H_0^1(\Omega)$ is the closure of $C_0^\infty(\Omega)$ in $H^1(\Omega)$. Functions in $H_0^1$ have "zero boundary values" in a generalized sense.

### Seminorm vs Norm

The **seminorm** on $H^1$:

$$|v|_{H^1} = \|v'\|_{L^2} = \left( \int_\Omega |v'|^2 \, dx \right)^{1/2}$$

This is only a seminorm because $|v|_{H^1} = 0$ doesn't imply $v = 0$ (constants have zero seminorm).

## 5. Fundamental Inequalities

### Poincaré Inequality

For $v \in H_0^1(\Omega)$ where $\Omega$ is bounded:

$$\|v\|_{L^2} \leq C_P \|v'\|_{L^2} = C_P |v|_{H^1}$$

**Consequence:** On $H_0^1$, the seminorm $|v|_{H^1}$ is equivalent to the full norm. This is crucial because it means the energy norm controls the $L^2$ norm.

### Sobolev Embedding (1D)

In one dimension, $H^1(\Omega) \subset C^0(\bar{\Omega})$. Functions in $H^1$ are automatically continuous. This fails in higher dimensions.

### Coercivity Estimate

For the bilinear form $a(u,v) = \int_\Omega u'v' \, dx$ on $H_0^1$:

$$a(v,v) = \|v'\|_{L^2}^2 = |v|_{H^1}^2 \geq \frac{1}{1 + C_P^2} \|v\|_{H^1}^2$$

## 6. Variational Formulation

### Deriving the Weak Form

Multiply $-u'' = f$ by test function $v \in H_0^1$ and integrate:

$$-\int_0^1 u'' v \, dx = \int_0^1 f v \, dx$$

Integrate by parts (boundary terms vanish since $v \in H_0^1$):

$$\int_0^1 u' v' \, dx = \int_0^1 f v \, dx$$

### Abstract Form

Find $u \in V$ such that:

$$a(u, v) = (f, v) \quad \forall v \in V$$

where:
- $V = H_0^1(\Omega)$ (solution space)
- $a(u,v) = \int_\Omega u'v' \, dx$ (bilinear form)
- $(f,v) = \int_\Omega fv \, dx$ (linear functional)

### Boundary Conditions

| Type | Classical | Weak Form Treatment |
|------|-----------|---------------------|
| Dirichlet | $u = g$ on $\partial\Omega$ | Built into space $V$ (essential BC) |
| Neumann | $u' = h$ on $\partial\Omega$ | Appears in $(f,v)$ (natural BC) |

### Equivalence Theorem

If $u$ is a classical solution ($u \in C^2$), then $u$ satisfies the weak form. Conversely, if $u$ satisfies the weak form and $u \in C^2$, then $u$ is a classical solution.

## 7. The Lax-Milgram Theorem

### Statement

Let $V$ be a Hilbert space, $a(\cdot, \cdot)$ a bilinear form, and $F$ a linear functional. If:

1. **Continuity:** $|a(u,v)| \leq M \|u\|_V \|v\|_V$
2. **Coercivity:** $a(v,v) \geq \alpha \|v\|_V^2$ for some $\alpha > 0$

Then there exists a unique $u \in V$ satisfying $a(u,v) = F(v)$ for all $v \in V$.

### Verification for Model Problem

On $V = H_0^1(0,1)$ with energy norm $\|v\|_E = |v|_{H^1}$:

**Continuity:** By Cauchy-Schwarz:
$$|a(u,v)| = \left| \int_0^1 u'v' \, dx \right| \leq \|u'\|_{L^2} \|v'\|_{L^2} = \|u\|_E \|v\|_E$$

So $M = 1$.

**Coercivity:**
$$a(v,v) = \int_0^1 (v')^2 \, dx = \|v\|_E^2$$

So $\alpha = 1$.

Lax-Milgram guarantees existence and uniqueness.

## 8. Galerkin Approximation

### The Discrete Problem

Choose a finite-dimensional subspace $S \subset V$ (e.g., piecewise linear functions). Find $u_S \in S$ such that:

$$a(u_S, w) = (f, w) \quad \forall w \in S$$

### Matrix Formulation

Let $\{\phi_1, \ldots, \phi_N\}$ be a basis for $S$. Write $u_S = \sum_{j=1}^N U_j \phi_j$.

The discrete problem becomes:

$$KU = F$$

where:
- $K_{ij} = a(\phi_j, \phi_i)$ (stiffness matrix)
- $F_i = (f, \phi_i)$ (load vector)
- $U = (U_1, \ldots, U_N)^T$ (coefficient vector)

### Properties of $K$

- **Symmetric:** $K_{ij} = a(\phi_j, \phi_i) = a(\phi_i, \phi_j) = K_{ji}$
- **Positive definite:** $U^T K U = a(u_S, u_S) > 0$ for $u_S \neq 0$
- **Sparse:** If basis functions have local support, most $K_{ij} = 0$

## 9. Error Analysis

### Galerkin Orthogonality

The error $e = u - u_S$ satisfies:

$$a(u - u_S, w) = 0 \quad \forall w \in S$$

The error is **orthogonal** to the approximation space in the energy inner product.

### Céa's Lemma (Best Approximation)

$$\|u - u_S\|_E = \min_{v \in S} \|u - v\|_E$$

**The Galerkin solution is the best approximation to $u$ from $S$ in the energy norm.**

This is remarkable: we automatically get the optimal approximation without explicitly minimizing.

### Proof Sketch

For any $v \in S$:
$$\|u - u_S\|_E^2 = a(u - u_S, u - u_S) = a(u - u_S, u - v) \leq \|u - u_S\|_E \|u - v\|_E$$

Dividing gives $\|u - u_S\|_E \leq \|u - v\|_E$.

## 10. Aubin-Nitsche Duality Argument

### The Dual Problem

To bound $\|e\|_{L^2}$ in terms of $\|e\|_E$, consider:

$$-w'' = e, \quad w(0) = w(1) = 0$$

### Duality Estimate

$$\|e\|_{L^2}^2 = (e, e) = a(w, e) = a(w - w_S, e) \leq \|w - w_S\|_E \|e\|_E$$

If interpolation gives $\|w - w_S\|_E \leq Ch\|w''\|_{L^2} = Ch\|e\|_{L^2}$, then:

$$\|e\|_{L^2} \leq Ch \|e\|_E$$

### Superconvergence

This explains why $L^2$ errors are often one order better than energy errors:
- Energy error: $O(h)$
- $L^2$ error: $O(h^2)$

## 11. Piecewise Linear Finite Elements

### The Finite Element Space

$$S = \{ v \in C^0[0,1] : v|_{[x_{i-1}, x_i]} \text{ is linear}, \, v(0) = v(1) = 0 \}$$

### Hat Function Basis

Define nodal basis functions:

$$\phi_i(x_j) = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

Explicitly:

$$\phi_i(x) = \begin{cases} \frac{x - x_{i-1}}{h} & x \in [x_{i-1}, x_i] \\ \frac{x_{i+1} - x}{h} & x \in [x_i, x_{i+1}] \\ 0 & \text{otherwise} \end{cases}$$

### Stiffness Matrix (Uniform Mesh)

For uniform spacing $h = 1/(N+1)$:

$$K = \frac{1}{h} \begin{pmatrix} 2 & -1 & & \\ -1 & 2 & -1 & \\ & \ddots & \ddots & \ddots \\ & & -1 & 2 \end{pmatrix}$$

This is exactly the finite difference matrix.

### Interpolation Error

For the nodal interpolant $u_I$ (matching $u$ at nodes):

$$\|u - u_I\|_E \leq Ch \|u''\|_{L^2}$$

$$\|u - u_I\|_{L^2} \leq Ch^2 \|u''\|_{L^2}$$

### Remarkable Property

For piecewise linears: $u_S = u_I$ when $f$ is piecewise constant.

The Galerkin solution equals the interpolant at nodes.

## 12. Notation Reference

| Symbol | Meaning |
|--------|---------|
| $a(u,v)$ | Energy inner product: $\int_\Omega u'v' dx$ |
| $(f,v)$ | $L^2$ inner product: $\int_\Omega fv \, dx$ |
| $\|v\|_E$ | Energy norm: $\sqrt{a(v,v)}$ |
| $\|v\|_{L^2}$ | $L^2$ norm: $\sqrt{(v,v)}$ |
| $\|v\|_{\max}$ | Maximum norm: $\max_x |v(x)|$ |
| $h$ | Mesh size: $\max_i (x_i - x_{i-1})$ |
| $V$ | Infinite-dimensional solution space |
| $S$ | Finite-dimensional approximation space |
| $u$ | Exact solution |
| $u_S$ | Galerkin approximation |
| $u_I$ | Interpolant |
| $H^k(\Omega)$ | Sobolev space $W_k^2(\Omega)$ |
| $C_0^\infty(\Omega)$ | Smooth functions with compact support |

## 13. Summary: The FEM Pipeline

$$\boxed{\text{Strong PDE} \to \text{Weak Form} \to \text{Lax-Milgram} \to \text{Choose } S \to KU = F \to \text{Solve} \to \text{Error Analysis}}$$

1. **Start** with strong form: $-u'' = f$
2. **Derive** weak form: $a(u,v) = (f,v)$
3. **Verify** Lax-Milgram conditions (existence/uniqueness)
4. **Choose** finite element space $S$
5. **Assemble** stiffness matrix $K$ and load vector $F$
6. **Solve** linear system $KU = F$
7. **Error** bounds come automatically from Céa's lemma

The mathematical framework guarantees convergence and provides error estimates without ad-hoc analysis.
