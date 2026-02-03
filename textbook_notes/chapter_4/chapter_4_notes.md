---
title: "Finite Element Spaces: Transformation Theory"
date: "2026-01-30"
tags: "Notes"
snippet: "Affine transformations, reference elements, Jacobians, and approximation properties for finite element spaces."
---

# Chapter 4: Finite Element Spaces

These notes cover the mathematical machinery for transforming between reference and physical elements, which is essential for implementing finite element methods.

## 4.1 The Affine Family

### Reference vs Physical Elements

In practice, we work with **two types of elements**:

| Element Type | Symbol | Description |
|--------------|--------|-------------|
| **Reference element** | $\widehat{K}$ | Standard element where we define basis functions |
| **Physical element** | $K$ | Actual element in the mesh |

**Key idea**: Compute everything on $\widehat{K}$, then transform to $K$ using an **affine map**.

### Affine Transformation

The affine map $F_K: \widehat{K} \to K$ is defined by:

$$F_K(\widehat{x}) = B_K \widehat{x} + b_K$$

where:
- $B_K \in \mathbb{R}^{d \times d}$ is an invertible matrix (linear part)
- $b_K \in \mathbb{R}^d$ is a translation vector

**Properties**:
- $F_K$ maps vertices to vertices
- $F_K$ maps edges to edges (preserves straight lines)
- $F_K$ is invertible: $F_K^{-1}(x) = B_K^{-1}(x - b_K)$

### Example: Triangle

For a triangle $K$ with vertices $v_1, v_2, v_3$ and reference triangle $\widehat{K}$ with vertices $(0,0), (1,0), (0,1)$:

$$B_K = \begin{pmatrix} v_2 - v_1 & v_3 - v_1 \end{pmatrix}, \quad b_K = v_1$$

**Verification**: 
- $F_K(0,0) = b_K = v_1$ 
- $F_K(1,0) = B_K(1,0)^T + b_K = (v_2 - v_1) + v_1 = v_2$ 
- $F_K(0,1) = B_K(0,1)^T + b_K = (v_3 - v_1) + v_1 = v_3$ 

### Jacobian Matrix

The **Jacobian** of $F_K$ is:

$$DF_K = B_K$$

(constant for affine maps!)

**Jacobian determinant**:
$$J_K = \det(B_K) = |K| / |\widehat{K}|$$

where $|K|$ denotes the volume (area in 2D) of the element.

### Transformation of Integrals

**Fundamental formula**:

$$\int_K f(x) \, dx = \int_{\widehat{K}} f(F_K(\widehat{x})) |J_K| \, d\widehat{x}$$

This is how we transform all finite element integrals.

## 4.2 Reference Elements and Basis Functions

### Pull-Back and Push-Forward

Given a function $v: K \to \mathbb{R}$, define its **pull-back** to the reference element:

$$\widehat{\,v\,} = v \circ F_K$$

That is, $\widehat{\,v\,}(\widehat{x}) = v(F_K(\widehat{x}))$ for $\widehat{x} \in \widehat{K}$.

**Inverse**: The **push-forward** from $\widehat{K}$ to $K$ is:

$$v = \widehat{\,v\,} \circ F_K^{-1}$$

### Transformation of Derivatives

**THE KEY FORMULA**: How derivatives transform under affine maps.

For $v: K \to \mathbb{R}$ and $\widehat{\,v\,} = v \circ F_K$:

$$\nabla v(x) = B_K^{-T} \nabla \widehat{\,v\,}(\widehat{x})$$

where $x = F_K(\widehat{x})$ and $B_K^{-T} = (B_K^{-1})^T = (B_K^T)^{-1}$.

**Proof**: By the chain rule,

$$\nabla v(x) = \nabla(\widehat{\,v\,} \circ F_K^{-1})(x) = (DF_K^{-1})^T \nabla \widehat{\,v\,}(F_K^{-1}(x)) = (B_K^{-1})^T \nabla \widehat{\,v\,}(\widehat{x})$$

### Transforming the Stiffness Matrix

Consider the bilinear form $a(u,v) = \int_K \nabla u \cdot \nabla v \, dx$.

The local stiffness matrix entry is:

$$K_{ij}^{(local)} = \int_K \nabla \phi_i \cdot \nabla \phi_j \, dx$$

**Transformation to reference element**:

Let $\widehat{\phi}_i = \phi_i \circ F_K$. Then:

$$K_{ij}^{(local)} = \int_{\widehat{K}} (B_K^{-T} \nabla \widehat{\phi}_i) \cdot (B_K^{-T} \nabla \widehat{\phi}_j) |J_K| \, d\widehat{x}$$

$$= |J_K| \int_{\widehat{K}} \nabla \widehat{\phi}_i^T (B_K^{-1} B_K^{-T}) \nabla \widehat{\phi}_j \, d\widehat{x}$$

Define the **metric tensor**:
$$G_K = B_K^{-1} B_K^{-T}$$

Then:
$$K_{ij}^{(local)} = |J_K| \int_{\widehat{K}} \nabla \widehat{\phi}_i^T G_K \nabla \widehat{\phi}_j \, d\widehat{x}$$

### Example: P₁ Triangle

For the reference triangle $\widehat{K}$ with vertices $(0,0), (1,0), (0,1)$:

**Reference basis functions**:
$$\widehat{\phi}_1(\widehat{x}, \widehat{\,y\,}) = 1 - \widehat{x} - \widehat{\,y\,}$$
$$\widehat{\phi}_2(\widehat{x}, \widehat{\,y\,}) = \widehat{x}$$
$$\widehat{\phi}_3(\widehat{x}, \widehat{\,y\,}) = \widehat{\,y\,}$$

**Gradients**:
$$\nabla \widehat{\phi}_1 = \begin{pmatrix} -1 \\ -1 \end{pmatrix}, \quad \nabla \widehat{\phi}_2 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \nabla \widehat{\phi}_3 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

**Reference stiffness matrix** (before transformation):
$$\widehat{K}_{ij} = \int_{\widehat{K}} \nabla \widehat{\phi}_i \cdot \nabla \widehat{\phi}_j \, d\widehat{x}$$

Since the gradients are constant and $|\widehat{K}| = 1/2$:

$$\widehat{K} = \frac{1}{2} \begin{pmatrix} 
2 & -1 & -1 \\
-1 & 1 & 0 \\
-1 & 0 & 1
\end{pmatrix}$$

**For physical element $K$**: Multiply by the metric tensor $G_K$ and Jacobian $|J_K|$.

## 4.3 Approximation Properties

### Interpolation Operator

Let $\Pi_K: C^0(K) \to P_k(K)$ be the interpolation operator that matches function values at the nodes.

**For $P_1$ elements**: $\Pi_K v$ matches $v$ at the three vertices.

### Bramble-Hilbert Lemma (Informal)

If $u \in H^{k+1}(K)$ and $\Pi_K$ reproduces polynomials of degree $\leq k$, then:

$$|u - \Pi_K u|_{H^m(K)} \leq C h_K^{k+1-m} |u|_{H^{k+1}(K)}$$

where:
- $h_K = \text{diam}(K)$ is the diameter of element $K$
- $m \leq k+1$ is the derivative order
- $C$ depends on the shape regularity of $K$

### Scaling Argument

**Key technique**: Estimates on reference element $\rightarrow$ estimates on physical element.

**Homogeneity**: If $v(\lambda x) = \lambda^\alpha v(x)$ for all $\lambda > 0$, then $v$ has homogeneity degree $\alpha$.

**Example**: For $v(x) = |x|^2$, we have $v(\lambda x) = \lambda^2 |x|^2$, so $\alpha = 2$.

**Scaling of norms**: For $\widehat{\,v\,}(\widehat{x}) = v(h\widehat{x})$:

$$\|\widehat{\,v\,}\|_{L^2(\widehat{K})} = h^{d/2} \|v\|_{L^2(K)}$$

$$|\widehat{\,v\,}|_{H^1(\widehat{K})} = h^{d/2 - 1} |v|_{H^1(K)}$$

**Application**: Interpolation error on $K$ relates to error on $\widehat{K}$ by:

$$\|u - \Pi_K u\|_{L^2(K)} \leq C h_K^2 |u|_{H^2(K)}$$

for $P_1$ elements.

### Convergence Rates

| Element | $L^2$ error | $H^1$ error |
|---------|-------------|-------------|
| $P_1$ (linear) | $O(h^2)$ | $O(h)$ |
| $P_2$ (quadratic) | $O(h^3)$ | $O(h^2)$ |
| $P_k$ (degree $k$) | $O(h^{k+1})$ | $O(h^k)$ |

## 4.4 Inverse Inequalities and Trace Theorems

### Inverse Inequality

**Opposite of approximation**: Bounds higher derivatives by lower derivatives.

**Theorem 4.4.4**: For $v \in P_k(K)$:

$$|v|_{H^m(K)} \leq C h_K^{s-m} |v|_{H^s(K)}$$

for $0 \leq s \leq m \leq k+1$.

**Key observation**: Note the $h_K^{s-m}$ with $s < m$, so this gives $h_K^{-\text{something}}$, meaning the bound **blows up** as $h_K \to 0$.

**Why "inverse"**: Normal estimates bound higher derivatives by lower ones. This goes backwards.

**Application**: Stability analysis for numerical methods.

### Shape Regularity

For inverse inequalities to hold with **uniform constants**, elements must be **shape regular**.

**Definition**: A family of triangulations $\{\mathcal{T}_h\}$ is **shape regular** if there exists $\gamma > 0$ such that:

$$\frac{h_K}{\rho_K} \leq \gamma$$

for all $K \in \mathcal{T}_h$ and all $h$, where:
- $h_K = \text{diam}(K)$ (diameter)
- $\rho_K = \sup\{\text{diameter of balls contained in } K\}$ (inradius)

**Interpretation**: No arbitrarily flat or skinny elements.

### Trace Theorem

**Problem**: How do $L^2$ boundary values relate to $H^1$ interior values?

**Trace Theorem**: There exists a bounded linear operator $\text{tr}: H^1(\Omega) \to L^2(\partial\Omega)$ such that $\text{tr}(u) = u|_{\partial\Omega}$ for $u \in C^0(\bar{\Omega})$.

**Inequality**:
$$\|u\|_{L^2(\partial K)} \leq C h_K^{-1/2} \|u\|_{L^2(K)} + C h_K^{1/2} |u|_{H^1(K)}$$

**Application**: Analyzing boundary integrals in weak formulations.

## Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Affine map | $F_K(\widehat{x}) = B_K \widehat{x} + b_K$ |
| Jacobian | $J_K = \det(B_K) = \frac{|K|}{|\widehat{K}|}$ |
| Derivative transform | $\nabla v = B_K^{-T} \nabla \widehat{\,v\,}$ |
| Integral transform | $\int_K f \, dx = \int_{\widehat{K}} f \circ F_K \cdot |J_K| \, d\widehat{x}$ |
| Metric tensor | $G_K = B_K^{-1} B_K^{-T}$ |
| Interpolation error | $\|u - \Pi_K u\|_{H^m(K)} \leq C h_K^{k+1-m} \|u\|_{H^{k+1}(K)}$ |
| Inverse inequality | $\|v\|_{H^m(K)} \leq C h_K^{s-m} \|v\|_{H^s(K)}$, $s < m$ |
| Shape regularity | $h_K / \rho_K \leq \gamma$ |

## Key Takeaways

1. **Reference elements** simplify computation - do everything on $\widehat{K}$
2. **Affine transformations** map reference to physical elements
3. **Jacobian** appears in all integral transformations
4. **Derivatives transform** via $B_K^{-T}$ (transpose-inverse)
5. **Interpolation error** scales as $O(h^{k+1})$ for $P_k$ elements
6. **Inverse inequalities** require shape regularity (no skinny elements)
7. **Shape regularity** is essential for uniform error estimates
8. **Scaling arguments** (homogeneity) are the key analytical technique

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\widehat{K}$ | Reference element |
| $K$ | Physical element |
| $F_K$ | Affine map: $\widehat{K} \to K$ |
| $B_K$ | Linear part of $F_K$ |
| $b_K$ | Translation part of $F_K$ |
| $J_K$ | Jacobian determinant |
| $G_K$ | Metric tensor $B_K^{-1} B_K^{-T}$ |
| $h_K$ | Diameter of $K$ |
| $\rho_K$ | Inradius of $K$ |
| $\Pi_K$ | Interpolation operator |
| $\widehat{\phi}_i$ | Reference basis function |
| $\phi_i$ | Physical basis function |

## Connection to Chapter 3

In Chapter 3, we learned **what** basis functions look like. In Chapter 4, we learn **how to compute with them**:

- Chapter 3: $\phi_i$ defined on specific triangle/rectangle
- Chapter 4: $\phi_i = \widehat{\phi}_i \circ F_K^{-1}$ via transformation
- Chapter 3: Showed partition of unity $\sum \phi_i = 1$
- Chapter 4: Proved error estimates $\|u - \sum u_i \phi_i\| = O(h^{k+1})$
- Chapter 3: Visualized basis functions
- Chapter 4: Shows how to assemble stiffness matrices

**Next step**: Chapter 5 applies these tools to prove convergence theorems for the finite element method.
