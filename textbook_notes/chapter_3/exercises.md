---
title: "Chapter 3 Interactive Exercises: Finite Element Construction"
date: "2026-01-30"
tags: "Exercises"
snippet: "Interactive visualizations of nodal basis functions for rectangles, triangles, and nonconforming elements"
---

# Chapter 3: Construction of Finite Element Spaces

## Exercise 3.x.9: Rectangular Element Basis Functions

**Problem**: Construct nodal basis functions for:
- $K = [-1,1] \times [0,1]$ (rectangle)
- $\mathcal{P} = Q_1$ (bilinear functions)
- $\mathcal{N} =$ evaluation at vertices

### Mathematical Solution

The space $Q_1$ consists of bilinear functions of the form:
$$p(x,y) = a_0 + a_1 x + a_2 y + a_3 xy$$

**Vertices**: $(-1,0), (1,0), (1,1), (-1,1)$

**Goal**: Find basis functions $\phi_i$ such that $\phi_i(v_j) = \delta_{ij}$.

**Construction on Reference Element**:

Using tensor product of 1D linear Lagrange polynomials:

$$\phi_1(x,y) = \frac{(1-x)(1-y)}{4}$$
$$\phi_2(x,y) = \frac{(1+x)(1-y)}{4}$$
$$\phi_3(x,y) = \frac{(1+x)(1+y)}{4}$$
$$\phi_4(x,y) = \frac{(1-x)(1+y)}{4}$$

**Verification**: These satisfy:
- $\phi_i \in Q_1$ (bilinear)
- $\phi_i(v_j) = \delta_{ij}$
- $\sum_{i=1}^4 \phi_i(x,y) = 1$ (partition of unity)

### Interactive Visualization

[codeContainer](./textbook_notes/chapter_3/ex_9_rectangle.js)

---

## Exercise 3.x.10: Quadratic Triangle Basis Functions

**Problem**: Construct nodal basis functions for:
- $K = $ triangle with vertices $(0,0), (1,0), (0,1)$
- $\mathcal{P} = P_2$ (quadratic polynomials)
- $\mathcal{N} =$ evaluation at vertices and edge midpoints

### Mathematical Solution

The space $P_2$ has dimension 6 (quadratics in 2D: $1, x, y, x^2, xy, y^2$).

**Nodes**:
1. Vertices: $v_1 = (0,0)$, $v_2 = (1,0)$, $v_3 = (0,1)$
2. Edge midpoints: $m_1 = (1/2, 0)$, $m_2 = (1/2, 1/2)$, $m_3 = (0, 1/2)$

**Using Barycentric Coordinates**: $(x, y) \to (\lambda_1, \lambda_2, \lambda_3)$ where:
$$\lambda_1 = 1 - x - y, \quad \lambda_2 = x, \quad \lambda_3 = y$$

**Vertex basis functions** (quadratic bubble):
$$\phi_1 = \lambda_1(2\lambda_1 - 1) = (1-x-y)(1-2x-2y)$$
$$\phi_2 = \lambda_2(2\lambda_2 - 1) = x(2x-1)$$
$$\phi_3 = \lambda_3(2\lambda_3 - 1) = y(2y-1)$$

**Edge midpoint basis functions** (quadratic bubbles on edges):
$$\phi_4 = 4\lambda_1\lambda_2 = 4x(1-x-y)$$
$$\phi_5 = 4\lambda_2\lambda_3 = 4xy$$
$$\phi_6 = 4\lambda_1\lambda_3 = 4y(1-x-y)$$

**Verification**:
- $\phi_i(node_j) = \delta_{ij}$
- Each $\phi_i \in P_2$
- $\sum_{i=1}^6 \phi_i = 1$

### Interactive Visualization

[codeContainer](./textbook_notes/chapter_3/ex_10_triangle.js)

---

## Exercise 3.x.14: Nonconforming Elements

**Problem**: Show that edge midpoints in a triangulation can parametrize the space of piecewise linear functions that are continuous at edge midpoints (but may be discontinuous at vertices). Generalize to quadratics.

### Mathematical Solution

**Part 1: Piecewise Linear Nonconforming Elements**

Let $\mathcal{T}$ be a triangulation of domain $\Omega$.

**Conforming space** (standard):
$$V_h^{conf} = \{v \in C^0(\Omega) : v|_T \in P_1 \text{ for each } T \in \mathcal{T}\}$$
- DOFs: values at vertices
- Dimension: $n_{vertices}$ (interior vertices for homogeneous BC)

**Nonconforming space** (Crouzeix-Raviart):
$$V_h^{NC} = \{v : v|_T \in P_1, \, v \text{ continuous at edge midpoints}\}$$
- DOFs: values at edge midpoints
- Dimension: $n_{edges}$
- Functions may "jump" at vertices!

**Why this works**:
1. On each triangle $T$, a linear function is determined by 3 DOFs
2. Two triangles sharing an edge have matching values at the edge midpoint
3. A linear function on an edge is determined by its value at the midpoint and its endpoint values
4. Matching at midpoint ensures "weak continuity"

**Example**: On two triangles sharing edge $e$:
- Triangle $T_1$: $v_1(x,y) = a_1 + b_1 x + c_1 y$
- Triangle $T_2$: $v_2(x,y) = a_2 + b_2 x + c_2 y$
- Constraint: $v_1(m_e) = v_2(m_e)$ where $m_e$ is edge midpoint

This gives continuity at $m_e$ but allows $v_1(vertex) \neq v_2(vertex)$.

**Part 2: Generalization to Quadratics**

For piecewise quadratics nonconforming:
$$V_h^{NC,P_2} = \{v : v|_T \in P_2, \, v \text{ continuous at 2 points per edge}\}$$

**DOFs**: Two points per edge (e.g., the two "third-points" at $t=1/3$ and $t=2/3$)

**Dimension**: $2 \times n_{edges}$

A quadratic function on an edge is determined by 3 values. If we enforce continuity at 2 interior points, the functions match on the entire edge.

### Interactive Visualization

[codeContainer](./textbook_notes/chapter_3/ex_14_nonconforming.js)

---

## Exercise 3.x.19: Lagrange Element DOF Count

**Problem**: Suppose that the nodes for the Lagrange element are chosen at the barycentric lattice points. Let $T$ be a triangle with vertices $p_k$ $(1 \leq k \leq 3)$ and $\ell, m, n$ be nonnegative integers with $\ell + m + n = r$. Show that the corresponding nodal basis functions for $\mathcal{P}_k$ can be written as a product of $k$ linear functions.

What is the corresponding formula for a tetrahedron?

### Mathematical Solution

**Setup**: For degree $r$ Lagrange elements, nodes are at barycentric coordinates:
$$\left(\frac{\ell}{r}, \frac{m}{r}, \frac{n}{r}\right) \text{ where } \ell + m + n = r$$

**Number of nodes (triangle)**:
$$N_{triangle}(r) = \binom{r+2}{2} = \frac{(r+1)(r+2)}{2}$$

**Derivation**: 
- Choose $\ell$ from $\{0, 1, \ldots, r\}$ ($r+1$ choices)
- Choose $m$ from $\{0, 1, \ldots, r-\ell\}$ ($r-\ell+1$ choices)
- Then $n = r - \ell - m$ (determined)
- Total: $\sum_{\ell=0}^r (r-\ell+1) = \sum_{j=1}^{r+1} j = \frac{(r+1)(r+2)}{2}$

**Basis function formula**:

For node at $(\ell/r, m/r, n/r)$, the basis function is:

$$\phi_{\ell,m,n}(\lambda_1, \lambda_2, \lambda_3) = \prod_{i=1}^{\ell} \frac{r\lambda_1 - i + 1}{i} \cdot \prod_{j=1}^{m} \frac{r\lambda_2 - j + 1}{j} \cdot \prod_{k=1}^{n} \frac{r\lambda_3 - k + 1}{k}$$

Or equivalently:
$$\phi_{\ell,m,n} = \frac{r!}{\ell! m! n!} \prod_{i=0}^{\ell-1} (r\lambda_1 - i) \prod_{j=0}^{m-1} (r\lambda_2 - j) \prod_{k=0}^{n-1} (r\lambda_3 - k) \cdot r^{-r}$$

**Special cases**:
- $r=1$ (linear): $\phi_{1,0,0} = \lambda_1$, $\phi_{0,1,0} = \lambda_2$, $\phi_{0,0,1} = \lambda_3$
- $r=2$ (quadratic): $\phi_{2,0,0} = \lambda_1(2\lambda_1-1)$, $\phi_{1,1,0} = 4\lambda_1\lambda_2$

**For a tetrahedron**:

With 4 vertices, nodes are at:
$$\left(\frac{\ell}{r}, \frac{m}{r}, \frac{n}{r}, \frac{p}{r}\right) \text{ where } \ell + m + n + p = r$$

**Number of nodes (tetrahedron)**:
$$N_{tet}(r) = \binom{r+3}{3} = \frac{(r+1)(r+2)(r+3)}{6}$$

**Derivation**: This is the number of ways to distribute $r$ identical balls into 4 distinct bins, which is the "stars and bars" formula.

**Examples**:
- $r=1$: $N = \binom{4}{3} = 4$ nodes (vertices)
- $r=2$: $N = \binom{5}{3} = 10$ nodes (4 vertices + 6 edge midpoints)
- $r=3$: $N = \binom{6}{3} = 20$ nodes

### Interactive Visualization

[codeContainer](./textbook_notes/chapter_3/ex_19_lagrange.js)

---

## Summary Table

| Exercise | Element Type | Polynomial Space | DOFs | Key Concept |
|----------|--------------|------------------|------|-------------|
| 3.x.9 | Rectangle | $Q_1$ (bilinear) | 4 vertices | Tensor product structure |
| 3.x.10 | Triangle | $P_2$ (quadratic) | 3 vertices + 3 midpoints | Barycentric basis |
| 3.x.14 | Triangle | $P_1$ nonconforming | Edge midpoints | Weak continuity |
| 3.x.19 | Triangle/Tet | $P_r$ (degree $r$) | $\binom{r+d}{d}$ | Lattice point counting |

## Key Takeaways

1. **Nodal basis functions** are constructed to satisfy $\phi_i(node_j) = \delta_{ij}$
2. **Barycentric coordinates** provide elegant formulas for simplex elements
3. **Nonconforming elements** relax continuity requirements for flexibility
4. **DOF counting** follows combinatorial formulas based on lattice points
5. **Tensor products** generate rectangular element bases from 1D bases
6. **Lagrange elements** of degree $r$ have $\binom{r+d}{d}$ nodes in $d$ dimensions
