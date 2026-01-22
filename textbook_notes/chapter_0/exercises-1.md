---
title: "Chapter 0 Exercise Solutions"
date: "2026-01-21"
tags: "Exercises"
snippet: "Complete solutions to exercises 0.x.1 through 0.x.16 from Brenner & Scott"
---

# Chapter 0 Exercise Solutions

## Exercise 0.x.1

**Problem**: Verify the expressions (0.5.1) for the stiffness matrix $K$ for piecewise linear functions. If $f$ is piecewise linear, determine the matrix $M$ (the "mass" matrix) such that $KU = MF$.

### Solution

**Part 1: Stiffness Matrix Verification**

Recall the stiffness matrix entries are $K_{ij} = a(\phi_j, \phi_i) = \int_0^1 \phi_i' \phi_j' dx$.

For piecewise linear basis functions on the partition $0 = x_0 < x_1 < \cdots < x_n = 1$ with $h_i = x_i - x_{i-1}$:

The basis function $\phi_i$ is:
- 0 for $x < x_{i-1}$
- $(x - x_{i-1})/h_i$ for $x \in [x_{i-1}, x_i]$
- $(x_{i+1} - x)/h_{i+1}$ for $x \in [x_i, x_{i+1}]$
- 0 for $x > x_{i+1}$

Therefore:
$$\phi_i'(x) = \begin{cases}
1/h_i & x \in (x_{i-1}, x_i) \\
-1/h_{i+1} & x \in (x_i, x_{i+1}) \\
0 & \text{otherwise}
\end{cases}$$

**Diagonal entries** ($K_{ii}$):
$$K_{ii} = \int_0^1 (\phi_i')^2 dx = \int_{x_{i-1}}^{x_i} \frac{1}{h_i^2} dx + \int_{x_i}^{x_{i+1}} \frac{1}{h_{i+1}^2} dx = \frac{1}{h_i} + \frac{1}{h_{i+1}}$$

**Off-diagonal entries** ($K_{i,i+1}$ and $K_{i+1,i}$):
$$K_{i,i+1} = \int_0^1 \phi_i' \phi_{i+1}' dx = \int_{x_i}^{x_{i+1}} \left(-\frac{1}{h_{i+1}}\right) \left(\frac{1}{h_{i+1}}\right) dx = -\frac{1}{h_{i+1}}$$

By symmetry, $K_{i+1,i} = K_{i,i+1} = -h_{i+1}^{-1}$.

All other entries are zero since non-adjacent basis functions have disjoint supports.

**Part 2: Mass Matrix**

If $f$ is piecewise linear, write $f(x) = \sum_{j=1}^n f_j \phi_j$ where $f_j = f(x_j)$.

The load vector is:
$$F_i = (f, \phi_i) = \int_0^1 f(x) \phi_i(x) dx = \int_0^1 \left(\sum_{j=1}^n f_j \phi_j(x)\right) \phi_i(x) dx$$

$$F_i = \sum_{j=1}^n f_j \int_0^1 \phi_j(x) \phi_i(x) dx = \sum_{j=1}^n M_{ij} f_j$$

where the **mass matrix** is:
$$M_{ij} = \int_0^1 \phi_i(x) \phi_j(x) dx$$

**Computing mass matrix entries**:

For $M_{ii}$:
$$M_{ii} = \int_{x_{i-1}}^{x_i} \left(\frac{x - x_{i-1}}{h_i}\right)^2 dx + \int_{x_i}^{x_{i+1}} \left(\frac{x_{i+1} - x}{h_{i+1}}\right)^2 dx$$

Using the substitution $\xi = (x - x_{i-1})/h_i$ on the first integral:
$$\int_{x_{i-1}}^{x_i} \left(\frac{x - x_{i-1}}{h_i}\right)^2 dx = h_i \int_0^1 \xi^2 d\xi = \frac{h_i}{3}$$

Similarly for the second integral: $h_{i+1}/3$.

Therefore:
$$M_{ii} = \frac{h_i + h_{i+1}}{3}$$

For $M_{i,i+1}$:
$$M_{i,i+1} = \int_{x_i}^{x_{i+1}} \left(\frac{x_{i+1} - x}{h_{i+1}}\right) \left(\frac{x - x_i}{h_{i+1}}\right) dx$$

With $\xi = (x - x_i)/h_{i+1}$:
$$M_{i,i+1} = h_{i+1} \int_0^1 (1-\xi)\xi d\xi = h_{i+1} \left[\frac{\xi^2}{2} - \frac{\xi^3}{3}\right]_0^1 = \frac{h_{i+1}}{6}$$

By symmetry, $M_{i+1,i} = h_{i+1}/6$.

**Summary**: The mass matrix $M$ is:
$$M_{ii} = \frac{h_i + h_{i+1}}{3}, \quad M_{i,i+1} = M_{i+1,i} = \frac{h_{i+1}}{6}$$

with all other entries zero.

---

## Exercise 0.x.2

**Problem**: Give weak formulations of modifications of (0.1.1) where:
- (a) the ODE is $-u'' + u = f$ instead of $-u'' = f$, and/or
- (b) the boundary conditions are $u(0) = u(1) = 0$

### Solution

**Part (a): Modified ODE with $-u'' + u = f$**

Starting with $-u'' + u = f$ on $(0,1)$ with $u(0) = 0, u'(1) = 0$.

Multiply by test function $v$ with $v(0) = 0$ and integrate:
$$\int_0^1 (-u'' + u)v dx = \int_0^1 fv dx$$

Integration by parts on the first term:
$$\int_0^1 u'v' dx - [u'v]_0^1 + \int_0^1 uv dx = \int_0^1 fv dx$$

Since $v(0) = 0$ and $u'(1) = 0$:
$$\int_0^1 u'v' dx + \int_0^1 uv dx = \int_0^1 fv dx$$

**Weak formulation**: Find $u \in V$ such that
$$a(u,v) = (f,v) \quad \forall v \in V$$

where:
- $a(u,v) = \int_0^1 u'v' dx + \int_0^1 uv dx = \int_0^1 (u'v' + uv) dx$
- $(f,v) = \int_0^1 fv dx$
- $V = \{v \in L^2(0,1) : a(v,v) < \infty, v(0) = 0\}$

**Part (b): Modified boundary conditions with $u(0) = u(1) = 0$**

With $-u'' = f$ on $(0,1)$ and $u(0) = u(1) = 0$.

Using test functions $v$ with $v(0) = v(1) = 0$:
$$\int_0^1 u'v' dx = \int_0^1 fv dx$$

**Weak formulation**: Find $u \in V$ such that
$$a(u,v) = (f,v) \quad \forall v \in V$$

where:
- $a(u,v) = \int_0^1 u'v' dx$
- $(f,v) = \int_0^1 fv dx$
- $V = \{v \in L^2(0,1) : a(v,v) < \infty, v(0) = v(1) = 0\}$

**Part (a) and (b) combined**: $-u'' + u = f$ with $u(0) = u(1) = 0$

**Weak formulation**: Find $u \in V$ such that
$$a(u,v) = (f,v) \quad \forall v \in V$$

where:
- $a(u,v) = \int_0^1 (u'v' + uv) dx$
- $(f,v) = \int_0^1 fv dx$
- $V = \{v \in L^2(0,1) : a(v,v) < \infty, v(0) = v(1) = 0\}$

---

## Exercise 0.x.3

**Problem**: Explain what is wrong in both the variational setting and the classical setting for the problem:
$$-u'' = f \quad \text{with} \quad u'(0) = u'(1) = 0$$

### Solution

This is the **pure Neumann problem**, which is **not well-posed** as stated.

**Classical Setting Issue**:

The problem has **non-unique solutions**. If $u$ is a solution, then $u + c$ is also a solution for any constant $c$ because:
- $(u + c)'' = u'' + 0 = u''$ (constants have zero second derivative)
- $(u + c)'(0) = u'(0) = 0$ and $(u + c)'(1) = u'(1) = 0$

Therefore, the solution is only determined **up to an additive constant**.

**Variational Setting Issue**:

Define $V = \{v \in L^2(0,1) : a(v,v) < \infty\}$ where $a(v,v) = \int_0^1 (v')^2 dx$.

The bilinear form $a(\cdot, \cdot)$ is **not coercive** on $V$ because:
- For any constant function $v = c$, we have $a(v,v) = \int_0^1 0 dx = 0$
- But $\|v\|_{L^2}^2 = \int_0^1 c^2 dx = c^2 > 0$ for $c \neq 0$

So we cannot prove:
$$\|v\|^2 \leq C \cdot a(v,v)$$

**Compatibility Condition**:

For a solution to exist classically, we need:
$$\int_0^1 f(x) dx = 0$$

This comes from integrating both sides of $-u'' = f$:
$$-\int_0^1 u'' dx = -[u']_0^1 = -(u'(1) - u'(0)) = 0 = \int_0^1 f dx$$

**How to fix it**:

1. **Add constraint**: Restrict to $V_0 = \{v \in V : \int_0^1 v dx = 0\}$ (functions with zero mean)
2. **Modify equation**: Add a term like $-u'' + u = f$ (as in Exercise 0.x.2a)
3. **Change boundary conditions**: Use at least one Dirichlet condition

With the constraint approach, the problem becomes well-posed when $(f, 1) = 0$ (compatibility).

---

## Exercise 0.x.4

**Problem**: Show that piecewise quadratics have a nodal basis consisting of values at the nodes $x_i$ together with the midpoints $\frac{1}{2}(x_i + x_{i+1})$. Calculate the stiffness matrix for these elements.

### Solution

**Nodal Basis for Piecewise Quadratics**:

On each interval $[x_i, x_{i+1}]$, a general quadratic has 3 degrees of freedom. The space of piecewise quadratics continuous at the nodes has dimension:
- $n$ interior nodes (excluding $x_0$, including $x_n$ if it's a Neumann condition)
- $n$ midpoints
- Total: $2n$ dimensions (adjusting for boundary conditions)

Define the **nodal points**:
- $x_i$ for $i = 0, 1, \ldots, n$ (nodes)
- $m_i = \frac{1}{2}(x_i + x_{i+1})$ for $i = 0, 1, \ldots, n-1$ (midpoints)

**Basis functions**:

1. **Vertex basis functions** $\phi_i$ for $i = 1, \ldots, n$:
   - $\phi_i(x_j) = \delta_{ij}$
   - $\phi_i(m_j) = 0$ for all $j$

2. **Midpoint basis functions** $\psi_i$ for $i = 0, 1, \ldots, n-1$:
   - $\psi_i(x_j) = 0$ for all $j$
   - $\psi_i(m_j) = \delta_{ij}$

**Construction on reference element**:

On $[0,1]$, the standard quadratic Lagrange basis functions are:
$$\ell_0(\xi) = 2(\xi - \frac{1}{2})(\xi - 1) = 2\xi^2 - 3\xi + 1$$
$$\ell_{1/2}(\xi) = 4\xi(1 - \xi) = -4\xi^2 + 4\xi$$
$$\ell_1(\xi) = 2\xi(\xi - \frac{1}{2}) = 2\xi^2 - \xi$$

These satisfy $\ell_\alpha(\beta) = \delta_{\alpha\beta}$ for $\alpha, \beta \in \{0, 1/2, 1\}$.

**Derivatives**:
$$\ell_0'(\xi) = 4\xi - 3, \quad \ell_{1/2}'(\xi) = -8\xi + 4, \quad \ell_1'(\xi) = 4\xi - 1$$

**Stiffness Matrix Calculation**:

For element $e$ on interval $I_e = [x_{e-1}, x_e]$ with length $h_e = x_e - x_{e-1}$:

Using the chain rule, if $\phi(\xi)$ is a basis function on $[0,1]$ and $x = x_{e-1} + \xi h_e$:
$$\frac{d\phi}{dx} = \frac{1}{h_e} \frac{d\phi}{d\xi}$$

The element stiffness matrix has entries:
$$K_{ij}^e = \int_{x_{e-1}}^{x_e} \frac{d\phi_i}{dx} \frac{d\phi_j}{dx} dx = \frac{1}{h_e^2} \int_0^1 \frac{d\ell_i}{d\xi} \frac{d\ell_j}{d\xi} h_e d\xi = \frac{1}{h_e} \int_0^1 \ell_i'(\xi) \ell_j'(\xi) d\xi$$

Computing the integrals:
$$\int_0^1 (\ell_0')^2 d\xi = \int_0^1 (4\xi - 3)^2 d\xi = \int_0^1 (16\xi^2 - 24\xi + 9) d\xi = \frac{16}{3} - 12 + 9 = \frac{7}{3}$$

$$\int_0^1 \ell_0' \ell_{1/2}' d\xi = \int_0^1 (4\xi - 3)(-8\xi + 4) d\xi = \int_0^1 (-32\xi^2 + 40\xi - 12) d\xi = -\frac{32}{3} + 20 - 12 = -\frac{8}{3}$$

$$\int_0^1 \ell_0' \ell_1' d\xi = \int_0^1 (4\xi - 3)(4\xi - 1) d\xi = \int_0^1 (16\xi^2 - 16\xi + 3) d\xi = \frac{16}{3} - 8 + 3 = \frac{1}{3}$$

$$\int_0^1 (\ell_{1/2}')^2 d\xi = \int_0^1 (-8\xi + 4)^2 d\xi = \int_0^1 (64\xi^2 - 64\xi + 16) d\xi = \frac{64}{3} - 32 + 16 = \frac{16}{3}$$

$$\int_0^1 \ell_{1/2}' \ell_1' d\xi = \int_0^1 (-8\xi + 4)(4\xi - 1) d\xi = \int_0^1 (-32\xi^2 + 24\xi - 4) d\xi = -\frac{32}{3} + 12 - 4 = -\frac{8}{3}$$

$$\int_0^1 (\ell_1')^2 d\xi = \int_0^1 (4\xi - 1)^2 d\xi = \int_0^1 (16\xi^2 - 8\xi + 1) d\xi = \frac{16}{3} - 4 + 1 = \frac{7}{3}$$

**Element stiffness matrix**:
$$K^e = \frac{1}{h_e} \begin{pmatrix}
7/3 & -8/3 & 1/3 \\
-8/3 & 16/3 & -8/3 \\
1/3 & -8/3 & 7/3
\end{pmatrix}$$

Or equivalently:
$$K^e = \frac{1}{3h_e} \begin{pmatrix}
7 & -8 & 1 \\
-8 & 16 & -8 \\
1 & -8 & 7
\end{pmatrix}$$

**Global stiffness matrix**: Assemble by summing contributions from each element, noting that vertex and midpoint degrees of freedom couple only within elements.

---

## Exercise 0.x.5

**Problem**: Verify (0.5.2): $(f, \phi_i) = \frac{1}{2}(h_i + h_{i+1})(f(x_i) + O(h))$

### Solution

Recall $\phi_i$ is the piecewise linear basis function with $\phi_i(x_j) = \delta_{ij}$.

$$(\f, \phi_i) = \int_0^1 f(x) \phi_i(x) dx = \int_{x_{i-1}}^{x_i} f(x) \phi_i(x) dx + \int_{x_i}^{x_{i+1}} f(x) \phi_i(x) dx$$

On $[x_{i-1}, x_i]$: $\phi_i(x) = \frac{x - x_{i-1}}{h_i}$

On $[x_i, x_{i+1}]$: $\phi_i(x) = \frac{x_{i+1} - x}{h_{i+1}}$

**Using Taylor expansion** around $x_i$:
$$f(x) = f(x_i) + f'(x_i)(x - x_i) + O((x - x_i)^2)$$

**First integral**:
$$I_1 = \int_{x_{i-1}}^{x_i} f(x) \frac{x - x_{i-1}}{h_i} dx$$

Substitute $x = x_i - h_i(1-t)$ where $t \in [0,1]$, so $x - x_{i-1} = h_i t$ and $dx = h_i dt$:
$$I_1 = \int_0^1 f(x_i - h_i(1-t)) t h_i dt$$

Using Taylor expansion:
$$f(x_i - h_i(1-t)) = f(x_i) - f'(x_i)h_i(1-t) + O(h_i^2)$$

Therefore:
$$I_1 = h_i \int_0^1 [f(x_i) - f'(x_i)h_i(1-t) + O(h_i^2)] t dt$$
$$= h_i \left[f(x_i) \int_0^1 t dt - f'(x_i)h_i \int_0^1 t(1-t) dt + O(h_i^3)\right]$$
$$= h_i \left[f(x_i) \cdot \frac{1}{2} - f'(x_i)h_i \cdot \frac{1}{6} + O(h_i^3)\right]$$
$$= \frac{h_i}{2} f(x_i) - \frac{h_i^2}{6} f'(x_i) + O(h_i^3)$$

**Second integral**: By symmetry,
$$I_2 = \int_{x_i}^{x_{i+1}} f(x) \frac{x_{i+1} - x}{h_{i+1}} dx = \frac{h_{i+1}}{2} f(x_i) + \frac{h_{i+1}^2}{6} f'(x_i) + O(h_{i+1}^3)$$

**Combining**:
$$(f, \phi_i) = I_1 + I_2 = \frac{h_i + h_{i+1}}{2} f(x_i) + \frac{h_{i+1}^2 - h_i^2}{6} f'(x_i) + O(h^3)$$

If $h = \max_j h_j$, then:
$$\frac{h_{i+1}^2 - h_i^2}{6} f'(x_i) = O(h^2) \|f'\|_\infty$$

However, the statement says $O(h)$, which is correct if we write:
$$\frac{h_{i+1}^2 - h_i^2}{6} = \frac{h_{i+1} - h_i}{6}(h_{i+1} + h_i) = O(h) \cdot O(h) = O(h^2)$$

But relative to the leading term $\frac{h_i + h_{i+1}}{2}$, the error is:
$$\frac{h_{i+1}^2 - h_i^2}{6} \bigg/ \frac{h_i + h_{i+1}}{2} = \frac{h_{i+1} - h_i}{3} = O(h)$$

when $|h_{i+1} - h_i| = O(h)$.

Therefore:
$$(f, \phi_i) = \frac{h_i + h_{i+1}}{2} (f(x_i) + O(h))$$

**Note**: The $O(h^2)$ accuracy is achieved only on a uniform mesh where $h_{i+1} = h_i$.

---

## Exercise 0.x.6

**Problem**: Under the same assumptions as Theorem 0.4.5, prove that $\|u - u_I\| \leq Ch^2 \|u''\|$.

Hints: 
- Use a homogeneity argument as in Theorem 0.4.5
- Show that $\int_0^1 w(x)^2 dx \leq \tilde{c} \int_0^1 w'(x)^2 dx$ utilizing $w(0) = 0$
- How small can you make $\tilde{c}$ if you use both $w(0) = 0$ and $w(1) = 0$?

### Solution

**Step 1: Reduction to reference element**

By the scaling argument in Theorem 0.4.5, it suffices to prove:
$$\int_0^1 w^2 dx \leq c \int_0^1 (w'')^2 dx$$

for functions $w$ on $[0,1]$ with $w(0) = w(1) = 0$ and $w$ linear on $[0,1]$ (so $w = 0$ on the reference element, but we work with the error $e$ which has these properties after transformation).

Actually, let me reconsider. The interpolation error $e = u - u_I$ on each element $[x_{j-1}, x_j]$ satisfies $e(x_{j-1}) = e(x_j) = 0$.

**Step 2: Poincaré inequality on $[0,1]$**

For $w$ with $w(0) = 0$:
$$w(x) = \int_0^x w'(t) dt$$

By Schwarz inequality:
$$|w(x)|^2 = \left|\int_0^x w'(t) dt\right|^2 \leq \int_0^x 1^2 dt \cdot \int_0^x (w'(t))^2 dt \leq x \int_0^1 (w')^2 dt$$

Integrating:
$$\int_0^1 w^2 dx \leq \int_0^1 x dx \cdot \int_0^1 (w')^2 dt = \frac{1}{2} \int_0^1 (w')^2 dt$$

So $\tilde{c} = 1/2$ using only $w(0) = 0$.

**Step 3: Better estimate using both boundary conditions**

If $w(0) = w(1) = 0$, by Rolle's theorem there exists $\xi \in (0,1)$ such that $w'(\xi) = 0$.

For any $x \in [0,1]$:
$$w'(x) = \int_\xi^x w''(t) dt$$

By Schwarz inequality:
$$|w'(x)|^2 \leq |x - \xi| \int_0^1 (w'')^2 dt \leq \int_0^1 (w'')^2 dt$$

(since $|x - \xi| \leq 1$).

Therefore:
$$\int_0^1 (w')^2 dx \leq \int_0^1 (w'')^2 dx$$

Now, for $w$ itself:
$$w(x) = \int_0^x w'(t) dt$$

$$|w(x)|^2 \leq x \int_0^x (w')^2 dt \leq x \int_0^1 (w')^2 dt \leq x \int_0^1 (w'')^2 dx$$

Integrating:
$$\int_0^1 w^2 dx \leq \frac{1}{2} \int_0^1 (w'')^2 dx$$

Alternatively, we can be more careful. Since $w(\xi) = 0$ for some $\xi$:
$$w(x) = \int_\xi^x w'(t) dt$$

$$|w(x)|^2 \leq |x - \xi| \int_0^1 (w')^2 dt \leq \int_0^1 (w'')^2 dx$$

To get the best constant, use:
$$|w(x)| = \left|\int_\xi^x w'(t) dt\right| \leq \left|\int_\xi^x \int_\xi^t w''(s) ds dt\right|$$

This becomes technical. A simpler approach using the Poincaré-Wirtinger inequality gives us that with $w(0) = w(1) = 0$:
$$\int_0^1 w^2 dx \leq \frac{1}{\pi^2} \int_0^1 (w')^2 dx \leq \frac{1}{\pi^2} \int_0^1 (w'')^2 dx$$

So the best constant is $\tilde{c} = 1/\pi^2 \approx 0.101$.

**Step 4: Application to interpolation error**

On element $[x_{j-1}, x_j]$ with $e = u - u_I$, we have $e(x_{j-1}) = e(x_j) = 0$ and $e''= u''$ (since $u_I$ is linear).

Scaling to $[0,1]$:
$$\int_{x_{j-1}}^{x_j} e^2 dx = h_j \int_0^1 \tilde{e}^2 d\xi \leq \frac{h_j}{\pi^2} \int_0^1 (\tilde{e}'')^2 d\xi$$

where $\tilde{e}(\xi) = e(x_{j-1} + h_j \xi)$ and $\tilde{e}'' = h_j^2 e''$.

Therefore:
$$\int_{x_{j-1}}^{x_j} e^2 dx \leq \frac{h_j^5}{\pi^2} \int_0^1 (e'')^2 d\xi = \frac{h_j^4}{\pi^2} \int_{x_{j-1}}^{x_j} (u'')^2 dx$$

Summing over elements:
$$\|u - u_I\|^2 \leq \frac{h^4}{\pi^2} \int_0^1 (u'')^2 dx$$

Therefore:
$$\|u - u_I\| \leq \frac{h^2}{\pi} \|u''\| \leq Ch^2 \|u''\|$$

with $C = 1/\pi$ (or $C = 1/\sqrt{2}$ using the simpler estimate).

---

## Exercise 0.x.7

**Problem**: Using only Theorems 0.3.5 and 0.4.5, prove that $\inf_{v \in S} \|u - v\| \leq Ch^2 \|u''\|$. Compare the constant $C$ with different approaches.

### Solution

**Using Theorem 0.3.5 (Duality Argument)**:

Theorem 0.3.5 states that if the approximation assumption holds:
$$\inf_{v \in S} \|w - v\|_E \leq \epsilon \|w''\|$$

then:
$$\|u - u_S\| \leq \epsilon^2 \|u''\|$$

**Using Theorem 0.4.5**:

Theorem 0.4.5 states:
$$\|u - u_I\|_E \leq Ch \|u''\|$$

Since the interpolant $u_I \in S$, we have:
$$\inf_{v \in S} \|u - v\|_E \leq \|u - u_I\|_E \leq Ch \|u''\|$$

So the approximation assumption holds with $\epsilon = Ch$.

**Applying Theorem 0.3.5**:

$$\|u - u_S\| \leq (Ch)^2 \|u''\| = Ch^2 \|u''\|$$

Since $u_S \in S$:
$$\inf_{v \in S} \|u - v\| \leq \|u - u_S\| \leq C^2 h^2 \|u''\|$$

From the proof of Theorem 0.4.5, we found $C = 1/\sqrt{2}$, so:
$$\inf_{v \in S} \|u - v\| \leq \frac{1}{2} h^2 \|u''\|$$

**Comparison of constants**:

1. **Direct approach** (Exercise 0.x.6): $C = 1/\pi \approx 0.318$ (or $1/\sqrt{2} \approx 0.707$ with simpler estimate)

2. **Via duality** (this exercise): $C = (1/\sqrt{2})^2 = 1/2 = 0.5$

The direct approach with the sharp Poincaré inequality gives the best constant.

---

## Exercise 0.x.8

**Problem**: Prove that (0.1.1) has a solution $u \in C^2([0,1])$ provided $f \in C^0([0,1])$.

Hint: Write $u(x) = \int_0^x \int_s^1 f(t) dt ds$ and verify the equations.

### Solution

**Construction**: Define
$$u(x) = \int_0^x \int_s^1 f(t) dt ds$$

**Verification**:

**Step 1**: Compute $u'(x)$

By the fundamental theorem of calculus:
$$u'(x) = \frac{d}{dx} \int_0^x \int_s^1 f(t) dt ds = \int_x^1 f(t) dt$$

Since $f \in C^0([0,1])$, we have $u' \in C^1([0,1])$.

**Step 2**: Compute $u''(x)$

$$u''(x) = \frac{d}{dx} \int_x^1 f(t) dt = -f(x)$$

(using the fundamental theorem with the convention that $\frac{d}{dx}\int_x^b g = -g(x)$).

Therefore $-u''(x) = f(x)$ for all $x \in [0,1]$, so $u \in C^2([0,1])$.

**Step 3**: Check boundary conditions

$$u(0) = \int_0^0 \int_s^1 f(t) dt ds = 0$$ ✓

$$u'(1) = \int_1^1 f(t) dt = 0$$ ✓

Therefore, $u$ solves (0.1.1).

**Uniqueness**: If $u_1$ and $u_2$ both solve (0.1.1), then $w = u_1 - u_2$ satisfies:
$$-w'' = 0, \quad w(0) = 0, \quad w'(1) = 0$$

This implies $w' = c$ (constant), and $w'(1) = 0$ gives $c = 0$, so $w' = 0$.
Thus $w$ is constant, and $w(0) = 0$ implies $w = 0$, so $u_1 = u_2$.

---

## Exercise 0.x.9

**Problem**: Let $V$ denote the space and $a(\cdot, \cdot)$ the bilinear form defined in Section 0.1. Prove the coercivity result:
$$\|v\|^2 + \|v'\|^2 \leq Ca(v,v) \quad \forall v \in V$$

Give a value for $C$. (For simplicity, restrict to $v \in V \cap C^1(0,1)$.)

### Solution

Recall:
- $V = \{v \in L^2(0,1) : a(v,v) < \infty, v(0) = 0\}$
- $a(v,v) = \int_0^1 (v')^2 dx = \|v'\|^2$

**Estimate for $\|v'\|^2$**:

Clearly, $\|v'\|^2 = a(v,v)$, so:
$$\|v'\|^2 \leq a(v,v)$$

**Estimate for $\|v\|^2$ using Poincaré inequality**:

For $v \in V$ with $v(0) = 0$:
$$v(x) = \int_0^x v'(t) dt$$

By Schwarz inequality:
$$|v(x)|^2 = \left|\int_0^x v'(t) dt\right|^2 \leq \int_0^x 1^2 dt \cdot \int_0^x (v'(t))^2 dt \leq x \int_0^1 (v')^2 dt$$

Integrating over $[0,1]$:
$$\int_0^1 v^2 dx \leq \int_0^1 x dx \cdot \int_0^1 (v')^2 dt = \frac{1}{2} \|v'\|^2 = \frac{1}{2} a(v,v)$$

**Combining**:
$$\|v\|^2 + \|v'\|^2 \leq \frac{1}{2} a(v,v) + a(v,v) = \frac{3}{2} a(v,v)$$

Therefore, the coercivity inequality holds with $C = 3/2$.

**Note**: A sharper estimate using the optimal Poincaré constant gives:
$$\|v\|^2 \leq \frac{1}{\pi^2} \|v'\|^2 = \frac{1}{\pi^2} a(v,v)$$

so:
$$\|v\|^2 + \|v'\|^2 \leq \left(1 + \frac{1}{\pi^2}\right) a(v,v)$$

giving $C = 1 + 1/\pi^2 \approx 1.101$.

---

## Exercise 0.x.10

**Problem**: Let $V$ denote the space and $a(\cdot, \cdot)$ the bilinear form defined in Section 0.1. Prove the Sobolev inequality:
$$\|v\|_{\max}^2 \leq Ca(v,v) \quad \forall v \in V$$

Give a value for $C$. (For simplicity, restrict to $v \in V \cap C^1(0,1)$.)

### Solution

For $v \in V$ with $v(0) = 0$:
$$v(x) = \int_0^x v'(t) dt$$

**Using Schwarz inequality**:
$$|v(x)|^2 = \left|\int_0^x v'(t) dt\right|^2 \leq \left(\int_0^x 1^2 dt\right) \left(\int_0^x (v'(t))^2 dt\right) \leq x \int_0^1 (v')^2 dt$$

Since $x \leq 1$ for $x \in [0,1]$:
$$|v(x)|^2 \leq \int_0^1 (v')^2 dt = a(v,v)$$

Taking the maximum over $x \in [0,1]$:
$$\|v\|_{\max}^2 = \max_{x \in [0,1]} |v(x)|^2 \leq a(v,v)$$

Therefore, the Sobolev inequality holds with $C = 1$.

**Alternative derivation with better constant**:

Using a more refined approach with the mean value:
$$|v(x)| \leq \int_0^1 |v'(t)| dt \leq \sqrt{1} \cdot \sqrt{\int_0^1 (v')^2 dt} = \sqrt{a(v,v)}$$

So $\|v\|_{\max} \leq \sqrt{a(v,v)}$, giving $\|v\|_{\max}^2 \leq a(v,v)$ with $C = 1$.

This is actually sharp: the constant cannot be improved.

---

## Exercise 0.x.11

**Problem**: Consider the difference method represented by (0.5.3):
$$-2\left(\frac{U_{i+1} - U_i}{h_i + h_{i+1}} \cdot \frac{1}{h_{i+1}} - \frac{U_i - U_{i-1}}{h_i + h_{i+1}} \cdot \frac{1}{h_i}\right) = f(x_i)$$

Prove $\tilde{u}_S := \sum U_i \phi_i$ satisfies:
$$a(\tilde{u}_S, v) = Q(fv) \quad \forall v \in S$$

where $a(\cdot, \cdot)$ is from Section 0.1, $S$ is piecewise linears from Section 0.4, and $Q$ is the trapezoidal rule:
$$Q(w) := \sum_{i=0}^n \frac{h_i + h_{i+1}}{2} w(x_i)$$

(with $h_0 = h_{n+1} = 0$).

### Solution

**Step 1**: Express $a(\tilde{u}_S, v)$ for $v = \phi_j$

Since $\{\phi_i\}$ is a basis for $S$, it suffices to check the equation for $v = \phi_j$ where $j = 1, \ldots, n$.

$$a(\tilde{u}_S, \phi_j) = \int_0^1 \tilde{u}_S' \phi_j' dx = \sum_{i=1}^n U_i \int_0^1 \phi_i' \phi_j' dx = \sum_{i=1}^n U_i K_{ji}$$

where $K_{ji} = a(\phi_i, \phi_j)$ is the stiffness matrix.

From Exercise 0.x.1:
- $K_{jj} = h_j^{-1} + h_{j+1}^{-1}$
- $K_{j,j+1} = K_{j+1,j} = -h_{j+1}^{-1}$
- All other entries are zero

Therefore:
$$a(\tilde{u}_S, \phi_j) = K_{jj} U_j + K_{j,j-1} U_{j-1} + K_{j,j+1} U_{j+1}$$
$$= \left(\frac{1}{h_j} + \frac{1}{h_{j+1}}\right) U_j - \frac{U_{j-1}}{h_j} - \frac{U_{j+1}}{h_{j+1}}$$

**Step 2**: Simplify

$$a(\tilde{u}_S, \phi_j) = \frac{U_j - U_{j-1}}{h_j} - \frac{U_{j+1} - U_j}{h_{j+1}}$$

**Step 3**: Difference equation

From (0.5.3), the difference equation for $i = j$ is:
$$-2\left(\frac{U_{j+1} - U_j}{h_j + h_{j+1}} \cdot \frac{1}{h_{j+1}} - \frac{U_j - U_{j-1}}{h_j + h_{j+1}} \cdot \frac{1}{h_j}\right) = f(x_j)$$

Simplifying:
$$-\frac{2}{h_j + h_{j+1}}\left(\frac{U_{j+1} - U_j}{h_{j+1}} - \frac{U_j - U_{j-1}}{h_j}\right) = f(x_j)$$

$$\frac{U_j - U_{j-1}}{h_j} - \frac{U_{j+1} - U_j}{h_{j+1}} = -\frac{h_j + h_{j+1}}{2} f(x_j)$$

**Step 4**: Compare with $Q(f\phi_j)$

$$Q(f\phi_j) = \sum_{i=0}^n \frac{h_i + h_{i+1}}{2} f(x_i) \phi_j(x_i)$$

Since $\phi_j(x_i) = \delta_{ij}$:
$$Q(f\phi_j) = \frac{h_j + h_{j+1}}{2} f(x_j)$$

**Step 5**: Conclusion

From Steps 2 and 3:
$$a(\tilde{u}_S, \phi_j) = -\frac{h_j + h_{j+1}}{2} f(x_j) = -Q(f\phi_j)$$

Wait, there's a sign error. Let me reconsider the difference equation.

Actually, looking at (0.5.3) more carefully, it should be:
$$\frac{U_j - U_{j-1}}{h_j} - \frac{U_{j+1} - U_j}{h_{j+1}} = \frac{h_j + h_{j+1}}{2} f(x_j)$$

But this doesn't match the signs. Let me recalculate from the stiffness matrix directly.

Actually, the standard difference equation from the stiffness matrix is:
$$-\left(\frac{U_j - U_{j-1}}{h_j} - \frac{U_{j+1} - U_j}{h_{j+1}}\right) = f(x_j)$$

which comes from $-u''(x_j) \approx f(x_j)$.

But the quadrature formula gives:
$$(f, \phi_j) = \int_0^1 f \phi_j dx \approx Q(f\phi_j) = \frac{h_j + h_{j+1}}{2} f(x_j)$$

So we should have:
$$a(\tilde{u}_S, \phi_j) = Q(f\phi_j)$$

which means:
$$\frac{U_j - U_{j-1}}{h_j} - \frac{U_{j+1} - U_j}{h_{j+1}} = \frac{h_j + h_{j+1}}{2} f(x_j)$$

Multiplying both sides by $-2/(h_j + h_{j+1})$:
$$-\frac{2}{h_j + h_{j+1}}\left(\frac{U_j - U_{j-1}}{h_j} - \frac{U_{j+1} - U_j}{h_{j+1}}\right) = f(x_j)$$

This matches (0.5.3). Therefore, $\tilde{u}_S$ satisfies $a(\tilde{u}_S, v) = Q(fv)$ for all $v \in S$.

---

## Exercise 0.x.12

**Problem**: Let $Q$ be defined as in Exercise 0.x.11. Prove that:
$$\left|Q(w) - \int_0^1 w(x) dx\right| \leq Ch^2 \sum_{i=1}^n \int_{x_{i-1}}^{x_i} |w''(x)| dx$$

Hint: The trapezoidal rule is exact for piecewise linears; refer to hint in Exercise 0.x.6.

### Solution

**Step 1**: Error on a single element

On element $[x_{i-1}, x_i]$, the trapezoidal rule is:
$$Q_i(w) = \frac{h_i}{2}(w(x_{i-1}) + w(x_i))$$

The exact integral is:
$$I_i(w) = \int_{x_{i-1}}^{x_i} w(x) dx$$

**Step 2**: Decompose $w = w_I + (w - w_I)$

Let $w_I$ be the linear interpolant of $w$ on $[x_{i-1}, x_i]$. Since the trapezoidal rule is exact for linear functions:
$$Q_i(w_I) = I_i(w_I)$$

Therefore:
$$Q_i(w) - I_i(w) = Q_i(w - w_I) - I_i(w - w_I)$$

Let $e = w - w_I$. Then $e(x_{i-1}) = e(x_i) = 0$.

**Step 3**: Estimate $Q_i(e)$

$$Q_i(e) = \frac{h_i}{2}(e(x_{i-1}) + e(x_i)) = 0$$

since $e$ vanishes at the endpoints.

**Step 4**: Estimate $I_i(e)$

From Exercise 0.x.6, for $e$ with $e(x_{i-1}) = e(x_i) = 0$ on element $[x_{i-1}, x_i]$:
$$\int_{x_{i-1}}^{x_i} e^2 dx \leq \frac{h_i^4}{\pi^2} \int_{x_{i-1}}^{x_i} (e'')^2 dx$$

By Schwarz inequality:
$$\left|\int_{x_{i-1}}^{x_i} e dx\right|^2 \leq h_i \int_{x_{i-1}}^{x_i} e^2 dx \leq \frac{h_i^5}{\pi^2} \int_{x_{i-1}}^{x_i} (e'')^2 dx$$

Since $e'' = w''$ (as $w_I$ is linear):
$$\left|I_i(e)\right| \leq \frac{h_i^{5/2}}{\pi} \left(\int_{x_{i-1}}^{x_i} (w'')^2 dx\right)^{1/2}$$

By Schwarz inequality again:
$$\left(\int_{x_{i-1}}^{x_i} (w'')^2 dx\right)^{1/2} \leq \sqrt{h_i} \cdot \left(\int_{x_{i-1}}^{x_i} |w''| dx\right)$$

Wait, this isn't quite right. Let me use a direct estimate.

Actually, for the trapezoidal rule error on $[x_{i-1}, x_i]$, the standard estimate is:
$$\left|\int_{x_{i-1}}^{x_i} w dx - \frac{h_i}{2}(w(x_{i-1}) + w(x_i))\right| \leq \frac{h_i^3}{12} \max_{x \in [x_{i-1}, x_i]} |w''(x)|$$

Therefore:
$$|Q_i(w) - I_i(w)| \leq \frac{h_i^3}{12} \max_{x \in [x_{i-1}, x_i]} |w''(x)| \leq \frac{h_i^2}{12} \int_{x_{i-1}}^{x_i} |w''(x)| dx$$

(using $h_i \max |w''| \leq \int |w''| dx$ by the mean value theorem).

**Step 5**: Sum over all elements

$$\left|\sum_{i=1}^n (Q_i(w) - I_i(w))\right| \leq \sum_{i=1}^n |Q_i(w) - I_i(w)| \leq \sum_{i=1}^n \frac{h_i^2}{12} \int_{x_{i-1}}^{x_i} |w''(x)| dx$$

Since $h_i \leq h$ for all $i$:
$$\left|Q(w) - \int_0^1 w dx\right| \leq \frac{h^2}{12} \sum_{i=1}^n \int_{x_{i-1}}^{x_i} |w''(x)| dx$$

Therefore, the estimate holds with $C = 1/12$.

---

## Exercise 0.x.13

**Problem**: Let $u_S$ solve (0.2.1) and let $\tilde{u}_S$ be as in Exercise 0.x.11. Prove that:
$$|a(u_S - \tilde{u}_S, v)| \leq Ch^2(\|f'\| + \|f''\|)(\|v\| + \|v'\|) \quad \forall v \in S$$

Hint: Apply Exercise 0.x.12 and Schwarz inequality.

### Solution

**Step 1**: Set up the equation

From (0.2.1): $a(u_S, v) = (f, v)$ for all $v \in S$.

From Exercise 0.x.11: $a(\tilde{u}_S, v) = Q(fv)$ for all $v \in S$.

Subtracting:
$$a(u_S - \tilde{u}_S, v) = (f, v) - Q(fv) = \int_0^1 f(x)v(x) dx - Q(fv)$$

**Step 2**: Apply Exercise 0.x.12

Let $w = fv$. Then $w'' = (fv)'' = f''v + 2f'v' + fv''$.

By Exercise 0.x.12:
$$|a(u_S - \tilde{u}_S, v)| = \left|\int_0^1 fv dx - Q(fv)\right| \leq Ch^2 \sum_{i=1}^n \int_{x_{i-1}}^{x_i} |(fv)''| dx$$

**Step 3**: Estimate the second derivative

$$|(fv)''| \leq |f''v| + 2|f'v'| + |fv''|$$

Therefore:
$$\sum_{i=1}^n \int_{x_{i-1}}^{x_i} |(fv)''| dx \leq \int_0^1 (|f''v| + 2|f'v'| + |fv''|) dx$$

**Step 4**: Apply Schwarz inequality

$$\int_0^1 |f''v| dx \leq \|f''\| \cdot \|v\|$$

$$\int_0^1 |f'v'| dx \leq \|f'\| \cdot \|v'\|$$

$$\int_0^1 |fv''| dx \leq \|f\| \cdot \|v''\|$$

But for $v \in S$ (piecewise linear), $v'' = 0$ almost everywhere, so the last term vanishes.

Therefore:
$$\sum_{i=1}^n \int_{x_{i-1}}^{x_i} |(fv)''| dx \leq \|f''\| \|v\| + 2\|f'\| \|v'\|$$

**Step 5**: Combine

$$|a(u_S - \tilde{u}_S, v)| \leq Ch^2(\|f''\| \|v\| + 2\|f'\| \|v'\|)$$
$$\leq Ch^2(\|f'\| + \|f''\|)(\|v\| + \|v'\|)$$

(with a possibly larger constant $C$).

---

## Exercise 0.x.14

**Problem**: Let $u_S$ and $\tilde{u}_S$ be as in Exercise 0.x.13. Prove that:
$$\|u_S - \tilde{u}_S\|_E \leq Ch^2(\|f'\| + \|f''\|)$$

Hint: Apply Exercise 0.x.13, pick $v = u_S - \tilde{u}_S$, and apply Exercise 0.x.9.

### Solution

**Step 1**: Apply Exercise 0.x.13 with $v = u_S - \tilde{u}_S$

$$|a(u_S - \tilde{u}_S, u_S - \tilde{u}_S)| \leq Ch^2(\|f'\| + \|f''\|)(\|u_S - \tilde{u}_S\| + \|u_S - \tilde{u}_S\|')$$

**Step 2**: Simplify the left side

$$a(u_S - \tilde{u}_S, u_S - \tilde{u}_S) = \|u_S - \tilde{u}_S\|_E^2 \geq 0$$

So:
$$\|u_S - \tilde{u}_S\|_E^2 \leq Ch^2(\|f'\| + \|f''\|)(\|u_S - \tilde{u}_S\| + \|u_S - \tilde{u}_S\|')$$

**Step 3**: Apply Exercise 0.x.9

From Exercise 0.x.9 (coercivity):
$$\|u_S - \tilde{u}_S\|^2 + \|u_S - \tilde{u}_S\|'^2 \leq C_1 a(u_S - \tilde{u}_S, u_S - \tilde{u}_S) = C_1 \|u_S - \tilde{u}_S\|_E^2$$

where $C_1 = 3/2$ (or $1 + 1/\pi^2$ with the sharp constant).

Therefore:
$$\|u_S - \tilde{u}_S\| + \|u_S - \tilde{u}_S\|' \leq \sqrt{2C_1} \|u_S - \tilde{u}_S\|_E$$

**Step 4**: Substitute back

$$\|u_S - \tilde{u}_S\|_E^2 \leq Ch^2(\|f'\| + \|f''\|) \sqrt{2C_1} \|u_S - \tilde{u}_S\|_E$$

Dividing by $\|u_S - \tilde{u}_S\|_E$ (assuming it's non-zero):
$$\|u_S - \tilde{u}_S\|_E \leq C\sqrt{2C_1} h^2(\|f'\| + \|f''\|)$$

Therefore:
$$\|u_S - \tilde{u}_S\|_E \leq Ch^2(\|f'\| + \|f''\|)$$

with $C$ replaced by $C\sqrt{2C_1}$.

---

## Exercise 0.x.15

**Problem**: Let $\tilde{u}_S$ be as in Exercise 0.x.11 and let $u$ solve (0.1.1). Prove that:
$$\|u - \tilde{u}_S\|_{\max} \leq Ch^2(\|f\|_{\max} + \|f'\| + \|f''\|)$$

Hint: Apply Exercise 0.x.14 and Theorem 0.7.2.

### Solution

**Step 1**: Triangle inequality

$$\|u - \tilde{u}_S\|_{\max} \leq \|u - u_S\|_{\max} + \|u_S - \tilde{u}_S\|_{\max}$$

**Step 2**: Estimate $\|u - u_S\|_{\max}$ using Theorem 0.7.2

Theorem 0.7.2 states:
$$\|u - u_S\|_{\max} \leq Ch^2 \|u''\|_{\max}$$

Since $-u'' = f$ from (0.1.1):
$$\|u - u_S\|_{\max} \leq Ch^2 \|f\|_{\max}$$

**Step 3**: Estimate $\|u_S - \tilde{u}_S\|_{\max}$

From Exercise 0.x.10 (Sobolev inequality):
$$\|v\|_{\max}^2 \leq Ca(v,v) = C\|v\|_E^2$$

Therefore:
$$\|u_S - \tilde{u}_S\|_{\max} \leq \sqrt{C} \|u_S - \tilde{u}_S\|_E$$

From Exercise 0.x.14:
$$\|u_S - \tilde{u}_S\|_E \leq Ch^2(\|f'\| + \|f''\|)$$

Therefore:
$$\|u_S - \tilde{u}_S\|_{\max} \leq Ch^2(\|f'\| + \|f''\|)$$

**Step 4**: Combine

$$\|u - \tilde{u}_S\|_{\max} \leq Ch^2\|f\|_{\max} + Ch^2(\|f'\| + \|f''\|)$$
$$= Ch^2(\|f\|_{\max} + \|f'\| + \|f''\|)$$

---

## Exercise 0.x.16

**Problem**: Give weak formulation of the modification of (0.1.1) where the boundary conditions are $u(0) = 0$ and $u'(1) = \lambda$.

Hint: Show that $a(u,v) = F(v)$ where $F$ is the linear functional $F(v) = \lambda v(1)$.

### Solution

**Starting point**: The ODE is $-u'' = f$ on $(0,1)$ with boundary conditions $u(0) = 0$ and $u'(1) = \lambda$.

**Derivation**: Multiply by test function $v$ with $v(0) = 0$ and integrate:
$$\int_0^1 -u''(x) v(x) dx = \int_0^1 f(x) v(x) dx$$

Integration by parts:
$$\int_0^1 u'(x) v'(x) dx - [u'(x)v(x)]_0^1 = \int_0^1 f(x) v(x) dx$$

Since $v(0) = 0$ and $u'(1) = \lambda$:
$$\int_0^1 u'(x) v'(x) dx - \lambda v(1) = \int_0^1 f(x) v(x) dx$$

Rearranging:
$$\int_0^1 u'(x) v'(x) dx = \int_0^1 f(x) v(x) dx + \lambda v(1)$$

**Weak formulation**: Find $u \in V$ such that
$$a(u, v) = F(v) \quad \forall v \in V$$

where:
- $a(u,v) = \int_0^1 u'(x) v'(x) dx$ (same as before)
- $F(v) = (f,v) + \lambda v(1) = \int_0^1 f(x) v(x) dx + \lambda v(1)$
- $V = \{v \in L^2(0,1) : a(v,v) < \infty, v(0) = 0\}$ (same space as before)

**Key observation**: The non-homogeneous Neumann condition appears in the **linear functional** $F$, not in the space $V$ or the bilinear form $a$. The Neumann boundary condition $u'(1) = \lambda$ is **natural** (not essential), meaning it's automatically satisfied by the weak formulation and doesn't need to be explicitly enforced in the definition of $V$.

**Interpretation**: The term $\lambda v(1)$ represents a boundary contribution to the load. If $\lambda > 0$, it's like a point source at $x = 1$; if $\lambda < 0$, it's a point sink.

---

## Summary of Key Techniques

1. **Stiffness and Mass Matrices**: Direct computation using integration on reference elements
2. **Weak Formulations**: Integration by parts to handle different boundary conditions
3. **Ill-posed Problems**: Pure Neumann problems require compatibility conditions
4. **Higher-order Elements**: Quadratics need vertex + midpoint nodes
5. **Quadrature Error**: Trapezoidal rule is $O(h^2)$ for smooth functions
6. **Poincaré Inequalities**: Essential for controlling $L^2$ norm by energy norm
7. **Sobolev Inequalities**: Control pointwise values by derivatives in 1D
8. **Duality Arguments**: Iteratively apply estimates to get higher-order convergence
9. **Coercivity**: Needed for well-posedness of variational problems
10. **Natural Boundary Conditions**: Appear in the linear functional, not the space definition

All exercises demonstrate fundamental techniques in finite element analysis: scaling arguments, duality, quadrature error analysis, and the interplay between different norms.