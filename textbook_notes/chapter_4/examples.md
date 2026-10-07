# Chapter 4 worked examples

Plots are generated locally; see the [figure instructions](../../docs/GENERATED_FILES.md).

Regenerate the figures with `python textbook_notes/generate_figures.py` from the repository root.

## Reference stiffness

On the reference triangle $(0,0),(1,0),(0,1)$, the P1 functions are $1-x-y,x,y$, with gradients $(-1,-1),(1,0),(0,1)$. Since the area is $1/2$,

$$\widehat K_{ij}=\frac12\nabla\widehat\phi_i\cdot\nabla\widehat\phi_j,
\qquad\widehat K=\frac12\begin{pmatrix}2&-1&-1\\-1&1&0\\-1&0&1\end{pmatrix}.$$

For example, entry $(1,2)$ is $[(-1)(1)+(-1)(0)]/2=-1/2$; entry $(2,3)$ is zero because its two gradients are perpendicular. Every entry follows this gradient-dot-product-times-area calculation.

Figure: Reference triangle gradients and stiffness matrix. Generated locally as `figures/reference_stiffness.png`.

The matrix is symmetric and **positive semidefinite**, with constant-vector nullspace. Boundary conditions are needed to remove the constant mode in a global Poisson problem. Physical elements require the Jacobian and the transformed gradient metric, not simply a copy of this matrix.

## Mesh quality

For edge lengths $a,b,c$ and semiperimeter $s=(a+b+c)/2$, compute

$$A=\sqrt{s(s-a)(s-b)(s-c)},\qquad \rho=A/s,\qquad h=\max(a,b,c).$$

Here $\rho$ is the inradius. Some texts use the inscribed-circle diameter instead, changing $h/\rho$ by a factor of two. Compute an angle between edge vectors $v,w$ with $\arccos[(v\cdot w)/(\lVert v\rVert\lVert w\rVert)]$, clipping roundoff to $[-1,1]$.

The following angle thresholds are illustrative quality labels. A small positive angle gives a thin triangle; degeneracy requires zero area.

| Minimum angle | Quality label |
|---|---|
| At least 30° | Excellent |
| 20° to below 30° | Good |
| 15° to below 20° | Acceptable |
| 5° to below 15° | Poor |
| Below 5° | Very thin |

Figure: Equilateral, right, thin, and nearly collinear triangles. Generated locally as `figures/mesh_quality.png`.

An equilateral triangle has $h/\rho=2\sqrt3$. As a triangle flattens, its area and inradius tend to zero while its longest edge can stay finite. Shape regularity excludes this limit uniformly over a mesh family.

## Homogeneity

Use the isotropic map $F(\widehat x)=s\widehat x+b$, where $s>0$. For $v=\widehat v\circ F^{-1}$ in dimension $d$,

$$|\det DF|=s^d,\qquad
\lVert v\rVert_{L^2(K)}=s^{d/2}\lVert\widehat v\rVert_{L^2(\widehat K)},\qquad
|v|_{H^m(K)}=s^{d/2-m}|\widehat v|_{H^m(\widehat K)}.$$

The reference right triangle has area $1/2$ and diameter $\sqrt2$. In two dimensions its scaled area is $s^2/2$, its diameter is $s\sqrt2$, and the $L^2,H^1,H^2$ scaling factors are $s,1,s^{-1}$. The plotted ratios use unit reference norms rather than a specified test function.

Figure: Triangle dilation and norm scaling factors. Generated locally as `figures/homogeneity.png`.

These factors enter interpolation estimates and inverse inequalities. For a general affine matrix $B$, derivative estimates depend on $B^{-1}$ as well as $\det B$; the scalar identities above describe isotropic scaling.

## Conditioning

For $n$ interior points on $(0,1)$ with homogeneous Dirichlet conditions, let $h=1/(n+1)$ and $K=h^{-1}\operatorname{tridiag}(-1,2,-1)$. Its eigenvalues are

$$\lambda_k=\frac4h\sin^2\!\left(\frac{k\pi}{2(n+1)}\right),\quad k=1,\ldots,n.$$

Thus $\lambda_{\min}\sim\pi^2h$, $\lambda_{\max}\sim4/h$, and $\kappa_2(K)\sim4/(\pi^2h^2)$. At fixed relative tolerance the standard CG bound scales with $\sqrt\kappa$, with a logarithmic tolerance factor. $\lceil\sqrt\kappa\rceil$ is a rough iteration indicator, not a measured count.

Figure: Eigenvalues and conditioning with mesh refinement. Generated locally as `figures/conditioning.png`.

Doubling resolution roughly quadruples the condition number. Preconditioning changes the relevant spectrum. A dense direct solve costs $O(n^3)$, but this tridiagonal problem admits an $O(n)$ direct solve.

## Quadrilateral mapping

For reference vertices $(-1,-1),(1,-1),(1,1),(-1,1)$, use

$$N=\frac14\big((1-\xi)(1-\eta),(1+\xi)(1-\eta),
(1+\xi)(1+\eta),(1-\xi)(1+\eta)\big),\qquad
F(\xi,\eta)=\sum_{i=1}^4N_i(\xi,\eta)p_i.$$

The columns of $J$ are $\sum_i(\partial_\xi N_i)p_i$ and $\sum_i(\partial_\eta N_i)p_i$. For example,

$$\partial_\xi N=\tfrac14(-(1-\eta),1-\eta,1+\eta,-(1+\eta)),
\quad\partial_\eta N=\tfrac14(-(1-\xi),-(1+\xi),1+\xi,1-\xi).$$

Compare a square, parallelogram, trapezoid, general convex quadrilateral, and concave quadrilateral. A parallelogram has an affine map and constant Jacobian; a general quadrilateral has a bilinear term. Edges map to straight edges. A zero or changing-sign determinant signals a singular or folded map.

Figure: Mapped reference grids and Jacobian ranges. Generated locally as `figures/quadrilateral_mapping.png`.

The figure samples a $6\times6$ grid including the corners and reports the center determinant, sampled minimum, and sampled maximum. For these straight-sided Q1 maps, $\det J$ is affine in $(\xi,\eta)$, so its extrema over the reference square occur at corners. A consistently oriented, convex physical element gives the usual valid configuration. Numerical quadrature is generally needed for physical stiffness integrals.
