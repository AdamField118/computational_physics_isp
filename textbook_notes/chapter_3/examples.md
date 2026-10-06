# Chapter 3 worked examples

These examples preserve the mathematical content of the exercise demonstrations. Regenerate their figures with `python textbook_notes/generate_figures.py` from the repository root.

## Rectangle

For the rectangle $[-1,1]\times[0,1]$, ordered counterclockwise from the lower-left corner, the nodal basis is

$$\phi_1=\frac{(1-x)(1-y)}2,\quad
\phi_2=\frac{(1+x)(1-y)}2,\quad
\phi_3=\frac{(1+x)y}2,\quad
\phi_4=\frac{(1-x)y}2.$$

Each function equals one at its own vertex and zero at the other three. Their sum is one; each is linear in either coordinate with the other held fixed. The corresponding surface is bilinear, rather than a plane in general.

![Four rectangular basis functions](figures/rectangle.png)

**Normalization note:** The original exercise text and demonstration used $(1\pm x)(1\pm y)/4$ while labelling the rectangle $[-1,1]\times[0,1]$. Those formulas belong to $[-1,1]^2$. The figure uses the formulas above, which match the stated rectangle. The original formulas remain in the exercise solution for comparison with this correction.

## Quadratic triangle

On vertices $(0,0),(1,0),(0,1)$, set $\lambda=(1-x-y,x,y)$. The six nodes are the vertices and $(1/2,0),(1/2,1/2),(0,1/2)$. The basis functions, in that order, are

$$\lambda_1(2\lambda_1-1),\quad\lambda_2(2\lambda_2-1),\quad
\lambda_3(2\lambda_3-1),\quad4\lambda_1\lambda_2,\quad
4\lambda_2\lambda_3,\quad4\lambda_1\lambda_3.$$

They satisfy the nodal Kronecker-delta property and sum to one. Vertex functions take negative values in parts of the triangle; the full signed range is retained in these plots.

![Six quadratic triangular basis functions](figures/quadratic_triangle.png)

## Nonconforming elements

The comparison mesh has vertices $(i/2,j/2)$ for $i,j=0,1,2$. Number them row by row. Its eight triangles are $(0,1,3),(1,4,3),(1,2,4),(2,5,4),(3,4,6),(4,7,6),(4,5,7),(5,8,7)$.

Conforming P1 functions use vertex values and agree along a shared edge. A Crouzeix–Raviart function uses one value per edge midpoint. Its local basis for the edge opposite vertex $i$ is

$$\phi_i^{\rm CR}=1-2\lambda_i.$$

It is one at that edge midpoint and zero at the other two midpoints. Values can jump elsewhere along an edge, including at vertices. The function reaches $-1$ at the opposite vertex, which should not be clipped away in a plot.

There are nine vertex DOFs and sixteen distinct edge DOFs on this mesh before boundary conditions. There are 24 local edge slots, but shared edges identify pairs of slots; the original display's “24 total” counted local slots rather than independent global DOFs. Its selectable local functions also did not assemble the matching contribution on the adjacent triangle.

![Conforming vertex DOFs and nonconforming edge DOFs](figures/nonconforming.png)

## Lagrange node counts

For degree $r\ge1$, enumerate nonnegative integer tuples whose sum is $r$, then divide each component by $r$ to obtain barycentric coordinates. On a triangle the tuples are $(i,j,k)$; on a tetrahedron they are $(i,j,k,\ell)$.

$$N_d(r)=\binom{r+d}{d}.$$

| Degree | Triangle | Tetrahedron |
|---|---:|---:|
| 1 | 3 | 4 |
| 2 | 6 | 10 |
| 3 | 10 | 20 |
| 4 | 15 | 35 |
| 5 | 21 | 56 |

A triangle has three vertex nodes, $3(r-1)$ edge-interior nodes, and $(r-1)(r-2)/2$ interior nodes. A tetrahedron has four vertex nodes, $6(r-1)$ edge-interior nodes, $2(r-1)(r-2)$ face-interior nodes, and $(r-1)(r-2)(r-3)/6$ interior nodes, with counts interpreted as zero when the degree is too small.

![Barycentric lattice nodes for degrees one through five](figures/lagrange_nodes.png)

The original tetrahedron display used approximately equilateral vertices $(0,0,0),(1,0,0),(0.5,0.866,0),(0.5,0.289,0.816)$. Geometry changes the positions of the nodes, not these counts.
