# P1 mesh, assembly, and solver examples

These examples retain the definitions and algorithms from the three elementary P1 demonstrations. The research implementation remains in [src](../src); these small examples are not replacements for it.

## Mesh and basis

On $[0,1]^2$, put nodes $(j/n,i/n)$ at global index $i(n+1)+j$, for $i,j=0,\ldots,n$. Split each cell into triangles $(n_0,n_1,n_2)$ and $(n_1,n_3,n_2)$, where $n_1=n_0+1$, $n_2=n_0+n+1$, and $n_3=n_2+1$. This gives $(n+1)^2$ nodes and $2n^2$ triangles. The demonstration started at $n=5$, with choices from 3 to 10.

The nodal basis $\phi_i$ equals one at node $i$ and zero at every other node. Within a triangle it is the corresponding barycentric coordinate. It vanishes on elements that do not contain $i$, so its support is the union of the incident triangles. Its gradient is constant within each element and generally differs across element boundaries. An average of neighboring gradients, as formerly displayed at a selected node, is a visualization summary rather than a unique nodal derivative.

Uniform refinement inserts one midpoint per distinct edge. A triangle $(a,b,c)$ becomes $(a,m_{ab},m_{ca})$, $(m_{ab},b,m_{bc})$, $(m_{ca},m_{bc},c)$, and $(m_{ab},m_{bc},m_{ca})$. Deduplicating midpoint nodes across shared edges keeps the mesh conforming.

![P1 mesh and nodal basis](figures/mesh_basis.png)

## Element assembly

The three-element example uses nodes $(0.2,0.2),(0.8,0.2),(0.5,0.8),(0.2,0.8),(0.8,0.8)$ and connectivity $(0,1,2),(0,2,3),(1,4,2)$.

For counterclockwise vertices $p_i=(x_i,y_i)$ and area $A$,

$$\nabla N_0=\frac{(y_1-y_2,x_2-x_1)}{2A},\quad
\nabla N_1=\frac{(y_2-y_0,x_0-x_2)}{2A},\quad
\nabla N_2=\frac{(y_0-y_1,x_1-x_0)}{2A},\qquad
K^e_{ij}=A\nabla N_i\cdot\nabla N_j.$$

Each triangle contributes its nine local entries to the corresponding global indices: $K_{I_iI_j}\mathrel{+}=K^e_{ij}$. Shared nodes receive contributions from several elements. The original display counted an entry as nonzero when its magnitude exceeded $10^{-10}$.

![Element-by-element assembly](figures/assembly.png)

Run `python weak_lensing_poisson/notes/generate_examples.py` from the repository root to reproduce both figures and the [element matrices](figures/assembly_matrices.json).

## P1 solver

The separate illustrative solver used a unit-square mesh, default $n=20$, zero boundary potential, and the convergence field

$$\kappa(x,y)=\sum_a\frac{m_a}{\sqrt{(x-x_a)^2+(y-y_a)^2+0.001}}.$$

Its preset was:

| $x_a$ | $y_a$ | $m_a$ |
|---:|---:|---:|
| 0.3 | 0.4 | 2.0 |
| 0.7 | 0.3 | 1.5 |
| 0.5 | 0.7 | 1.0 |

New sources had default strength 1.0. The demonstration accumulated stiffness entries in coordinate lists, converted them to a dense matrix, and approximated the element load with the centroid value, assigning $2\kappa(x_c,y_c)A/3$ to each node. Boundary rows and columns were zeroed, diagonal entries set to one, and boundary loads set to zero.

**Sign convention:** That positive load solves $-\Delta\psi=2\kappa$ under the standard stiffness convention. For the lensing convention $\Delta\psi=2\kappa$, the load must instead be negative. This records the original demonstration's behavior rather than presenting it as a validated lensing solve.

The unpreconditioned CG iteration started at $x=0$, $r=f$, $p=r$. Each step formed $Ap=Kp$, set $\alpha=(r^Tr)/(p^TAp)$, updated $x\leftarrow x+\alpha p$ and $r\leftarrow r-\alpha Ap$, and then set $\beta=(r_{\rm new}^Tr_{\rm new})/(r_{\rm old}^Tr_{\rm old})$ and $p\leftarrow r+\beta p$. It stopped at residual norm below $10^{-6}$ or 1000 iterations. It also guarded against a nearly zero $p^TAp$.

The deflection display formed $\nabla\psi|_e=\sum_i\psi_i\nabla N_i$ and averaged incident-element gradients at each node with equal weights. It offered convergence, potential, and deflection views and reported iteration count, residual, maximum absolute potential, and maximum deflection magnitude. P1 has zero element-interior Hessians, so this example does not provide a pointwise shear field.
