## Weak Formulation (Brenner & Scott §2.3)

Starting from strong form: $-\Delta u = f$ in $\Omega$, $u = 0$ on $\partial\Omega$

**Derivation**: Multiply by test function $v \in H^1_0(\Omega)$, integrate by parts:
$$\int_\Omega \nabla u \cdot \nabla v \, dx = \int_\Omega f v \, dx$$

**Discrete**: Choose $V_h \subset H^1_0(\Omega)$ (piecewise linear), find $u_h \in V_h$:
$$a(u_h, v_h) = (f, v_h) \quad \forall v_h \in V_h$$

**Expected convergence** (Theorem 4.4.3):
- $\|u - u_h\|_{L^2(\Omega)} \leq Ch^2 \|u\|_{H^2(\Omega)}$
- $\|u - u_h\|_{H^1(\Omega)} \leq Ch \|u\|_{H^2(\Omega)}$