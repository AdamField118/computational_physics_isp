# Small-system simulation example

The illustrative particle simulation was separate from the benchmark implementations. Its computational setup is retained here; use the Python implementations in [nbody](../nbody) for experiments.

| Parameter | Value or distribution |
|---|---|
| Particle count | Default 50; range 10–200 in steps of 10 |
| Initial position components | Independent uniform values in $[-10,10)$ |
| Initial velocity components | Independent uniform values in $[-0.05,0.05)$ |
| Mass | Independent uniform values in $[0.5,1)$ |
| Gravitational constant | $G=0.01$ |
| Timestep | $0.1s$, with speed parameter $s$ from 0.1 to 3.0 |
| Softening term added to squared distance | 0.1 |

For each particle it computed

$$a_i=G\sum_{j\ne i}\frac{m_j(r_j-r_i)}{(|r_j-r_i|^2+0.1)^{3/2}},$$

then updated $v_i\leftarrow v_i+a_i\Delta t$ and $r_i\leftarrow r_i+v_i\Delta t$. The loop updated particles in place, so later particles saw some positions from the new step. Although its comment called this simplified Velocity Verlet, it was a sequential kick-and-drift update without the second acceleration evaluation. It should not be used as evidence for the accuracy or speed of the benchmark's Velocity Verlet implementations.

Initial conditions were randomly regenerated on reset without a stored seed. The original display allowed pause, restart, particle-count changes, and timestep changes. These affect the experiment; camera movement, colors, and display lighting do not.

The [illustrative sample table](../results/illustrative_sample.json) is also separate from measured results. It supplied substitute values when the saved result file could not be loaded. Its timestamp was generated at display time, so no measurement timestamp can be recovered from it.
