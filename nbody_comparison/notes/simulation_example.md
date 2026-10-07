# Small-system simulation example

This example uses a sequential kick-and-drift update. The [benchmark implementations](../nbody) use Velocity Verlet, so their timings and accuracy should be assessed separately.

| Parameter | Value or distribution |
|---|---|
| Particle count | Default 50; range 10–200 in steps of 10 |
| Initial position components | Independent uniform values in $[-10,10)$ |
| Initial velocity components | Independent uniform values in $[-0.05,0.05)$ |
| Mass | Independent uniform values in $[0.5,1)$ |
| Gravitational constant | $G=0.01$ |
| Timestep | $0.1s$, with speed parameter $s$ from 0.1 to 3.0 |
| Softening term added to squared distance | 0.1 |

For each particle, compute

$$a_i=G\sum_{j\ne i}\frac{m_j(r_j-r_i)}{(|r_j-r_i|^2+0.1)^{3/2}},$$

then update $v_i\leftarrow v_i+a_i\Delta t$ and $r_i\leftarrow r_i+v_i\Delta t$. Updating particles in place means later particles see some positions from the new step. This differs from Velocity Verlet, which evaluates acceleration again after updating all positions.

The initial-condition distributions above do not specify a random seed. Set and record one when implementing this example so that runs can be compared.

The [illustrative sample table](../results/illustrative_sample.json) contains example timings, not measurements. It has no measurement timestamp; use [benchmark_results.json](../results/benchmark_results.json) for recorded timings.
