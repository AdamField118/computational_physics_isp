# N-body numerical setup

## N-Body Physics Equations

### Gravitational Force
For particle i with mass m_i at position r_i:

F_i = G * Σ(j≠i) [ (m_i * m_j * (r_j - r_i)) / |r_j - r_i|³ ]

### Equations of Motion
dv_i/dt = F_i / m_i
dr_i/dt = v_i

### Numerical Integration (Velocity Verlet)
r(t + Δt) = r(t) + v(t)*Δt + 0.5*a(t)*Δt²
v(t + Δt) = v(t) + 0.5*(a(t) + a(t + Δt))*Δt

## Computational Complexity
- Direct summation: O(N²) per timestep
- Barnes-Hut tree O(N log N)

## Checks

Compare forces and trajectories at matched precision, measure energy drift as the timestep decreases, and locate the CPU/GPU crossover from timings.

## Benchmark comparisons
- At what N does GPU become advantageous?
- How do OpenMP parallel CPU implementations compare?
- Does compiler optimization matter? (gcc -O0 vs -O3)
- Memory bandwidth vs compute bound?
- Single vs double precision tradeoffs?
