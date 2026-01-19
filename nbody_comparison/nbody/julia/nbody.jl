#=
N-Body Gravitational Simulation - Julia Implementation
- Auto-selects best algorithm based on N
- SIMD-optimized loops for large N
- Minimal allocations

Adam Field - Computational Physics ISP
=#

using LinearAlgebra

"""
Loop-based acceleration - with SIMD
This is fastest for N > 150 due to minimal allocations
"""
function compute_acceleration_simd(positions::Matrix{Float64}, 
                                   masses::Vector{Float64},
                                   G::Float64,
                                   softening::Float64)
    N = size(positions, 1)
    accelerations = zeros(Float64, N, 3)
    
    soft_sq = softening * softening
    
    # Pairwise force calculation with aggressive optimization
    @inbounds for i in 1:N
        ax, ay, az = 0.0, 0.0, 0.0
        
        # Cache particle i's position
        px_i = positions[i, 1]
        py_i = positions[i, 2]
        pz_i = positions[i, 3]
        
        # Inner loop with SIMD
        @simd for j in 1:N
            if i != j
                # Displacement vector
                dx = positions[j, 1] - px_i
                dy = positions[j, 2] - py_i
                dz = positions[j, 3] - pz_i
                
                # Distance with softening
                dist_sq = dx*dx + dy*dy + dz*dz + soft_sq
                dist = sqrt(dist_sq)
                
                # 1/r³ term
                inv_dist_cubed = 1.0 / (dist * dist_sq)
                
                # Accumulate acceleration
                factor = G * masses[j] * inv_dist_cubed
                ax += factor * dx
                ay += factor * dy
                az += factor * dz
            end
        end
        
        accelerations[i, 1] = ax
        accelerations[i, 2] = ay
        accelerations[i, 3] = az
    end
    
    return accelerations
end


"""
Vectorized acceleration (good for small N < 150)
"""
function compute_acceleration_vectorized(positions::Matrix{Float64},
                                         masses::Vector{Float64},
                                         G::Float64,
                                         softening::Float64)
    N = size(positions, 1)
    
    # Compute all pairwise displacements
    dx = positions[:, 1]' .- positions[:, 1]
    dy = positions[:, 2]' .- positions[:, 2]
    dz = positions[:, 3]' .- positions[:, 3]
    
    # Distances with softening
    dist_sq = dx.^2 .+ dy.^2 .+ dz.^2 .+ softening^2
    dist = sqrt.(dist_sq)
    
    # 1/r³ term
    inv_dist_cubed = 1.0 ./ (dist .* dist_sq)
    
    # Set diagonal to zero (no self-interaction)
    @inbounds for i in 1:N
        inv_dist_cubed[i, i] = 0.0
    end
    
    # Compute accelerations
    mass_factor = inv_dist_cubed .* masses'
    
    ax = G * vec(sum(dx .* mass_factor, dims=2))
    ay = G * vec(sum(dy .* mass_factor, dims=2))
    az = G * vec(sum(dz .* mass_factor, dims=2))
    
    return hcat(ax, ay, az)
end


"""
Adaptive algorithm selection based on problem size
"""
function compute_acceleration(positions::Matrix{Float64}, 
                              masses::Vector{Float64},
                              G::Float64,
                              softening::Float64)
    N = size(positions, 1)
    
    # Crossover point: vectorized faster for small N, SIMD loops for large N
    if N < 150
        return compute_acceleration_vectorized(positions, masses, G, softening)
    else
        return compute_acceleration_simd(positions, masses, G, softening)
    end
end


"""
Single timestep using Velocity Verlet integration
"""
function velocity_verlet_step!(positions::Matrix{Float64},
                               velocities::Matrix{Float64},
                               masses::Vector{Float64},
                               G::Float64,
                               softening::Float64,
                               dt::Float64,
                               use_vectorized::Bool=true)
    dt_sq = dt * dt
    
    # Compute current acceleration (auto-selects algorithm)
    acc_current = compute_acceleration(positions, masses, G, softening)
    
    # Update positions with SIMD
    @inbounds @simd for i in eachindex(positions)
        positions[i] += velocities[i] * dt + 0.5 * acc_current[i] * dt_sq
    end
    
    # Compute new acceleration
    acc_new = compute_acceleration(positions, masses, G, softening)
    
    # Update velocities with SIMD
    @inbounds @simd for i in eachindex(velocities)
        velocities[i] += 0.5 * (acc_current[i] + acc_new[i]) * dt
    end
    
    return nothing
end


"""
Run full N-body simulation for n_steps
"""
function simulate(positions_init::Matrix{Float64},
                 velocities_init::Matrix{Float64},
                 masses::Vector{Float64},
                 n_steps::Int,
                 G::Float64,
                 softening::Float64,
                 dt::Float64,
                 use_vectorized::Bool=true)
    # Copy initial conditions
    positions = copy(positions_init)
    velocities = copy(velocities_init)
    
    # Time integration loop
    @inbounds for step in 1:n_steps
        velocity_verlet_step!(positions, velocities, masses, 
                             G, softening, dt, use_vectorized)
    end
    
    return positions, velocities
end


"""
Compute total energy
"""
function compute_energy(positions::Matrix{Float64},
                       velocities::Matrix{Float64},
                       masses::Vector{Float64},
                       G::Float64,
                       softening::Float64)
    N = size(positions, 1)
    
    # Kinetic energy with SIMD
    ke = 0.0
    @inbounds @simd for i in 1:N
        v_sq = velocities[i, 1]^2 + velocities[i, 2]^2 + velocities[i, 3]^2
        ke += 0.5 * masses[i] * v_sq
    end
    
    # Potential energy
    pe = 0.0
    soft_sq = softening * softening
    
    @inbounds for i in 1:N
        for j in (i+1):N
            dx = positions[j, 1] - positions[i, 1]
            dy = positions[j, 2] - positions[i, 2]
            dz = positions[j, 3] - positions[i, 3]
            
            dist = sqrt(dx*dx + dy*dy + dz*dz + soft_sq)
            
            pe -= G * masses[i] * masses[j] / dist
        end
    end
    
    return ke, pe, ke + pe
end


# Command-line interface
if abspath(PROGRAM_FILE) == @__FILE__
    using Random
    
    println("Julia N-Body Simulation")
    println("="^50)
    
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
    n_steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1000
    
    Random.seed!(42)
    positions = rand(N, 3) .* 20.0 .- 10.0
    velocities = randn(N, 3) .* 0.5
    masses = rand(N) .* 0.9 .+ 0.1
    
    G = 1.0
    softening = 0.1
    dt = 0.01
    
    ke0, pe0, E0 = compute_energy(positions, velocities, masses, G, softening)
    println("Initial energy: KE=$ke0, PE=$pe0, Total=$E0")
    
    println("\nWarm-up run...")
    simulate(positions, velocities, masses, 10, G, softening, dt, true)
    
    println("\nRunning $n_steps steps (N=$N)...")
    println("Algorithm: ", N < 150 ? "Vectorized" : "SIMD Loops")
    
    t_start = time()
    pos_final, vel_final = simulate(positions, velocities, masses, 
                                    n_steps, G, softening, dt, true)
    elapsed = time() - t_start
    
    println("Runtime: $elapsed seconds")
    println("Time per step: $(elapsed/n_steps*1000) ms")
    
    ke_f, pe_f, E_f = compute_energy(pos_final, vel_final, masses, G, softening)
    println("\nFinal energy: KE=$ke_f, PE=$pe_f, Total=$E_f")
    println("Energy drift: $(abs(E_f - E0)/abs(E0)*100)%")
end