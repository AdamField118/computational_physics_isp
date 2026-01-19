#=
N-Body Gravitational Simulation - Julia Implementation (OPTIMIZED)
Drop-in replacement with @inbounds, @simd, and better vectorization

Adam Field - Computational Physics ISP
=#

using LinearAlgebra

"""
Compute gravitational accelerations - OPTIMIZED with @inbounds and @simd
Uses vectorized operations with compiler hints for maximum performance
"""
function compute_acceleration_vectorized(positions::Matrix{Float64},
                                         masses::Vector{Float64},
                                         G::Float64,
                                         softening::Float64)
    N = size(positions, 1)
    
    # Compute all pairwise displacements using broadcasting
    dx = positions[:, 1]' .- positions[:, 1]
    dy = positions[:, 2]' .- positions[:, 2]
    dz = positions[:, 3]' .- positions[:, 3]
    
    # Distances with softening
    dist_sq = dx.^2 .+ dy.^2 .+ dz.^2 .+ softening^2
    dist = sqrt.(dist_sq)
    
    # 1/r³ term
    inv_dist_cubed = 1.0 ./ (dist .* dist_sq)
    
    # Set diagonal to zero (no self-interaction) with @inbounds
    @inbounds for i in 1:N
        inv_dist_cubed[i, i] = 0.0
    end
    
    # Compute accelerations with optimized broadcasting
    mass_factor = inv_dist_cubed .* masses'
    
    # Use @inbounds to skip bounds checking in hot loop
    ax = zeros(Float64, N)
    ay = zeros(Float64, N)
    az = zeros(Float64, N)
    
    @inbounds begin
        ax = G * vec(sum(dx .* mass_factor, dims=2))
        ay = G * vec(sum(dy .* mass_factor, dims=2))
        az = G * vec(sum(dz .* mass_factor, dims=2))
    end
    
    return hcat(ax, ay, az)
end


"""
Loop-based acceleration (kept for compatibility, also optimized)
"""
function compute_acceleration(positions::Matrix{Float64}, 
                              masses::Vector{Float64},
                              G::Float64,
                              softening::Float64)
    N = size(positions, 1)
    accelerations = zeros(Float64, N, 3)
    
    # Pairwise force calculation with @inbounds
    @inbounds for i in 1:N
        for j in 1:N
            if i != j
                # Displacement vector
                dx = positions[j, 1] - positions[i, 1]
                dy = positions[j, 2] - positions[i, 2]
                dz = positions[j, 3] - positions[i, 3]
                
                # Distance with softening
                dist_sq = dx*dx + dy*dy + dz*dz + softening*softening
                dist = sqrt(dist_sq)
                
                # 1/r³ term
                inv_dist_cubed = 1.0 / (dist * dist_sq)
                
                # Accumulate acceleration
                factor = G * masses[j] * inv_dist_cubed
                accelerations[i, 1] += factor * dx
                accelerations[i, 2] += factor * dy
                accelerations[i, 3] += factor * dz
            end
        end
    end
    
    return accelerations
end


"""
Single timestep using Velocity Verlet integration - OPTIMIZED
"""
function velocity_verlet_step!(positions::Matrix{Float64},
                               velocities::Matrix{Float64},
                               masses::Vector{Float64},
                               G::Float64,
                               softening::Float64,
                               dt::Float64,
                               use_vectorized::Bool=true)
    dt_sq = dt * dt
    
    # Current acceleration
    acc_current = if use_vectorized
        compute_acceleration_vectorized(positions, masses, G, softening)
    else
        compute_acceleration(positions, masses, G, softening)
    end
    
    # Update positions with @inbounds and @simd
    @inbounds @simd for i in eachindex(positions)
        positions[i] = positions[i] + velocities[i] * dt + 0.5 * acc_current[i] * dt_sq
    end
    
    # Compute new acceleration
    acc_new = if use_vectorized
        compute_acceleration_vectorized(positions, masses, G, softening)
    else
        compute_acceleration(positions, masses, G, softening)
    end
    
    # Update velocities with @inbounds and @simd
    @inbounds @simd for i in eachindex(velocities)
        velocities[i] = velocities[i] + 0.5 * (acc_current[i] + acc_new[i]) * dt
    end
    
    return nothing
end


"""
Run full N-body simulation for n_steps - OPTIMIZED
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
    
    # Time integration loop with @inbounds
    @inbounds for step in 1:n_steps
        velocity_verlet_step!(positions, velocities, masses, 
                             G, softening, dt, use_vectorized)
    end
    
    return positions, velocities
end


"""
Compute total energy of the system - OPTIMIZED
Returns (kinetic_energy, potential_energy, total_energy)
"""
function compute_energy(positions::Matrix{Float64},
                       velocities::Matrix{Float64},
                       masses::Vector{Float64},
                       G::Float64,
                       softening::Float64)
    N = size(positions, 1)
    
    # Kinetic energy with @inbounds and @simd
    ke = 0.0
    @inbounds @simd for i in 1:N
        v_sq = velocities[i, 1]^2 + velocities[i, 2]^2 + velocities[i, 3]^2
        ke += 0.5 * masses[i] * v_sq
    end
    
    # Potential energy with @inbounds
    pe = 0.0
    @inbounds for i in 1:N
        for j in (i+1):N
            dx = positions[j, 1] - positions[i, 1]
            dy = positions[j, 2] - positions[i, 2]
            dz = positions[j, 3] - positions[i, 3]
            
            dist = sqrt(dx*dx + dy*dy + dz*dz + softening*softening)
            
            pe -= G * masses[i] * masses[j] / dist
        end
    end
    
    return ke, pe, ke + pe
end


# Command-line interface for benchmarking
if abspath(PROGRAM_FILE) == @__FILE__
    using Random
    
    println("Julia N-Body Simulation (OPTIMIZED)")
    println("="^50)
    
    # Parse command-line arguments or use defaults
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
    n_steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1000
    
    # Create test system
    Random.seed!(42)
    positions = rand(N, 3) .* 20.0 .- 10.0  # Uniform in [-10, 10]
    velocities = randn(N, 3) .* 0.5
    masses = rand(N) .* 0.9 .+ 0.1  # Uniform in [0.1, 1.0]
    
    # Physics parameters
    G = 1.0
    softening = 0.1
    dt = 0.01
    
    # Compute initial energy
    ke0, pe0, E0 = compute_energy(positions, velocities, masses, G, softening)
    println("Initial energy: KE=$ke0, PE=$pe0, Total=$E0")
    
    # Warm-up run (compilation)
    println("\nWarm-up run...")
    simulate(positions, velocities, masses, 10, G, softening, dt, true)
    
    # Timed run
    println("\nRunning $n_steps steps (N=$N)...")
    t_start = time()
    pos_final, vel_final = simulate(positions, velocities, masses, 
                                    n_steps, G, softening, dt, true)
    elapsed = time() - t_start
    
    println("Runtime: $elapsed seconds")
    println("Time per step: $(elapsed/n_steps*1000) ms")
    
    # Compute final energy
    ke_f, pe_f, E_f = compute_energy(pos_final, vel_final, masses, G, softening)
    println("\nFinal energy: KE=$ke_f, PE=$pe_f, Total=$E_f")
    println("Energy drift: $(abs(E_f - E0)/abs(E0)*100)%")
    
    println("\n" * "="^50)
    println("Performance notes:")
    println("- Using @inbounds (no bounds checking)")
    println("- Using @simd (SIMD vectorization)")
    println("- Vectorized pairwise operations")
    println("="^50)
end