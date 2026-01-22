"""
1D FEM Assembly in Julia with multi-threading

Usage from Python via PyJulia:
    from julia import Main
    Main.include("fem_assembly.jl")
    K, F = Main.assemble_system(n, f_vals)
"""

using Base.Threads

function assemble_system(n::Int, f_vals::Vector{Float64})
    """
    Assemble stiffness matrix and load vector (parallel version)
    
    Parameters:
    -----------
    n : Int
        Number of elements
    f_vals : Vector{Float64}
        Source function values at nodes (length n+1)
    
    Returns:
    --------
    K : Matrix{Float64}
        Stiffness matrix (n x n)
    F : Vector{Float64}
        Load vector (length n)
    """
    h = 1.0 / n
    k_local = 1.0 / h
    
    # Allocate arrays (column-major order in Julia)
    K = zeros(Float64, n, n)
    F = zeros(Float64, n)
    
    # Thread-safe assembly using locks
    lock = ReentrantLock()
    
    # Assemble stiffness matrix in parallel
    @threads for e in 1:n
        i_left = e
        i_right = e + 1
        
        if i_left >= 2
            idx_left = i_left - 1
            idx_right = i_right - 1
            
            lock(() -> begin
                K[idx_left, idx_left] += k_local
                K[idx_left, idx_right] -= k_local
                K[idx_right, idx_left] -= k_local
            end, lock)
        end
        
        idx_right = i_right - 1
        lock(() -> K[idx_right, idx_right] += k_local, lock)
    end
    
    # Assemble load vector in parallel
    @threads for i in 1:(n-1)
        F[i] = (h / 2.0) * (f_vals[i] + f_vals[i+2])
    end
    F[n] = (h / 2.0) * (f_vals[n] + f_vals[n+1])
    
    return K, F
end


function assemble_system_serial(n::Int, f_vals::Vector{Float64})
    """
    Assemble stiffness matrix and load vector (serial version)
    """
    h = 1.0 / n
    k_local = 1.0 / h
    
    K = zeros(Float64, n, n)
    F = zeros(Float64, n)
    
    # Serial assembly
    for e in 1:n
        i_left = e
        i_right = e + 1
        
        if i_left >= 2
            idx_left = i_left - 1
            idx_right = i_right - 1
            
            K[idx_left, idx_left] += k_local
            K[idx_left, idx_right] -= k_local
            K[idx_right, idx_left] -= k_local
        end
        
        idx_right = i_right - 1
        K[idx_right, idx_right] += k_local
    end
    
    for i in 1:(n-1)
        F[i] = (h / 2.0) * (f_vals[i] + f_vals[i+2])
    end
    F[n] = (h / 2.0) * (f_vals[n] + f_vals[n+1])
    
    return K, F
end


# Optimized version without locks (reduce-then-combine pattern)
function assemble_system_optimized(n::Int, f_vals::Vector{Float64})
    """
    Optimized parallel assembly using per-thread storage
    """
    h = 1.0 / n
    k_local = 1.0 / h
    
    # Allocate per-thread storage
    num_threads = nthreads()
    K_local = [zeros(Float64, n, n) for _ in 1:num_threads]
    
    # Parallel assembly into thread-local matrices
    @threads for e in 1:n
        tid = threadid()
        i_left = e
        i_right = e + 1
        
        if i_left >= 2
            idx_left = i_left - 1
            idx_right = i_right - 1
            
            K_local[tid][idx_left, idx_left] += k_local
            K_local[tid][idx_left, idx_right] -= k_local
            K_local[tid][idx_right, idx_left] -= k_local
        end
        
        idx_right = i_right - 1
        K_local[tid][idx_right, idx_right] += k_local
    end
    
    # Combine thread-local matrices
    K = sum(K_local)
    
    # Assemble load vector in parallel
    F = zeros(Float64, n)
    @threads for i in 1:(n-1)
        F[i] = (h / 2.0) * (f_vals[i] + f_vals[i+2])
    end
    F[n] = (h / 2.0) * (f_vals[n] + f_vals[n+1])
    
    return K, F
end


# Print thread info for verification
println("Julia FEM assembly module loaded")
println("Number of threads available: ", nthreads())