"""
1D FEM Assembly in Julia (Allocates and returns - PyCall compatible)

CRITICAL: PyCall doesn't reliably support in-place modification of NumPy arrays
Solution: Julia allocates its own arrays and returns them, Python copies back

Usage from Python via PyJulia:
    from julia import Main
    import numpy as np
    Main.include("fem_assembly.jl")
    
    K = np.zeros((n, n), order='F')
    F = np.zeros(n)
    K_julia, F_julia = Main.assemble_system(n, f_vals)
    K[:] = np.array(K_julia)
    F[:] = np.array(F_julia)
"""

function assemble_system(n::Int, f_vals::Vector{Float64})
    """
    Assemble stiffness matrix and load vector
    
    Allocates its own arrays and returns them (PyCall safe)
    
    Parameters:
    -----------
    n : Int
        Number of elements
    f_vals : Vector{Float64}
        Source function values at nodes (length n+1)
    
    Returns:
    --------
    K : Matrix{Float64}
        Stiffness matrix (n x n, column-major)
    F : Vector{Float64}
        Load vector (length n)
    """
    h = 1.0 / n
    k_local = 1.0 / h
    
    # Allocate arrays (Julia uses column-major by default)
    K = zeros(Float64, n, n)
    F = zeros(Float64, n)
    
    # Assemble load vector
    @inbounds for i in 1:(n-1)
        F[i] = (h / 2.0) * (f_vals[i] + f_vals[i+2])
    end
    @inbounds F[n] = (h / 2.0) * f_vals[n]
    
    # Assemble stiffness matrix
    @inbounds K[1, 1] = k_local
    
    # Main assembly loop
    @inbounds for e in 2:n
        i = e - 1
        K[i, i]     += k_local
        K[i+1, i]   -= k_local
        K[i, i+1]   -= k_local
        K[i+1, i+1] += k_local
    end
    
    return K, F
end

println("Julia FEM assembly module loaded (allocates and returns - PyCall compatible)")