"""
Python wrapper for Julia N-body implementation
Uses subprocess to call Julia and pass data via JSON

Adam Field - Computational Physics ISP
"""

import numpy as np
import json
import subprocess
import tempfile
import os
from pathlib import Path

# Get Julia executable
JULIA_CMD = "julia"

# Get path to Julia implementation
JULIA_DIR = Path(__file__).parent
JULIA_SCRIPT = JULIA_DIR / "nbody.jl"

# Julia wrapper script
JULIA_WRAPPER = '''
using JSON

# Load the main implementation
include(ARGS[1])

# Read input data
input_data = JSON.parsefile(ARGS[2])

positions = Matrix{Float64}(hcat([input_data["positions"][i] for i in 1:length(input_data["positions"])]...)')
velocities = Matrix{Float64}(hcat([input_data["velocities"][i] for i in 1:length(input_data["velocities"])]...)')
masses = Vector{Float64}(input_data["masses"])
n_steps = input_data["n_steps"]
G = input_data["G"]
softening = input_data["softening"]
dt = input_data["dt"]

# Run simulation
pos_final, vel_final = simulate(positions, velocities, masses, n_steps, G, softening, dt, true)

# Compute energy
ke0, pe0, E0 = compute_energy(positions, velocities, masses, G, softening)
ke_f, pe_f, E_f = compute_energy(pos_final, vel_final, masses, G, softening)

# Prepare output
output = Dict(
    "positions" => [pos_final[i, :] for i in 1:size(pos_final, 1)],
    "velocities" => [vel_final[i, :] for i in 1:size(vel_final, 1)],
    "initial_energy" => Dict("ke" => ke0, "pe" => pe0, "total" => E0),
    "final_energy" => Dict("ke" => ke_f, "pe" => pe_f, "total" => E_f)
)

# Write output
open(ARGS[3], "w") do f
    JSON.print(f, output)
end
'''


def check_julia_available():
    """Check if Julia is available"""
    try:
        result = subprocess.run([JULIA_CMD, "--version"], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def simulate(positions_init, velocities_init, masses, n_steps, 
            G=1.0, softening=0.1, dt=0.01):
    """
    Run N-body simulation using Julia backend
    
    Args:
        positions_init: (N, 3) array
        velocities_init: (N, 3) array  
        masses: (N,) array
        n_steps: number of timesteps
        G, softening, dt: physics parameters
    
    Returns:
        final_positions, final_velocities (both (N, 3) arrays)
    """
    if not check_julia_available():
        raise RuntimeError("Julia not found. Please install Julia and ensure it's in PATH")
    
    # Prepare input data
    input_data = {
        "positions": positions_init.tolist(),
        "velocities": velocities_init.tolist(),
        "masses": masses.tolist(),
        "n_steps": n_steps,
        "G": G,
        "softening": softening,
        "dt": dt
    }
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input.json")
        output_file = os.path.join(tmpdir, "output.json")
        wrapper_file = os.path.join(tmpdir, "wrapper.jl")
        
        # Write input data
        with open(input_file, 'w') as f:
            json.dump(input_data, f)
        
        # Write wrapper script
        with open(wrapper_file, 'w') as f:
            f.write(JULIA_WRAPPER)
        
        # Run Julia
        cmd = [JULIA_CMD, wrapper_file, str(JULIA_SCRIPT), input_file, output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Julia execution failed:\n{result.stderr}")
        
        # Read output
        with open(output_file, 'r') as f:
            output_data = json.load(f)
    
    # Convert back to numpy arrays
    positions = np.array(output_data["positions"])
    velocities = np.array(output_data["velocities"])
    
    return positions, velocities


def compute_energy(positions, velocities, masses, G=1.0, softening=0.1):
    """
    Compute energy using Julia backend
    
    Returns:
        (kinetic_energy, potential_energy, total_energy)
    """
    if not check_julia_available():
        raise RuntimeError("Julia not found")
    
    # For energy computation, just run 0 steps and return initial energy
    input_data = {
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "masses": masses.tolist(),
        "n_steps": 0,
        "G": G,
        "softening": softening,
        "dt": 0.01
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input.json")
        output_file = os.path.join(tmpdir, "output.json")
        wrapper_file = os.path.join(tmpdir, "wrapper.jl")
        
        with open(input_file, 'w') as f:
            json.dump(input_data, f)
        
        with open(wrapper_file, 'w') as f:
            f.write(JULIA_WRAPPER)
        
        cmd = [JULIA_CMD, wrapper_file, str(JULIA_SCRIPT), input_file, output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Julia execution failed:\n{result.stderr}")
        
        with open(output_file, 'r') as f:
            output_data = json.load(f)
    
    energy = output_data["initial_energy"]
    return energy["ke"], energy["pe"], energy["total"]


if __name__ == "__main__":
    print("Testing Julia wrapper...")
    
    if not check_julia_available():
        print("ERROR: Julia not found!")
        print("Please install Julia from https://julialang.org/")
        exit(1)
    
    # Quick test
    N = 10
    np.random.seed(42)
    positions = np.random.uniform(-10, 10, (N, 3))
    velocities = np.random.randn(N, 3) * 0.5
    masses = np.random.uniform(0.1, 1.0, N)
    
    print(f"Testing with N={N} particles...")
    
    ke, pe, E = compute_energy(positions, velocities, masses)
    print(f"Initial energy: KE={ke:.4f}, PE={pe:.4f}, Total={E:.4f}")
    
    pos_final, vel_final = simulate(positions, velocities, masses, 100)
    
    ke_f, pe_f, E_f = compute_energy(pos_final, vel_final, masses)
    print(f"Final energy: KE={ke_f:.4f}, PE={pe_f:.4f}, Total={E_f:.4f}")
    print(f"Energy drift: {abs(E_f - E)/abs(E)*100:.6f}%")
    print("\nJulia wrapper works!")