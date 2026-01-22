#!/bin/bash
# Build script for all FEM implementations

set -e  # Exit on error

echo "======================================================================"
echo "Building FEM 1D Benchmark Implementations"
echo "======================================================================"
echo

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Function to print status
print_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 built successfully"
    else
        echo "✗ $1 build failed"
        return 1
    fi
}

# Build C implementation
echo "Building C implementation..."
cd ~/computational_physics_isp/fem_1d_benchmark/c
gcc -O3 -fPIC -shared -fopenmp -o fem_c.so fem_assembly.c -lgomp
print_status "C"
cd ..
echo

# Build C++ implementation
echo "Building C++ implementation..."
cd ~/computational_physics_isp/fem_1d_benchmark/cpp
if command -v python3 &> /dev/null; then
    c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp \
        $(python3 -m pybind11 --includes) \
        fem_assembly.cpp -o fem_cpp$(python3-config --extension-suffix) -lgomp
    print_status "C++"
else
    echo "✗ Python3 not found, skipping C++"
fi
cd ..
echo

# Build Fortran implementation
echo "Building Fortran implementation..."
cd ~/computational_physics_isp/fem_1d_benchmark/fortran
if command -v f2py &> /dev/null; then
    f2py -c -m fem_fortran fem_assembly.f90 --f90flags="-fopenmp -O3" -lgomp
    print_status "Fortran"
else
    echo "✗ f2py not found, skipping Fortran"
fi
cd ..
echo

# Build Rust implementation
echo "Building Rust implementation..."
cd ~/computational_physics_isp/fem_1d_benchmark/rust
module load rust
if command -v cargo &> /dev/null; then
    # Install maturin if not present
    if ! command -v maturin &> /dev/null; then
        echo "Installing maturin..."
        pip install maturin
    fi
    
    maturin develop --release
    print_status "Rust"
else
    echo "✗ Cargo not found, skipping Rust"
fi
cd ~/computational_physics_isp/fem_1d_benchmark/..
echo

# Setup Julia
echo "Setting up Julia..."
if command -v julia &> /dev/null; then
    # Install PyJulia if not present
    python3 -c "import julia" 2>/dev/null || pip install julia
    
    # Initialize Julia if needed
    python3 -c "from julia import Main; Main.eval('println(\"Julia ready\")')" 2>/dev/null || \
        python3 -c "import julia; julia.install()"
    
    print_status "Julia"
else
    echo "✗ Julia not found, skipping Julia"
fi
echo

echo "======================================================================"
echo "Build Complete!"
echo "======================================================================"
echo
echo "To run benchmarks:"
echo "  cd ~/computational_physics_isp/fem_1d_benchmark/benchmark"
echo "  python3 benchmark.py"
echo