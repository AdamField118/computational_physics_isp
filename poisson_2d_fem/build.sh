#!/bin/bash
# Build script for Poisson 2D FEM solver

set -e

echo "========================================="
echo "Building Poisson 2D FEM Solver"
echo "========================================="

# Navigate to Fortran directory
cd fortran

echo ""
echo "Compiling Fortran modules..."

# Compile modules in order (respecting dependencies)
gfortran -c -O3 -fPIC types_module.f90
gfortran -c -O3 -fPIC reference_element.f90
gfortran -c -O3 -fPIC assembly.f90
gfortran -c -O3 -fPIC boundary_conditions.f90
gfortran -c -O3 -fPIC solver.f90

echo ""
echo "Creating f2py wrapper..."

# Create combined module with f2py
f2py -c -m fem_fortran \
    types_module.f90 \
    reference_element.f90 \
    assembly.f90 \
    boundary_conditions.f90 \
    solver.f90 \
    --f90flags="-O3" \
    -llapack -lblas

# Move .so to Python directory
mv fem_fortran*.so ../python/

echo ""
echo "========================================="
echo "Build complete!"
echo "========================================="
echo ""
echo "To test:"
echo "  cd python"
echo "  python convergence_study.py"