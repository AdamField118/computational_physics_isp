"""
CRITICAL DIAGNOSTIC: Is the solution even close to correct?
"""
import numpy as np
import sys
sys.path.append('../fortran')

import fem_fortran
from mesh_generator import SimpleMesh
from manufactured_solutions import SineSolution

print("=" * 70)
print("CRITICAL DIAGNOSTIC")
print("=" * 70)

# Create medium-sized mesh
mesh = SimpleMesh('unit_square', max_area=0.01)
mesh.generate()

nodes = np.asfortranarray(mesh.nodes)
elements = np.asfortranarray(mesh.elements)
boundary = np.asfortranarray(mesh.boundary)

# Solve
try:
    solution = fem_fortran.python_interface.solve_poisson_2d(
        nodes, elements, boundary,
        mesh.nodes.shape[0], mesh.elements.shape[0], mesh.boundary.shape[0]
    )
except AttributeError:
    solution = fem_fortran.solve_poisson_2d(
        nodes, elements, boundary,
        mesh.nodes.shape[0], mesh.elements.shape[0], mesh.boundary.shape[0]
    )

# Get exact solution
mms = SineSolution()
u_exact = np.array([mms.u_exact(x, y) for x, y in mesh.nodes])

print(f"\nMesh: {mesh.nodes.shape[0]} nodes")
print("=" * 70)

# Test 1: Check solution magnitude
print("\nTEST 1: Solution Magnitude")
print("-" * 70)
print(f"Numerical solution:")
print(f"  Min:  {solution.min():.6f}")
print(f"  Max:  {solution.max():.6f}")
print(f"  Mean: {solution.mean():.6f}")
print(f"\nExact solution:")
print(f"  Min:  {u_exact.min():.6f}")
print(f"  Max:  {u_exact.max():.6f}")
print(f"  Mean: {u_exact.mean():.6f}")

magnitude_ratio = solution.max() / u_exact.max()
print(f"\nMagnitude ratio (num/exact): {magnitude_ratio:.3f}")

if abs(magnitude_ratio - 1.0) > 0.5:
    print("❌ WRONG MAGNITUDE - solution is scaled incorrectly!")
    print(f"   Off by factor of {magnitude_ratio:.2f}")
else:
    print("✓ Magnitude is reasonable")

# Test 2: Check solution at center
print("\n" + "=" * 70)
print("TEST 2: Solution at Center Point (0.5, 0.5)")
print("-" * 70)

# Find node closest to (0.5, 0.5)
center = np.array([0.5, 0.5])
distances = np.linalg.norm(mesh.nodes - center, axis=1)
center_idx = np.argmin(distances)
x_c, y_c = mesh.nodes[center_idx]

u_num_center = solution[center_idx]
u_exact_center = mms.u_exact(0.5, 0.5)  # Exact at (0.5, 0.5) = 1.0

print(f"Closest node to (0.5, 0.5): ({x_c:.4f}, {y_c:.4f})")
print(f"Numerical solution: {u_num_center:.6f}")
print(f"Exact at (0.5, 0.5): {u_exact_center:.6f}")
print(f"Error: {abs(u_num_center - u_exact_center):.6f}")

# Test 3: Check boundary conditions
print("\n" + "=" * 70)
print("TEST 3: Boundary Conditions")
print("-" * 70)

boundary_vals = solution[mesh.boundary - 1]
print(f"Boundary values (should all be ~0):")
print(f"  Max abs: {np.abs(boundary_vals).max():.6e}")
print(f"  Mean abs: {np.abs(boundary_vals).mean():.6e}")

if np.abs(boundary_vals).max() > 1e-6:
    print("❌ BOUNDARY CONDITIONS NOT SATISFIED!")
    print("   This is a critical bug in apply_dirichlet_zero()")
    print("\nSample boundary values:")
    for i in range(min(5, len(boundary_vals))):
        idx = mesh.boundary[i] - 1
        x, y = mesh.nodes[idx]
        print(f"   ({x:.3f}, {y:.3f}): {boundary_vals[i]:.6e}")
else:
    print("✓ Boundary conditions satisfied")

# Test 4: Check if solution is symmetric
print("\n" + "=" * 70)
print("TEST 4: Solution Symmetry")
print("-" * 70)
print("For u = sin(πx)sin(πy), solution should be symmetric about (0.5, 0.5)")

# Find pairs of symmetric points
symmetric_errors = []
for i in range(min(20, len(mesh.nodes))):
    x, y = mesh.nodes[i]
    # Symmetric point
    x_sym, y_sym = 1.0 - x, 1.0 - y
    
    # Find closest node to symmetric point
    sym_point = np.array([x_sym, y_sym])
    distances = np.linalg.norm(mesh.nodes - sym_point, axis=1)
    sym_idx = np.argmin(distances)
    
    if distances[sym_idx] < 0.05:  # Close enough
        diff = abs(solution[i] - solution[sym_idx])
        symmetric_errors.append(diff)

if symmetric_errors:
    mean_sym_error = np.mean(symmetric_errors)
    print(f"Mean symmetry error: {mean_sym_error:.6e}")
    if mean_sym_error > 0.01:
        print("❌ Solution is NOT symmetric - assembly might be wrong")
    else:
        print("✓ Solution appears symmetric")

# Test 5: Compare error distribution
print("\n" + "=" * 70)
print("TEST 5: Error Distribution")
print("-" * 70)

error = solution - u_exact
print(f"Error statistics:")
print(f"  Min:   {error.min():.6f}")
print(f"  Max:   {error.max():.6f}")
print(f"  Mean:  {error.mean():.6f}")  # Should be ~0 if unbiased
print(f"  Std:   {error.std():.6f}")
print(f"  L2:    {np.sqrt(np.mean(error**2)):.6f}")

if abs(error.mean()) > 0.1:
    print(f"\n❌ ERROR IS BIASED! Mean error = {error.mean():.3f}")
    print("   Solution is systematically offset from correct answer")
    print("   Possible causes:")
    print("   - Wrong sign in stiffness or load assembly")
    print("   - Wrong factor (missing 1/2, wrong quadrature weight, etc.)")
else:
    print("\n✓ Error appears unbiased")

# Test 6: Scaling test
print("\n" + "=" * 70)
print("TEST 6: Solution Scaling Test")
print("-" * 70)

# Check if solution * constant gives better match
scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
best_scale = 1.0
best_error = np.sqrt(np.mean(error**2))

print(f"{'Scale':<10} {'L2 error':<12}")
print("-" * 25)
for scale in scales:
    scaled_solution = solution * scale
    scaled_error = scaled_solution - u_exact
    scaled_L2 = np.sqrt(np.mean(scaled_error**2))
    print(f"{scale:<10.2f} {scaled_L2:<12.6f}")
    
    if scaled_L2 < best_error:
        best_error = scaled_L2
        best_scale = scale

if abs(best_scale - 1.0) > 0.1:
    print(f"\n❌ SCALING PROBLEM DETECTED!")
    print(f"   Solution should be multiplied by {best_scale:.2f}")
    print(f"   Check for missing factors in:")
    print(f"   - Stiffness matrix assembly (det_B factor?)")
    print(f"   - Load vector assembly (quadrature weights?)")
    print(f"   - Reference element formulas")
else:
    print(f"\n✓ No obvious scaling problem")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("If you see:")
print("  - Wrong magnitude → check det_B, quadrature weights")
print("  - BC not satisfied → check apply_dirichlet_zero()")
print("  - Biased error → check signs in assembly")
print("  - Scaling factor ≠ 1 → missing factor in K or F")
print("=" * 70)