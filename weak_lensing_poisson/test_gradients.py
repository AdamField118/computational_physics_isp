"""Test if P3 gradients satisfy constant field reproduction"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import jax.numpy as jnp
from p3_shape_functions import compute_p3_shape_functions, compute_p3_shape_gradients_reference

# Test: For linear field u = ax + by, gradients should give [a, b] exactly
# Reference triangle vertices
coords_ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

# Test point
xi, eta = 0.3, 0.4

# Get shape functions and gradients
N = compute_p3_shape_functions(xi, eta)
dN_dxi_ref = compute_p3_shape_gradients_reference(xi, eta)

print("Testing constant gradient reproduction...")
print(f"Test point: ξ={xi}, η={eta}")
print(f"\nShape functions sum: {jnp.sum(N):.10f} (should be 1.0)")
print(f"Gradient sum ∂/∂ξ: {jnp.sum(dN_dxi_ref[:,0]):.10f} (should be 0.0)")
print(f"Gradient sum ∂/∂η: {jnp.sum(dN_dxi_ref[:,1]):.10f} (should be 0.0)")

# For physical triangle with vertices at (0,0), (1,0), (0,1)
# This is the SAME as reference, so J = I and gradients should match
print("\nFor identity mapping (J=I):")
print("∂/∂ξ should equal ∂/∂x")
print("∂/∂η should equal ∂/∂y")