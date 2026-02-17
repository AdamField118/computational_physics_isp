"""Simplest possible P3 test - single element"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import jax.numpy as jnp
from p3_shape_functions import compute_p3_shape_functions, compute_p3_shape_gradients_reference

# Reference triangle
coords = jnp.array([
    [0.0, 0.0],  # vertex 0
    [1.0, 0.0],  # vertex 1
    [0.0, 1.0],  # vertex 2
    [1/3, 0.0],  # edge 0-1, t=1/3
    [2/3, 0.0],  # edge 0-1, t=2/3
    [2/3, 1/3],  # edge 1-2, t=1/3
    [1/3, 2/3],  # edge 1-2, t=2/3
    [0.0, 2/3],  # edge 2-0, t=1/3
    [0.0, 1/3],  # edge 2-0, t=2/3
    [1/3, 1/3],  # interior
])

# Linear field: u = x + 2y
u_values = coords[:,0] + 2*coords[:,1]

# At center point (1/3, 1/3):
xi, eta = 1/3, 1/3
N = compute_p3_shape_functions(xi, eta)
u_interp = jnp.dot(N, u_values)
u_exact = 1/3 + 2*(1/3)
print(f"Linear field interpolation test:")
print(f"Interpolated u = {u_interp:.10f}")
print(f"Exact u = {u_exact:.10f}")
print(f"Error = {abs(u_interp - u_exact):.2e}")

if abs(u_interp - u_exact) < 1e-10:
    print("✅ P3 correctly reproduces linear functions")
else:
    print("❌ P3 FAILS to reproduce linear functions!")