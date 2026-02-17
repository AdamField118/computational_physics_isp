"""Test P3 at simpler coordinates"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import jax.numpy as jnp
from p3_shape_functions import compute_p3_shape_functions

# Reference triangle node coordinates
coords = jnp.array([
    [0.0, 0.0],  # vertex 0
    [1.0, 0.0],  # vertex 1
    [0.0, 1.0],  # vertex 2
    [1/3, 0.0],  # edge nodes
    [2/3, 0.0],
    [2/3, 1/3],
    [1/3, 2/3],
    [0.0, 2/3],
    [0.0, 1/3],
    [1/3, 1/3],  # interior
])

# Linear field: u = x + 2y
u_values = coords[:,0] + 2*coords[:,1]

# Test at SIMPLER points
test_points = [
    (0.5, 0.25, 0.5 + 2*0.25),  # xi, eta, expected
    (0.25, 0.25, 0.25 + 2*0.25),
    (0.1, 0.2, 0.1 + 2*0.2),
]

print("Testing linear interpolation at simple points:")
for xi, eta, expected in test_points:
    N = compute_p3_shape_functions(xi, eta)
    u_interp = jnp.dot(N, u_values)
    error = abs(u_interp - expected)
    status = "✅" if error < 1e-12 else "❌"
    print(f"  ({xi:.2f}, {eta:.2f}): interp={u_interp:.15f}, exact={expected:.15f}, err={error:.2e} {status}")