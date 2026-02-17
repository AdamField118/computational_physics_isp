"""Test Jacobian computation"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import jax.numpy as jnp

# Simple right triangle: (0,0), (1,0), (0,1)
vertex_coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

x0, y0 = vertex_coords[0]
x1, y1 = vertex_coords[1]
x2, y2 = vertex_coords[2]

J = jnp.array([[x1 - x0, y1 - y0],
               [x2 - x0, y2 - y0]])

print("Jacobian for reference triangle:")
print(J)
print(f"\ndet(J) = {jnp.linalg.det(J)}")
print(f"Expected: 1.0 (area of reference triangle is 0.5, |J|=1)")

# For a right triangle with area A, we expect |J| = 2A
# Reference triangle has area 0.5, so |J| = 1

# Test on a different triangle
vertex_coords2 = jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
x0, y0 = vertex_coords2[0]
x1, y1 = vertex_coords2[1]
x2, y2 = vertex_coords2[2]

J2 = jnp.array([[x1 - x0, y1 - y0],
                [x2 - x0, y2 - y0]])

area = 0.5 * abs(x1*y2 - x2*y1)  # Triangle area formula
print(f"\n\nTriangle 2:")
print(f"Vertices: (0,0), (2,0), (0,3)")
print(f"Area = {area}")
print(f"det(J) = {jnp.linalg.det(J2)}")
print(f"Expected: {2*area}")