"""Check if any elements have negative Jacobian determinant"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import jax.numpy as jnp
from p3_mesh_generator import generate_p3_structured_mesh

mesh = generate_p3_structured_mesh(4, 4, 0, 1, 0, 1)
nodes = np.array(mesh.nodes)
elements = np.array(mesh.elements)

print("Checking Jacobian determinants...")
neg_count = 0
for elem in elements:
    coords = nodes[elem]
    v0, v1, v2 = coords[0], coords[1], coords[2]
    
    J = np.array([[v1[0] - v0[0], v1[1] - v0[1]],
                  [v2[0] - v0[0], v2[1] - v0[1]]])
    detJ = np.linalg.det(J)
    
    if detJ < 0:
        neg_count += 1
        print(f"  Element with vertices {elem[:3]}: det(J) = {detJ:.6f}")

if neg_count > 0:
    print(f"\n❌ {neg_count}/{len(elements)} elements have NEGATIVE det(J)!")
    print("   Need to use abs(det(J)) in assembly!")
else:
    print(f"\n✅ All {len(elements)} elements have positive det(J)")