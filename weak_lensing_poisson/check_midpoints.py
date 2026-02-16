"""
Check if edge midpoints are actually at the midpoints of edges
"""

import numpy as np
import jax.numpy as jnp


def check_midpoint_placement():
    """
    Verify that edge midpoints are correctly placed
    """
    print("=" * 70)
    print("CHECKING EDGE MIDPOINT PLACEMENT")
    print("=" * 70)
    
    from src.mesh_generator import generate_p2_structured_mesh
    
    mesh = generate_p2_structured_mesh(5, 5, xmin=0, xmax=1, ymin=0, ymax=1)
    
    print("\nP2 element node ordering:")
    print("  Nodes 0,1,2: Vertices")
    print("  Node 3: Midpoint of edge 0-1")
    print("  Node 4: Midpoint of edge 1-2")
    print("  Node 5: Midpoint of edge 2-0")
    
    # Check element 0 (works) and element 2 (fails)
    for elem_idx in [0, 2]:
        print(f"\n{'='*70}")
        print(f"ELEMENT {elem_idx}:")
        print(f"{'='*70}")
        
        nodes_idx = mesh.elements[elem_idx]
        coords = mesh.nodes[nodes_idx]
        
        print(f"\nNode indices: {nodes_idx}")
        print(f"\nNode coordinates:")
        for i in range(6):
            print(f"  Node {i} (global {nodes_idx[i]:3d}): ({coords[i,0]:7.4f}, {coords[i,1]:7.4f})")
        
        # Check if midpoints are actually midpoints
        print(f"\nVerifying edge midpoints:")
        
        # Node 3 should be midpoint of nodes 0-1
        v0, v1, v2 = coords[0], coords[1], coords[2]
        m01_actual = coords[3]
        m01_expected = (v0 + v1) / 2
        error_01 = np.linalg.norm(m01_actual - m01_expected)
        status_01 = "✓" if error_01 < 1e-10 else "✗"
        print(f"  Edge 0-1 midpoint (node 3):")
        print(f"    Expected: ({m01_expected[0]:7.4f}, {m01_expected[1]:7.4f})")
        print(f"    Actual:   ({m01_actual[0]:7.4f}, {m01_actual[1]:7.4f})")
        print(f"    Error:    {error_01:.6e} {status_01}")
        
        # Node 4 should be midpoint of nodes 1-2
        m12_actual = coords[4]
        m12_expected = (v1 + v2) / 2
        error_12 = np.linalg.norm(m12_actual - m12_expected)
        status_12 = "✓" if error_12 < 1e-10 else "✗"
        print(f"  Edge 1-2 midpoint (node 4):")
        print(f"    Expected: ({m12_expected[0]:7.4f}, {m12_expected[1]:7.4f})")
        print(f"    Actual:   ({m12_actual[0]:7.4f}, {m12_actual[1]:7.4f})")
        print(f"    Error:    {error_12:.6e} {status_12}")
        
        # Node 5 should be midpoint of nodes 2-0
        m20_actual = coords[5]
        m20_expected = (v2 + v0) / 2
        error_20 = np.linalg.norm(m20_actual - m20_expected)
        status_20 = "✓" if error_20 < 1e-10 else "✗"
        print(f"  Edge 2-0 midpoint (node 5):")
        print(f"    Expected: ({m20_expected[0]:7.4f}, {m20_expected[1]:7.4f})")
        print(f"    Actual:   ({m20_actual[0]:7.4f}, {m20_actual[1]:7.4f})")
        print(f"    Error:    {error_20:.6e} {status_20}")
        
        if error_01 > 1e-10 or error_12 > 1e-10 or error_20 > 1e-10:
            print(f"\n  ✗ MIDPOINTS ARE WRONG FOR ELEMENT {elem_idx}!")
            print(f"    → This explains the shear errors!")
        else:
            print(f"\n  ✓ All midpoints correct for element {elem_idx}")
    
    print("\n" + "=" * 70)
    
    # Also visualize the mesh structure
    print("\nMesh structure (first 2 cells):")
    print("\nCell 0 (elements 0, 1):")
    print("  Should form a square from (0,0) to (0.2, 0.2)")
    
    print("\nCell 1 (elements 2, 3):")
    print("  Should form a square from (0.2,0) to (0.4, 0.2)")
    
    # Get all unique nodes in first 4 elements
    all_nodes = set()
    for elem_idx in range(4):
        nodes_idx = mesh.elements[elem_idx]
        all_nodes.update(nodes_idx.tolist())
    
    print(f"\nTotal unique nodes in first 4 elements: {len(all_nodes)}")
    print("Expected: ~15-20 (some shared on boundaries)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    check_midpoint_placement()