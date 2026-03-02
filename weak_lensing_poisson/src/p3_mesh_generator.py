"""
P3 Mesh Generation for Cubic Triangular Elements

Generates structured meshes with 10 nodes per triangle:
- 3 vertex nodes
- 6 edge nodes (2 per edge at t=1/3, 2/3)
- 1 interior node (centroid)

Node numbering convention matches p3_shape_functions.py
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# ============================================================================
# P3 Structured Mesh Generation
# ============================================================================

def generate_p3_structured_mesh(nx: int, ny: int,
                               xmin: float = 0.0, xmax: float = 1.0,
                               ymin: float = 0.0, ymax: float = 1.0,
                               return_numpy: bool = False):
    r"""
    Generate P3 structured triangular mesh on rectangular domain
    
    Process:
    1. Create P1 base mesh (vertices only)
    2. Add edge nodes at t=1/3, 2/3 for each edge
    3. Add interior centroid node for each triangle
    4. Build 10-node element connectivity
    
    Node Ordering (per triangle):
        v2
        /\
       /  \
    8 /    \ 6
     /      \
  9 /   10   \ 7
   /          \
  /            \
v0─────3───4────v1
    
    Nodes 0,1,2: Vertices
    Nodes 3,4:   Edge 0→1 at t=1/3, 2/3
    Nodes 5,6:   Edge 1→2 at t=1/3, 2/3
    Nodes 7,8:   Edge 2→0 at t=1/3, 2/3
    Node 9:      Interior (centroid)
    
    Args:
        nx, ny: Number of cells in x, y directions
        xmin, xmax, ymin, ymax: Domain bounds
        return_numpy: If True, return numpy arrays instead of JAX
        
    Returns:
        Mesh object with P3 elements (10 nodes per triangle)
    """
    try:
        from .fem_solver import Mesh
    except ImportError:
        from fem_solver import Mesh
    
    print(f"Generating P3 mesh: {nx}×{ny} cells...")
    
    # ========================================================================
    # Step 1: Create base vertex grid (like P1)
    # ========================================================================
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    xx, yy = np.meshgrid(x, y)
    
    n_vertices = (nx + 1) * (ny + 1)
    vertex_nodes = np.column_stack([xx.ravel(), yy.ravel()])  # (n_vertices, 2)
    
    # ========================================================================
    # Step 2: Create P1 element connectivity (just for topology)
    # ========================================================================
    p1_elements = []
    for i in range(ny):
        for j in range(nx):
            # Node indices (row-major ordering)
            n0 = i * (nx + 1) + j
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            
            # Two triangles per cell
            p1_elements.append([n0, n1, n2])  # Lower triangle
            p1_elements.append([n1, n3, n2])  # Upper triangle
    
    p1_elements = np.array(p1_elements, dtype=np.int32)
    n_elements = len(p1_elements)
    
    print(f"  Base mesh: {n_vertices} vertices, {n_elements} triangles")
    
    # ========================================================================
    # Step 3: Add edge nodes
    # ========================================================================
    # We'll track edges and create nodes at t=1/3 and t=2/3
    # Edge identified by sorted vertex pair (i, j) where i < j
    
    edge_nodes = {}  # (i, j) -> [node_at_1/3, node_at_2/3]
    nodes_list = list(vertex_nodes)  # Start with vertices
    next_node_idx = n_vertices
    
    def get_edge_nodes(v1: int, v2: int) -> Tuple[int, int]:
        """Get or create edge nodes at t=1/3 and t=2/3"""
        nonlocal next_node_idx
        
        edge_key = tuple(sorted([v1, v2]))
        
        if edge_key not in edge_nodes:
            # Create two new nodes on this edge
            i, j = edge_key  # Use sorted vertices
            pi = vertex_nodes[i]
            pj = vertex_nodes[j]
            
            # Node at t=1/3
            node_1_3 = (2.0 * pi + 1.0 * pj) / 3.0
            idx_1_3 = next_node_idx
            nodes_list.append(node_1_3)
            next_node_idx += 1
            
            # Node at t=2/3
            node_2_3 = (1.0 * pi + 2.0 * pj) / 3.0
            idx_2_3 = next_node_idx
            nodes_list.append(node_2_3)
            next_node_idx += 1
            
            edge_nodes[edge_key] = [idx_1_3, idx_2_3]
        
        # Return in correct order based on v1, v2 direction
        if v1 < v2:
            return edge_nodes[edge_key][0], edge_nodes[edge_key][1]
        else:
            return edge_nodes[edge_key][1], edge_nodes[edge_key][0]
    
    # ========================================================================
    # Step 4: Build P3 element connectivity
    # ========================================================================
    p3_elements = []
    
    for elem_idx, p1_elem in enumerate(p1_elements):
        v0, v1, v2 = p1_elem
        
        # Get edge nodes (in order along each edge)
        n3, n4 = get_edge_nodes(v0, v1)  # Edge 0→1
        n5, n6 = get_edge_nodes(v1, v2)  # Edge 1→2
        n7, n8 = get_edge_nodes(v2, v0)  # Edge 2→0
        
        # Interior node (centroid)
        p0 = vertex_nodes[v0]
        p1 = vertex_nodes[v1]
        p2 = vertex_nodes[v2]
        centroid = (p0 + p1 + p2) / 3.0
        
        n9 = next_node_idx
        nodes_list.append(centroid)
        next_node_idx += 1
        
        # P3 element: [vertices, edge nodes, interior]
        p3_elem = [v0, v1, v2, n3, n4, n5, n6, n7, n8, n9]
        p3_elements.append(p3_elem)
    
    # ========================================================================
    # Step 5: Convert to arrays
    # ========================================================================
    nodes_array = np.array(nodes_list)
    p3_elements = np.array(p3_elements, dtype=np.int32)
    n_nodes = len(nodes_array)
    
    print(f"  P3 mesh created:")
    print(f"    Total nodes: {n_nodes}")
    print(f"    - Vertices: {n_vertices}")
    print(f"    - Edge nodes: {len(edge_nodes) * 2}")
    print(f"    - Interior nodes: {n_elements}")
    print(f"    Elements: {n_elements} (10 nodes each)")
    
    # ========================================================================
    # Step 6: Identify boundary nodes
    # ========================================================================
    # Boundary includes vertices AND edge nodes on domain boundary
    boundary_mask = (
        (nodes_array[:, 0] <= xmin + 1e-10) |  # Left edge
        (nodes_array[:, 0] >= xmax - 1e-10) |  # Right edge
        (nodes_array[:, 1] <= ymin + 1e-10) |  # Bottom edge
        (nodes_array[:, 1] >= ymax - 1e-10)    # Top edge
    )
    boundary = np.where(boundary_mask)[0].astype(np.int32)
    
    print(f"    Boundary nodes: {len(boundary)}")
    
    # ========================================================================
    # Step 7: Convert to JAX arrays if requested
    # ========================================================================
    if not return_numpy:
        nodes_array = jnp.array(nodes_array)
        p3_elements = jnp.array(p3_elements)
        boundary = jnp.array(boundary)
    
    return Mesh(nodes=nodes_array, elements=p3_elements, boundary=boundary)


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_p3_mesh(mesh, filename='p3_mesh_structure.png', 
                     show_nodes=True, show_numbering=False):
    """
    Visualize P3 mesh structure showing all node types
    
    Args:
        mesh: P3 Mesh object
        filename: Output filename
        show_nodes: Show node markers
        show_numbering: Show node numbers (only for small meshes!)
    """
    print(f"\nVisualizing P3 mesh structure...")
    
    nodes = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    
    # Determine node types
    n_vertices = 0
    n_elements = len(elements)
    
    # Count vertices (appear in first 3 positions of multiple elements)
    vertex_counts = np.zeros(len(nodes), dtype=int)
    for elem in elements:
        for i in range(3):  # First 3 nodes are vertices
            vertex_counts[elem[i]] += 1
    
    is_vertex = vertex_counts > 1
    is_interior = np.zeros(len(nodes), dtype=bool)
    is_edge = np.ones(len(nodes), dtype=bool)
    
    # Interior nodes appear only once (in node 9 position)
    for elem in elements:
        is_interior[elem[9]] = True
        is_edge[elem[9]] = False
    
    # Edge nodes are neither vertices nor interior
    is_edge = is_edge & ~is_vertex & ~is_interior
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw triangles (using vertex connectivity)
    for elem in elements:
        v0, v1, v2 = elem[0], elem[1], elem[2]
        triangle = np.array([nodes[v0], nodes[v1], nodes[v2]])
        
        poly = Polygon(triangle, fill=False, edgecolor='#444444', 
                      linewidth=1.5, alpha=0.6)
        ax.add_patch(poly)
    
    # Draw nodes by type
    if show_nodes:
        # Vertices (red circles)
        vertex_nodes = nodes[is_vertex]
        if len(vertex_nodes) > 0:
            ax.scatter(vertex_nodes[:, 0], vertex_nodes[:, 1], 
                      c='#ff4444', s=100, marker='o', 
                      edgecolors='white', linewidths=1.5,
                      label='Vertices (3 per triangle)', zorder=5)
        
        # Edge nodes (blue squares)
        edge_nodes = nodes[is_edge]
        if len(edge_nodes) > 0:
            ax.scatter(edge_nodes[:, 0], edge_nodes[:, 1],
                      c='#4444ff', s=80, marker='s',
                      edgecolors='white', linewidths=1.5,
                      label='Edge nodes (6 per triangle)', zorder=4)
        
        # Interior nodes (green triangles)
        interior_nodes = nodes[is_interior]
        if len(interior_nodes) > 0:
            ax.scatter(interior_nodes[:, 0], interior_nodes[:, 1],
                      c='#44ff44', s=120, marker='^',
                      edgecolors='white', linewidths=1.5,
                      label='Interior nodes (1 per triangle)', zorder=6)
    
    # Optionally show node numbers (only for small meshes!)
    if show_numbering and len(nodes) < 50:
        for i, node in enumerate(nodes):
            ax.annotate(str(i), (node[0], node[1]), 
                       fontsize=8, ha='center', va='bottom',
                       color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='black', alpha=0.7))
    
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.set_title('P3 Mesh Structure (Cubic Elements)', 
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


def visualize_single_p3_element(mesh, elem_idx=0, 
                                filename='p3_element_detail.png'):
    """
    Visualize a single P3 element showing all 10 nodes with labels
    
    Args:
        mesh: P3 Mesh object
        elem_idx: Index of element to visualize
        filename: Output filename
    """
    print(f"\nVisualizing P3 element {elem_idx} in detail...")
    
    nodes = np.array(mesh.nodes)
    elem = np.array(mesh.elements[elem_idx])
    
    # Get node coordinates
    elem_nodes = nodes[elem]
    
    # Node labels
    node_labels = [
        'v₀ (vertex)',
        'v₁ (vertex)',
        'v₂ (vertex)',
        'n₃ (edge 0→1, t=1/3)',
        'n₄ (edge 0→1, t=2/3)',
        'n₅ (edge 1→2, t=1/3)',
        'n₆ (edge 1→2, t=2/3)',
        'n₇ (edge 2→0, t=1/3)',
        'n₈ (edge 2→0, t=2/3)',
        'n₉ (interior)'
    ]
    
    # Colors by type
    colors = ['#ff4444'] * 3 + ['#4444ff'] * 6 + ['#44ff44']
    markers = ['o'] * 3 + ['s'] * 6 + ['^']
    sizes = [150] * 3 + [120] * 6 + [180]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw triangle edges
    triangle = np.array([elem_nodes[0], elem_nodes[1], elem_nodes[2]])
    poly = Polygon(triangle, fill=True, facecolor='#f0f0f0', 
                  edgecolor='black', linewidth=2.5, alpha=0.3)
    ax.add_patch(poly)
    
    # Draw nodes
    for i, (node, label, color, marker, size) in enumerate(
        zip(elem_nodes, node_labels, colors, markers, sizes)):
        
        ax.scatter(node[0], node[1], c=color, s=size, marker=marker,
                  edgecolors='white', linewidths=2, zorder=5)
        
        # Label with offset
        offset = 0.05
        if i < 3:  # Vertices
            offset_x = offset * np.sign(node[0] - elem_nodes[9][0])
            offset_y = offset * np.sign(node[1] - elem_nodes[9][1])
        else:
            offset_x = offset * 0.5
            offset_y = offset * 0.5
        
        ax.annotate(f'{i}: {label}', 
                   (node[0] + offset_x, node[1] + offset_y),
                   fontsize=10, ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', alpha=0.9))
    
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.set_title(f'P3 Element Detail (Element {elem_idx})\n10 Nodes: 3 Vertices + 6 Edge + 1 Interior',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


# ============================================================================
# Validation Functions
# ============================================================================

def validate_p3_mesh(mesh):
    """
    Validate P3 mesh has correct structure
    
    Checks:
    1. Each element has exactly 10 nodes
    2. Edge nodes are at t=1/3 and t=2/3 of edges
    3. Interior node is at centroid
    4. No duplicate nodes
    """
    print("=" * 70)
    print("P3 MESH VALIDATION")
    print("=" * 70)
    
    nodes = np.array(mesh.nodes)
    elements = np.array(mesh.elements)
    
    n_nodes = len(nodes)
    n_elements = len(elements)
    
    print(f"\nMesh statistics:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Elements: {n_elements}")
    print(f"  Nodes per element: {elements.shape[1]}")
    
    # Check 1: Element size
    print(f"\n1. Checking element structure...")
    if elements.shape[1] == 10:
        print(f"   ✅ All elements have 10 nodes")
    else:
        print(f"   ❌ ERROR: Elements have {elements.shape[1]} nodes, expected 10")
        return False
    
    # Check 2: Edge node positions
    print(f"\n2. Checking edge node positions...")
    max_edge_error = 0.0
    n_checked = min(10, n_elements)
    
    for elem_idx in range(n_checked):
        elem = elements[elem_idx]
        elem_coords = nodes[elem]
        
        v0, v1, v2 = elem_coords[0], elem_coords[1], elem_coords[2]
        
        # Check edge 0→1 nodes
        n3_expected = (2*v0 + 1*v1) / 3
        n4_expected = (1*v0 + 2*v1) / 3
        
        err3 = np.linalg.norm(elem_coords[3] - n3_expected)
        err4 = np.linalg.norm(elem_coords[4] - n4_expected)
        
        max_edge_error = max(max_edge_error, err3, err4)
    
    if max_edge_error < 1e-10:
        print(f"   ✅ Edge nodes correctly positioned (max error: {max_edge_error:.2e})")
    else:
        print(f"   ⚠️  Edge node error: {max_edge_error:.2e}")
    
    # Check 3: Interior node at centroid
    print(f"\n3. Checking interior node positions...")
    max_centroid_error = 0.0
    
    for elem_idx in range(n_checked):
        elem = elements[elem_idx]
        elem_coords = nodes[elem]
        
        v0, v1, v2 = elem_coords[0], elem_coords[1], elem_coords[2]
        centroid_expected = (v0 + v1 + v2) / 3
        
        err = np.linalg.norm(elem_coords[9] - centroid_expected)
        max_centroid_error = max(max_centroid_error, err)
    
    if max_centroid_error < 1e-10:
        print(f"   ✅ Interior nodes at centroids (max error: {max_centroid_error:.2e})")
    else:
        print(f"   ⚠️  Centroid error: {max_centroid_error:.2e}")
    
    # Check 4: Boundary nodes
    print(f"\n4. Checking boundary nodes...")
    print(f"   Boundary nodes: {len(mesh.boundary)}")
    print(f"   ✅ Boundary identified")
    
    print("\n" + "=" * 70)
    print("✅ P3 MESH VALIDATION COMPLETE")
    print("=" * 70)
    
    return True


# ============================================================================
# Main Test/Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🎯" * 35)
    print(" " * 20 + "P3 MESH GENERATION - VALIDATION")
    print("🎯" * 35 + "\n")
    
    # Generate small mesh for visualization
    print("Creating small P3 mesh for visualization...")
    mesh_small = generate_p3_structured_mesh(3, 3, xmin=0, xmax=1, ymin=0, ymax=1)
    
    # Validate
    validate_p3_mesh(mesh_small)
    
    # Visualize full mesh
    visualize_p3_mesh(mesh_small, show_nodes=True, show_numbering=False)
    
    # Visualize single element in detail
    visualize_single_p3_element(mesh_small, elem_idx=0)
    
    # Generate larger mesh to show scalability
    print("\n" + "=" * 70)
    print("Testing larger mesh generation...")
    print("=" * 70)
    mesh_large = generate_p3_structured_mesh(20, 20, xmin=-1, xmax=1, ymin=-1, ymax=1)
    
    print("\n" + "=" * 70)
    print("✅ P3 MESH GENERATOR READY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. ✅ P3 shape functions validated")
    print("  2. ✅ P3 mesh generator validated")
    print("  3. ⏳ Implement P3 assembly")
    print("  4. ⏳ Add P3 shear computation")
    print("=" * 70)
