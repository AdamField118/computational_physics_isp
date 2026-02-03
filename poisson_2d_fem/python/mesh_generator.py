"""
Mesh generation using Triangle library
Outputs mesh compatible with Fortran
"""
import numpy as np
import triangle as tr
import matplotlib.pyplot as plt

class SimpleMesh:
    """Simple triangular mesh for FEM"""
    
    def __init__(self, domain='unit_square', max_area=0.01):
        self.domain = domain
        self.max_area = max_area
        self.nodes = None      # (n_nodes, 2) - coordinates
        self.elements = None   # (n_elem, 3) - connectivity (1-indexed for Fortran!)
        self.boundary = None   # (n_boundary,) - node IDs on boundary
        
    def generate(self):
        """Generate mesh using Triangle"""
        if self.domain == 'unit_square':
            # Define vertices and segments
            vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
            
            # Triangle dictionary
            A = dict(vertices=vertices, segments=segments)
            
            # Triangulate with quality mesh
            B = tr.triangulate(A, f'pq30a{self.max_area}')
            
            self.nodes = B['vertices']  # (n_nodes, 2)
            # Convert to 1-indexed for Fortran
            self.elements = B['triangles'] + 1  # (n_elem, 3)
            
            # Find boundary nodes
            self.boundary = self._find_boundary_nodes()
            
        elif self.domain == 'lshaped':
            # L-shaped domain for singularity testing
            # ... implement later
            pass
    
    def _find_boundary_nodes(self):
        """Identify nodes on domain boundary"""
        tol = 1e-10
        boundary_flags = (
            (np.abs(self.nodes[:, 0] - 0.0) < tol) |  # x = 0
            (np.abs(self.nodes[:, 0] - 1.0) < tol) |  # x = 1
            (np.abs(self.nodes[:, 1] - 0.0) < tol) |  # y = 0
            (np.abs(self.nodes[:, 1] - 1.0) < tol)    # y = 1
        )
        return np.where(boundary_flags)[0] + 1  # 1-indexed
    
    def plot(self):
        """Visualize mesh"""
        fig, ax = plt.subplots(figsize=(8, 8))
        tr.plot(ax, vertices=self.nodes, triangles=self.elements - 1)
        ax.set_aspect('equal')
        plt.show()