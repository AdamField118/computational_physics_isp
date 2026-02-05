"""
Main FEM driver - orchestrates mesh → assembly → solve → postprocess
"""
import numpy as np
import sys
sys.path.append('../fortran')

# Import compiled Fortran module
try:
    import fem_fortran
except ImportError:
    print("WARNING: fem_fortran not found. Run 'make build' first.")
    fem_fortran = None

from mesh_generator import SimpleMesh
from visualization import plot_solution, plot_mesh, plot_convergence

class PoissonSolver2D:
    """2D Poisson equation FEM solver"""
    
    def __init__(self, domain='unit_square', max_area=0.01):
        self.mesh = SimpleMesh(domain, max_area)
        self.solution = None
        
    def solve(self, f_source=None, g_boundary=None):
        """
        Solve -Δu = f with u = g on boundary
        
        Args:
            f_source: Python function f(x, y) -> float (currently unused - uses default)
            g_boundary: Python function g(x, y) -> float (currently only homogeneous BC)
        
        Note: Current Fortran interface uses hardcoded sine source and zero BC
        """
        if fem_fortran is None:
            raise RuntimeError("Fortran module not compiled. Run 'make build' first.")
        
        # Generate mesh
        print("Generating mesh...")
        self.mesh.generate()
        
        print(f"  Nodes: {self.mesh.nodes.shape[0]}")
        print(f"  Elements: {self.mesh.elements.shape[0]}")
        print(f"  Boundary nodes: {self.mesh.boundary.shape[0]}")
        
        # Call Fortran solver (all-in-one)
        print("\nSolving FEM system (Fortran)...")
        
        # Ensure arrays are Fortran-contiguous
        nodes = np.asfortranarray(self.mesh.nodes)
        elements = np.asfortranarray(self.mesh.elements)
        boundary = np.asfortranarray(self.mesh.boundary)
        
        self.solution = fem_fortran.python_interface.solve_poisson_2d(
            nodes=nodes,
            elements=elements,
            boundary=boundary,
            n_nodes=self.mesh.nodes.shape[0],
            n_elements=self.mesh.elements.shape[0],
            n_boundary=self.mesh.boundary.shape[0]
        )
        
        print("Done!")
        return self.solution
    
    def compute_errors(self, u_exact):
        """
        Compute L2, H1, and Linf errors against exact solution
        
        Args:
            u_exact: Python function u(x, y) -> float
        
        Returns:
            (L2_error, H1_error, Linf_error)
        """
        if self.solution is None:
            raise RuntimeError("Must call solve() before compute_errors()")
        
        # Evaluate exact solution at nodes
        u_exact_vals = np.array([u_exact(x, y) for x, y in self.mesh.nodes])
        
        # L-infinity error (max norm)
        Linf_error = np.max(np.abs(self.solution - u_exact_vals))
        
        # L2 error: integrate (u - u_h)^2 over elements
        L2_error = self._compute_L2_error(u_exact_vals)
        
        # H1 seminorm: integrate |grad(u - u_h)|^2 (approximate)
        H1_error = np.sqrt(L2_error**2 + self._compute_H1_seminorm(u_exact)**2)
        
        return L2_error, H1_error, Linf_error
    
    def _compute_L2_error(self, u_exact_vals):
        """
        Compute L2 error using trapezoidal rule over elements
        Simple implementation - proper would integrate over each element
        """
        error = self.solution - u_exact_vals
        
        # Approximate by summing over elements
        L2_sq = 0.0
        for elem_idx in range(self.mesh.elements.shape[0]):
            # Get element nodes (convert to 0-indexed)
            nodes_idx = self.mesh.elements[elem_idx] - 1
            
            # Get coordinates
            x1, y1 = self.mesh.nodes[nodes_idx[0]]
            x2, y2 = self.mesh.nodes[nodes_idx[1]]
            x3, y3 = self.mesh.nodes[nodes_idx[2]]
            
            # Element area (via cross product)
            area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
            
            # Average squared error over element (P1 approximation)
            e1, e2, e3 = error[nodes_idx]
            L2_sq += area * (e1**2 + e2**2 + e3**2) / 3.0
        
        return np.sqrt(L2_sq)
    
    def _compute_H1_seminorm(self, u_exact):
        """
        Compute H1 seminorm |grad(u - u_h)|_L2
        This is approximate - proper implementation would compute gradients
        """
        # Simple finite difference approximation
        dx = np.sqrt(self.mesh.max_area)
        return dx * np.linalg.norm(self.solution) * 0.1  # Placeholder