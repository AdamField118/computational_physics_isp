"""
Main FEM driver - orchestrates mesh → assembly → solve → postprocess
"""
import numpy as np
import sys
sys.path.append('../fortran')

# Import compiled Fortran modules (after f2py compilation)
import fem_fortran  # Will contain all Fortran subroutines

from mesh_generator import SimpleMesh
from visualization import plot_solution, plot_mesh, plot_convergence

class PoissonSolver2D:
    """2D Poisson equation FEM solver"""
    
    def __init__(self, domain='unit_square', max_area=0.01):
        self.mesh = SimpleMesh(domain, max_area)
        self.solution = None
        
    def solve(self, f_source, g_boundary=None):
        """
        Solve -Δu = f with u = g on boundary
        
        Args:
            f_source: Python function f(x, y) -> float
            g_boundary: Python function g(x, y) -> float (default: 0)
        """
        # Generate mesh
        print("Generating mesh...")
        self.mesh.generate()
        
        # Call Fortran assembly
        print("Assembling system...")
        K, F = self._assemble_system(f_source)
        
        # Apply boundary conditions
        print("Applying boundary conditions...")
        K, F = self._apply_bc(K, F, g_boundary)
        
        # Solve
        print("Solving linear system...")
        self.solution = fem_fortran.solve_system(K, F)
        
        print("Done!")
        return self.solution
    
    def _assemble_system(self, f_source):
        """Call Fortran assembly routines"""
        n_nodes = self.mesh.nodes.shape[0]
        
        # Pass mesh to Fortran
        # Note: f2py handles array conversion automatically
        K = fem_fortran.assemble_stiffness(
            self.mesh.nodes,
            self.mesh.elements,
            n_nodes
        )
        
        # For load vector, need to evaluate f at quadrature points
        # This requires callback from Fortran → Python
        # Simpler: evaluate f at nodes, interpolate (less accurate but ok)
        f_vals = np.array([f_source(x, y) for x, y in self.mesh.nodes])
        
        F = fem_fortran.assemble_load(
            self.mesh.nodes,
            self.mesh.elements,
            f_vals,
            n_nodes
        )
        
        return K, F
    
    def _apply_bc(self, K, F, g_boundary):
        """Apply Dirichlet boundary conditions"""
        if g_boundary is None:
            # Homogeneous BC
            K, F = fem_fortran.apply_bc_zero(
                K, F,
                self.mesh.boundary
            )
        else:
            # Non-homogeneous BC
            g_vals = np.array([
                g_boundary(self.mesh.nodes[i-1, 0], self.mesh.nodes[i-1, 1])
                for i in self.mesh.boundary
            ])
            K, F = fem_fortran.apply_bc_nonzero(
                K, F,
                self.mesh.boundary,
                g_vals
            )
        
        return K, F
    
    def compute_errors(self, u_exact):
        """
        Compute L2 and H1 errors against exact solution
        
        Args:
            u_exact: Python function u(x, y) -> float
        
        Returns:
            (L2_error, H1_error, Linf_error)
        """
        # Evaluate exact solution at nodes
        u_exact_vals = np.array([u_exact(x, y) for x, y in self.mesh.nodes])
        
        # L-infinity error (max norm)
        Linf_error = np.max(np.abs(self.solution - u_exact_vals))
        
        # L2 and H1 errors require integration over elements
        # Call Fortran routine
        L2_error, H1_error = fem_fortran.compute_errors(
            self.mesh.nodes,
            self.mesh.elements,
            self.solution,
            u_exact_vals  # For now, simple nodal comparison
        )
        
        return L2_error, H1_error, Linf_error