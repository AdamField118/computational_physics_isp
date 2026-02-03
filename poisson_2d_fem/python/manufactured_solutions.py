"""
Manufactured solutions for verification
"""
import numpy as np

class ManufacturedSolution:
    """Base class for manufactured solutions"""
    
    def u_exact(self, x, y):
        """Exact solution"""
        raise NotImplementedError
    
    def f_source(self, x, y):
        """Source term f = -Δu"""
        raise NotImplementedError
    
    def g_boundary(self, x, y):
        """Boundary condition"""
        return self.u_exact(x, y)

class SineSolution(ManufacturedSolution):
    """
    u(x,y) = sin(πx) sin(πy)
    f(x,y) = 2π² sin(πx) sin(πy)
    """
    
    def u_exact(self, x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def f_source(self, x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

class PolynomialSolution(ManufacturedSolution):
    """
    u(x,y) = x(1-x) y(1-y)
    f(x,y) = 2x(1-x) + 2y(1-y)
    """
    
    def u_exact(self, x, y):
        return x * (1 - x) * y * (1 - y)
    
    def f_source(self, x, y):
        return 2*x*(1-x) + 2*y*(1-y)