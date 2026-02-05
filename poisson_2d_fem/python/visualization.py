"""
Visualization utilities for 2D Poisson FEM solver
Professional quality plots with dark theme
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import cm

# Set dark theme
plt.style.use('dark_background')

def plot_solution(mesh, solution, title='FEM Solution', cmap='viridis', save_path=None):
    """
    Plot FEM solution on triangular mesh
    
    Args:
        mesh: SimpleMesh object with nodes and elements
        solution: Solution vector at nodes
        title: Plot title
        cmap: Colormap name
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create matplotlib triangulation (0-indexed)
    triang = Triangulation(
        mesh.nodes[:, 0], 
        mesh.nodes[:, 1], 
        mesh.elements - 1  # Convert from 1-indexed to 0-indexed
    )
    
    # Contour plot
    levels = np.linspace(solution.min(), solution.max(), 20)
    tcf = ax.tricontourf(triang, solution, levels=levels, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(tcf, ax=ax)
    cbar.set_label('u(x,y)', fontsize=12)
    
    # Mesh overlay (light)
    ax.triplot(triang, 'w-', alpha=0.1, linewidth=0.5)
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(title, fontsize=16, color='#00ff41')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='#1a1a1a')
    
    plt.show()
    return fig, ax


def plot_mesh(mesh, highlight_boundary=True, save_path=None):
    """
    Visualize mesh structure
    
    Args:
        mesh: SimpleMesh object
        highlight_boundary: Whether to highlight boundary nodes
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create triangulation
    triang = Triangulation(
        mesh.nodes[:, 0], 
        mesh.nodes[:, 1], 
        mesh.elements - 1
    )
    
    # Plot mesh edges
    ax.triplot(triang, 'w-', linewidth=0.8, alpha=0.6)
    
    # Plot nodes
    ax.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], 'o', 
            color='#00ff41', markersize=3, alpha=0.5, label='Interior nodes')
    
    # Highlight boundary nodes
    if highlight_boundary and mesh.boundary is not None:
        boundary_coords = mesh.nodes[mesh.boundary - 1]  # Convert to 0-indexed
        ax.plot(boundary_coords[:, 0], boundary_coords[:, 1], 'o',
                color='#ff4444', markersize=5, label='Boundary nodes')
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(f'Mesh: {mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} elements', 
                 fontsize=16, color='#00ff41')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='#1a1a1a')
    
    plt.show()
    return fig, ax


def plot_convergence(h_vals, errors, labels, title='Convergence Study', save_path=None):
    """
    Plot convergence rates on log-log scale
    
    Args:
        h_vals: Array of mesh sizes
        errors: List of error arrays (one per norm)
        labels: List of labels for each error type
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Markers and colors
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#00ff41', '#00aaff', '#ffaa00', '#ff4444', '#aa00ff']
    
    # Plot each error type
    for i, (err, label) in enumerate(zip(errors, labels)):
        ax.loglog(h_vals, err, markers[i % len(markers)] + '-', 
                 label=label, linewidth=2, markersize=8,
                 color=colors[i % len(colors)])
    
    # Reference slopes
    h_ref = np.array([h_vals[0], h_vals[-1]])
    
    # O(h²) reference
    ref_scale = errors[0][0] / h_vals[0]**2
    ax.loglog(h_ref, ref_scale * h_ref**2, 'w--', 
             label='O(h²)', alpha=0.5, linewidth=1.5)
    
    # O(h) reference
    ref_scale = errors[0][0] / h_vals[0]
    ax.loglog(h_ref, ref_scale * h_ref, 'w:', 
             label='O(h)', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Mesh size h', fontsize=14)
    ax.set_ylabel('Error', fontsize=14)
    ax.set_title(title, fontsize=16, color='#00ff41')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='#1a1a1a')
    
    plt.show()
    return fig, ax


def plot_error_distribution(mesh, solution, exact_solution_func, save_path=None):
    """
    Plot pointwise error distribution on mesh
    
    Args:
        mesh: SimpleMesh object
        solution: Numerical solution vector
        exact_solution_func: Function u_exact(x, y)
        save_path: Optional path to save figure
    """
    # Compute exact solution at nodes
    exact = np.array([exact_solution_func(x, y) for x, y in mesh.nodes])
    error = np.abs(solution - exact)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Create triangulation
    triang = Triangulation(
        mesh.nodes[:, 0], 
        mesh.nodes[:, 1], 
        mesh.elements - 1
    )
    
    # Plot numerical solution
    tcf1 = ax1.tricontourf(triang, solution, levels=20, cmap='viridis')
    ax1.set_title('Numerical Solution', fontsize=14, color='#00ff41')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_aspect('equal')
    plt.colorbar(tcf1, ax=ax1)
    
    # Plot error distribution
    tcf2 = ax2.tricontourf(triang, error, levels=20, cmap='hot')
    ax2.set_title('Pointwise Error |u - u_h|', fontsize=14, color='#00ff41')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_aspect('equal')
    cbar2 = plt.colorbar(tcf2, ax=ax2)
    cbar2.set_label('Error', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='#1a1a1a')
    
    plt.show()
    return fig, (ax1, ax2)


def plot_3d_solution(mesh, solution, title='3D Solution View', save_path=None):
    """
    3D surface plot of solution
    
    Args:
        mesh: SimpleMesh object
        solution: Solution vector
        title: Plot title
        save_path: Optional path to save figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create triangulation
    triang = Triangulation(
        mesh.nodes[:, 0], 
        mesh.nodes[:, 1], 
        mesh.elements - 1
    )
    
    # Surface plot
    ax.plot_trisurf(triang, solution, cmap='viridis', 
                    edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('u(x,y)', fontsize=12)
    ax.set_title(title, fontsize=16, color='#00ff41', pad=20)
    
    # Dark background for 3D
    ax.set_facecolor('#1a1a1a')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='#1a1a1a')
    
    plt.show()
    return fig, ax