"""
N-Body Simulation Visualization & Analysis
Creates animations and analysis plots for trajectory data

Adam Field - Computational Physics ISP
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from pathlib import Path
import json
import sys

sys.path.append('../jax')
from nbody_jax import (
    create_random_system, create_solar_system, NBodyState, NBodyConfig, 
    simulate as jax_simulate, compute_energy
)

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    print("Warning: JAX not available")
    HAS_JAX = False


class NBodyVisualizer:
    """Visualization tools for N-body simulations"""
    
    def __init__(self, output_dir='../../web/data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up nice plotting style
        plt.style.use('dark_background')
        self.colors = plt.cm.viridis(np.linspace(0, 1, 100))
    
    def create_trajectory_animation_2d(self, times, positions, masses, 
                                       output_file='nbody_2d.gif',
                                       fps=30, trail_length=50):
        """
        Create 2D animation of particle trajectories (XY projection)
        
        Args:
            times: (n_frames,) array of timestamps
            positions: (n_frames, N, 3) array of positions
            masses: (N,) array of particle masses
            output_file: filename for output GIF
            fps: frames per second
            trail_length: number of past positions to show as trails
        """
        print(f"Creating 2D trajectory animation...")
        
        n_frames, n_particles, _ = positions.shape
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('#0a0a0a')
        fig.patch.set_facecolor('#0a0a0a')
        
        # Set axis limits (with some padding)
        all_pos = positions.reshape(-1, 3)
        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
        padding = 0.1 * max(x_max - x_min, y_max - y_min)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_aspect('equal')
        
        # Particle sizes based on mass
        sizes = 100 * (masses / masses.max())**0.5
        
        # Initialize scatter plot with first frame data (not empty arrays)
        scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1], 
                           s=sizes, c=self.colors[:n_particles], 
                           alpha=0.9, edgecolors='white', linewidths=0.5)
        
        # Initialize trail lines
        trails = [ax.plot([], [], '-', alpha=0.3, linewidth=1, 
                         color=self.colors[i % len(self.colors)])[0] 
                 for i in range(n_particles)]
        
        # Title
        title = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=14, verticalalignment='top',
                       color='white', family='monospace')
        
        ax.set_xlabel('X Position', fontsize=12, color='white')
        ax.set_ylabel('Y Position', fontsize=12, color='white')
        ax.tick_params(colors='white')
        
        # Grid
        ax.grid(True, alpha=0.2, linestyle='--')
        
        def update(frame):
            # Update particle positions
            scatter.set_offsets(positions[frame, :, :2])
            
            # Update trails
            start_idx = max(0, frame - trail_length)
            for i in range(n_particles):
                trail_x = positions[start_idx:frame+1, i, 0]
                trail_y = positions[start_idx:frame+1, i, 1]
                trails[i].set_data(trail_x, trail_y)
            
            # Update title
            title.set_text(f'N-Body Simulation (N={n_particles})\n'
                          f'Time: {times[frame]:.2f}  |  Frame: {frame}/{n_frames-1}')
            
            return [scatter] + trails + [title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=n_frames, 
                           interval=1000/fps, blit=True)
        
        # Save
        output_path = self.output_dir / output_file
        print(f"  Saving to {output_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        print(f"  ✓ Saved!")
        
        plt.close()
        return output_path
    
    def create_trajectory_animation_3d(self, times, positions, masses,
                                       output_file='nbody_3d.gif',
                                       fps=30, trail_length=50):
        """
        Create 3D animation of particle trajectories
        
        Args:
            times: (n_frames,) array of timestamps
            positions: (n_frames, N, 3) array of positions
            masses: (N,) array of particle masses
            output_file: filename for output GIF
            fps: frames per second
            trail_length: number of past positions to show as trails
        """
        print(f"Creating 3D trajectory animation...")
        
        n_frames, n_particles, _ = positions.shape
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0a0a0a')
        fig.patch.set_facecolor('#0a0a0a')
        
        # Set axis limits
        all_pos = positions.reshape(-1, 3)
        for i, label in enumerate(['X', 'Y', 'Z']):
            min_val, max_val = all_pos[:, i].min(), all_pos[:, i].max()
            padding = 0.1 * (max_val - min_val)
            getattr(ax, f'set_{label.lower()}lim')(min_val - padding, max_val + padding)
            getattr(ax, f'set_{label.lower()}label')(f'{label} Position', fontsize=12, color='white')
        
        # Particle sizes based on mass
        sizes = 100 * (masses / masses.max())**0.5
        
        # Initialize scatter plot with first frame data
        scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2],
                           s=sizes, c=self.colors[:n_particles],
                           alpha=0.9, edgecolors='white', linewidths=0.5)
        
        # Initialize trails
        trails = [ax.plot([], [], [], '-', alpha=0.3, linewidth=1,
                         color=self.colors[i % len(self.colors)])[0]
                 for i in range(n_particles)]
        
        # Title
        title = ax.text2D(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=14, verticalalignment='top',
                         color='white', family='monospace')
        
        # Styling
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        
        def update(frame):
            # Update particle positions
            scatter._offsets3d = (positions[frame, :, 0],
                                 positions[frame, :, 1],
                                 positions[frame, :, 2])
            
            # Update trails
            start_idx = max(0, frame - trail_length)
            for i in range(n_particles):
                trail_x = positions[start_idx:frame+1, i, 0]
                trail_y = positions[start_idx:frame+1, i, 1]
                trail_z = positions[start_idx:frame+1, i, 2]
                trails[i].set_data(trail_x, trail_y)
                trails[i].set_3d_properties(trail_z)
            
            # Update title
            title.set_text(f'N-Body Simulation (N={n_particles})\n'
                          f'Time: {times[frame]:.2f}  |  Frame: {frame}/{n_frames-1}')
            
            # Rotate view
            ax.view_init(elev=20, azim=frame * 0.5)
            
            return [scatter] + trails + [title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=n_frames,
                           interval=1000/fps, blit=False)
        
        # Save
        output_path = self.output_dir / output_file
        print(f"  Saving to {output_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        print(f"  ✓ Saved!")
        
        plt.close()
        return output_path
    
    def plot_energy_conservation(self, times, energies, output_file='energy_conservation.png'):
        """
        Plot kinetic, potential, and total energy over time
        
        Args:
            times: (n_frames,) array of timestamps
            energies: (n_frames, 3) array of [KE, PE, Total] energies
            output_file: output filename
        """
        print(f"Creating energy conservation plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.patch.set_facecolor('#0a0a0a')
        
        # Plot energies
        ax1.plot(times, energies[:, 0], 'c-', linewidth=2, label='Kinetic Energy', alpha=0.8)
        ax1.plot(times, energies[:, 1], 'm-', linewidth=2, label='Potential Energy', alpha=0.8)
        ax1.plot(times, energies[:, 2], 'y-', linewidth=2, label='Total Energy', alpha=0.8)
        
        ax1.set_xlabel('Time', fontsize=12, color='white')
        ax1.set_ylabel('Energy', fontsize=12, color='white')
        ax1.set_title('Energy Evolution', fontsize=14, fontweight='bold', color='white')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#0a0a0a')
        ax1.tick_params(colors='white')
        
        # Plot energy drift (relative)
        E_initial = energies[0, 2]
        drift = (energies[:, 2] - E_initial) / np.abs(E_initial) * 100
        
        ax2.plot(times, drift, 'r-', linewidth=2, alpha=0.8)
        ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Time', fontsize=12, color='white')
        ax2.set_ylabel('Energy Drift (%)', fontsize=12, color='white')
        ax2.set_title('Total Energy Conservation', fontsize=14, fontweight='bold', color='white')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#0a0a0a')
        ax2.tick_params(colors='white')
        
        # Add statistics
        max_drift = np.abs(drift).max()
        final_drift = drift[-1]
        stats_text = f'Max drift: {max_drift:.6f}%\nFinal drift: {final_drift:.6f}%'
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                color='white', family='monospace')
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
        
        return output_path
    
    def plot_speedup_analysis(self, benchmark_file='benchmark_results.json',
                              output_file='speedup_analysis.png'):
        """
        Create speedup plots relative to Python baseline
        
        Args:
            benchmark_file: JSON file with benchmark results
            output_file: output filename
        """
        print(f"Creating speedup analysis plot...")
        
        # Load benchmark data
        benchmark_path = self.output_dir / benchmark_file
        if not benchmark_path.exists():
            print(f"  Warning: {benchmark_path} not found, skipping speedup analysis")
            return None
            
        with open(benchmark_path, 'r') as f:
            data = json.load(f)
        
        results = data['results']
        
        # Organize by implementation and N
        implementations = {}
        for r in results:
            impl = r['implementation']
            if impl not in implementations:
                implementations[impl] = {'N': [], 'time_per_step': []}
            implementations[impl]['N'].append(r['n_particles'])
            implementations[impl]['time_per_step'].append(r['time_per_step'] * 1000)  # ms
        
        # Get Python baseline
        python_data = implementations.get('Python (NumPy)', None)
        if python_data is None:
            print("  Warning: No Python baseline found")
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#0a0a0a')
        
        # Plot 1: Speedup vs N
        for impl, data_dict in implementations.items():
            if impl == 'Python (NumPy)':
                continue
            
            N_vals = np.array(data_dict['N'])
            times = np.array(data_dict['time_per_step'])
            
            # Compute speedup relative to Python
            python_times = np.array(python_data['time_per_step'])
            speedup = python_times / times
            
            ax1.loglog(N_vals, speedup, 'o-', linewidth=2, markersize=8, 
                      label=impl, alpha=0.8)
        
        ax1.set_xlabel('Number of Particles (N)', fontsize=12, color='white')
        ax1.set_ylabel('Speedup vs Python', fontsize=12, color='white')
        ax1.set_title('Performance Speedup Analysis', fontsize=14, fontweight='bold', color='white')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3, which='both')
        ax1.set_facecolor('#0a0a0a')
        ax1.tick_params(colors='white')
        ax1.axhline(1, color='red', linestyle='--', alpha=0.5, label='Python baseline')
        
        # Plot 2: Absolute performance
        for impl, data_dict in implementations.items():
            N_vals = np.array(data_dict['N'])
            times = np.array(data_dict['time_per_step'])
            
            ax2.loglog(N_vals, times, 'o-', linewidth=2, markersize=8,
                      label=impl, alpha=0.8)
        
        ax2.set_xlabel('Number of Particles (N)', fontsize=12, color='white')
        ax2.set_ylabel('Time per Step (ms)', fontsize=12, color='white')
        ax2.set_title('Absolute Performance', fontsize=14, fontweight='bold', color='white')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.set_facecolor('#0a0a0a')
        ax2.tick_params(colors='white')
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
        
        return output_path
    
    def plot_trajectory_snapshots(self, positions, masses, times,
                                  snapshot_indices=[0, 25, 50, 75, 99],
                                  output_file='trajectory_snapshots.png'):
        """
        Create panel of trajectory snapshots at different times
        
        Args:
            positions: (n_frames, N, 3) array
            masses: (N,) array
            times: (n_frames,) array
            snapshot_indices: list of frame indices to plot
            output_file: output filename
        """
        print(f"Creating trajectory snapshots...")
        
        n_snapshots = len(snapshot_indices)
        fig, axes = plt.subplots(1, n_snapshots, figsize=(4*n_snapshots, 4))
        fig.patch.set_facecolor('#0a0a0a')
        
        if n_snapshots == 1:
            axes = [axes]
        
        n_particles = positions.shape[1]
        sizes = 100 * (masses / masses.max())**0.5
        
        for idx, (ax, frame_idx) in enumerate(zip(axes, snapshot_indices)):
            ax.set_facecolor('#0a0a0a')
            
            # Plot particles
            ax.scatter(positions[frame_idx, :, 0], positions[frame_idx, :, 1],
                      s=sizes, c=self.colors[:n_particles], alpha=0.9,
                      edgecolors='white', linewidths=0.5)
            
            # Plot trails (last 20 frames)
            trail_start = max(0, frame_idx - 20)
            for i in range(n_particles):
                ax.plot(positions[trail_start:frame_idx+1, i, 0],
                       positions[trail_start:frame_idx+1, i, 1],
                       '-', alpha=0.3, linewidth=1, color=self.colors[i % len(self.colors)])
            
            ax.set_title(f't = {times[frame_idx]:.2f}', fontsize=12, color='white')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            ax.tick_params(colors='white')
            
            if idx == 0:
                ax.set_ylabel('Y Position', fontsize=11, color='white')
            ax.set_xlabel('X Position', fontsize=11, color='white')
        
        plt.suptitle(f'N-Body Evolution (N={n_particles})', 
                    fontsize=14, fontweight='bold', color='white', y=0.98)
        plt.tight_layout()
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
        
        return output_path


def run_simulation_and_visualize(N=50, n_steps=1000, save_every=10, system_type='random'):
    """
    Run a simulation and create all visualizations
    
    Args:
        N: number of particles
        n_steps: number of timesteps
        save_every: save state every N steps
        system_type: 'random' or 'solar'
    """
    if not HAS_JAX:
        print("JAX not available, cannot run simulation")
        return
    
    print("=" * 70)
    print(f"N-Body Simulation & Visualization")
    print("=" * 70)
    print(f"System: {system_type}, N={N}, steps={n_steps}")
    print()
    
    # Create initial conditions
    config = NBodyConfig(G=1.0, softening=0.1, dt=0.01)
    
    if system_type == 'solar':
        initial_state = create_solar_system()
        N = initial_state.masses.shape[0]
    else:
        key = jax.random.PRNGKey(42)
        initial_state = create_random_system(N, key, position_scale=10.0, velocity_scale=0.5)
    
    print(f"Running simulation...")
    times, positions, velocities = jax_simulate(initial_state, config, n_steps, save_every)
    print(f"  ✓ Simulation complete!")
    
    # Compute energies over time
    print(f"\nComputing energy evolution...")
    n_frames = len(times)
    energies = np.zeros((n_frames, 3))
    
    for i in range(n_frames):
        state = NBodyState(
            positions=positions[i],
            velocities=velocities[i],
            masses=initial_state.masses,
            time=times[i]
        )
        ke, pe, total = compute_energy(state, config)
        energies[i] = [ke, pe, total]
    
    print(f"  ✓ Energies computed!")
    
    # Create visualizations
    viz = NBodyVisualizer()
    
    print(f"\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    
    # 1. Energy conservation plot
    viz.plot_energy_conservation(np.array(times), energies)
    
    # 2. Trajectory snapshots
    viz.plot_trajectory_snapshots(np.array(positions), np.array(initial_state.masses), 
                                  np.array(times))
    
    # 3. Speedup analysis
    viz.plot_speedup_analysis()
    
    # 4. 2D animation
    viz.create_trajectory_animation_2d(np.array(times), np.array(positions), 
                                       np.array(initial_state.masses),
                                       fps=30, trail_length=30)
    
    # 5. 3D animation (optional - can be slow)
    print("\nCreate 3D animation? (can be slow) [y/N]: ", end='')
    # Auto-skip for now, user can uncomment
    # response = input().lower()
    # if response == 'y':
    #     viz.create_trajectory_animation_3d(np.array(times), np.array(positions),
    #                                        np.array(initial_state.masses),
    #                                        fps=20, trail_length=30)
    
    print("\n" + "=" * 70)
    print("✓ All visualizations created successfully!")
    print("=" * 70)
    print(f"\nOutputs saved to: {viz.output_dir.absolute()}")


if __name__ == "__main__":
    # Run example visualization
    run_simulation_and_visualize(N=50, n_steps=1000, save_every=10, system_type='random')