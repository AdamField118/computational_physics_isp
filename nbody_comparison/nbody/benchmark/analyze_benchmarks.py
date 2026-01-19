"""
Enhanced Analysis of N-Body Benchmark Results
Generates comprehensive plots and statistics from benchmark data

Adam Field - Computational Physics ISP
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches


class BenchmarkAnalyzer:
    """Analyze and visualize benchmark results"""
    
    def __init__(self, data_dir='../../web/data'):
        self.data_dir = Path(data_dir)
        self.results = None
        self.implementations = None
        
        # Set plotting style
        plt.style.use('dark_background')
    
    def load_results(self, filename='benchmark_results.json'):
        """Load benchmark results from JSON"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"Error: {filepath} not found")
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.results = data['results']
        print(f"Loaded {len(self.results)} benchmark results")
        print(f"Timestamp: {data['timestamp']}")
        
        # Organize by implementation
        self.implementations = {}
        for r in self.results:
            impl = r['implementation']
            if impl not in self.implementations:
                self.implementations[impl] = []
            self.implementations[impl].append(r)
        
        print(f"Implementations found: {list(self.implementations.keys())}")
        return True
    
    def create_comprehensive_report(self):
        """Generate all analysis plots"""
        if not self.results:
            print("No results loaded. Call load_results() first.")
            return
        
        print("\n" + "=" * 70)
        print("Generating Comprehensive Analysis Report")
        print("=" * 70 + "\n")
        
        self.plot_scaling_comparison()
        self.plot_efficiency_heatmap()
        self.plot_crossover_analysis()
        self.plot_energy_drift_comparison()
        self.generate_summary_table()
        
        print("\n" + "=" * 70)
        print("✓ Analysis complete!")
        print("=" * 70)
    
    def plot_scaling_comparison(self):
        """Enhanced scaling plot with theoretical O(N²) lines"""
        print("Creating scaling comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle('N-Body Performance Scaling Analysis', 
                    fontsize=16, fontweight='bold', color='white', y=0.995)
        
        # Get unique N values
        N_vals = sorted(list(set(r['n_particles'] for r in self.results)))
        
        # Define colors for each implementation
        colors = {
            'JAX (GPU)': '#00ff00',
            'Fortran (OpenMP)': '#ff00ff',
            'C++': '#ff8800',
            'C': '#00ffff',
            'Python (NumPy)': '#ff0000'
        }
        
        # Plot 1: Time per step vs N (log-log)
        ax1 = axes[0, 0]
        for impl, impl_results in self.implementations.items():
            N = [r['n_particles'] for r in impl_results]
            times = [r['time_per_step'] * 1000 for r in impl_results]  # ms
            
            color = colors.get(impl, '#ffffff')
            ax1.loglog(N, times, 'o-', linewidth=2.5, markersize=10, 
                      label=impl, color=color, alpha=0.9)
        
        # Add theoretical O(N²) line
        N_theory = np.array(N_vals)
        baseline = 0.001  # Adjust this to fit the plot
        ax1.loglog(N_theory, baseline * N_theory**2, '--', 
                  color='white', alpha=0.5, linewidth=2, label='O(N²) scaling')
        
        ax1.set_xlabel('Number of Particles (N)', fontsize=13, color='white')
        ax1.set_ylabel('Time per Step (ms)', fontsize=13, color='white')
        ax1.set_title('Computational Complexity', fontsize=14, fontweight='bold', color='white')
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, which='both', linestyle='--')
        ax1.set_facecolor('#0a0a0a')
        ax1.tick_params(colors='white', labelsize=11)
        
        # Plot 2: Total runtime vs N
        ax2 = axes[0, 1]
        for impl, impl_results in self.implementations.items():
            N = [r['n_particles'] for r in impl_results]
            runtime = [r['runtime'] for r in impl_results]
            
            color = colors.get(impl, '#ffffff')
            ax2.loglog(N, runtime, 'o-', linewidth=2.5, markersize=10,
                      label=impl, color=color, alpha=0.9)
        
        ax2.set_xlabel('Number of Particles (N)', fontsize=13, color='white')
        ax2.set_ylabel('Total Runtime (s)', fontsize=13, color='white')
        ax2.set_title('Total Execution Time', fontsize=14, fontweight='bold', color='white')
        ax2.legend(loc='best', fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3, which='both', linestyle='--')
        ax2.set_facecolor('#0a0a0a')
        ax2.tick_params(colors='white', labelsize=11)
        
        # Plot 3: Speedup vs Python baseline
        ax3 = axes[1, 0]
        python_data = self.implementations.get('Python (NumPy)', [])
        
        if python_data:
            python_times = {r['n_particles']: r['time_per_step'] 
                          for r in python_data}
            
            for impl, impl_results in self.implementations.items():
                if impl == 'Python (NumPy)':
                    continue
                
                N = []
                speedup = []
                for r in impl_results:
                    n = r['n_particles']
                    if n in python_times:
                        N.append(n)
                        speedup.append(python_times[n] / r['time_per_step'])
                
                color = colors.get(impl, '#ffffff')
                ax3.semilogx(N, speedup, 'o-', linewidth=2.5, markersize=10,
                           label=impl, color=color, alpha=0.9)
        
        ax3.axhline(1, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax3.set_xlabel('Number of Particles (N)', fontsize=13, color='white')
        ax3.set_ylabel('Speedup Factor', fontsize=13, color='white')
        ax3.set_title('Performance vs Python Baseline', fontsize=14, fontweight='bold', color='white')
        ax3.legend(loc='best', fontsize=10, framealpha=0.9)
        ax3.grid(True, alpha=0.3, which='both', linestyle='--')
        ax3.set_facecolor('#0a0a0a')
        ax3.tick_params(colors='white', labelsize=11)
        
        # Plot 4: Performance bar chart for largest N
        ax4 = axes[1, 1]
        largest_N = max(N_vals)
        
        impl_names = []
        times_at_max = []
        colors_list = []
        
        for impl, impl_results in self.implementations.items():
            for r in impl_results:
                if r['n_particles'] == largest_N:
                    impl_names.append(impl.replace(' ', '\n'))
                    times_at_max.append(r['time_per_step'] * 1000)
                    colors_list.append(colors.get(impl, '#ffffff'))
                    break
        
        bars = ax4.barh(impl_names, times_at_max, color=colors_list, alpha=0.8, edgecolor='white')
        ax4.set_xlabel('Time per Step (ms)', fontsize=13, color='white')
        ax4.set_title(f'Performance at N={largest_N}', fontsize=14, fontweight='bold', color='white')
        ax4.set_facecolor('#0a0a0a')
        ax4.tick_params(colors='white', labelsize=11)
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.set_xscale('log')
        
        # Add value labels on bars
        for bar, val in zip(bars, times_at_max):
            ax4.text(val, bar.get_y() + bar.get_height()/2, 
                    f'  {val:.3f} ms', va='center', ha='left',
                    color='white', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.data_dir / 'comprehensive_scaling.png'
        plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
    
    def plot_efficiency_heatmap(self):
        """Create heatmap showing efficiency across N and implementations"""
        print("Creating efficiency heatmap...")
        
        # Get unique N values and implementations
        N_vals = sorted(list(set(r['n_particles'] for r in self.results)))
        impl_names = sorted(self.implementations.keys())
        
        # Create matrix of times
        times_matrix = np.zeros((len(impl_names), len(N_vals)))
        
        for i, impl in enumerate(impl_names):
            for r in self.implementations[impl]:
                j = N_vals.index(r['n_particles'])
                times_matrix[i, j] = r['time_per_step'] * 1000  # ms
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#0a0a0a')
        
        # Plot heatmap (log scale)
        im = ax.imshow(np.log10(times_matrix), cmap='hot', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(N_vals)))
        ax.set_yticks(np.arange(len(impl_names)))
        ax.set_xticklabels(N_vals, color='white', fontsize=11)
        ax.set_yticklabels(impl_names, color='white', fontsize=11)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log₁₀(Time per Step [ms])', rotation=270, labelpad=20, 
                      fontsize=12, color='white')
        cbar.ax.tick_params(colors='white', labelsize=11)
        
        # Add text annotations
        for i in range(len(impl_names)):
            for j in range(len(N_vals)):
                text = ax.text(j, i, f'{times_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white", 
                             fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Number of Particles (N)', fontsize=13, color='white')
        ax.set_ylabel('Implementation', fontsize=13, color='white')
        ax.set_title('Performance Heatmap (Time per Step in ms)', 
                    fontsize=14, fontweight='bold', color='white')
        
        plt.tight_layout()
        
        output_path = self.data_dir / 'efficiency_heatmap.png'
        plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
    
    def plot_crossover_analysis(self):
        """Find and plot crossover points where GPU becomes faster"""
        print("Creating crossover analysis...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#0a0a0a')
        
        # Get JAX data
        jax_data = self.implementations.get('JAX (GPU)', [])
        if not jax_data:
            print("  No JAX data found, skipping crossover analysis")
            return
        
        jax_dict = {r['n_particles']: r['time_per_step'] * 1000 for r in jax_data}
        
        # Compare with other implementations
        colors = {'Fortran (OpenMP)': '#ff00ff', 'C++': '#ff8800', 
                 'C': '#00ffff', 'Python (NumPy)': '#ff0000'}
        
        for impl, impl_results in self.implementations.items():
            if impl == 'JAX (GPU)':
                continue
            
            N = []
            ratio = []  # Other / JAX (>1 means JAX is faster)
            
            for r in impl_results:
                n = r['n_particles']
                if n in jax_dict:
                    N.append(n)
                    ratio.append(r['time_per_step'] * 1000 / jax_dict[n])
            
            color = colors.get(impl, '#ffffff')
            ax.semilogx(N, ratio, 'o-', linewidth=2.5, markersize=10,
                       label=f'{impl} / JAX', color=color, alpha=0.9)
        
        ax.axhline(1, color='#00ff00', linestyle='--', linewidth=2, alpha=0.7,
                  label='JAX crossover (ratio=1)')
        ax.fill_between([min(N), max(N)], 0, 1, alpha=0.2, color='red',
                       label='JAX slower region')
        ax.fill_between([min(N), max(N)], 1, ax.get_ylim()[1], alpha=0.2, color='green',
                       label='JAX faster region')
        
        ax.set_xlabel('Number of Particles (N)', fontsize=13, color='white')
        ax.set_ylabel('Performance Ratio (Other / JAX)', fontsize=13, color='white')
        ax.set_title('GPU Crossover Analysis: When Does JAX Win?', 
                    fontsize=14, fontweight='bold', color='white')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white', labelsize=11)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        output_path = self.data_dir / 'crossover_analysis.png'
        plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
    
    def plot_energy_drift_comparison(self):
        """Compare energy conservation across implementations"""
        print("Creating energy drift comparison...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#0a0a0a')
        
        colors = {'JAX (GPU)': '#00ff00', 'Fortran (OpenMP)': '#ff00ff',
                 'C++': '#ff8800', 'C': '#00ffff', 'Python (NumPy)': '#ff0000'}
        
        for impl, impl_results in self.implementations.items():
            N = [r['n_particles'] for r in impl_results]
            drift = [r['energy_drift'] for r in impl_results]
            
            color = colors.get(impl, '#ffffff')
            ax.loglog(N, drift, 'o-', linewidth=2.5, markersize=10,
                     label=impl, color=color, alpha=0.9)
        
        ax.set_xlabel('Number of Particles (N)', fontsize=13, color='white')
        ax.set_ylabel('Energy Drift (%)', fontsize=13, color='white')
        ax.set_title('Energy Conservation Quality', 
                    fontsize=14, fontweight='bold', color='white')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white', labelsize=11)
        
        # Add acceptable drift region
        ax.axhline(0.1, color='yellow', linestyle='--', alpha=0.5, linewidth=2,
                  label='0.1% drift threshold')
        ax.fill_between(ax.get_xlim(), 0, 0.1, alpha=0.1, color='green')
        
        plt.tight_layout()
        
        output_path = self.data_dir / 'energy_drift.png'
        plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
    
    def generate_summary_table(self):
        """Generate text summary table"""
        print("Generating summary table...")
        
        output_path = self.data_dir / 'performance_summary.txt'
        
        with open(output_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("N-BODY SIMULATION: PERFORMANCE SUMMARY\n")
            f.write("=" * 100 + "\n\n")
            
            # Table header
            f.write(f"{'Implementation':<20} {'N':<8} {'Runtime (s)':<15} {'ms/step':<15} "
                   f"{'Energy Drift (%)':<18} {'vs Python':<12}\n")
            f.write("-" * 100 + "\n")
            
            # Get Python baseline
            python_times = {}
            if 'Python (NumPy)' in self.implementations:
                for r in self.implementations['Python (NumPy)']:
                    python_times[r['n_particles']] = r['time_per_step']
            
            # Print results
            for impl in sorted(self.implementations.keys()):
                for r in sorted(self.implementations[impl], key=lambda x: x['n_particles']):
                    speedup = ""
                    if r['n_particles'] in python_times and impl != 'Python (NumPy)':
                        speedup = f"{python_times[r['n_particles']] / r['time_per_step']:.1f}x"
                    
                    f.write(f"{impl:<20} {r['n_particles']:<8} {r['runtime']:<15.4f} "
                           f"{r['time_per_step']*1000:<15.4f} {r['energy_drift']:<18.6f} "
                           f"{speedup:<12}\n")
                f.write("\n")
            
            f.write("=" * 100 + "\n\n")
            
            # Key findings
            f.write("KEY FINDINGS:\n")
            f.write("-" * 100 + "\n\n")
            
            # Find fastest at N=1000
            results_1000 = [r for r in self.results if r['n_particles'] == 1000]
            if results_1000:
                fastest = min(results_1000, key=lambda x: x['time_per_step'])
                slowest = max(results_1000, key=lambda x: x['time_per_step'])
                
                f.write(f"At N=1000 particles:\n")
                f.write(f"  • Fastest: {fastest['implementation']} "
                       f"({fastest['time_per_step']*1000:.4f} ms/step)\n")
                f.write(f"  • Slowest: {slowest['implementation']} "
                       f"({slowest['time_per_step']*1000:.4f} ms/step)\n")
                f.write(f"  • Speedup: {slowest['time_per_step']/fastest['time_per_step']:.1f}x\n\n")
            
            # GPU vs CPU comparison
            if 'JAX (GPU)' in self.implementations and 'Fortran (OpenMP)' in self.implementations:
                jax_1000 = next((r for r in self.implementations['JAX (GPU)'] 
                               if r['n_particles'] == 1000), None)
                fort_1000 = next((r for r in self.implementations['Fortran (OpenMP)']
                                if r['n_particles'] == 1000), None)
                
                if jax_1000 and fort_1000:
                    f.write(f"GPU vs Multi-core CPU (N=1000):\n")
                    f.write(f"  • JAX (GPU): {jax_1000['time_per_step']*1000:.4f} ms/step\n")
                    f.write(f"  • Fortran (OpenMP): {fort_1000['time_per_step']*1000:.4f} ms/step\n")
                    f.write(f"  • GPU Advantage: {fort_1000['time_per_step']/jax_1000['time_per_step']:.1f}x\n\n")
        
        print(f"  ✓ Saved to {output_path}")
        
        # Also print to console
        with open(output_path, 'r') as f:
            print("\n" + f.read())


if __name__ == "__main__":
    analyzer = BenchmarkAnalyzer()
    
    if analyzer.load_results():
        analyzer.create_comprehensive_report()