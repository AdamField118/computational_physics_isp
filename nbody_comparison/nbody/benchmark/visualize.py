#!/usr/bin/env python3
"""
N-Body Simulation Visualization Script
Comprehensive performance analysis and visualization
Updated with Rust and Julia support
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Set style
plt.style.use('dark_background')

# Updated color scheme with Rust and Julia
COLORS = {
    'JAX (GPU)': '#00ff41',
    'Fortran (OpenMP)': '#ff00ff',
    'Rust': '#ff6b35',  # Orange for Rust
    'Julia': '#9558b2',  # Purple for Julia
    'C++': '#ff8800',
    'C': '#00ffff',
    'Python (NumPy)': '#ff0000'
}

# Implementation order for consistent plotting
IMPL_ORDER = [
    'JAX (GPU)',
    'Fortran (OpenMP)',
    'Rust',
    'Julia',
    'C++',
    'C',
    'Python (NumPy)'
]

def load_benchmark_data(filepath):
    """Load benchmark results from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['results'])

def plot_scaling(df, output_path):
    """Plot time per step vs N for all implementations"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for impl in IMPL_ORDER:
        impl_data = df[df['implementation'] == impl].sort_values('n_particles')
        if len(impl_data) > 0:
            ax.loglog(
                impl_data['n_particles'], 
                impl_data['time_per_step'] * 1000,  # Convert to ms
                'o-', 
                label=impl, 
                color=COLORS[impl],
                linewidth=2.5,
                markersize=8
            )
    
    ax.set_xlabel('Number of Particles (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time per Step (ms)', fontsize=14, fontweight='bold')
    ax.set_title('N-Body Simulation: Performance Scaling', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#0a0a0a')
    plt.close()
    print(f"✓ Saved: {output_path}")

def plot_speedup(df, output_path):
    """Plot speedup vs Python baseline"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get Python baseline
    python_data = df[df['implementation'] == 'Python (NumPy)'].set_index('n_particles')['time_per_step']
    
    for impl in IMPL_ORDER:
        if impl == 'Python (NumPy)':
            continue
            
        impl_data = df[df['implementation'] == impl].sort_values('n_particles')
        if len(impl_data) > 0:
            speedups = []
            n_values = []
            
            for _, row in impl_data.iterrows():
                n = row['n_particles']
                if n in python_data.index:
                    speedup = python_data[n] / row['time_per_step']
                    speedups.append(speedup)
                    n_values.append(n)
            
            if speedups:
                ax.loglog(
                    n_values, 
                    speedups,
                    'o-', 
                    label=impl, 
                    color=COLORS[impl],
                    linewidth=2.5,
                    markersize=8
                )
    
    ax.set_xlabel('Number of Particles (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup vs Python', fontsize=14, fontweight='bold')
    ax.set_title('N-Body Simulation: Speedup vs Python Baseline', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#0a0a0a')
    plt.close()
    print(f"✓ Saved: {output_path}")

def plot_energy_conservation(df, output_path):
    """Plot energy conservation quality"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for impl in IMPL_ORDER:
        impl_data = df[df['implementation'] == impl].sort_values('n_particles')
        if len(impl_data) > 0:
            ax.loglog(
                impl_data['n_particles'], 
                impl_data['energy_drift'],
                'o-', 
                label=impl, 
                color=COLORS[impl],
                linewidth=2.5,
                markersize=8
            )
    
    ax.set_xlabel('Number of Particles (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Energy Drift (%)', fontsize=14, fontweight='bold')
    ax.set_title('N-Body Simulation: Energy Conservation Quality', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#0a0a0a')
    plt.close()
    print(f"✓ Saved: {output_path}")

def plot_comprehensive_scaling(df, output_path):
    """Create comprehensive 2x2 plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Time per step (log-log)
    ax = axes[0, 0]
    for impl in IMPL_ORDER:
        impl_data = df[df['implementation'] == impl].sort_values('n_particles')
        if len(impl_data) > 0:
            ax.loglog(impl_data['n_particles'], impl_data['time_per_step'] * 1000,
                     'o-', label=impl, color=COLORS[impl], linewidth=2.5, markersize=8)
    ax.set_xlabel('Number of Particles (N)', fontweight='bold')
    ax.set_ylabel('Time per Step (ms)', fontweight='bold')
    ax.set_title('Performance Scaling', fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # 2. Speedup vs Python
    ax = axes[0, 1]
    python_data = df[df['implementation'] == 'Python (NumPy)'].set_index('n_particles')['time_per_step']
    for impl in IMPL_ORDER:
        if impl == 'Python (NumPy)':
            continue
        impl_data = df[df['implementation'] == impl].sort_values('n_particles')
        if len(impl_data) > 0:
            speedups = []
            n_values = []
            for _, row in impl_data.iterrows():
                n = row['n_particles']
                if n in python_data.index:
                    speedup = python_data[n] / row['time_per_step']
                    speedups.append(speedup)
                    n_values.append(n)
            if speedups:
                ax.loglog(n_values, speedups, 'o-', label=impl, color=COLORS[impl],
                         linewidth=2.5, markersize=8)
    ax.set_xlabel('Number of Particles (N)', fontweight='bold')
    ax.set_ylabel('Speedup vs Python', fontweight='bold')
    ax.set_title('Speedup Analysis', fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # 3. Energy conservation
    ax = axes[1, 0]
    for impl in IMPL_ORDER:
        impl_data = df[df['implementation'] == impl].sort_values('n_particles')
        if len(impl_data) > 0:
            ax.loglog(impl_data['n_particles'], impl_data['energy_drift'],
                     'o-', label=impl, color=COLORS[impl], linewidth=2.5, markersize=8)
    ax.set_xlabel('Number of Particles (N)', fontweight='bold')
    ax.set_ylabel('Energy Drift (%)', fontweight='bold')
    ax.set_title('Energy Conservation', fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # 4. Performance comparison bar chart at N=1000
    ax = axes[1, 1]
    n1000_data = df[df['n_particles'] == 1000].sort_values('time_per_step')
    if len(n1000_data) > 0:
        impls = n1000_data['implementation'].values
        times = n1000_data['time_per_step'].values * 1000  # Convert to ms
        colors_list = [COLORS[impl] for impl in impls]
        
        bars = ax.barh(impls, times, color=colors_list, alpha=0.8)
        ax.set_xlabel('Time per Step (ms)', fontweight='bold')
        ax.set_title('Performance at N=1000', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, time in zip(bars, times):
            width = bar.get_width()
            ax.text(width * 1.1, bar.get_y() + bar.get_height()/2,
                   f'{time:.4f}' if time < 1 else f'{time:.2f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#0a0a0a')
    plt.close()
    print(f"✓ Saved: {output_path}")

def create_performance_table(df, output_path):
    """Create a formatted performance table"""
    pivot = df.pivot_table(
        values='time_per_step',
        index='implementation',
        columns='n_particles'
    )
    
    # Reorder rows
    pivot = pivot.reindex([impl for impl in IMPL_ORDER if impl in pivot.index])
    
    # Convert to milliseconds
    pivot = pivot * 1000
    
    # Create markdown table
    with open(output_path, 'w') as f:
        f.write("# Performance Results\n\n")
        f.write("## Time per Step (milliseconds)\n\n")
        f.write(pivot.to_markdown(floatfmt=".4f"))
        f.write("\n\n## Speedup vs Python\n\n")
        
        if 'Python (NumPy)' in pivot.index:
            speedup = pivot.div(pivot.loc['Python (NumPy)'])
            speedup = speedup.drop('Python (NumPy)')
            f.write(speedup.to_markdown(floatfmt=".1f"))
    
    print(f"✓ Saved: {output_path}")

def main():
    """Main visualization pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize N-body benchmark results')
    parser.add_argument('input', help='Input JSON file with benchmark results')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_benchmark_data(args.input)
    
    print(f"\nFound {len(df['implementation'].unique())} implementations:")
    for impl in df['implementation'].unique():
        print(f"  - {impl}")
    
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_scaling(df, output_dir / 'performance_summary.png')
    plot_speedup(df, output_dir / 'speedup_analysis.png')
    plot_energy_conservation(df, output_dir / 'energy_conservation.png')
    plot_comprehensive_scaling(df, output_dir / 'comprehensive_scaling.png')
    create_performance_table(df, output_dir / 'performance_summary.md')
    
    print("\n✅ All visualizations complete!")
    print(f"📁 Output directory: {output_dir.absolute()}")

if __name__ == '__main__':
    main()