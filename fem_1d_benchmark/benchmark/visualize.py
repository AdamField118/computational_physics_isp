#!/usr/bin/env python3
"""
Generate interactive HTML dashboard from benchmark results
"""

import json
import sys
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FEM 1D Benchmark Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        
        h1 {{
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        
        .metadata {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }}
        
        .chart-container {{
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .chart-title {{
            font-size: 1.5em;
            color: #667eea;
            margin-bottom: 15px;
            text-align: center;
        }}
        
        canvas {{
            max-height: 400px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: #f5f5f5;
        }}
        
        .fastest {{
            background: #d4edda;
            font-weight: bold;
        }}
        
        .speedup {{
            color: #28a745;
            font-weight: bold;
        }}
        
        @media (prefers-color-scheme: dark) {{
            body {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            }}
            
            .container {{
                background: #1e1e1e;
                color: #e0e0e0;
            }}
            
            h1 {{
                color: #9b59b6;
            }}
            
            .subtitle {{
                color: #aaa;
            }}
            
            .metadata {{
                background: #2a2a2a;
                border-left-color: #9b59b6;
            }}
            
            .chart-container {{
                background: #2a2a2a;
            }}
            
            .chart-title {{
                color: #9b59b6;
            }}
            
            table {{
                background: #2a2a2a;
            }}
            
            th {{
                background: #9b59b6;
            }}
            
            td {{
                color: #e0e0e0;
            }}
            
            tr:hover {{
                background: #333;
            }}
            
            .fastest {{
                background: #1e4620;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>1D FEM Multi-Language Benchmark</h1>
        <p class="subtitle">Performance comparison of piecewise linear finite element assembly</p>
        
        <div class="metadata">
            <strong>Problem:</strong> {problem}<br>
            <strong>Description:</strong> {description}<br>
            <strong>Date:</strong> {date}<br>
            <strong>Implementations:</strong> {implementations}
        </div>
        
        <div class="chart-container">
            <h2 class="chart-title">Parallel Performance: Assembly Time vs Problem Size</h2>
            <canvas id="performanceChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2 class="chart-title">Speedup vs Python (Parallel)</h2>
            <canvas id="speedupChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2 class="chart-title">Parallel vs Serial Performance</h2>
            <canvas id="parallelChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2 class="chart-title">Performance Summary (n={max_n})</h2>
            {summary_table}
        </div>
    </div>
    
    <script>
        const data = {data_json};
        
        // Performance chart
        {{
            const ctx = document.getElementById('performanceChart').getContext('2d');
            const datasets = data.benchmarks.map((bench, idx) => ({{
                label: bench.name + ' (Parallel)',
                data: bench.results.parallel.map(r => ({{x: r.n, y: r.mean * 1000}})),
                borderColor: `hsl(${{idx * 360 / data.benchmarks.length}}, 70%, 50%)`,
                backgroundColor: `hsla(${{idx * 360 / data.benchmarks.length}}, 70%, 50%, 0.1)`,
                borderWidth: 2,
                pointRadius: 4,
                tension: 0.1
            }}));
            
            new Chart(ctx, {{
                type: 'line',
                data: {{ datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {{
                        x: {{
                            type: 'logarithmic',
                            title: {{ display: true, text: 'Number of elements (n)' }}
                        }},
                        y: {{
                            type: 'logarithmic',
                            title: {{ display: true, text: 'Assembly time (ms)' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ position: 'top' }},
                        tooltip: {{
                            callbacks: {{
                                label: (context) => `${{context.dataset.label}}: ${{context.parsed.y.toFixed(3)}} ms`
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Speedup chart
        {{
            const ctx = document.getElementById('speedupChart').getContext('2d');
            const pythonData = data.benchmarks.find(b => b.name === 'Python').results.parallel;
            
            const datasets = data.benchmarks
                .filter(b => b.name !== 'Python')
                .map((bench, idx) => ({{
                    label: bench.name,
                    data: bench.results.parallel.map((r, i) => ({{
                        x: r.n,
                        y: pythonData[i].mean / r.mean
                    }})),
                    borderColor: `hsl(${{idx * 360 / (data.benchmarks.length - 1)}}, 70%, 50%)`,
                    backgroundColor: `hsla(${{idx * 360 / (data.benchmarks.length - 1)}}, 70%, 50%, 0.1)`,
                    borderWidth: 2,
                    pointRadius: 4,
                    tension: 0.1
                }}));
            
            new Chart(ctx, {{
                type: 'line',
                data: {{ datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {{
                        x: {{
                            type: 'logarithmic',
                            title: {{ display: true, text: 'Number of elements (n)' }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'Speedup vs Python' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ position: 'top' }},
                        tooltip: {{
                            callbacks: {{
                                label: (context) => `${{context.dataset.label}}: ${{context.parsed.y.toFixed(2)}}x`
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Parallel vs Serial chart
        {{
            const ctx = document.getElementById('parallelChart').getContext('2d');
            const n_values = data.benchmarks[0].results.parallel.map(r => r.n);
            
            const datasets = [];
            data.benchmarks.forEach((bench, idx) => {{
                if (bench.parallel && bench.results.serial) {{
                    const color = `hsl(${{idx * 360 / data.benchmarks.length}}, 70%, 50%)`;
                    
                    datasets.push({{
                        label: bench.name + ' (Serial)',
                        data: bench.results.serial.map((r, i) => ({{
                            x: r.n,
                            y: bench.results.serial[i].mean / bench.results.parallel[i].mean
                        }})),
                        borderColor: color,
                        backgroundColor: `hsla(${{idx * 360 / data.benchmarks.length}}, 70%, 50%, 0.1)`,
                        borderWidth: 2,
                        pointRadius: 4,
                        borderDash: [5, 5],
                        tension: 0.1
                    }});
                }}
            }});
            
            new Chart(ctx, {{
                type: 'line',
                data: {{ datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {{
                        x: {{
                            type: 'logarithmic',
                            title: {{ display: true, text: 'Number of elements (n)' }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'Parallel Speedup (Serial / Parallel)' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ position: 'top' }},
                        tooltip: {{
                            callbacks: {{
                                label: (context) => `${{context.dataset.label}}: ${{context.parsed.y.toFixed(2)}}x speedup`
                            }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
"""


def generate_summary_table(data):
    """Generate HTML table with benchmark summary"""
    # Get largest problem size
    max_n = data['benchmarks'][0]['results']['parallel'][-1]['n']
    
    # Get Python time for reference
    python_bench = next(b for b in data['benchmarks'] if b['name'] == 'Python')
    python_time = python_bench['results']['parallel'][-1]['mean']
    
    # Sort by parallel time
    sorted_benches = sorted(data['benchmarks'], 
                           key=lambda x: x['results']['parallel'][-1]['mean'])
    
    fastest_time = sorted_benches[0]['results']['parallel'][-1]['mean']
    
    rows = []
    for bench in sorted_benches:
        name = bench['name']
        parallel_result = bench['results']['parallel'][-1]
        parallel_time = parallel_result['mean'] * 1000
        
        is_fastest = parallel_result['mean'] == fastest_time
        row_class = ' class="fastest"' if is_fastest else ''
        
        speedup_python = python_time / parallel_result['mean']
        
        if bench['parallel'] and bench['results']['serial']:
            serial_result = bench['results']['serial'][-1]
            serial_time = serial_result['mean'] * 1000
            speedup_parallel = serial_result['mean'] / parallel_result['mean']
            
            rows.append(
                f'<tr{row_class}>'
                f'<td>{name}{"🏆" if is_fastest else ""}</td>'
                f'<td>{parallel_time:.3f}</td>'
                f'<td>{serial_time:.3f}</td>'
                f'<td class="speedup">{speedup_parallel:.2f}x</td>'
                f'<td class="speedup">{speedup_python:.2f}x</td>'
                '</tr>'
            )
        else:
            rows.append(
                f'<tr{row_class}>'
                f'<td>{name}{"🏆" if is_fastest else ""}</td>'
                f'<td>{parallel_time:.3f}</td>'
                f'<td>N/A</td>'
                f'<td>N/A</td>'
                f'<td class="speedup">{speedup_python:.2f}x</td>'
                '</tr>'
            )
    
    return f"""
    <table>
        <thead>
            <tr>
                <th>Implementation</th>
                <th>Parallel (ms)</th>
                <th>Serial (ms)</th>
                <th>Parallel Speedup</th>
                <th>vs Python</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """, max_n


def main():
    # Load JSON results
    results_file = Path(__file__).parent.parent / 'results' / 'fem_benchmark_results.json'
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Run benchmark.py first to generate results")
        sys.exit(1)
    
    with open(results_file) as f:
        data = json.load(f)
    
    # Generate summary table
    summary_table, max_n = generate_summary_table(data)
    
    # Get implementation names
    impl_names = ', '.join(b['name'] for b in data['benchmarks'])
    
    # Generate HTML
    html = HTML_TEMPLATE.format(
        problem=data['metadata']['problem'],
        description=data['metadata']['description'],
        date=data['metadata']['date'],
        implementations=impl_names,
        max_n=max_n,
        summary_table=summary_table,
        data_json=json.dumps(data)
    )
    
    # Save HTML
    output_file = Path(__file__).parent.parent / 'web' / 'benchmark_results.html'
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✓ Dashboard generated: {output_file}")
    print(f"  Open in browser to view results")


if __name__ == '__main__':
    main()