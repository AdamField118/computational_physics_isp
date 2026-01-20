/**
 * N-Body Performance Dashboard - Code Container Version
 * Loads charts and performance metrics dynamically
 */

(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for performance dashboard');
        return;
    }

    // Create HTML structure
    container.innerHTML = `
        <div class="performance-dashboard">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="total-particles">-</div>
                    <div class="stat-label">Max Particles Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="total-steps">-</div>
                    <div class="stat-label">Timesteps per Benchmark</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="energy-conservation">-</div>
                    <div class="stat-label">Avg Energy Drift (N=50)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="gpu-speedup">-</div>
                    <div class="stat-label">GPU Speedup (N=1000)</div>
                </div>
            </div>

            <div class="chart-grid">
                <div class="chart-container">
                    <h3>Scaling: Time per Step vs N</h3>
                    <canvas id="scalingChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Speedup vs Python Baseline</h3>
                    <canvas id="speedupChart"></canvas>
                </div>
            </div>

            <div class="chart-container">
                <h3>Energy Conservation Quality</h3>
                <canvas id="energyChart"></canvas>
            </div>
        </div>

        <style>
            .performance-dashboard {
                width: 100%;
                max-width: 1400px;
                margin: 20px auto;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: linear-gradient(135deg, #2a2a2a 0%, #353535 100%);
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                transition: transform 0.3s ease, border-color 0.3s ease;
            }
            .stat-card:hover {
                border-color: #00ff41;
                transform: scale(1.05);
            }
            .stat-value {
                font-size: 2.5rem;
                font-weight: 700;
                color: #00ff41;
                margin-bottom: 8px;
            }
            .stat-label {
                font-size: 0.9rem;
                color: #b0b0b0;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .chart-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .chart-container {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                min-height: 400px;
            }
            .chart-container h3 {
                color: #00ff41;
                margin-bottom: 15px;
                font-size: 1.2rem;
            }
            .chart-container canvas {
                width: 100% !important;
                height: auto !important;
                max-height: 500px;
            }
            @media (max-width: 768px) {
                .chart-grid {
                    grid-template-columns: 1fr;
                }
                .chart-container {
                    min-height: 300px;
                }
            }
        </style>
    `;

    // Color scheme
    const colors = {
        jax: '#00ff41',
        fortran: '#ff00ff',
        rust: '#ff6b35',
        julia: '#9558b2',
        cpp: '#ff8800',
        c: '#00ffff',
        python: '#ff0000'
    };

    // Load and process data
    async function loadBenchmarkData() {
        try {
            const response = await fetch('./nbody_comparison/web/data/benchmark_results.json');
            if (!response.ok) {
                console.warn('Using sample data');
                return generateSampleData();
            }
            return await response.json();
        } catch (error) {
            console.warn('Error loading data:', error);
            return generateSampleData();
        }
    }

    function generateSampleData() {
        return {
            timestamp: new Date().toISOString(),
            results: [
                { implementation: 'JAX (GPU)', n_particles: 10, n_steps: 1000, runtime: 0.084, time_per_step: 0.000084, energy_drift: 0.00175 },
                { implementation: 'JAX (GPU)', n_particles: 50, n_steps: 1000, runtime: 0.096, time_per_step: 0.000096, energy_drift: 0.027 },
                { implementation: 'JAX (GPU)', n_particles: 100, n_steps: 1000, runtime: 0.125, time_per_step: 0.000125, energy_drift: 0.046 },
                { implementation: 'JAX (GPU)', n_particles: 500, n_steps: 1000, runtime: 0.088, time_per_step: 0.000088, energy_drift: 0.139 },
                { implementation: 'JAX (GPU)', n_particles: 1000, n_steps: 1000, runtime: 0.080, time_per_step: 0.000080, energy_drift: 3.316 },
                { implementation: 'Fortran (OpenMP)', n_particles: 10, n_steps: 1000, runtime: 0.015, time_per_step: 0.000015, energy_drift: 0.000023 },
                { implementation: 'Fortran (OpenMP)', n_particles: 50, n_steps: 1000, runtime: 0.020, time_per_step: 0.000020, energy_drift: 0.0093 },
                { implementation: 'Fortran (OpenMP)', n_particles: 100, n_steps: 1000, runtime: 0.034, time_per_step: 0.000034, energy_drift: 0.022 },
                { implementation: 'Fortran (OpenMP)', n_particles: 500, n_steps: 1000, runtime: 0.394, time_per_step: 0.000394, energy_drift: 0.935 },
                { implementation: 'Fortran (OpenMP)', n_particles: 1000, n_steps: 1000, runtime: 1.373, time_per_step: 0.001373, energy_drift: 3.940 },
                { implementation: 'Rust', n_particles: 10, n_steps: 1000, runtime: 0.012, time_per_step: 0.000012, energy_drift: 0.000023 },
                { implementation: 'Rust', n_particles: 50, n_steps: 1000, runtime: 0.018, time_per_step: 0.000018, energy_drift: 0.0093 },
                { implementation: 'Rust', n_particles: 100, n_steps: 1000, runtime: 0.030, time_per_step: 0.000030, energy_drift: 0.022 },
                { implementation: 'Rust', n_particles: 500, n_steps: 1000, runtime: 0.350, time_per_step: 0.000350, energy_drift: 0.935 },
                { implementation: 'Rust', n_particles: 1000, n_steps: 1000, runtime: 1.250, time_per_step: 0.001250, energy_drift: 3.940 },
                { implementation: 'Julia', n_particles: 10, n_steps: 1000, runtime: 0.013, time_per_step: 0.000013, energy_drift: 0.000023 },
                { implementation: 'Julia', n_particles: 50, n_steps: 1000, runtime: 0.019, time_per_step: 0.000019, energy_drift: 0.0093 },
                { implementation: 'Julia', n_particles: 100, n_steps: 1000, runtime: 0.032, time_per_step: 0.000032, energy_drift: 0.022 },
                { implementation: 'Julia', n_particles: 500, n_steps: 1000, runtime: 0.370, time_per_step: 0.000370, energy_drift: 0.935 },
                { implementation: 'Julia', n_particles: 1000, n_steps: 1000, runtime: 1.300, time_per_step: 0.001300, energy_drift: 3.940 },
                { implementation: 'Python (NumPy)', n_particles: 10, n_steps: 1000, runtime: 0.035, time_per_step: 0.000035, energy_drift: 0.000023 },
                { implementation: 'Python (NumPy)', n_particles: 50, n_steps: 1000, runtime: 0.225, time_per_step: 0.000225, energy_drift: 0.0093 },
                { implementation: 'Python (NumPy)', n_particles: 100, n_steps: 1000, runtime: 1.257, time_per_step: 0.001257, energy_drift: 0.022 },
                { implementation: 'Python (NumPy)', n_particles: 500, n_steps: 1000, runtime: 27.46, time_per_step: 0.02746, energy_drift: 0.934 },
                { implementation: 'Python (NumPy)', n_particles: 1000, n_steps: 1000, runtime: 110.84, time_per_step: 0.11084, energy_drift: 4.331 },
            ]
        };
    }

    function processBenchmarkData(data) {
        const implementations = {};
        data.results.forEach(result => {
            const impl = result.implementation;
            if (!implementations[impl]) {
                implementations[impl] = [];
            }
            implementations[impl].push(result);
        });
        
        Object.keys(implementations).forEach(impl => {
            implementations[impl].sort((a, b) => a.n_particles - b.n_particles);
        });
        
        return implementations;
    }

    function updateStats(data) {
        const allResults = data.results;
        
        const maxParticles = Math.max(...allResults.map(r => r.n_particles));
        container.querySelector('#total-particles').textContent = maxParticles.toLocaleString();
        
        const steps = allResults[0]?.n_steps || 1000;
        container.querySelector('#total-steps').textContent = steps.toLocaleString();
        
        const n50Results = allResults.filter(r => r.n_particles === 50);
        if (n50Results.length > 0) {
            const avgDrift = n50Results.reduce((sum, r) => sum + r.energy_drift, 0) / n50Results.length;
            container.querySelector('#energy-conservation').textContent = `${avgDrift.toFixed(3)}%`;
        }
        
        const jaxResult = allResults.find(r => r.implementation.includes('JAX') && r.n_particles === 1000);
        const pythonResult = allResults.find(r => r.implementation.includes('Python') && r.n_particles === 1000);
        
        if (jaxResult && pythonResult) {
            const speedup = pythonResult.time_per_step / jaxResult.time_per_step;
            container.querySelector('#gpu-speedup').textContent = `${speedup.toFixed(0)}×`;
        }
    }

    function createScalingChart(implementations) {
        const canvas = container.querySelector('#scalingChart');
        const ctx = canvas.getContext('2d');
        
        const datasets = [];
        const implOrder = ['JAX (GPU)', 'Fortran (OpenMP)', 'Rust', 'Julia', 'C++', 'C', 'Python (NumPy)'];
        const colorMap = {
            'JAX (GPU)': colors.jax,
            'Fortran (OpenMP)': colors.fortran,
            'Rust': colors.rust,
            'Julia': colors.julia,
            'C++': colors.cpp,
            'C': colors.c,
            'Python (NumPy)': colors.python
        };
        
        implOrder.forEach(implName => {
            if (!implementations[implName]) return;
            
            const data = implementations[implName].map(r => ({
                x: r.n_particles,
                y: r.time_per_step * 1000
            }));
            
            datasets.push({
                label: implName,
                data: data,
                borderColor: colorMap[implName],
                backgroundColor: colorMap[implName] + '33',
                borderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8,
            });
        });
        
        new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                aspectRatio: 1.8,
                scales: {
                    x: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Number of Particles (N)',
                            color: '#fff',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: { color: '#b0b0b0', font: { size: 12 } },
                        grid: { color: '#333' }
                    },
                    y: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Time per Step (ms)',
                            color: '#fff',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: { color: '#b0b0b0', font: { size: 12 } },
                        grid: { color: '#333' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#fff', font: { size: 12 }, padding: 15 }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(4)} ms`;
                            }
                        }
                    }
                }
            }
        });
    }

    function createSpeedupChart(implementations) {
        const canvas = container.querySelector('#speedupChart');
        const ctx = canvas.getContext('2d');
        
        const pythonData = implementations['Python (NumPy)'];
        if (!pythonData) return;
        
        const pythonTimes = {};
        pythonData.forEach(r => {
            pythonTimes[r.n_particles] = r.time_per_step;
        });
        
        const datasets = [];
        const implOrder = ['JAX (GPU)', 'Fortran (OpenMP)', 'Rust', 'Julia', 'C++', 'C'];
        const colorMap = {
            'JAX (GPU)': colors.jax,
            'Fortran (OpenMP)': colors.fortran,
            'Rust': colors.rust,
            'Julia': colors.julia,
            'C++': colors.cpp,
            'C': colors.c
        };
        
        implOrder.forEach(implName => {
            if (!implementations[implName]) return;
            
            const data = implementations[implName]
                .filter(r => pythonTimes[r.n_particles])
                .map(r => ({
                    x: r.n_particles,
                    y: pythonTimes[r.n_particles] / r.time_per_step
                }));
            
            datasets.push({
                label: implName,
                data: data,
                borderColor: colorMap[implName],
                backgroundColor: colorMap[implName] + '33',
                borderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8,
            });
        });
        
        new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                aspectRatio: 1.8,
                scales: {
                    x: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Number of Particles (N)',
                            color: '#fff',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: { color: '#b0b0b0', font: { size: 12 } },
                        grid: { color: '#333' }
                    },
                    y: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Speedup vs Python',
                            color: '#fff',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: { color: '#b0b0b0', font: { size: 12 } },
                        grid: { color: '#333' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#fff', font: { size: 12 }, padding: 15 }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}× faster`;
                            }
                        }
                    }
                }
            }
        });
    }

    function createEnergyChart(implementations) {
        const canvas = container.querySelector('#energyChart');
        const ctx = canvas.getContext('2d');
        
        const datasets = [];
        const implOrder = ['JAX (GPU)', 'Fortran (OpenMP)', 'Rust', 'Julia', 'C++', 'C', 'Python (NumPy)'];
        const colorMap = {
            'JAX (GPU)': colors.jax,
            'Fortran (OpenMP)': colors.fortran,
            'Rust': colors.rust,
            'Julia': colors.julia,
            'C++': colors.cpp,
            'C': colors.c,
            'Python (NumPy)': colors.python
        };
        
        implOrder.forEach(implName => {
            if (!implementations[implName]) return;
            
            const data = implementations[implName].map(r => ({
                x: r.n_particles,
                y: r.energy_drift
            }));
            
            datasets.push({
                label: implName,
                data: data,
                borderColor: colorMap[implName],
                backgroundColor: colorMap[implName] + '33',
                borderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8,
            });
        });
        
        new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                aspectRatio: 1.8,
                scales: {
                    x: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Number of Particles (N)',
                            color: '#fff',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: { color: '#b0b0b0', font: { size: 12 } },
                        grid: { color: '#333' }
                    },
                    y: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Energy Drift (%)',
                            color: '#fff',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: { color: '#b0b0b0', font: { size: 12 } },
                        grid: { color: '#333' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#fff', font: { size: 12 }, padding: 15 }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(4)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Initialize dashboard
    async function init() {
        console.log('Loading performance dashboard...');
        
        const benchmarkData = await loadBenchmarkData();
        const implementations = processBenchmarkData(benchmarkData);
        
        updateStats(benchmarkData);
        createScalingChart(implementations);
        createSpeedupChart(implementations);
        createEnergyChart(implementations);
        
        console.log('Performance dashboard loaded successfully!');
    }

    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js';
        script.onload = () => init();
        document.head.appendChild(script);
    } else {
        init();
    }
})();