/**
 * FEM Benchmark Performance Dashboard
 * Interactive visualization of multi-language FEM assembly performance
 */

(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for FEM dashboard');
        return;
    }

    // Benchmark data (embedded from results)
    const benchmarkData = {
        "metadata": {
            "date": "2026-02-02T20:02:05.476412",
            "problem": "1D FEM Assembly",
            "description": "Piecewise linear finite element method benchmark"
        },
        "benchmarks": [
            {
            "name": "Python",
            "results": [
                {
                "n": 500,
                "mean": 0.000593616598052904,
                "std": 8.589619867565167e-06,
                "min": 0.0005871730099897832,
                "max": 0.0006101219914853573
                },
                {
                "n": 1000,
                "mean": 0.0018055107968393714,
                "std": 0.00026667238973685853,
                "min": 0.0015776619839016348,
                "max": 0.0022467829985544086
                },
                {
                "n": 5000,
                "mean": 0.018514598003821447,
                "std": 6.25880620129247e-05,
                "min": 0.01844810100737959,
                "max": 0.01861304900376126
                },
                {
                "n": 10000,
                "mean": 0.05207458459190093,
                "std": 0.0017352417064776038,
                "min": 0.050135537981987,
                "max": 0.054837100993609056
                },
                {
                "n": 20000,
                "mean": 0.15629533539758994,
                "std": 0.0004139306369100561,
                "min": 0.15557291998993605,
                "max": 0.15670586199848913
                }
            ]
            },
            {
            "name": "C",
            "results": [
                {
                "n": 500,
                "mean": 2.31220037676394e-05,
                "std": 3.183621177035916e-06,
                "min": 2.0570994820445776e-05,
                "max": 2.8790993383154273e-05
                },
                {
                "n": 1000,
                "mean": 2.3774197325110434e-05,
                "std": 1.0089203691911201e-06,
                "min": 2.283899812027812e-05,
                "max": 2.5716988602653146e-05
                },
                {
                "n": 5000,
                "mean": 8.562459843233227e-05,
                "std": 6.494935777175291e-06,
                "min": 7.979400106705725e-05,
                "max": 9.794699144549668e-05
                },
                {
                "n": 10000,
                "mean": 0.00022092979052104055,
                "std": 9.283337854581965e-06,
                "min": 0.00021192498388700187,
                "max": 0.00023767098900862038
                },
                {
                "n": 20000,
                "mean": 0.0005656382010784001,
                "std": 8.24541541389446e-06,
                "min": 0.0005541400169022381,
                "max": 0.0005766579997725785
                }
            ]
            },
            {
            "name": "C++",
            "results": [
                {
                "n": 500,
                "mean": 9.56220319494605e-06,
                "std": 2.200889186783406e-06,
                "min": 8.319999324157834e-06,
                "max": 1.3960001524537802e-05
                },
                {
                "n": 1000,
                "mean": 1.134300255216658e-05,
                "std": 3.593657007904862e-07,
                "min": 1.0910996934399009e-05,
                "max": 1.1872005416080356e-05
                },
                {
                "n": 5000,
                "mean": 6.904079928062856e-05,
                "std": 5.43114863645525e-06,
                "min": 6.216100882738829e-05,
                "max": 7.749599171802402e-05
                },
                {
                "n": 10000,
                "mean": 0.00020532679045572876,
                "std": 7.834871928223838e-06,
                "min": 0.0001967559801414609,
                "max": 0.00021719498909078538
                },
                {
                "n": 20000,
                "mean": 0.0005466944014187903,
                "std": 1.3567072084354774e-05,
                "min": 0.0005308320105541497,
                "max": 0.0005709179968107492
                }
            ]
            },
            {
            "name": "Fortran",
            "results": [
                {
                "n": 500,
                "mean": 6.095797289162874e-06,
                "std": 1.139031513658827e-06,
                "min": 5.27501106262207e-06,
                "max": 8.347997209057212e-06
                },
                {
                "n": 1000,
                "mean": 8.102599531412125e-06,
                "std": 2.2655679772412866e-07,
                "min": 7.79700349085033e-06,
                "max": 8.487026207149029e-06
                },
                {
                "n": 5000,
                "mean": 5.625380435958505e-05,
                "std": 6.166468169632746e-06,
                "min": 5.063600838184357e-05,
                "max": 6.739000673405826e-05
                },
                {
                "n": 10000,
                "mean": 0.00018802059930749236,
                "std": 6.053479974976451e-06,
                "min": 0.00018060198635794222,
                "max": 0.00019892200361937284
                },
                {
                "n": 20000,
                "mean": 0.0005314596055541188,
                "std": 3.888236307537854e-06,
                "min": 0.0005271779955364764,
                "max": 0.0005386130069382489
                }
            ]
            },
            {
            "name": "Julia",
            "results": [
                {
                "n": 500,
                "mean": 0.0013166116084903478,
                "std": 0.00025346846462801886,
                "min": 0.0008142820151988417,
                "max": 0.0015042430022731423
                },
                {
                "n": 1000,
                "mean": 0.018855592398904265,
                "std": 0.030580407458093217,
                "min": 0.003480137005681172,
                "max": 0.08001623500604182
                },
                {
                "n": 5000,
                "mean": 0.11593856800463982,
                "std": 0.002044943908744592,
                "min": 0.11245493000023998,
                "max": 0.11837882400141098
                },
                {
                "n": 10000,
                "mean": 0.46529327940079385,
                "std": 0.011587739233386652,
                "min": 0.4524745979870204,
                "max": 0.48653084502439015
                },
                {
                "n": 20000,
                "mean": 1.8453022646019235,
                "std": 0.027807226242947153,
                "min": 1.8122279990056995,
                "max": 1.880953537998721
                }
            ]
            },
            {
            "name": "Rust",
            "results": [
                {
                "n": 500,
                "mean": 6.317807128652931e-06,
                "std": 2.0753330626303248e-06,
                "min": 4.986999556422234e-06,
                "max": 1.0431016562506557e-05
                },
                {
                "n": 1000,
                "mean": 6.9185975007712844e-06,
                "std": 1.1241252355651236e-06,
                "min": 5.826994311064482e-06,
                "max": 9.08499350771308e-06
                },
                {
                "n": 5000,
                "mean": 7.722500013187527e-05,
                "std": 5.074652147149179e-06,
                "min": 7.264799205586314e-05,
                "max": 8.651698590256274e-05
                },
                {
                "n": 10000,
                "mean": 0.00020718720625154675,
                "std": 1.1384167560704993e-05,
                "min": 0.00019904301734641194,
                "max": 0.0002293430152349174
                },
                {
                "n": 20000,
                "mean": 0.0005311559943947941,
                "std": 8.393169261543415e-06,
                "min": 0.0005198699946049601,
                "max": 0.000543628993909806
                }
            ]
            }
        ]
    };

    // Create HTML structure matching nbody dashboard style
    container.innerHTML = `
        <div class="fem-performance-dashboard">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="max-elements">-</div>
                    <div class="stat-label">Max Elements Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="fastest-impl">-</div>
                    <div class="stat-label">Fastest Implementation</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="max-speedup">-</div>
                    <div class="stat-label">Max Speedup vs Python</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="complexity">O(n)</div>
                    <div class="stat-label">Theoretical Complexity</div>
                </div>
            </div>

            <div class="chart-grid">
                <div class="chart-container">
                    <h3>Assembly Time vs Problem Size</h3>
                    <canvas id="scalingChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Speedup vs Python Baseline</h3>
                    <canvas id="speedupChart"></canvas>
                </div>
            </div>

            <div class="chart-container">
                <h3>Performance Comparison (n=20,000 elements)</h3>
                <canvas id="barChart"></canvas>
            </div>
        </div>

        <style>
            .fem-performance-dashboard {
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

    // Color scheme matching nbody style
    const colors = {
        'Python': '#ff0000',
        'C': '#00ffff',
        'C++': '#ff8800',
        'Fortran': '#ff00ff',
        'Rust': '#ff6b35',
        'Julia': '#9558b2'
    };

    function updateStats(data) {
        const allResults = data.benchmarks.flatMap(b => b.results);
        
        const maxN = Math.max(...allResults.map(r => r.n));
        container.querySelector('#max-elements').textContent = maxN.toLocaleString();
        
        // Find fastest at n=20000
        const n20kResults = data.benchmarks.map(b => ({
            name: b.name,
            time: b.results.find(r => r.n === 20000)?.mean || Infinity
        })).sort((a, b) => a.time - b.time);
        
        container.querySelector('#fastest-impl').textContent = n20kResults[0].name;
        
        // Calculate max speedup
        const pythonTime = data.benchmarks.find(b => b.name === 'Python')
            .results.find(r => r.n === 20000).mean;
        const fastestTime = n20kResults[0].time;
        const maxSpeedup = pythonTime / fastestTime;
        
        container.querySelector('#max-speedup').textContent = `${maxSpeedup.toFixed(0)}×`;
    }

    function createScalingChart(data) {
        const canvas = container.querySelector('#scalingChart');
        const ctx = canvas.getContext('2d');
        
        const datasets = data.benchmarks
            .map(bench => ({
                label: bench.name,
                data: bench.results.map(r => ({
                    x: r.n,
                    y: r.mean * 1000
                })),
                borderColor: colors[bench.name],
                backgroundColor: colors[bench.name] + '33',
                borderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8,
            }));
        
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
                            text: 'Number of Elements (n)',
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
                            text: 'Assembly Time (ms)',
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

    function createSpeedupChart(data) {
        const canvas = container.querySelector('#speedupChart');
        const ctx = canvas.getContext('2d');
        
        const pythonData = data.benchmarks.find(b => b.name === 'Python');
        const pythonTimes = {};
        pythonData.results.forEach(r => {
            pythonTimes[r.n] = r.mean;
        });
        
        const datasets = data.benchmarks
            .filter(b => b.name !== 'Python')
            .map(bench => ({
                label: bench.name,
                data: bench.results
                    .filter(r => pythonTimes[r.n])
                    .map(r => ({
                        x: r.n,
                        y: pythonTimes[r.n] / r.mean
                    })),
                borderColor: colors[bench.name],
                backgroundColor: colors[bench.name] + '33',
                borderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8,
            }));
        
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
                            text: 'Number of Elements (n)',
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

    function createBarChart(data) {
        const canvas = container.querySelector('#barChart');
        const ctx = canvas.getContext('2d');
        
        const n20kData = data.benchmarks
            .map(b => ({
                name: b.name,
                time: b.results.find(r => r.n === 20000)?.mean * 1000 || 0
            }))
            .sort((a, b) => a.time - b.time);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: n20kData.map(d => d.name),
                datasets: [{
                    label: 'Assembly Time (ms)',
                    data: n20kData.map(d => d.time),
                    backgroundColor: n20kData.map(d => colors[d.name] + 'aa'),
                    borderColor: n20kData.map(d => colors[d.name]),
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                aspectRatio: 2.5,
                scales: {
                    x: {
                        ticks: { color: '#b0b0b0', font: { size: 12 } },
                        grid: { display: false }
                    },
                    y: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Assembly Time (ms)',
                            color: '#fff',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: { color: '#b0b0b0', font: { size: 12 } },
                        grid: { color: '#333' }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.parsed.y.toFixed(4)} ms`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Initialize dashboard
    async function init() {
        console.log('Loading FEM performance dashboard...');
        
        updateStats(benchmarkData);
        createScalingChart(benchmarkData);
        createSpeedupChart(benchmarkData);
        createBarChart(benchmarkData);
        
        console.log('FEM performance dashboard loaded successfully!');
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