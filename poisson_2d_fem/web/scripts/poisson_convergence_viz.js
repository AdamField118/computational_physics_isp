/**
 * 2D Poisson FEM Convergence Visualization
 * Shows L² and H¹ error convergence with theoretical reference slopes
 */

(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for convergence visualization');
        return;
    }

    // Convergence data from actual results
    const convergenceData = {
        h: [0.31623, 0.22361, 0.15811, 0.11180, 0.07906],
        nodes: [13, 23, 41, 77, 147],
        elements: [16, 28, 64, 123, 260],
        L2_error: [2.448e-03, 1.228e-03, 6.147e-04, 3.077e-04, 1.539e-04],
        H1_error: [2.458e-02, 1.740e-02, 1.230e-02, 8.697e-03, 6.152e-03],
        Linf_error: [4.551e-01, 5.119e-01, 5.058e-01, 5.020e-01, 5.029e-01]
    };

    // Calculate convergence rates
    function calculateRates(h, errors) {
        const rates = [0]; // First point has no rate
        for (let i = 1; i < h.length; i++) {
            const rate = Math.log(errors[i-1] / errors[i]) / Math.log(h[i-1] / h[i]);
            rates.push(rate);
        }
        return rates;
    }

    const L2_rates = calculateRates(convergenceData.h, convergenceData.L2_error);
    const H1_rates = calculateRates(convergenceData.h, convergenceData.H1_error);

    // Create HTML structure
    container.innerHTML = `
        <div class="convergence-dashboard">
            <div class="stats-row">
                <div class="stat-box">
                    <div class="stat-value">${convergenceData.nodes[convergenceData.nodes.length-1]}</div>
                    <div class="stat-label">Max Nodes Tested</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">2.0</div>
                    <div class="stat-label">L² Convergence Rate</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">1.0</div>
                    <div class="stat-label">H¹ Convergence Rate</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">O(h²)</div>
                    <div class="stat-label">Theoretical L² Rate</div>
                </div>
            </div>

            <div class="chart-wrapper">
                <h3>Convergence Rates: 2D Poisson Equation (P1 Elements)</h3>
                <canvas id="convergenceChart"></canvas>
            </div>

            <div class="rate-table">
                <h3>Detailed Convergence Analysis</h3>
                <table>
                    <thead>
                        <tr>
                            <th>h</th>
                            <th>Nodes</th>
                            <th>L² Error</th>
                            <th>L² Rate</th>
                            <th>H¹ Error</th>
                            <th>H¹ Rate</th>
                        </tr>
                    </thead>
                    <tbody id="rateTableBody"></tbody>
                </table>
            </div>
        </div>

        <style>
            .convergence-dashboard {
                width: 100%;
                max-width: 1200px;
                margin: 20px auto;
            }
            .stats-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }
            .stat-box {
                background: linear-gradient(135deg, #2a2a2a 0%, #353535 100%);
                border: 1px solid #00ff41;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .stat-box:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 20px rgba(0, 255, 65, 0.3);
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
            .chart-wrapper {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
            }
            .chart-wrapper h3 {
                color: #00ff41;
                margin-bottom: 15px;
                font-size: 1.3rem;
                text-align: center;
            }
            .chart-wrapper canvas {
                width: 100% !important;
                height: 500px !important;
            }
            .rate-table {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 20px;
            }
            .rate-table h3 {
                color: #00ff41;
                margin-bottom: 15px;
                font-size: 1.2rem;
            }
            .rate-table table {
                width: 100%;
                border-collapse: collapse;
            }
            .rate-table th {
                background: #1a1a1a;
                color: #00ff41;
                padding: 12px;
                text-align: left;
                border-bottom: 2px solid #00ff41;
            }
            .rate-table td {
                padding: 10px 12px;
                border-bottom: 1px solid #333;
                color: #e0e0e0;
            }
            .rate-table tr:hover {
                background: #333;
            }
            .rate-good {
                color: #00ff41;
                font-weight: bold;
            }
            @media (max-width: 768px) {
                .stats-row {
                    grid-template-columns: 1fr 1fr;
                }
                .chart-wrapper canvas {
                    height: 350px !important;
                }
            }
        </style>
    `;

    // Populate rate table
    const tableBody = container.querySelector('#rateTableBody');
    for (let i = 0; i < convergenceData.h.length; i++) {
        const row = document.createElement('tr');
        const l2RateClass = (i > 0 && L2_rates[i] >= 1.9) ? 'rate-good' : '';
        const h1RateClass = (i > 0 && H1_rates[i] >= 0.9) ? 'rate-good' : '';
        
        row.innerHTML = `
            <td>${convergenceData.h[i].toFixed(5)}</td>
            <td>${convergenceData.nodes[i]}</td>
            <td>${convergenceData.L2_error[i].toExponential(3)}</td>
            <td class="${l2RateClass}">${i === 0 ? '-' : L2_rates[i].toFixed(2)}</td>
            <td>${convergenceData.H1_error[i].toExponential(3)}</td>
            <td class="${h1RateClass}">${i === 0 ? '-' : H1_rates[i].toFixed(2)}</td>
        `;
        tableBody.appendChild(row);
    }

    // Create convergence chart
    function createChart() {
        const canvas = container.querySelector('#convergenceChart');
        const ctx = canvas.getContext('2d');

        // Prepare data points
        const h_values = convergenceData.h;
        const L2_data = convergenceData.L2_error;
        const H1_data = convergenceData.H1_error;

        // Reference slopes (O(h²) and O(h))
        const h_ref = [h_values[0], h_values[h_values.length - 1]];
        const L2_ref_scale = L2_data[1] / (h_values[1] ** 2);
        const H1_ref_scale = H1_data[1] / h_values[1];
        
        const L2_reference = h_ref.map(h => L2_ref_scale * h * h);
        const H1_reference = h_ref.map(h => H1_ref_scale * h);

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: h_values.map(h => h.toFixed(3)),
                datasets: [
                    {
                        label: 'L² Error',
                        data: L2_data,
                        borderColor: '#00ff41',
                        backgroundColor: 'rgba(0, 255, 65, 0.1)',
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        tension: 0.1
                    },
                    {
                        label: 'H¹ Error',
                        data: H1_data,
                        borderColor: '#00aaff',
                        backgroundColor: 'rgba(0, 170, 255, 0.1)',
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        tension: 0.1
                    },
                    {
                        label: 'O(h²) Reference',
                        data: [L2_reference[0], null, null, null, L2_reference[1]],
                        borderColor: '#ffffff',
                        borderWidth: 2,
                        borderDash: [10, 5],
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'O(h) Reference',
                        data: [H1_reference[0], null, null, null, H1_reference[1]],
                        borderColor: '#ffffff',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Mesh Size h',
                            color: '#fff',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: { color: '#b0b0b0', font: { size: 11 } },
                        grid: { color: '#333' }
                    },
                    y: {
                        type: 'logarithmic',
                        title: {
                            display: true,
                            text: 'Error',
                            color: '#fff',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: { 
                            color: '#b0b0b0',
                            font: { size: 11 },
                            callback: function(value) {
                                return value.toExponential(0);
                            }
                        },
                        grid: { color: '#333' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { 
                            color: '#fff',
                            font: { size: 12 },
                            padding: 15,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;
                                if (value === null) return null;
                                return `${label}: ${value.toExponential(3)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Initialize
    if (typeof Chart === 'undefined') {
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js';
        script.onload = () => {
            console.log('Chart.js loaded, creating convergence visualization');
            createChart();
        };
        document.head.appendChild(script);
    } else {
        createChart();
    }
})();