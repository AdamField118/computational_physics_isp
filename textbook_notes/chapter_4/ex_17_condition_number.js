// Exercise 4.x.17: Condition Number of Stiffness Matrix
(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for ex_17_condition_number');
        return;
    }

    container.innerHTML = `
    <div style="max-width: 1000px; margin: 0 auto;">
        <h3 style="color: #00ff41;">Condition Number: Why Fine Meshes Are Hard to Solve</h3>
        <p style="color: #e4e4e4;">Watch κ(K) grow as O(h⁻²) with mesh refinement</p>
        
        <div style="margin: 20px 0;">
            <label for="mesh-slider" style="color: #e4e4e4;">
                Number of Intervals: <strong id="n-value">4</strong> (h = <strong id="h-value">0.25</strong>)
            </label>
            <input type="range" id="mesh-slider" min="2" max="20" value="4" 
                   style="width: 400px; margin-left: 10px;">
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
            <div>
                <canvas id="canvas-mesh" width="500" height="200" 
                        style="border: 1px solid #ccc; background: white;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">
                    1D Mesh Visualization
                </p>
                
                <canvas id="canvas-eigenvalues" width="500" height="300" 
                        style="border: 1px solid #ccc; background: white; margin-top: 20px;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">
                    Eigenvalue Spectrum of Stiffness Matrix
                </p>
            </div>
            
            <div style="flex: 1; min-width: 350px;">
                <div style="background: #ffebee; padding: 20px; border-radius: 5px; margin-bottom: 15px; border: 3px solid #f44336;">
                    <h4 style="color: #000;">Condition Number</h4>
                    <div style="font-size: 32px; text-align: center; color: #f44336; font-weight: bold; margin: 20px 0;" id="cond-display">
                    </div>
                    <div style="text-align: center; color: #666; font-size: 14px;">
                        κ(K) = λ<sub>max</sub> / λ<sub>min</sub>
                    </div>
                </div>
                
                <div style="background: #e3f2fd; padding: 20px; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="color: #000;">Eigenvalues</h4>
                    <div id="eigenvalue-info"></div>
                </div>
                
                <div style="background: #fff3cd; padding: 20px; border-radius: 5px;">
                    <h4 style="color: #000;">Solver Implications</h4>
                    <div id="solver-info"></div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 20px; background: #f0f0f0; border-radius: 5px;">
            <h4 style="color: #000;">📊 Scaling Analysis</h4>
            <canvas id="canvas-scaling" width="900" height="300" 
                    style="border: 1px solid #ccc; background: white; max-width: 100%;"></canvas>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 5px;">
            <h4 style="color: #000;">Key Facts</h4>
            <ul style="color: #000; line-height: 1.8;">
                <li><strong>Eigenvalue bounds:</strong> λ<sub>min</sub> ~ π²h, λ<sub>max</sub> ~ 4/h</li>
                <li><strong>Condition number:</strong> κ(K) = O(h⁻²)</li>
                <li><strong>Doubling refinement:</strong> 2× more unknowns, 4× worse conditioning</li>
                <li><strong>CG iterations:</strong> Converges in O(√κ) iterations</li>
                <li><strong>Solution:</strong> Use preconditioners (e.g., multigrid, ILU)</li>
            </ul>
        </div>
    </div>
`;

const canvasMesh = container.querySelector('#canvas-mesh');
const ctxMesh = canvasMesh.getContext('2d');
const canvasEig = container.querySelector('#canvas-eigenvalues');
const ctxEig = canvasEig.getContext('2d');
const canvasScale = container.querySelector('#canvas-scaling');
const ctxScale = canvasScale.getContext('2d');

const slider = container.querySelector('#mesh-slider');
const nValue = container.querySelector('#n-value');
const hValue = container.querySelector('#h-value');

// Store history for scaling plot
let history = [];

function computeEigenvalues(n) {
    // For 1D Poisson: λ_k = (4/h) * sin²(kπ/(2(n+1)))
    const h = 1 / (n + 1);
    const eigenvalues = [];
    
    for (let k = 1; k <= n; k++) {
        const lambda = (4 / h) * Math.pow(Math.sin(k * Math.PI / (2 * (n + 1))), 2);
        eigenvalues.push(lambda);
    }
    
    return eigenvalues.sort((a, b) => a - b);
}

function drawMesh(n) {
    ctxMesh.clearRect(0, 0, canvasMesh.width, canvasMesh.height);
    
    const padding = 50;
    const width = canvasMesh.width - 2 * padding;
    const y = canvasMesh.height / 2;
    
    // Draw domain line
    ctxMesh.strokeStyle = '#666';
    ctxMesh.lineWidth = 2;
    ctxMesh.beginPath();
    ctxMesh.moveTo(padding, y);
    ctxMesh.lineTo(canvasMesh.width - padding, y);
    ctxMesh.stroke();
    
    // Draw nodes
    for (let i = 0; i <= n + 1; i++) {
        const x = padding + (i / (n + 1)) * width;
        
        // Node circle
        if (i === 0 || i === n + 1) {
            // Boundary nodes (gray)
            ctxMesh.fillStyle = '#999';
            ctxMesh.strokeStyle = 'black';
        } else {
            // Interior nodes (blue)
            ctxMesh.fillStyle = '#2196f3';
            ctxMesh.strokeStyle = 'black';
        }
        
        ctxMesh.beginPath();
        ctxMesh.arc(x, y, 6, 0, 2*Math.PI);
        ctxMesh.fill();
        ctxMesh.lineWidth = 2;
        ctxMesh.stroke();
        
        // Label
        ctxMesh.fillStyle = 'black';
        ctxMesh.font = '12px Arial';
        ctxMesh.textAlign = 'center';
        ctxMesh.fillText(`${i}`, x, y + 25);
    }
    
    // Draw h annotation
    if (n >= 2) {
        const x1 = padding + width / (n + 1);
        const x2 = padding + 2 * width / (n + 1);
        const yArrow = y + 50;
        
        ctxMesh.strokeStyle = '#ff9800';
        ctxMesh.fillStyle = '#ff9800';
        ctxMesh.lineWidth = 2;
        
        // Line
        ctxMesh.beginPath();
        ctxMesh.moveTo(x1, yArrow);
        ctxMesh.lineTo(x2, yArrow);
        ctxMesh.stroke();
        
        // Arrows
        ctxMesh.beginPath();
        ctxMesh.moveTo(x1, yArrow);
        ctxMesh.lineTo(x1 + 8, yArrow - 4);
        ctxMesh.lineTo(x1 + 8, yArrow + 4);
        ctxMesh.fill();
        
        ctxMesh.beginPath();
        ctxMesh.moveTo(x2, yArrow);
        ctxMesh.lineTo(x2 - 8, yArrow - 4);
        ctxMesh.lineTo(x2 - 8, yArrow + 4);
        ctxMesh.fill();
        
        // Label
        ctxMesh.font = 'bold 14px Arial';
        ctxMesh.fillText('h', (x1 + x2) / 2, yArrow + 20);
    }
}

function drawEigenvalues(eigenvalues) {
    ctxEig.clearRect(0, 0, canvasEig.width, canvasEig.height);
    
    if (eigenvalues.length === 0) return;
    
    const padding = 50;
    const width = canvasEig.width - 2 * padding;
    const height = canvasEig.height - 2 * padding;
    
    const maxEig = Math.max(...eigenvalues);
    
    // Draw axes
    ctxEig.strokeStyle = 'black';
    ctxEig.lineWidth = 2;
    ctxEig.beginPath();
    ctxEig.moveTo(padding, padding);
    ctxEig.lineTo(padding, canvasEig.height - padding);
    ctxEig.lineTo(canvasEig.width - padding, canvasEig.height - padding);
    ctxEig.stroke();
    
    // Draw bars
    const barWidth = width / eigenvalues.length * 0.8;
    
    eigenvalues.forEach((eig, i) => {
        const x = padding + (i + 0.5) * width / eigenvalues.length - barWidth / 2;
        const barHeight = (eig / maxEig) * height;
        const y = canvasEig.height - padding - barHeight;
        
        // Color code by size
        let color = '#4caf50';
        if (i === 0) color = '#2196f3'; // Minimum
        if (i === eigenvalues.length - 1) color = '#f44336'; // Maximum
        
        ctxEig.fillStyle = color;
        ctxEig.fillRect(x, y, barWidth, barHeight);
        
        // Border
        ctxEig.strokeStyle = 'black';
        ctxEig.lineWidth = 1;
        ctxEig.strokeRect(x, y, barWidth, barHeight);
    });
    
    // Labels
    ctxEig.fillStyle = 'black';
    ctxEig.font = '12px Arial';
    ctxEig.textAlign = 'center';
    ctxEig.fillText('Index k', canvasEig.width / 2, canvasEig.height - 10);
    
    ctxEig.save();
    ctxEig.translate(15, canvasEig.height / 2);
    ctxEig.rotate(-Math.PI / 2);
    ctxEig.fillText('Eigenvalue λₖ', 0, 0);
    ctxEig.restore();
    
    // Legend
    ctxEig.fillStyle = '#2196f3';
    ctxEig.fillRect(padding + 10, 15, 15, 15);
    ctxEig.fillStyle = 'black';
    ctxEig.font = '11px Arial';
    ctxEig.textAlign = 'left';
    ctxEig.fillText('λ_min', padding + 30, 27);
    
    ctxEig.fillStyle = '#f44336';
    ctxEig.fillRect(padding + 80, 15, 15, 15);
    ctxEig.fillStyle = 'black';
    ctxEig.fillText('λ_max', padding + 100, 27);
}

function updateInfo(n, eigenvalues) {
    const h = 1 / (n + 1);
    const lambda_min = eigenvalues[0];
    const lambda_max = eigenvalues[eigenvalues.length - 1];
    const kappa = lambda_max / lambda_min;
    
    // Update display
    nValue.textContent = n;
    hValue.textContent = h.toFixed(4);
    
    // Condition number
    const condDisplay = container.querySelector('#cond-display');
    condDisplay.textContent = kappa.toFixed(2);
    
    // Eigenvalue info
    const eigInfo = container.querySelector('#eigenvalue-info');
    eigInfo.innerHTML = `
        <table style="width: 100%; line-height: 1.8; color: #000;">
            <tr>
                <td><strong>λ<sub>min</sub>:</strong></td>
                <td style="text-align: right; font-family: monospace;">${lambda_min.toFixed(4)}</td>
            </tr>
            <tr>
                <td><strong>λ<sub>max</sub>:</strong></td>
                <td style="text-align: right; font-family: monospace;">${lambda_max.toFixed(4)}</td>
            </tr>
            <tr style="background: #ffffcc;">
                <td><strong>Ratio:</strong></td>
                <td style="text-align: right; font-family: monospace;">${(lambda_max/lambda_min).toFixed(2)}</td>
            </tr>
            <tr>
                <td><strong>Theory λ<sub>min</sub>:</strong></td>
                <td style="text-align: right; font-family: monospace;">π²h ≈ ${(Math.PI * Math.PI * h).toFixed(4)}</td>
            </tr>
            <tr>
                <td><strong>Theory λ<sub>max</sub>:</strong></td>
                <td style="text-align: right; font-family: monospace;">4/h ≈ ${(4/h).toFixed(4)}</td>
            </tr>
        </table>
    `;
    
    // Solver info
    const iterations_cg = Math.ceil(Math.sqrt(kappa));
    const solverInfo = container.querySelector('#solver-info');
    
    let quality = 'Good';
    let qualityColor = '#4caf50';
    if (kappa > 100) { quality = 'Challenging'; qualityColor = '#ff9800'; }
    if (kappa > 1000) { quality = 'Difficult'; qualityColor = '#f44336'; }
    
    solverInfo.innerHTML = `
        <table style="width: 100%; line-height: 1.8; color: #000;">
            <tr>
                <td><strong>CG iterations:</strong></td>
                <td style="text-align: right;">~${iterations_cg}</td>
            </tr>
            <tr>
                <td><strong>Direct solve:</strong></td>
                <td style="text-align: right;">O(n³) = ${Math.pow(n, 3)}</td>
            </tr>
            <tr style="background: ${qualityColor}30;">
                <td><strong>Difficulty:</strong></td>
                <td style="text-align: right; color: ${qualityColor}; font-weight: bold;">${quality}</td>
            </tr>
        </table>
        <p style="color: #666; margin-top: 10px; font-size: 12px; font-style: italic;">
            ${kappa > 1000 ? '⚠️ Preconditioning strongly recommended!' : 'Direct solver acceptable for this size.'}
        </p>
    `;
    
    // Update history
    history.push({ h, kappa });
    if (history.length > 20) history.shift();
    
    drawScalingPlot();
}

function drawScalingPlot() {
    ctxScale.clearRect(0, 0, canvasScale.width, canvasScale.height);
    
    if (history.length < 2) return;
    
    const padding = 60;
    const width = canvasScale.width - 2 * padding;
    const height = canvasScale.height - 2 * padding;
    
    // Find ranges
    const h_vals = history.map(d => d.h);
    const kappa_vals = history.map(d => d.kappa);
    const h_min = Math.min(...h_vals);
    const h_max = Math.max(...h_vals);
    const kappa_max = Math.max(...kappa_vals);
    
    // Draw axes
    ctxScale.strokeStyle = 'black';
    ctxScale.lineWidth = 2;
    ctxScale.beginPath();
    ctxScale.moveTo(padding, padding);
    ctxScale.lineTo(padding, canvasScale.height - padding);
    ctxScale.lineTo(canvasScale.width - padding, canvasScale.height - padding);
    ctxScale.stroke();
    
    // Draw theoretical O(h^-2) line
    ctxScale.strokeStyle = '#ff9800';
    ctxScale.lineWidth = 2;
    ctxScale.setLineDash([5, 5]);
    ctxScale.beginPath();
    
    for (let i = 0; i <= 100; i++) {
        const h = h_min + (h_max - h_min) * i / 100;
        const kappa_theory = 4 / (Math.PI * Math.PI * h * h);
        
        const x = padding + ((h - h_min) / (h_max - h_min)) * width;
        const y = canvasScale.height - padding - (kappa_theory / kappa_max) * height;
        
        if (i === 0) ctxScale.moveTo(x, y);
        else ctxScale.lineTo(x, y);
    }
    ctxScale.stroke();
    ctxScale.setLineDash([]);
    
    // Draw actual data points
    ctxScale.strokeStyle = '#2196f3';
    ctxScale.fillStyle = '#2196f3';
    ctxScale.lineWidth = 2;
    ctxScale.beginPath();
    
    history.forEach((d, i) => {
        const x = padding + ((d.h - h_min) / (h_max - h_min)) * width;
        const y = canvasScale.height - padding - (d.kappa / kappa_max) * height;
        
        if (i === 0) ctxScale.moveTo(x, y);
        else ctxScale.lineTo(x, y);
    });
    ctxScale.stroke();
    
    // Draw points
    history.forEach(d => {
        const x = padding + ((d.h - h_min) / (h_max - h_min)) * width;
        const y = canvasScale.height - padding - (d.kappa / kappa_max) * height;
        
        ctxScale.beginPath();
        ctxScale.arc(x, y, 5, 0, 2*Math.PI);
        ctxScale.fill();
    });
    
    // Labels
    ctxScale.fillStyle = 'black';
    ctxScale.font = '14px Arial';
    ctxScale.textAlign = 'center';
    ctxScale.fillText('Mesh size h', canvasScale.width / 2, canvasScale.height - 15);
    
    ctxScale.save();
    ctxScale.translate(20, canvasScale.height / 2);
    ctxScale.rotate(-Math.PI / 2);
    ctxScale.fillText('Condition number κ(K)', 0, 0);
    ctxScale.restore();
    
    // Legend
    ctxScale.fillStyle = '#2196f3';
    ctxScale.fillRect(padding + 10, 15, 20, 3);
    ctxScale.fillStyle = 'black';
    ctxScale.font = '12px Arial';
    ctxScale.textAlign = 'left';
    ctxScale.fillText('Actual κ(K)', padding + 35, 20);
    
    ctxScale.strokeStyle = '#ff9800';
    ctxScale.setLineDash([5, 5]);
    ctxScale.lineWidth = 2;
    ctxScale.beginPath();
    ctxScale.moveTo(padding + 130, 16);
    ctxScale.lineTo(padding + 150, 16);
    ctxScale.stroke();
    ctxScale.setLineDash([]);
    ctxScale.fillStyle = 'black';
    ctxScale.fillText('Theory O(h⁻²)', padding + 155, 20);
}

function update() {
    const n = parseInt(slider.value);
    const eigenvalues = computeEigenvalues(n);
    
    drawMesh(n);
    drawEigenvalues(eigenvalues);
    updateInfo(n, eigenvalues);
}

slider.oninput = update;

// Initial draw
update();
})();