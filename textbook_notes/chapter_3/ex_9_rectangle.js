// Exercise 3.x.9: Rectangular Bilinear Element
(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for ex_9_rectangle');
        return;
    }

    container.innerHTML = `
    <div style="max-width: 900px; margin: 0 auto;">
        <h3 style="color: #00ff41;">Bilinear Basis Functions on Rectangle [-1,1] × [0,1]</h3>
        <p style="color: #e4e4e4;">Select a basis function to visualize:</p>
        
        <div style="display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap;">
            <button id="btn-phi1" class="basis-btn">φ₁ (bottom-left)</button>
            <button id="btn-phi2" class="basis-btn">φ₂ (bottom-right)</button>
            <button id="btn-phi3" class="basis-btn">φ₃ (top-right)</button>
            <button id="btn-phi4" class="basis-btn">φ₄ (top-left)</button>
            <button id="btn-all" class="basis-btn">All (sum=1)</button>
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div>
                <canvas id="canvas-rect" width="400" height="400" style="border: 1px solid #ccc; background: white;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">2D View (lighter = higher value)</p>
            </div>
            <div>
                <canvas id="canvas-3d" width="400" height="400" style="border: 1px solid #ccc; background: white;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">3D Perspective View</p>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px;">
            <h4 id="formula-title" style="color: #000;">Basis Function Formula</h4>
            <div id="formula-content" style="font-family: monospace; font-size: 16px; color: #000;"></div>
            <div id="properties" style="margin-top: 10px; color: #000;"></div>
        </div>
    </div>
`;

const canvas2d = container.querySelector('#canvas-rect');
const ctx2d = canvas2d.getContext('2d');
const canvas3d = container.querySelector('#canvas-3d');
const ctx3d = canvas3d.getContext('2d');

// Bilinear basis functions
const phi = [
    (x, y) => (1-x)*(1-y)/4,  // φ₁: bottom-left vertex
    (x, y) => (1+x)*(1-y)/4,  // φ₂: bottom-right vertex
    (x, y) => (1+x)*(1+y)/4,  // φ₃: top-right vertex
    (x, y) => (1-x)*(1+y)/4   // φ₄: top-left vertex
];

const vertices = [
    [-1, 0],  // v1
    [1, 0],   // v2
    [1, 1],   // v3
    [-1, 1]   // v4
];

const formulas = [
    'φ₁(x,y) = (1-x)(1-y)/4',
    'φ₂(x,y) = (1+x)(1-y)/4',
    'φ₃(x,y) = (1+x)(1+y)/4',
    'φ₄(x,y) = (1-x)(1+y)/4'
];

const descriptions = [
    'Bottom-left vertex: (-1, 0)',
    'Bottom-right vertex: (1, 0)',
    'Top-right vertex: (1, 1)',
    'Top-left vertex: (-1, 1)'
];

function draw2D(basisIdx) {
    ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);
    
    const resolution = 40;
    const w = canvas2d.width;
    const h = canvas2d.height;
    
    // Draw basis function values as color map
    for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
            const x = -1 + (2 * i / resolution);
            const y = 0 + (1 * j / resolution);
            
            let value;
            if (basisIdx === 'all') {
                value = phi[0](x,y) + phi[1](x,y) + phi[2](x,y) + phi[3](x,y);
            } else {
                value = phi[basisIdx](x, y);
            }
            
            // Map value [0,1] to color (blue to red)
            const intensity = Math.floor(255 * value);
            ctx2d.fillStyle = `rgb(${255-intensity}, ${255-intensity}, 255)`;
            
            const px = (x + 1) / 2 * w;
            const py = h - (y / 1 * h);
            const dx = w / resolution;
            const dy = h / resolution;
            
            // +1 to avoid gaps between cells
            ctx2d.fillRect(px, py, dx + 1, dy + 1);
        }
    }
    
    // Draw rectangle boundary (thicker)
    ctx2d.strokeStyle = 'black';
    ctx2d.lineWidth = 3;
    ctx2d.strokeRect(0, 0, w, h);
    
    // Draw vertices (larger and with borders)
    vertices.forEach((v, i) => {
        const px = (v[0] + 1) / 2 * w;
        const py = h - (v[1] / 1 * h);
        
        // Fill with color
        ctx2d.fillStyle = (basisIdx === i || basisIdx === 'all') ? 'red' : 'blue';
        ctx2d.beginPath();
        ctx2d.arc(px, py, 8, 0, 2*Math.PI);
        ctx2d.fill();
        
        // Black border for visibility
        ctx2d.strokeStyle = 'black';
        ctx2d.lineWidth = 2;
        ctx2d.stroke();
        
        // Label (adjusted position)
        ctx2d.fillStyle = 'black';
        ctx2d.font = 'bold 14px Arial';
        const labelX = (i === 0 || i === 3) ? px - 30 : px + 12;  // Left side labels go left
        const labelY = (i < 2) ? py + 20 : py - 10;  // Bottom vertices below, top above
        ctx2d.fillText(`v${i+1}`, labelX, labelY);
    });
}

function draw3D(basisIdx) {
    ctx3d.clearRect(0, 0, canvas3d.width, canvas3d.height);
    
    const w = canvas3d.width;
    const h = canvas3d.height;
    const resolution = 30;
    
    // 3D projection parameters - better angle
    const scale = 120;
    const angleX = 1.0;  // More tilt
    const angleZ = 0.8;   // More rotation
    
    function project3D(x, y, z) {
        // Rotate around X axis
        const y1 = y * Math.cos(angleX) - z * Math.sin(angleX);
        const z1 = y * Math.sin(angleX) + z * Math.cos(angleX);
        
        // Rotate around Z axis
        const x2 = x * Math.cos(angleZ) - y1 * Math.sin(angleZ);
        const y2 = x * Math.sin(angleZ) + y1 * Math.cos(angleZ);
        
        return {
            x: w/2 + x2 * scale,
            y: h/2 - y2 * scale - z1 * scale * 1.5 + 50  // Shifted down by 50px
        };
    }
    
    // Draw grid surface with DARKER lines
    ctx3d.strokeStyle = 'rgba(0,0,0,0.6)';  // Much darker
    ctx3d.lineWidth = 1.0;  // Thicker
    
    for (let i = 0; i <= resolution; i++) {
        const x = -1 + (2 * i / resolution);
        ctx3d.beginPath();
        for (let j = 0; j <= resolution; j++) {
            const y = 0 + (1 * j / resolution);
            let z;
            if (basisIdx === 'all') {
                z = phi[0](x,y) + phi[1](x,y) + phi[2](x,y) + phi[3](x,y);
            } else {
                z = phi[basisIdx](x, y);
            }
            
            const p = project3D(x, y, z);
            if (j === 0) ctx3d.moveTo(p.x, p.y);
            else ctx3d.lineTo(p.x, p.y);
        }
        ctx3d.stroke();
    }
    
    for (let j = 0; j <= resolution; j++) {
        const y = 0 + (1 * j / resolution);
        ctx3d.beginPath();
        for (let i = 0; i <= resolution; i++) {
            const x = -1 + (2 * i / resolution);
            let z;
            if (basisIdx === 'all') {
                z = phi[0](x,y) + phi[1](x,y) + phi[2](x,y) + phi[3](x,y);
            } else {
                z = phi[basisIdx](x, y);
            }
            
            const p = project3D(x, y, z);
            if (i === 0) ctx3d.moveTo(p.x, p.y);
            else ctx3d.lineTo(p.x, p.y);
        }
        ctx3d.stroke();
    }
    
    // Draw axes with thicker lines
    ctx3d.strokeStyle = 'black';
    ctx3d.lineWidth = 2;
    
    // X axis
    ctx3d.beginPath();
    let p = project3D(-1.2, 0, 0);
    ctx3d.moveTo(p.x, p.y);
    p = project3D(1.2, 0, 0);
    ctx3d.lineTo(p.x, p.y);
    ctx3d.stroke();
    
    // Y axis
    ctx3d.beginPath();
    p = project3D(0, -0.2, 0);
    ctx3d.moveTo(p.x, p.y);
    p = project3D(0, 1.2, 0);
    ctx3d.lineTo(p.x, p.y);
    ctx3d.stroke();
    
    // Z axis
    ctx3d.beginPath();
    p = project3D(0, 0, 0);
    ctx3d.moveTo(p.x, p.y);
    p = project3D(0, 0, 1.2);
    ctx3d.lineTo(p.x, p.y);
    ctx3d.stroke();
    
    // Labels
    ctx3d.fillStyle = 'black';
    ctx3d.font = '14px Arial';
    p = project3D(1.3, 0, 0);
    ctx3d.fillText('x', p.x, p.y);
    p = project3D(0, 1.3, 0);
    ctx3d.fillText('y', p.x, p.y);
    p = project3D(0, 0, 1.3);
    ctx3d.fillText('φ', p.x, p.y);
}

function updateFormula(idx) {
    const formulaContent = container.querySelector('#formula-content');
    const properties = container.querySelector('#properties');
    
    if (idx === 'all') {
        formulaContent.textContent = 'Σφᵢ(x,y) = 1 (partition of unity)';
        properties.innerHTML = `
            <strong>Properties:</strong><br>
            • All basis functions sum to 1 at every point<br>
            • This is the "partition of unity" property<br>
            • Essential for finite element interpolation
        `;
    } else {
        formulaContent.textContent = formulas[idx];
        properties.innerHTML = `
            <strong>Properties:</strong><br>
            • ${descriptions[idx]}<br>
            • φ${idx+1}(v${idx+1}) = 1, φ${idx+1}(vⱼ) = 0 for j ≠ ${idx+1}<br>
            • φ${idx+1} ∈ Q₁ (bilinear)<br>
            • φ${idx+1} ∈ [0, 1] on the rectangle
        `;
    }
}

function visualize(idx) {
    draw2D(idx);
    draw3D(idx);
    updateFormula(idx);
}

// Set up button handlers
container.querySelector('#btn-phi1').onclick = () => visualize(0);
container.querySelector('#btn-phi2').onclick = () => visualize(1);
container.querySelector('#btn-phi3').onclick = () => visualize(2);
container.querySelector('#btn-phi4').onclick = () => visualize(3);
container.querySelector('#btn-all').onclick = () => visualize('all');

// Initial visualization
visualize(0);
})();