// Exercise 3.x.10: Quadratic Triangle Element
(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for ex_10_triangle');
        return;
    }

    container.innerHTML = `
    <div style="max-width: 900px; margin: 0 auto;">
        <h3 style="color: #00ff41;">Quadratic Basis Functions on Reference Triangle</h3>
        <p style="color: #e4e4e4;">Select a basis function to visualize (P₂ space, 6 DOFs):</p>
        
        <div style="display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap;">
            <button id="btn-phi1" class="basis-btn">φ₁ (v₁)</button>
            <button id="btn-phi2" class="basis-btn">φ₂ (v₂)</button>
            <button id="btn-phi3" class="basis-btn">φ₃ (v₃)</button>
            <button id="btn-phi4" class="basis-btn">φ₄ (m₁₂)</button>
            <button id="btn-phi5" class="basis-btn">φ₅ (m₂₃)</button>
            <button id="btn-phi6" class="basis-btn">φ₆ (m₁₃)</button>
            <button id="btn-all" class="basis-btn">Sum=1</button>
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div>
                <canvas id="canvas-tri" width="400" height="400" style="border: 1px solid #ccc; background: white;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">2D View (blue = 0, red = 1)</p>
            </div>
            <div>
                <canvas id="canvas-3d-tri" width="400" height="400" style="border: 1px solid #ccc; background: white;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">3D Perspective View</p>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px;">
            <h4 style="color: #000;">Barycentric Formula</h4>
            <div id="formula-tri" style="font-family: monospace; font-size: 14px; margin-top: 10px; color: #000;"></div>
            <div id="cartesian-tri" style="font-family: monospace; font-size: 13px; margin-top: 10px; color: #555;"></div>
            <div id="properties-tri" style="margin-top: 10px; color: #000;"></div>
        </div>
    </div>
`;

const canvas2d = container.querySelector('#canvas-tri');
const ctx2d = canvas2d.getContext('2d');
const canvas3d = container.querySelector('#canvas-3d-tri');
const ctx3d = canvas3d.getContext('2d');

// Barycentric to Cartesian: λ₁ = 1-x-y, λ₂ = x, λ₃ = y
// Vertices of reference triangle
const vertices = [
    [0, 0],      // v1
    [1, 0],      // v2
    [0, 1]       // v3
];

const midpoints = [
    [0.5, 0],    // m12 (edge v1-v2)
    [0.5, 0.5],  // m23 (edge v2-v3)
    [0, 0.5]     // m13 (edge v1-v3)
];

// Quadratic basis functions in barycentric coordinates
const phi = [
    // Vertex basis functions
    (x, y) => {
        const l1 = 1 - x - y;
        return l1 * (2*l1 - 1);
    },
    (x, y) => {
        const l2 = x;
        return l2 * (2*l2 - 1);
    },
    (x, y) => {
        const l3 = y;
        return l3 * (2*l3 - 1);
    },
    // Edge midpoint basis functions
    (x, y) => {
        const l1 = 1 - x - y;
        const l2 = x;
        return 4 * l1 * l2;
    },
    (x, y) => {
        const l2 = x;
        const l3 = y;
        return 4 * l2 * l3;
    },
    (x, y) => {
        const l1 = 1 - x - y;
        const l3 = y;
        return 4 * l1 * l3;
    }
];

const barycentricFormulas = [
    'φ₁ = λ₁(2λ₁ - 1)',
    'φ₂ = λ₂(2λ₂ - 1)',
    'φ₃ = λ₃(2λ₃ - 1)',
    'φ₄ = 4λ₁λ₂',
    'φ₅ = 4λ₂λ₃',
    'φ₆ = 4λ₁λ₃'
];

const cartesianFormulas = [
    'φ₁ = (1-x-y)(1-2x-2y)',
    'φ₂ = x(2x-1)',
    'φ₃ = y(2y-1)',
    'φ₄ = 4x(1-x-y)',
    'φ₅ = 4xy',
    'φ₆ = 4y(1-x-y)'
];

const descriptions = [
    'Vertex v₁ = (0,0): quadratic bubble, 1 at v₁, 0 at other nodes',
    'Vertex v₂ = (1,0): quadratic bubble, 1 at v₂, 0 at other nodes',
    'Vertex v₃ = (0,1): quadratic bubble, 1 at v₃, 0 at other nodes',
    'Edge midpoint m₁₂ = (1/2,0): 1 at m₁₂, 0 at other nodes',
    'Edge midpoint m₂₃ = (1/2,1/2): 1 at m₂₃, 0 at other nodes',
    'Edge midpoint m₁₃ = (0,1/2): 1 at m₁₃, 0 at other nodes'
];

function isInTriangle(x, y) {
    return x >= 0 && y >= 0 && x + y <= 1;
}

function draw2D(basisIdx) {
    ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);
    
    const resolution = 100;
    const w = canvas2d.width;
    const h = canvas2d.height;
    const padding = 50;
    const scale = w - 2*padding;
    
    // Draw basis function values
    for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
            const x = i / resolution;
            const y = j / resolution;
            
            if (!isInTriangle(x, y)) continue;
            
            let value;
            if (basisIdx === 'all') {
                value = 0;
                for (let k = 0; k < 6; k++) {
                    value += phi[k](x, y);
                }
            } else {
                value = phi[basisIdx](x, y);
            }
            
            // Color map: blue (0) to red (1)
            const r = Math.floor(255 * value);
            const b = Math.floor(255 * (1 - value));
            ctx2d.fillStyle = `rgb(${r}, 0, ${b})`;
            
            const px = padding + x * scale;
            const py = h - padding - y * scale;
            const dx = scale / resolution;
            const dy = scale / resolution;
            
            ctx2d.fillRect(px, py, dx, dy);
        }
    }
    
    // Draw triangle outline
    ctx2d.strokeStyle = 'black';
    ctx2d.lineWidth = 2;
    ctx2d.beginPath();
    ctx2d.moveTo(padding, h - padding);
    ctx2d.lineTo(padding + scale, h - padding);
    ctx2d.lineTo(padding, h - padding - scale);
    ctx2d.closePath();
    ctx2d.stroke();
    
    // Draw vertices
    vertices.forEach((v, i) => {
        const px = padding + v[0] * scale;
        const py = h - padding - v[1] * scale;
        
        ctx2d.fillStyle = (basisIdx === i || basisIdx === 'all') ? 'red' : 'black';
        ctx2d.beginPath();
        ctx2d.arc(px, py, 6, 0, 2*Math.PI);
        ctx2d.fill();
        
        ctx2d.fillStyle = 'black';
        ctx2d.font = 'bold 14px Arial';
        ctx2d.fillText(`v${i+1}`, px + 10, py - 10);
    });
    
    // Draw midpoints
    midpoints.forEach((m, i) => {
        const px = padding + m[0] * scale;
        const py = h - padding - m[1] * scale;
        
        ctx2d.fillStyle = (basisIdx === i+3 || basisIdx === 'all') ? 'red' : 'green';
        ctx2d.beginPath();
        ctx2d.arc(px, py, 5, 0, 2*Math.PI);
        ctx2d.fill();
        
        ctx2d.fillStyle = 'black';
        ctx2d.font = '12px Arial';
        const label = ['m₁₂', 'm₂₃', 'm₁₃'][i];
        ctx2d.fillText(label, px + 10, py - 10);
    });
}

function draw3D(basisIdx) {
    ctx3d.clearRect(0, 0, canvas3d.width, canvas3d.height);
    
    const w = canvas3d.width;
    const h = canvas3d.height;
    const resolution = 40;
    
    // 3D projection - improved angle
    const scale = 200;
    const angleX = 1.1;  // Better tilt
    const angleZ = 0.7;  // Better rotation
    
    function project3D(x, y, z) {
        const y1 = y * Math.cos(angleX) - z * Math.sin(angleX);
        const z1 = y * Math.sin(angleX) + z * Math.cos(angleX);
        
        const x2 = x * Math.cos(angleZ) - y1 * Math.sin(angleZ);
        const y2 = x * Math.sin(angleZ) + y1 * Math.cos(angleZ);
        
        return {
            x: w/2 + x2 * scale - 50,  // Shifted left (was +30)
            y: h/2 - y2 * scale - z1 * scale * 1.5 + 80  // Shifted down (was +20)
        };
    }
    
    // Draw mesh - MUCH DARKER
    ctx3d.strokeStyle = 'rgba(0,0,0,0.7)';  // Very dark
    ctx3d.lineWidth = 1.0;  // Thicker
    
    // Parallel to x-axis
    for (let j = 0; j <= resolution; j++) {
        const y = j / resolution;
        ctx3d.beginPath();
        let firstPoint = true;
        for (let i = 0; i <= resolution; i++) {
            const x = i / resolution;
            if (!isInTriangle(x, y)) continue;
            
            let z;
            if (basisIdx === 'all') {
                z = 0;
                for (let k = 0; k < 6; k++) {
                    z += phi[k](x, y);
                }
            } else {
                z = phi[basisIdx](x, y);
            }
            
            const p = project3D(x, y, z);
            if (firstPoint) {
                ctx3d.moveTo(p.x, p.y);
                firstPoint = false;
            } else {
                ctx3d.lineTo(p.x, p.y);
            }
        }
        ctx3d.stroke();
    }
    
    // Parallel to y-axis
    for (let i = 0; i <= resolution; i++) {
        const x = i / resolution;
        ctx3d.beginPath();
        let firstPoint = true;
        for (let j = 0; j <= resolution; j++) {
            const y = j / resolution;
            if (!isInTriangle(x, y)) continue;
            
            let z;
            if (basisIdx === 'all') {
                z = 0;
                for (let k = 0; k < 6; k++) {
                    z += phi[k](x, y);
                }
            } else {
                z = phi[basisIdx](x, y);
            }
            
            const p = project3D(x, y, z);
            if (firstPoint) {
                ctx3d.moveTo(p.x, p.y);
                firstPoint = false;
            } else {
                ctx3d.lineTo(p.x, p.y);
            }
        }
        ctx3d.stroke();
    }
    
    // Draw triangle base - thicker
    ctx3d.strokeStyle = 'black';
    ctx3d.lineWidth = 3;
    ctx3d.beginPath();
    let p = project3D(0, 0, 0);
    ctx3d.moveTo(p.x, p.y);
    p = project3D(1, 0, 0);
    ctx3d.lineTo(p.x, p.y);
    p = project3D(0, 1, 0);
    ctx3d.lineTo(p.x, p.y);
    ctx3d.closePath();
    ctx3d.stroke();
}

function updateFormula(idx) {
    const formulaDiv = container.querySelector('#formula-tri');
    const cartesianDiv = container.querySelector('#cartesian-tri');
    const propertiesDiv = container.querySelector('#properties-tri');
    
    if (idx === 'all') {
        formulaDiv.textContent = 'Σᵢ φᵢ(x,y) = 1 (partition of unity)';
        cartesianDiv.textContent = '';
        propertiesDiv.innerHTML = `
            <strong>Verification:</strong> All 6 basis functions sum to 1 everywhere on the triangle.<br>
            This confirms the partition of unity property for P₂ elements.
        `;
    } else {
        formulaDiv.innerHTML = `<strong>Barycentric:</strong> ${barycentricFormulas[idx]}<br>
                                where λ₁ = 1-x-y, λ₂ = x, λ₃ = y`;
        cartesianDiv.innerHTML = `<strong>Cartesian:</strong> ${cartesianFormulas[idx]}`;
        propertiesDiv.innerHTML = `<strong>Node:</strong> ${descriptions[idx]}`;
    }
}

function visualize(idx) {
    draw2D(idx);
    draw3D(idx);
    updateFormula(idx);
}

// Button handlers
for (let i = 0; i < 6; i++) {
    container.querySelector(`#btn-phi${i+1}`).onclick = () => visualize(i);
}
container.querySelector('#btn-all').onclick = () => visualize('all');

// Initial
visualize(0);
})();