// Exercise 4.x.10: Minimum Angle Condition and Mesh Quality
(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for ex_10_angle_checker');
        return;
    }

    container.innerHTML = `
    <div style="max-width: 1000px; margin: 0 auto;">
        <h3 style="color: #00ff41;">Mesh Quality: Minimum Angle Condition</h3>
        <p style="color: #e4e4e4;">Drag vertices to see how angles affect shape regularity</p>
        
        <div style="margin: 20px 0; display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;">
            <button id="btn-equilateral" class="basis-btn">Equilateral (60° each)</button>
            <button id="btn-right" class="basis-btn">Right Triangle (90°, 45°, 45°)</button>
            <button id="btn-skinny" class="basis-btn">Skinny (10° bad!)</button>
            <button id="btn-needle" class="basis-btn">Needle (2° very bad!)</button>
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
            <div>
                <canvas id="canvas-angle" width="500" height="500" 
                        style="border: 1px solid #ccc; background: white; cursor: pointer;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">
                    Click and drag vertices to modify triangle
                </p>
            </div>
            <div style="flex: 1; min-width: 350px;">
                <div id="quality-display" style="padding: 20px; border-radius: 5px; margin-bottom: 15px;">
                </div>
                
                <div style="background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="color: #000;">Angle Measurements</h4>
                    <div id="angle-table"></div>
                </div>
                
                <div style="background: #fff3cd; padding: 20px; border-radius: 5px;">
                    <h4 style="color: #000;">Shape Regularity Metrics</h4>
                    <div id="metrics-display"></div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 5px;">
            <h4 style="color: #000;">Quality Guidelines</h4>
            <table style="width: 100%; margin-top: 10px;">
                <tr>
                    <td style="padding: 8px; color: #000;"><strong>Excellent:</strong></td>
                    <td style="padding: 8px; color: #000;">All angles 30° - 90°</td>
                    <td style="padding: 8px; background: #4caf50; color: white; text-align: center; border-radius: 3px;">h/ρ ≤ 6</td>
                </tr>
                <tr>
                    <td style="padding: 8px; color: #000;"><strong>Good:</strong></td>
                    <td style="padding: 8px; color: #000;">Min angle ≥ 20°</td>
                    <td style="padding: 8px; background: #8bc34a; color: white; text-align: center; border-radius: 3px;">h/ρ ≤ 12</td>
                </tr>
                <tr>
                    <td style="padding: 8px; color: #000;"><strong>Acceptable:</strong></td>
                    <td style="padding: 8px; color: #000;">Min angle ≥ 15°</td>
                    <td style="padding: 8px; background: #ffc107; color: black; text-align: center; border-radius: 3px;">h/ρ ≤ 20</td>
                </tr>
                <tr>
                    <td style="padding: 8px; color: #000;"><strong>Poor:</strong></td>
                    <td style="padding: 8px; color: #000;">Min angle < 15°</td>
                    <td style="padding: 8px; background: #ff9800; color: white; text-align: center; border-radius: 3px;">h/ρ > 20</td>
                </tr>
                <tr>
                    <td style="padding: 8px; color: #000;"><strong>Degenerate:</strong></td>
                    <td style="padding: 8px; color: #000;">Min angle < 5°</td>
                    <td style="padding: 8px; background: #f44336; color: white; text-align: center; border-radius: 3px;">h/ρ > 50</td>
                </tr>
            </table>
        </div>
        

    </div>
`;

const canvas = container.querySelector('#canvas-angle');
const ctx = canvas.getContext('2d');

let vertices = [
    [150, 400],
    [450, 400],
    [250, 150]
];

let dragging = -1;

function toCanvas(x, y) {
    return { x, y };
}

function dist(p1, p2) {
    return Math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2);
}

function angle(v1, vertex, v2) {
    // Angle at vertex between edges to v1 and v2
    const vec1 = [v1[0] - vertex[0], v1[1] - vertex[1]];
    const vec2 = [v2[0] - vertex[0], v2[1] - vertex[1]];
    
    const dot = vec1[0]*vec2[0] + vec1[1]*vec2[1];
    const mag1 = Math.sqrt(vec1[0]**2 + vec1[1]**2);
    const mag2 = Math.sqrt(vec2[0]**2 + vec2[1]**2);
    
    const cosAngle = dot / (mag1 * mag2);
    return Math.acos(Math.max(-1, Math.min(1, cosAngle))) * 180 / Math.PI;
}

function computeAngles() {
    return [
        angle(vertices[1], vertices[0], vertices[2]),
        angle(vertices[0], vertices[1], vertices[2]),
        angle(vertices[0], vertices[2], vertices[1])
    ];
}

function computeMetrics() {
    const [a, b, c] = [
        dist(vertices[0], vertices[1]),
        dist(vertices[1], vertices[2]),
        dist(vertices[2], vertices[0])
    ];
    
    const h = Math.max(a, b, c); // diameter
    const s = (a + b + c) / 2;   // semi-perimeter
    const area = Math.sqrt(s * (s-a) * (s-b) * (s-c)); // Heron's formula
    const rho = area / s;         // inradius
    
    return { h, rho, ratio: h / rho, area };
}

function getQuality(minAngle) {
    if (minAngle >= 30) return { label: 'Excellent', color: '#4caf50', bg: '#e8f5e9' };
    if (minAngle >= 20) return { label: 'Good', color: '#8bc34a', bg: '#f1f8e9' };
    if (minAngle >= 15) return { label: 'Acceptable', color: '#ffc107', bg: '#fff3cd' };
    if (minAngle >= 5) return { label: 'Poor', color: '#ff9800', bg: '#fff3e0' };
    return { label: 'Degenerate', color: '#f44336', bg: '#ffebee' };
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const angles = computeAngles();
    const minAngle = Math.min(...angles);
    const quality = getQuality(minAngle);
    
    // Fill triangle
    ctx.fillStyle = quality.bg;
    ctx.beginPath();
    ctx.moveTo(vertices[0][0], vertices[0][1]);
    ctx.lineTo(vertices[1][0], vertices[1][1]);
    ctx.lineTo(vertices[2][0], vertices[2][1]);
    ctx.closePath();
    ctx.fill();
    
    // Draw triangle edges
    ctx.strokeStyle = quality.color;
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Draw angle arcs
    const radius = 40;
    vertices.forEach((v, i) => {
        const v1 = vertices[(i+2)%3];
        const v2 = vertices[(i+1)%3];
        
        const vec1 = [v1[0] - v[0], v1[1] - v[1]];
        const vec2 = [v2[0] - v[0], v2[1] - v[1]];
        
        const angle1 = Math.atan2(vec1[1], vec1[0]);
        const angle2 = Math.atan2(vec2[1], vec2[0]);
        
        // Ensure arc goes the right way
        let startAngle = angle1;
        let endAngle = angle2;
        if (endAngle < startAngle) endAngle += 2*Math.PI;
        if (endAngle - startAngle > Math.PI) {
            [startAngle, endAngle] = [endAngle, startAngle + 2*Math.PI];
        }
        
        ctx.strokeStyle = quality.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(v[0], v[1], radius, startAngle, endAngle);
        ctx.stroke();
        
        // Angle label
        const midAngle = (startAngle + endAngle) / 2;
        const labelX = v[0] + Math.cos(midAngle) * (radius + 20);
        const labelY = v[1] + Math.sin(midAngle) * (radius + 20);
        
        ctx.fillStyle = quality.color;
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`${angles[i].toFixed(1)}°`, labelX, labelY);
    });
    
    // Draw vertices
    vertices.forEach((v, i) => {
        ctx.fillStyle = dragging === i ? '#ff5722' : '#2196f3';
        ctx.beginPath();
        ctx.arc(v[0], v[1], 8, 0, 2*Math.PI);
        ctx.fill();
        
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ctx.fillStyle = 'black';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`v${i+1}`, v[0], v[1] - 15);
    });
    
    updateDisplay();
}

function updateDisplay() {
    const angles = computeAngles();
    const minAngle = Math.min(...angles);
    const maxAngle = Math.max(...angles);
    const quality = getQuality(minAngle);
    const metrics = computeMetrics();
    
    // Quality banner
    const qualityDiv = container.querySelector('#quality-display');
    qualityDiv.style.background = quality.bg;
    qualityDiv.style.border = `3px solid ${quality.color}`;
    qualityDiv.innerHTML = `
        <h3 style="color: ${quality.color}; margin: 0;">
            Quality: ${quality.label}
        </h3>
        <p style="color: #000; margin: 10px 0 0 0; font-size: 18px;">
            Minimum Angle: <strong>${minAngle.toFixed(2)}°</strong>
        </p>
    `;
    
    // Angle table
    const angleTable = container.querySelector('#angle-table');
    angleTable.innerHTML = `
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <tr style="background: #e0e0e0;">
                <th style="padding: 8px; border: 1px solid #ccc; color: #000;">Vertex</th>
                <th style="padding: 8px; border: 1px solid #ccc; color: #000;">Angle</th>
                <th style="padding: 8px; border: 1px solid #ccc; color: #000;">Status</th>
            </tr>
            ${angles.map((a, i) => `
                <tr style="background: ${a === minAngle ? '#ffeb3b' : 'white'};">
                    <td style="padding: 8px; border: 1px solid #ccc; text-align: center; color: #000;">v${i+1}</td>
                    <td style="padding: 8px; border: 1px solid #ccc; text-align: center; color: #000; font-weight: bold;">
                        ${a.toFixed(2)}°
                    </td>
                    <td style="padding: 8px; border: 1px solid #ccc; text-align: center; color: #000;">
                        ${a === minAngle ? 'Min' : a === maxAngle ? 'Max' : '✓'}
                    </td>
                </tr>
            `).join('')}
        </table>
    `;
    
    // Metrics
    const metricsDiv = container.querySelector('#metrics-display');
    metricsDiv.innerHTML = `
        <table style="width: 100%; line-height: 1.8;">
            <tr>
                <td style="color: #000;"><strong>Diameter (h):</strong></td>
                <td style="color: #000; text-align: right;">${metrics.h.toFixed(2)} px</td>
            </tr>
            <tr>
                <td style="color: #000;"><strong>Inradius (ρ):</strong></td>
                <td style="color: #000; text-align: right;">${metrics.rho.toFixed(2)} px</td>
            </tr>
            <tr style="background: ${quality.bg};">
                <td style="color: #000;"><strong>Ratio (h/ρ):</strong></td>
                <td style="color: #000; text-align: right; font-size: 18px;">
                    <strong>${metrics.ratio.toFixed(2)}</strong>
                </td>
            </tr>
            <tr>
                <td style="color: #000;"><strong>Area:</strong></td>
                <td style="color: #000; text-align: right;">${metrics.area.toFixed(0)} px²</td>
            </tr>
        </table>
        <p style="color: #666; margin-top: 10px; font-size: 12px; font-style: italic;">
            Shape regularity constant γ = h/ρ should be small (≤ 20 for good meshes)
        </p>
    `;
}

// Mouse handlers
canvas.onmousedown = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    for (let i = 0; i < 3; i++) {
        if (dist([x, y], vertices[i]) < 15) {
            dragging = i;
            break;
        }
    }
};

canvas.onmousemove = (e) => {
    if (dragging >= 0) {
        const rect = canvas.getBoundingClientRect();
        const x = Math.max(20, Math.min(canvas.width - 20, e.clientX - rect.left));
        const y = Math.max(20, Math.min(canvas.height - 20, e.clientY - rect.top));
        vertices[dragging] = [x, y];
        draw();
    }
};

canvas.onmouseup = () => {
    dragging = -1;
    draw();
};

// Preset buttons
container.querySelector('#btn-equilateral').onclick = () => {
    const cx = 300, cy = 300, r = 150;
    vertices = [
        [cx, cy - r],
        [cx + r * Math.cos(7*Math.PI/6), cy + r * Math.sin(7*Math.PI/6)],
        [cx + r * Math.cos(-Math.PI/6), cy + r * Math.sin(-Math.PI/6)]
    ];
    draw();
};

container.querySelector('#btn-right').onclick = () => {
    vertices = [
        [150, 400],
        [450, 400],
        [150, 150]
    ];
    draw();
};

container.querySelector('#btn-skinny').onclick = () => {
    vertices = [
        [100, 400],
        [460, 400],
        [280, 370]
    ];
    draw();
};

container.querySelector('#btn-needle').onclick = () => {
    vertices = [
        [100, 400],
        [480, 400],
        [290, 390]
    ];
    draw();
};

// Initial draw
draw();
})();