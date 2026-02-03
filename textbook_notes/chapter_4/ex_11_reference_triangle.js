// Exercise 4.x.11: Reference Triangle Stiffness Matrix (INTERACTIVE)
(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for ex_11_reference_triangle');
        return;
    }

    container.innerHTML = `
    <div style="max-width: 1000px; margin: 0 auto;">
        <h3 style="color: #00ff41;">Reference Triangle: Basis Functions and Stiffness Matrix</h3>
        <p style="color: #e4e4e4;">Click matrix entries to see computation details • Hover over gradients for info</p>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
            <div>
                <canvas id="canvas-triangle" width="400" height="400" 
                        style="border: 1px solid #ccc; background: white; cursor: pointer;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">
                    Reference Triangle with Gradients
                </p>
            </div>
            <div style="flex: 1; min-width: 400px;">
                <div style="background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px;">
                    <h4 style="color: #000;">Basis Functions (Barycentric)</h4>
                    <div style="font-family: monospace; font-size: 14px; line-height: 1.8; color: #000;">
                        φ₁(x,y) = 1 - x - y<br>
                        φ₂(x,y) = x<br>
                        φ₃(x,y) = y
                    </div>
                </div>
                
                <div style="background: #e8f5e9; padding: 20px; border-radius: 5px; margin-bottom: 20px;">
                    <h4 style="color: #000;">Gradients (Constant!)</h4>
                    <div style="font-family: monospace; font-size: 14px; line-height: 1.8; color: #000;">
                        <div id="grad-1" style="padding: 5px; cursor: pointer; border-radius: 3px;">∇φ₁ = [-1, -1]ᵀ</div>
                        <div id="grad-2" style="padding: 5px; cursor: pointer; border-radius: 3px;">∇φ₂ = [ 1,  0]ᵀ</div>
                        <div id="grad-3" style="padding: 5px; cursor: pointer; border-radius: 3px;">∇φ₃ = [ 0,  1]ᵀ</div>
                    </div>
                </div>
                
                <div style="background: #fff3cd; padding: 20px; border-radius: 5px;">
                    <h4 style="color: #000;">Reference Stiffness Matrix K̂ <span style="font-size: 12px; font-weight: normal;">(click entries!)</span></h4>
                    <div id="stiffness-display" style="font-family: 'Courier New', monospace; font-size: 16px; color: #000; margin-top: 10px;">
                    </div>
                    <p style="color: #666; margin-top: 10px; font-size: 13px;">
                        K̂ᵢⱼ = ∫<sub>K̂</sub> ∇φᵢ · ∇φⱼ dx = (area) × (∇φᵢ · ∇φⱼ)
                    </p>
                </div>
            </div>
        </div>
        
        <div id="computation-detail" style="margin-top: 20px; padding: 20px; background: #e3f2fd; border: 2px solid #2196f3; border-radius: 5px; display: none;">
            <h4 style="color: #000; margin-top: 0;">Computing K̂<sub id="entry-label">ij</sub></h4>
            <div id="detail-content" style="color: #000; font-size: 14px; line-height: 1.8;">
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #fff; border: 2px solid #00458b; border-radius: 5px;">
            <h4 style="color: #000;">Interactive Features</h4>
            <ul style="color: #000; line-height: 1.8;">
                <li><strong>Click matrix entries:</strong> See step-by-step computation</li>
                <li><strong>Hover gradients (left):</strong> Highlight on triangle</li>
                <li><strong>Hover gradient labels (right):</strong> Emphasize that gradient</li>
            </ul>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 5px;">
            <h4 style="color: #000;">Key Implementation Facts</h4>
            <ul style="color: #000; line-height: 1.8;">
                <li><strong>Area of reference triangle:</strong> |K̂| = 1/2</li>
                <li><strong>This matrix is universal:</strong> Same for ALL P₁ triangles in reference space</li>
                <li><strong>Physical elements:</strong> Transform with Jacobian and metric tensor</li>
                <li><strong>Sparsity:</strong> Only adjacent vertices couple (off-diagonal -1)</li>
                <li><strong>Symmetry:</strong> K̂ᵀ = K̂ (from symmetry of inner product)</li>
                <li><strong>Positive definite:</strong> xᵀK̂x > 0 for x ≠ 0</li>
            </ul>
        </div>
    </div>
`;

const canvas = container.querySelector('#canvas-triangle');
const ctx = canvas.getContext('2d');

// Reference triangle vertices
const vertices = [
    [0, 0],      // v1
    [1, 0],      // v2
    [0, 1]       // v3
];

// Gradients
const gradients = [
    [-1, -1],    // ∇φ₁
    [1, 0],      // ∇φ₂
    [0, 1]       // ∇φ₃
];

const colors = ['#d32f2f', '#388e3c', '#1976d2'];
let highlightGrad = -1;
let selectedEntry = null;

function toCanvas(x, y) {
    const padding = 50;
    const scale = 300;
    return {
        x: padding + x * scale,
        y: canvas.height - padding - y * scale
    };
}

function drawTriangle() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw filled triangle (light)
    ctx.fillStyle = 'rgba(200, 230, 255, 0.3)';
    ctx.beginPath();
    const p1 = toCanvas(vertices[0][0], vertices[0][1]);
    const p2 = toCanvas(vertices[1][0], vertices[1][1]);
    const p3 = toCanvas(vertices[2][0], vertices[2][1]);
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.lineTo(p3.x, p3.y);
    ctx.closePath();
    ctx.fill();
    
    // Draw triangle outline
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.lineTo(p3.x, p3.y);
    ctx.closePath();
    ctx.stroke();
    
    // Draw vertices
    vertices.forEach((v, i) => {
        const p = toCanvas(v[0], v[1]);
        
        // Circle
        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, 2*Math.PI);
        ctx.fill();
        
        // Border
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Label
        ctx.fillStyle = 'black';
        ctx.font = 'bold 16px Arial';
        const labelX = v[0] === 0 ? p.x - 35 : p.x + 12;
        const labelY = v[1] === 0 ? p.y + 25 : p.y - 10;
        ctx.fillText(`v${i+1} = (${v[0]}, ${v[1]})`, labelX, labelY);
    });
    
    // Draw gradient vectors (from centroid)
    const centroid = toCanvas(1/3, 1/3);
    const scale = 60;
    
    const labels = ['∇φ₁', '∇φ₂', '∇φ₃'];
    
    gradients.forEach((grad, i) => {
        const endX = centroid.x + grad[0] * scale;
        const endY = centroid.y - grad[1] * scale;
        
        const isHighlighted = highlightGrad === i;
        
        // Arrow
        ctx.strokeStyle = colors[i];
        ctx.fillStyle = colors[i];
        ctx.lineWidth = isHighlighted ? 5 : 3;
        ctx.globalAlpha = isHighlighted ? 1.0 : 0.8;
        
        // Line
        ctx.beginPath();
        ctx.moveTo(centroid.x, centroid.y);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // Arrowhead
        const angle = Math.atan2(-(grad[1]), grad[0]);
        const headlen = 12;
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - headlen * Math.cos(angle - Math.PI/6),
                   endY + headlen * Math.sin(angle - Math.PI/6));
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - headlen * Math.cos(angle + Math.PI/6),
                   endY + headlen * Math.sin(angle + Math.PI/6));
        ctx.stroke();
        
        // Label
        ctx.fillStyle = colors[i];
        ctx.font = isHighlighted ? 'bold 16px Arial' : 'bold 14px Arial';
        ctx.fillText(labels[i], endX + 5, endY - 5);
        
        ctx.globalAlpha = 1.0;
    });
    
    // Draw axes
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    
    const origin = toCanvas(0, 0);
    const xEnd = toCanvas(1.3, 0);
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(xEnd.x, xEnd.y);
    ctx.stroke();
    ctx.fillStyle = 'black';
    ctx.font = '14px Arial';
    ctx.fillText('x', xEnd.x + 5, xEnd.y + 5);
    
    const yEnd = toCanvas(0, 1.3);
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(yEnd.x, yEnd.y);
    ctx.stroke();
    ctx.fillText('y', yEnd.x - 15, yEnd.y - 5);
    
    ctx.setLineDash([]);
}

function computeStiffness() {
    const K = Array(3).fill(0).map(() => Array(3).fill(0));
    const area = 0.5;
    
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            const dot = gradients[i][0] * gradients[j][0] + 
                       gradients[i][1] * gradients[j][1];
            K[i][j] = area * dot;
        }
    }
    
    return K;
}

function displayStiffness() {
    const K = computeStiffness();
    const display = container.querySelector('#stiffness-display');
    
    let html = '<div style="text-align: center;">';
    html += 'K̂ = <div style="display: inline-block; vertical-align: middle;">';
    html += '<div style="border-left: 2px solid black; border-right: 2px solid black; padding: 5px 10px;">';
    for (let i = 0; i < 3; i++) {
        html += '<div style="display: flex; gap: 15px; justify-content: center;">';
        for (let j = 0; j < 3; j++) {
            const val = K[i][j];
            const displayVal = val === 0 ? ' 0   ' : 
                             val === 1 ? ' 1   ' :
                             val === 0.5 ? ' 1/2 ' :
                             val === -0.5 ? '-1/2' :
                             val.toFixed(1);
            const isSelected = selectedEntry && selectedEntry[0] === i && selectedEntry[1] === j;
            const bg = isSelected ? '#ffd700' : 'transparent';
            html += `<span class="matrix-entry" data-i="${i}" data-j="${j}" 
                          style="width: 50px; text-align: center; cursor: pointer; 
                                 padding: 5px; border-radius: 3px; background: ${bg};
                                 transition: background 0.2s;">${displayVal}</span>`;
        }
        html += '</div>';
    }
    html += '</div></div></div>';
    
    display.innerHTML = html;
    
    // Add click handlers
    display.querySelectorAll('.matrix-entry').forEach(entry => {
        entry.onmouseover = () => {
            entry.style.background = '#e3f2fd';
        };
        entry.onmouseout = () => {
            const i = parseInt(entry.dataset.i);
            const j = parseInt(entry.dataset.j);
            const isSelected = selectedEntry && selectedEntry[0] === i && selectedEntry[1] === j;
            entry.style.background = isSelected ? '#ffd700' : 'transparent';
        };
        entry.onclick = () => {
            const i = parseInt(entry.dataset.i);
            const j = parseInt(entry.dataset.j);
            selectedEntry = [i, j];
            displayStiffness();
            showComputation(i, j);
        };
    });
}

function showComputation(i, j) {
    const detailDiv = container.querySelector('#computation-detail');
    const labelSpan = container.querySelector('#entry-label');
    const contentDiv = container.querySelector('#detail-content');
    
    detailDiv.style.display = 'block';
    labelSpan.textContent = `${i+1},${j+1}`;
    
    const grad_i = gradients[i];
    const grad_j = gradients[j];
    const dot = grad_i[0] * grad_j[0] + grad_i[1] * grad_j[1];
    const area = 0.5;
    const result = area * dot;
    
    const gradColor_i = colors[i];
    const gradColor_j = colors[j];
    
    contentDiv.innerHTML = `
        <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <strong>Step 1: Identify the gradients</strong><br>
            <span style="color: ${gradColor_i};">∇φ${i+1} = [${grad_i[0]}, ${grad_i[1]}]ᵀ</span><br>
            <span style="color: ${gradColor_j};">∇φ${j+1} = [${grad_j[0]}, ${grad_j[1]}]ᵀ</span>
        </div>
        
        <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <strong>Step 2: Compute dot product</strong><br>
            <span style="color: ${gradColor_i};">∇φ${i+1}</span> · <span style="color: ${gradColor_j};">∇φ${j+1}</span> = 
            (${grad_i[0]})(${grad_j[0]}) + (${grad_i[1]})(${grad_j[1]})<br>
            = ${grad_i[0] * grad_j[0]} + ${grad_i[1] * grad_j[1]}<br>
            = <strong>${dot}</strong>
        </div>
        
        <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <strong>Step 3: Multiply by area</strong><br>
            K̂<sub>${i+1},${j+1}</sub> = ∫<sub>K̂</sub> ∇φ${i+1} · ∇φ${j+1} dx<br>
            = (area of K̂) × (dot product)<br>
            = (1/2) × (${dot})<br>
            = <strong style="font-size: 18px; color: #2196f3;">${result === 0 ? '0' : result === 0.5 ? '1/2' : result === -0.5 ? '-1/2' : result}</strong>
        </div>
        
        <div style="background: #ffffcc; padding: 10px; border-radius: 5px;">
            <strong>Why this works:</strong> Gradients are constant over K̂, so the integral is just 
            (value) × (area). This is unique to P₁ elements on triangles!
        </div>
    `;
    
    // Scroll to detail
    detailDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Gradient hover handlers
for (let i = 1; i <= 3; i++) {
    const gradDiv = container.querySelector(`#grad-${i}`);
    gradDiv.onmouseenter = () => {
        highlightGrad = i - 1;
        gradDiv.style.background = colors[i-1] + '30';
        gradDiv.style.fontWeight = 'bold';
        drawTriangle();
    };
    gradDiv.onmouseleave = () => {
        highlightGrad = -1;
        gradDiv.style.background = 'transparent';
        gradDiv.style.fontWeight = 'normal';
        drawTriangle();
    };
}

// Initial draw
drawTriangle();
displayStiffness();
})();