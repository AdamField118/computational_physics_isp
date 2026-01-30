// Exercise 3.x.14: Nonconforming Elements
(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for ex_14_nonconforming');
        return;
    }

    container.innerHTML = `
    <div style="max-width: 1000px; margin: 0 auto;">
        <h3 style="color: #00ff41;">Conforming vs Nonconforming Finite Elements</h3>
        <p style="color: #e4e4e4;">Compare standard (conforming) and Crouzeix-Raviart (nonconforming) P₁ elements</p>
        
        <div style="display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap;">
            <button id="btn-conforming" class="basis-btn">Conforming (vertex DOFs)</button>
            <button id="btn-nonconforming" class="basis-btn">Nonconforming (edge DOFs)</button>
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
            <div style="flex: 1; min-width: 400px;">
                <canvas id="canvas-mesh" width="450" height="450" style="border: 1px solid #ccc; background: white; display: block; width: 100%;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">Mesh with DOF locations</p>
            </div>
            <div style="flex: 1; min-width: 400px;">
                <canvas id="canvas-func" width="450" height="450" style="border: 1px solid #ccc; background: white; display: block; width: 100%;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">Sample function (click vertex/edge)</p>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px;">
            <div id="info-box" style="color: #000;"></div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;">
            <h4 style="color: #000;">Key Difference</h4>
            <p id="difference-text" style="color: #000;"></p>
        </div>
    </div>
`;

const canvasMesh = container.querySelector('#canvas-mesh');
const ctxMesh = canvasMesh.getContext('2d');
const canvasFunc = container.querySelector('#canvas-func');
const ctxFunc = canvasFunc.getContext('2d');

// Create a simple triangulation of a square
const vertices = [
    [0, 0], [1, 0], [2, 0],
    [0, 1], [1, 1], [2, 1],
    [0, 2], [1, 2], [2, 2]
];

const triangles = [
    [0, 1, 3], [1, 4, 3],  // Bottom-left square
    [1, 2, 4], [2, 5, 4],  // Bottom-right square
    [3, 4, 6], [4, 7, 6],  // Top-left square
    [4, 5, 7], [5, 8, 7]   // Top-right square
];

// Edge midpoints for each triangle
function getEdgeMidpoints(tri) {
    const [i, j, k] = tri;
    const [v0, v1, v2] = [vertices[i], vertices[j], vertices[k]];
    return [
        [(v0[0] + v1[0])/2, (v0[1] + v1[1])/2],  // mid of edge 0-1
        [(v1[0] + v2[0])/2, (v1[1] + v2[1])/2],  // mid of edge 1-2
        [(v2[0] + v0[0])/2, (v2[1] + v0[1])/2]   // mid of edge 2-0
    ];
}

let currentMode = 'conforming';
let selectedDOF = -1;

function toCanvas(x, y, canvas) {
    const padding = 40;
    const scale = (canvas.width - 2*padding) / 2;
    return {
        x: padding + x * scale,
        y: canvas.height - padding - y * scale
    };
}

function drawMesh() {
    ctxMesh.clearRect(0, 0, canvasMesh.width, canvasMesh.height);
    
    // Draw triangles
    ctxMesh.strokeStyle = '#333';
    ctxMesh.lineWidth = 1.5;
    triangles.forEach(tri => {
        ctxMesh.beginPath();
        tri.forEach((vi, i) => {
            const p = toCanvas(vertices[vi][0], vertices[vi][1], canvasMesh);
            if (i === 0) ctxMesh.moveTo(p.x, p.y);
            else ctxMesh.lineTo(p.x, p.y);
        });
        ctxMesh.closePath();
        ctxMesh.stroke();
    });
    
    if (currentMode === 'conforming') {
        // Draw vertex DOFs
        ctxMesh.fillStyle = 'blue';
        ctxMesh.font = 'bold 12px Arial';
        vertices.forEach((v, i) => {
            const p = toCanvas(v[0], v[1], canvasMesh);
            ctxMesh.beginPath();
            ctxMesh.arc(p.x, p.y, 6, 0, 2*Math.PI);
            ctxMesh.fill();
            
            if (i === selectedDOF) {
                ctxMesh.strokeStyle = 'red';
                ctxMesh.lineWidth = 3;
                ctxMesh.beginPath();
                ctxMesh.arc(p.x, p.y, 10, 0, 2*Math.PI);
                ctxMesh.stroke();
            }
            
            ctxMesh.fillStyle = 'black';
            ctxMesh.fillText(`${i}`, p.x - 15, p.y - 10);
        });
    } else {
        // Draw edge midpoint DOFs
        ctxMesh.fillStyle = 'green';
        ctxMesh.font = 'bold 12px Arial';
        let dofIndex = 0;
        triangles.forEach((tri, tIdx) => {
            const mids = getEdgeMidpoints(tri);
            mids.forEach((m, mIdx) => {
                const p = toCanvas(m[0], m[1], canvasMesh);
                ctxMesh.beginPath();
                ctxMesh.arc(p.x, p.y, 5, 0, 2*Math.PI);
                ctxMesh.fill();
                
                if (dofIndex === selectedDOF) {
                    ctxMesh.strokeStyle = 'red';
                    ctxMesh.lineWidth = 3;
                    ctxMesh.beginPath();
                    ctxMesh.arc(p.x, p.y, 10, 0, 2*Math.PI);
                    ctxMesh.stroke();
                }
                
                ctxMesh.fillStyle = 'black';
                ctxMesh.fillText(`e${dofIndex}`, p.x + 8, p.y - 8);
                dofIndex++;
            });
        });
    }
    
    // Labels
    ctxMesh.fillStyle = 'black';
    ctxMesh.font = '14px Arial';
    const labelP = toCanvas(1, -0.3, canvasMesh);
    ctxMesh.fillText(currentMode === 'conforming' ? 
        'DOFs at vertices (blue circles)' : 
        'DOFs at edge midpoints (green circles)', 
        labelP.x - 100, labelP.y);
}

function drawFunction() {
    ctxFunc.clearRect(0, 0, canvasFunc.width, canvasFunc.height);
    
    if (selectedDOF === -1) {
        ctxFunc.fillStyle = '#666';
        ctxFunc.font = '16px Arial';
        ctxFunc.textAlign = 'center';
        ctxFunc.fillText('Click a DOF on the left to see its basis function', 
                        canvasFunc.width/2, canvasFunc.height/2);
        return;
    }
    
    const resolution = 50;
    
    // Draw basis function
    triangles.forEach((tri, tIdx) => {
        const [i, j, k] = tri;
        const [v0, v1, v2] = [vertices[i], vertices[j], vertices[k]];
        
        // Compute basis function on this triangle
        for (let u = 0; u < resolution; u++) {
            for (let v = 0; v < resolution; v++) {
                const s = u / resolution;
                const t = v / resolution;
                if (s + t > 1) continue;
                
                // Barycentric interpolation
                const x = (1-s-t)*v0[0] + s*v1[0] + t*v2[0];
                const y = (1-s-t)*v0[1] + s*v1[1] + t*v2[1];
                
                let value = 0;
                
                if (currentMode === 'conforming') {
                    // Standard P1: value at vertices
                    if (i === selectedDOF) value = 1-s-t;
                    else if (j === selectedDOF) value = s;
                    else if (k === selectedDOF) value = t;
                } else {
                    // Nonconforming: value at edge midpoints
                    const mids = getEdgeMidpoints(tri);
                    const edgeIdx = tIdx * 3;
                    
                    // Basis functions for edge midpoints
                    // These are NOT continuous at vertices!
                    if (edgeIdx === selectedDOF) {
                        // Edge 0-1: linear, 1 at midpoint, 0 at opposite vertex
                        value = 1 - 2*t;
                    } else if (edgeIdx + 1 === selectedDOF) {
                        // Edge 1-2
                        value = 1 - 2*(1-s-t);
                    } else if (edgeIdx + 2 === selectedDOF) {
                        // Edge 2-0
                        value = 1 - 2*s;
                    }
                }
                
                value = Math.max(0, Math.min(1, value));
                
                // Color map
                const r = Math.floor(255 * value);
                const b = Math.floor(255 * (1-value));
                ctxFunc.fillStyle = `rgb(${r}, 100, ${b})`;
                
                const p = toCanvas(x, y, canvasFunc);
                const dx = (canvasFunc.width - 80) / 2 / resolution;
                ctxFunc.fillRect(p.x, p.y, dx, dx);
            }
        }
    });
    
    // Draw mesh overlay
    ctxFunc.strokeStyle = 'rgba(0,0,0,0.3)';
    ctxFunc.lineWidth = 1;
    triangles.forEach(tri => {
        ctxFunc.beginPath();
        tri.forEach((vi, idx) => {
            const p = toCanvas(vertices[vi][0], vertices[vi][1], canvasFunc);
            if (idx === 0) ctxFunc.moveTo(p.x, p.y);
            else ctxFunc.lineTo(p.x, p.y);
        });
        ctxFunc.closePath();
        ctxFunc.stroke();
    });
}

function updateInfo() {
    const infoBox = container.querySelector('#info-box');
    const diffText = container.querySelector('#difference-text');
    
    if (currentMode === 'conforming') {
        infoBox.innerHTML = `
            <h4>Standard Conforming P₁ Elements</h4>
            <p><strong>DOFs:</strong> Function values at vertices (9 total)</p>
            <p><strong>Continuity:</strong> Functions are continuous everywhere (C⁰)</p>
            <p><strong>Space:</strong> V_h = {v ∈ C⁰(Ω) : v|_T ∈ P₁ for each triangle T}</p>
            <p><strong>Basis function:</strong> Each vertex has a "hat" function that equals 1 at that vertex and 0 at all other vertices</p>
        `;
        
        diffText.innerHTML = `
            In conforming elements, functions are <strong>continuous everywhere</strong>. 
            Basis functions associated with shared vertices automatically ensure continuity across element boundaries.
        `;
    } else {
        infoBox.innerHTML = `
            <h4>Crouzeix-Raviart Nonconforming P₁ Elements</h4>
            <p><strong>DOFs:</strong> Function values at edge midpoints (24 total, 3 per triangle)</p>
            <p><strong>Continuity:</strong> Continuous at edge midpoints only (may jump at vertices!)</p>
            <p><strong>Space:</strong> V_h^NC = {v : v|_T ∈ P₁, v continuous at edge midpoints}</p>
            <p><strong>Basis function:</strong> Each edge midpoint has a function that equals 1 at that midpoint and 0 at other midpoints (but nonzero at vertices!)</p>
        `;
        
        diffText.innerHTML = `
            Nonconforming elements <strong>relax the continuity requirement</strong>. Functions are only required to match at edge midpoints, 
            allowing discontinuities at vertices. This provides more flexibility but requires careful analysis to ensure convergence.
            Note: Click different edge midpoints to see how functions can have different values at the same vertex across triangles!
        `;
    }
}

// Handle canvas clicks
canvasMesh.onclick = (e) => {
    const rect = canvasMesh.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    if (currentMode === 'conforming') {
        // Check vertices
        let minDist = Infinity;
        let closest = -1;
        vertices.forEach((v, i) => {
            const p = toCanvas(v[0], v[1], canvasMesh);
            const dist = Math.sqrt((p.x - mouseX)**2 + (p.y - mouseY)**2);
            if (dist < 15 && dist < minDist) {
                minDist = dist;
                closest = i;
            }
        });
        if (closest !== -1) {
            selectedDOF = closest;
            drawMesh();
            drawFunction();
        }
    } else {
        // Check edge midpoints
        let minDist = Infinity;
        let closest = -1;
        let dofIndex = 0;
        triangles.forEach((tri, tIdx) => {
            const mids = getEdgeMidpoints(tri);
            mids.forEach((m, mIdx) => {
                const p = toCanvas(m[0], m[1], canvasMesh);
                const dist = Math.sqrt((p.x - mouseX)**2 + (p.y - mouseY)**2);
                if (dist < 15 && dist < minDist) {
                    minDist = dist;
                    closest = dofIndex;
                }
                dofIndex++;
            });
        });
        if (closest !== -1) {
            selectedDOF = closest;
            drawMesh();
            drawFunction();
        }
    }
};

// Button handlers
container.querySelector('#btn-conforming').onclick = () => {
    currentMode = 'conforming';
    selectedDOF = -1;
    drawMesh();
    drawFunction();
    updateInfo();
};

container.querySelector('#btn-nonconforming').onclick = () => {
    currentMode = 'nonconforming';
    selectedDOF = -1;
    drawMesh();
    drawFunction();
    updateInfo();
};

// Initial draw
drawMesh();
drawFunction();
updateInfo();
})();