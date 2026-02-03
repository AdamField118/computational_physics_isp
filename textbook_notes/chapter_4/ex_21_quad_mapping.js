// Exercise 4.x.21: Bilinear Quadrilateral Mapping
(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for ex_21_quad_mapping');
        return;
    }

    container.innerHTML = `
    <div style="max-width: 1200px; margin: 0 auto;">
        <h3 style="color: #00ff41;">Bilinear Quadrilateral Mapping F: K̂ → K</h3>
        <p style="color: #e4e4e4;">Drag vertices of physical quad to see the bilinear transformation</p>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
            <div>
                <canvas id="canvas-ref" width="350" height="350" 
                        style="border: 1px solid #ccc; background: white;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">
                    Reference Square K̂ = [-1,1]²
                </p>
            </div>
            <div>
                <canvas id="canvas-phys" width="350" height="350" 
                        style="border: 1px solid #ccc; background: white; cursor: pointer;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">
                    Physical Quad K (drag vertices!)
                </p>
            </div>
            <div style="flex: 1; min-width: 300px;">
                <div style="background: #e3f2fd; padding: 20px; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="color: #000;">Bilinear Shape Functions</h4>
                    <div style="font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.8; color: #000;">
                        N₁(ξ,η) = (1-ξ)(1-η)/4<br>
                        N₂(ξ,η) = (1+ξ)(1-η)/4<br>
                        N₃(ξ,η) = (1+ξ)(1+η)/4<br>
                        N₄(ξ,η) = (1-ξ)(1+η)/4
                    </div>
                </div>
                
                <div style="background: #fff3cd; padding: 20px; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="color: #000;">Mapping Formula</h4>
                    <div style="font-family: 'Courier New', monospace; font-size: 13px; color: #000;">
                        F(ξ,η) = Σᵢ Nᵢ(ξ,η) · vᵢ
                    </div>
                    <div id="jacobian-info" style="margin-top: 10px; color: #000;">
                    </div>
                </div>
                
                <div id="convexity-warning" style="padding: 15px; border-radius: 5px;">
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 5px;">
            <h4 style="color: #000;">Key Properties</h4>
            <ul style="color: #000; line-height: 1.8;">
                <li><strong>NOT affine:</strong> Contains ξη term (bilinear!)</li>
                <li><strong>Jacobian varies:</strong> det(J) not constant across element</li>
                <li><strong>Convexity required:</strong> det(J) > 0 everywhere ensures invertibility</li>
                <li><strong>Quadrature needed:</strong> Must integrate numerically (unlike triangles)</li>
                <li><strong>Maps edges to edges:</strong> Preserves quadrilateral structure</li>
            </ul>
        </div>
        
        <div style="margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;">
            <button id="btn-square" class="basis-btn">Square</button>
            <button id="btn-parallelogram" class="basis-btn">Parallelogram</button>
            <button id="btn-trapezoid" class="basis-btn">Trapezoid</button>
            <button id="btn-general" class="basis-btn">General Convex</button>
            <button id="btn-nonconvex" class="basis-btn">Non-convex (bad!)</button>
        </div>
    </div>
`;

const canvasRef = container.querySelector('#canvas-ref');
const ctxRef = canvasRef.getContext('2d');
const canvasPhys = container.querySelector('#canvas-phys');
const ctxPhys = canvasPhys.getContext('2d');

// Reference square vertices (in order: bottom-left, bottom-right, top-right, top-left)
const refVertices = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1]
];

// Physical quadrilateral vertices (draggable)
let physVertices = [
    [100, 300],
    [300, 300],
    [280, 100],
    [120, 120]
];

let dragging = -1;

function bilinearShapes(xi, eta) {
    return [
        (1 - xi) * (1 - eta) / 4,
        (1 + xi) * (1 - eta) / 4,
        (1 + xi) * (1 + eta) / 4,
        (1 - xi) * (1 + eta) / 4
    ];
}

function bilinearMap(xi, eta, vertices) {
    const N = bilinearShapes(xi, eta);
    let x = 0, y = 0;
    for (let i = 0; i < 4; i++) {
        x += N[i] * vertices[i][0];
        y += N[i] * vertices[i][1];
    }
    return [x, y];
}

function toCanvasRef(xi, eta) {
    const padding = 50;
    const scale = (canvasRef.width - 2 * padding) / 2;
    return {
        x: padding + (xi + 1) * scale,
        y: canvasRef.height - padding - (eta + 1) * scale
    };
}

function drawReferenceSquare() {
    ctxRef.clearRect(0, 0, canvasRef.width, canvasRef.height);
    
    const resolution = 10;
    
    // Draw grid
    ctxRef.strokeStyle = '#e0e0e0';
    ctxRef.lineWidth = 1;
    
    for (let i = 0; i <= resolution; i++) {
        const xi = -1 + 2 * i / resolution;
        const eta1 = toCanvasRef(xi, -1);
        const eta2 = toCanvasRef(xi, 1);
        
        ctxRef.beginPath();
        ctxRef.moveTo(eta1.x, eta1.y);
        ctxRef.lineTo(eta2.x, eta2.y);
        ctxRef.stroke();
    }
    
    for (let j = 0; j <= resolution; j++) {
        const eta = -1 + 2 * j / resolution;
        const xi1 = toCanvasRef(-1, eta);
        const xi2 = toCanvasRef(1, eta);
        
        ctxRef.beginPath();
        ctxRef.moveTo(xi1.x, xi1.y);
        ctxRef.lineTo(xi2.x, xi2.y);
        ctxRef.stroke();
    }
    
    // Draw boundary
    ctxRef.strokeStyle = '#2196f3';
    ctxRef.lineWidth = 3;
    ctxRef.beginPath();
    refVertices.forEach((v, i) => {
        const p = toCanvasRef(v[0], v[1]);
        if (i === 0) ctxRef.moveTo(p.x, p.y);
        else ctxRef.lineTo(p.x, p.y);
    });
    ctxRef.closePath();
    ctxRef.stroke();
    
    // Draw vertices
    refVertices.forEach((v, i) => {
        const p = toCanvasRef(v[0], v[1]);
        
        ctxRef.fillStyle = '#2196f3';
        ctxRef.beginPath();
        ctxRef.arc(p.x, p.y, 6, 0, 2*Math.PI);
        ctxRef.fill();
        
        ctxRef.strokeStyle = 'black';
        ctxRef.lineWidth = 2;
        ctxRef.stroke();
        
        // Label
        ctxRef.fillStyle = 'black';
        ctxRef.font = 'bold 12px Arial';
        ctxRef.textAlign = 'center';
        const labels = ['(-1,-1)', '(1,-1)', '(1,1)', '(-1,1)'];
        const offsetY = i < 2 ? 20 : -10;
        ctxRef.fillText(labels[i], p.x, p.y + offsetY);
    });
    
    // Axes labels
    ctxRef.fillStyle = 'black';
    ctxRef.font = '14px Arial';
    const center = toCanvasRef(0, -1.3);
    ctxRef.fillText('ξ', center.x, center.y);
    const yLabel = toCanvasRef(-1.3, 0);
    ctxRef.fillText('η', yLabel.x, yLabel.y);
}

function computeJacobian(xi, eta, vertices) {
    // Derivatives of shape functions
    const dN_dxi = [
        -(1 - eta) / 4,
        (1 - eta) / 4,
        (1 + eta) / 4,
        -(1 + eta) / 4
    ];
    
    const dN_deta = [
        -(1 - xi) / 4,
        -(1 + xi) / 4,
        (1 + xi) / 4,
        (1 - xi) / 4
    ];
    
    let dx_dxi = 0, dx_deta = 0, dy_dxi = 0, dy_deta = 0;
    
    for (let i = 0; i < 4; i++) {
        dx_dxi += dN_dxi[i] * vertices[i][0];
        dx_deta += dN_deta[i] * vertices[i][0];
        dy_dxi += dN_dxi[i] * vertices[i][1];
        dy_deta += dN_deta[i] * vertices[i][1];
    }
    
    const det = dx_dxi * dy_deta - dx_deta * dy_dxi;
    return { det, dx_dxi, dx_deta, dy_dxi, dy_deta };
}

function checkConvexity(vertices) {
    // Sample det(J) at several points
    const samples = 5;
    let minDet = Infinity;
    let maxDet = -Infinity;
    
    for (let i = 0; i <= samples; i++) {
        for (let j = 0; j <= samples; j++) {
            const xi = -1 + 2 * i / samples;
            const eta = -1 + 2 * j / samples;
            const J = computeJacobian(xi, eta, vertices);
            minDet = Math.min(minDet, J.det);
            maxDet = Math.max(maxDet, J.det);
        }
    }
    
    return { minDet, maxDet, isConvex: minDet > 0 };
}

function drawPhysicalQuad() {
    ctxPhys.clearRect(0, 0, canvasPhys.width, canvasPhys.height);
    
    const resolution = 10;
    const convexity = checkConvexity(physVertices);
    
    // Draw mapped grid
    ctxPhys.strokeStyle = convexity.isConvex ? '#e0f2f1' : '#ffebee';
    ctxPhys.lineWidth = 1;
    
    // Constant xi lines
    for (let i = 0; i <= resolution; i++) {
        const xi = -1 + 2 * i / resolution;
        ctxPhys.beginPath();
        
        for (let j = 0; j <= resolution; j++) {
            const eta = -1 + 2 * j / resolution;
            const [x, y] = bilinearMap(xi, eta, physVertices);
            
            if (j === 0) ctxPhys.moveTo(x, y);
            else ctxPhys.lineTo(x, y);
        }
        ctxPhys.stroke();
    }
    
    // Constant eta lines
    for (let j = 0; j <= resolution; j++) {
        const eta = -1 + 2 * j / resolution;
        ctxPhys.beginPath();
        
        for (let i = 0; i <= resolution; i++) {
            const xi = -1 + 2 * i / resolution;
            const [x, y] = bilinearMap(xi, eta, physVertices);
            
            if (i === 0) ctxPhys.moveTo(x, y);
            else ctxPhys.lineTo(x, y);
        }
        ctxPhys.stroke();
    }
    
    // Draw boundary
    ctxPhys.strokeStyle = convexity.isConvex ? '#f44336' : '#ff5722';
    ctxPhys.lineWidth = convexity.isConvex ? 3 : 4;
    ctxPhys.beginPath();
    physVertices.forEach((v, i) => {
        if (i === 0) ctxPhys.moveTo(v[0], v[1]);
        else ctxPhys.lineTo(v[0], v[1]);
    });
    ctxPhys.closePath();
    ctxPhys.stroke();
    
    // Draw vertices
    physVertices.forEach((v, i) => {
        ctxPhys.fillStyle = dragging === i ? '#ff9800' : '#f44336';
        ctxPhys.beginPath();
        ctxPhys.arc(v[0], v[1], 8, 0, 2*Math.PI);
        ctxPhys.fill();
        
        ctxPhys.strokeStyle = 'black';
        ctxPhys.lineWidth = 2;
        ctxPhys.stroke();
        
        // Label
        ctxPhys.fillStyle = 'black';
        ctxPhys.font = 'bold 12px Arial';
        ctxPhys.textAlign = 'center';
        const offsetY = i < 2 ? 20 : -10;
        ctxPhys.fillText(`v${i+1}`, v[0], v[1] + offsetY);
    });
    
    updateInfo(convexity);
}

function updateInfo(convexity) {
    const J_center = computeJacobian(0, 0, physVertices);
    
    // Jacobian info
    const jacInfo = container.querySelector('#jacobian-info');
    jacInfo.innerHTML = `
        <div style="font-size: 12px; line-height: 1.6;">
            <strong>At center (ξ=0, η=0):</strong><br>
            det(J) = ${J_center.det.toFixed(2)}<br>
            <br>
            <strong>Over element:</strong><br>
            min det(J) = ${convexity.minDet.toFixed(2)}<br>
            max det(J) = ${convexity.maxDet.toFixed(2)}
        </div>
    `;
    
    // Convexity warning
    const warning = container.querySelector('#convexity-warning');
    if (convexity.isConvex) {
        warning.style.background = '#e8f5e9';
        warning.style.border = '2px solid #4caf50';
        warning.innerHTML = `
            <h4 style="color: #4caf50; margin: 0 0 10px 0;">✓ Valid Mapping</h4>
            <p style="color: #000; margin: 0;">det(J) > 0 everywhere → Mapping is invertible</p>
        `;
    } else {
        warning.style.background = '#ffebee';
        warning.style.border = '2px solid #f44336';
        warning.innerHTML = `
            <h4 style="color: #f44336; margin: 0 0 10px 0;">⚠️ Invalid Mapping!</h4>
            <p style="color: #000; margin: 0;">
                det(J) ≤ 0 detected → Element folds back on itself!<br>
                <strong>This will cause assembly errors.</strong> Make quad convex.
            </p>
        `;
    }
}

function dist(p1, p2) {
    return Math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2);
}

// Mouse handlers
canvasPhys.onmousedown = (e) => {
    const rect = canvasPhys.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    for (let i = 0; i < 4; i++) {
        if (dist([x, y], physVertices[i]) < 15) {
            dragging = i;
            break;
        }
    }
};

canvasPhys.onmousemove = (e) => {
    if (dragging >= 0) {
        const rect = canvasPhys.getBoundingClientRect();
        const x = Math.max(20, Math.min(canvasPhys.width - 20, e.clientX - rect.left));
        const y = Math.max(20, Math.min(canvasPhys.height - 20, e.clientY - rect.top));
        physVertices[dragging] = [x, y];
        drawPhysicalQuad();
    }
};

canvasPhys.onmouseup = () => {
    dragging = -1;
    drawPhysicalQuad();
};

// Preset buttons
container.querySelector('#btn-square').onclick = () => {
    physVertices = [
        [100, 300],
        [300, 300],
        [300, 100],
        [100, 100]
    ];
    drawPhysicalQuad();
};

container.querySelector('#btn-parallelogram').onclick = () => {
    physVertices = [
        [80, 300],
        [280, 300],
        [320, 100],
        [120, 100]
    ];
    drawPhysicalQuad();
};

container.querySelector('#btn-trapezoid').onclick = () => {
    physVertices = [
        [80, 300],
        [320, 300],
        [280, 100],
        [120, 100]
    ];
    drawPhysicalQuad();
};

container.querySelector('#btn-general').onclick = () => {
    physVertices = [
        [100, 300],
        [300, 280],
        [280, 100],
        [120, 120]
    ];
    drawPhysicalQuad();
};

container.querySelector('#btn-nonconvex').onclick = () => {
    physVertices = [
        [100, 300],
        [300, 300],
        [120, 120],
        [280, 100]
    ];
    drawPhysicalQuad();
};

// Initial draw
drawReferenceSquare();
drawPhysicalQuad();
})();
