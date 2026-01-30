// Exercise 3.x.19: Lagrange Elements and DOF Counting
(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for ex_19_lagrange');
        return;
    }

    container.innerHTML = `
    <div style="max-width: 1000px; margin: 0 auto;">
        <h3 style="color: #00ff41;">Lagrange Elements: Barycentric Lattice Points</h3>
        <p style="color: #e4e4e4;">Explore how polynomial degree determines the number of DOFs</p>
        
        <div style="margin: 20px 0;">
            <label for="degree-slider" style="color: #e4e4e4;">Polynomial Degree r: <strong id="degree-value">1</strong></label>
            <input type="range" id="degree-slider" min="1" max="5" value="1" 
                   style="width: 300px; margin-left: 10px;">
        </div>
        
        <div style="display: flex; gap: 30px; flex-wrap: wrap; justify-content: center;">
            <div>
                <canvas id="canvas-triangle" width="400" height="400" 
                        style="border: 1px solid #ccc; background: white;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">
                    <strong>Triangle (2D)</strong><br>
                    Nodes: <span id="tri-nodes">3</span>
                </p>
            </div>
            <div>
                <canvas id="canvas-tetrahedron" width="400" height="400" 
                        style="border: 1px solid #ccc; background: white;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">
                    <strong>Tetrahedron (3D)</strong><br>
                    Nodes: <span id="tet-nodes">4</span>
                </p>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px;">
            <h4 style="color: #000;">Node Count Formulas</h4>
            <div id="formulas" style="font-size: 16px; line-height: 2; color: #000;"></div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 5px;">
            <h4 style="color: #000;">Lattice Point Explanation</h4>
            <p id="explanation" style="color: #000;"></p>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #fff; border: 2px solid #00458b; border-radius: 5px;">
            <h4 style="color: #000;">📊 Comparison Table</h4>
            <table id="comparison-table" style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <thead style="background: #00458b; color: white;">
                    <tr>
                        <th style="padding: 8px; border: 1px solid #ddd; color: white;">Degree r</th>
                        <th style="padding: 8px; border: 1px solid #ddd; color: white;">Triangle Nodes</th>
                        <th style="padding: 8px; border: 1px solid #ddd; color: white;">Tetrahedron Nodes</th>
                        <th style="padding: 8px; border: 1px solid #ddd; color: white;">Location Description</th>
                    </tr>
                </thead>
                <tbody id="table-body"></tbody>
            </table>
        </div>
    </div>
`;

const canvasTri = container.querySelector('#canvas-triangle');
const ctxTri = canvasTri.getContext('2d');
const canvasTet = container.querySelector('#canvas-tetrahedron');
const ctxTet = canvasTet.getContext('2d');
const slider = container.querySelector('#degree-slider');
const degreeValue = container.querySelector('#degree-value');

function binomial(n, k) {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    
    let result = 1;
    for (let i = 0; i < k; i++) {
        result *= (n - i);
        result /= (i + 1);
    }
    return Math.round(result);
}

function countTriangleNodes(r) {
    // C(r+2, 2) = (r+1)(r+2)/2
    return binomial(r + 2, 2);
}

function countTetrahedronNodes(r) {
    // C(r+3, 3) = (r+1)(r+2)(r+3)/6
    return binomial(r + 3, 3);
}

function drawTriangle(r) {
    ctxTri.clearRect(0, 0, canvasTri.width, canvasTri.height);
    
    const padding = 50;
    const size = canvasTri.width - 2 * padding;
    
    // Draw triangle
    ctxTri.strokeStyle = 'black';
    ctxTri.lineWidth = 2;
    ctxTri.beginPath();
    ctxTri.moveTo(padding, canvasTri.height - padding);
    ctxTri.lineTo(padding + size, canvasTri.height - padding);
    ctxTri.lineTo(padding, canvasTri.height - padding - size);
    ctxTri.closePath();
    ctxTri.stroke();
    
    // Draw lattice points
    const points = [];
    for (let i = 0; i <= r; i++) {
        for (let j = 0; j <= r - i; j++) {
            const k = r - i - j;
            
            // Barycentric coordinates (i/r, j/r, k/r)
            // Cartesian: x = j/r, y = k/r
            const x = j / r;
            const y = k / r;
            
            const px = padding + x * size;
            const py = canvasTri.height - padding - y * size;
            points.push([px, py, i, j, k]);
        }
    }
    
    // Color code by type
    points.forEach(([px, py, i, j, k]) => {
        let color = 'blue';
        let radius = 6;
        
        // Vertices (one barycentric coord = r)
        if ((i === r) || (j === r) || (k === r)) {
            color = 'red';
            radius = 7;
        }
        // Edge points (one coord = 0, but not a vertex)
        else if ((i === 0 && j > 0 && k > 0) || 
                 (j === 0 && i > 0 && k > 0) || 
                 (k === 0 && i > 0 && j > 0)) {
            color = 'green';
            radius = 6;
        }
        // Interior points
        else {
            color = 'purple';
            radius = 5;
        }
        
        ctxTri.fillStyle = color;
        ctxTri.beginPath();
        ctxTri.arc(px, py, radius, 0, 2 * Math.PI);
        ctxTri.fill();
        
        // Subtle border for clarity
        ctxTri.strokeStyle = 'rgba(0,0,0,0.3)';
        ctxTri.lineWidth = 0.5;
        ctxTri.stroke();
        
        // Label for low degree (clearer placement)
        if (r <= 3) {
            ctxTri.fillStyle = 'black';
            ctxTri.font = 'bold 10px Arial';
            const label = `(${i},${j},${k})`;
            // Smart label placement
            let offsetX = 10, offsetY = -10;
            if (k === r) { offsetX = -30; offsetY = 10; }  // Top vertex
            if (j === r) { offsetX = 10; offsetY = 5; }     // Right vertex
            if (i === r) { offsetX = -30; offsetY = -10; }  // Left vertex
            ctxTri.fillText(label, px + offsetX, py + offsetY);
        }
    });
    
    // Legend (clearer)
    ctxTri.font = '12px Arial';
    ctxTri.fillStyle = 'red';
    ctxTri.beginPath();
    ctxTri.arc(padding + 20, 30, 6, 0, 2*Math.PI);
    ctxTri.fill();
    ctxTri.fillStyle = 'black';
    ctxTri.fillText('Vertices', padding + 30, 35);
    
    ctxTri.fillStyle = 'green';
    ctxTri.beginPath();
    ctxTri.arc(padding + 110, 30, 6, 0, 2*Math.PI);
    ctxTri.fill();
    ctxTri.fillStyle = 'black';
    ctxTri.fillText('Edge nodes', padding + 120, 35);
    
    if (r >= 3) {
        ctxTri.fillStyle = 'purple';
        ctxTri.beginPath();
        ctxTri.arc(padding + 210, 30, 5, 0, 2*Math.PI);
        ctxTri.fill();
        ctxTri.fillStyle = 'black';
        ctxTri.fillText('Interior', padding + 220, 35);
    }
    
    // Update count
    container.querySelector('#tri-nodes').textContent = points.length;
}

function drawTetrahedron(r) {
    ctxTet.clearRect(0, 0, canvasTet.width, canvasTet.height);
    
    const w = canvasTet.width;
    const h = canvasTet.height;
    const scale = 180;  // Larger scale
    
    // Better 3D projection (more isometric)
    function project(x, y, z) {
        // Isometric-style projection
        const px = w/2 + (x - z) * scale * 0.866 + 20;
        const py = h/2 - y * scale * 0.9 - (x + z) * scale * 0.4 + 30;
        return [px, py, x + y + z];  // Return depth for sorting
    }
    
    // Tetrahedron vertices in better positions
    const v = [
        [0, 0, 0],           // vertex 0
        [1, 0, 0],           // vertex 1
        [0.5, 0.866, 0],     // vertex 2
        [0.5, 0.289, 0.816]  // vertex 3 (top)
    ];
    
    // Draw edges (lighter, thinner)
    ctxTet.strokeStyle = 'rgba(0,0,0,0.25)';
    ctxTet.lineWidth = 1.0;
    const edges = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]];
    edges.forEach(([i, j]) => {
        const [x1, y1] = project(...v[i]);
        const [x2, y2] = project(...v[j]);
        ctxTet.beginPath();
        ctxTet.moveTo(x1, y1);
        ctxTet.lineTo(x2, y2);
        ctxTet.stroke();
    });
    
    // Collect all lattice points with depth
    const pointsWithDepth = [];
    for (let i = 0; i <= r; i++) {
        for (let j = 0; j <= r - i; j++) {
            for (let k = 0; k <= r - i - j; k++) {
                const l = r - i - j - k;
                
                // Barycentric to Cartesian
                const x = (j * v[1][0] + k * v[2][0] + l * v[3][0]) / r;
                const y = (j * v[1][1] + k * v[2][1] + l * v[3][1]) / r;
                const z = (j * v[1][2] + k * v[2][2] + l * v[3][2]) / r;
                
                const [px, py, depth] = project(x, y, z);
                
                // Determine type
                let type = 'interior';
                let radius = 4;
                let color = 'purple';
                
                // Vertices (one barycentric coord = r)
                if ((i === r) || (j === r) || (k === r) || (l === r)) {
                    type = 'vertex';
                    color = 'red';
                    radius = 6;
                }
                // Edge points (two coords = 0)
                else if ((i === 0 && j === 0) || (i === 0 && k === 0) || 
                         (i === 0 && l === 0) || (j === 0 && k === 0) ||
                         (j === 0 && l === 0) || (k === 0 && l === 0)) {
                    type = 'edge';
                    color = 'green';
                    radius = 5;
                }
                // Face points (one coord = 0, but not edge)
                else if ((i === 0) || (j === 0) || (k === 0) || (l === 0)) {
                    type = 'face';
                    color = 'orange';
                    radius = 4;
                }
                
                pointsWithDepth.push({ px, py, depth, color, radius, type });
            }
        }
    }
    
    // Sort by depth (back to front)
    pointsWithDepth.sort((a, b) => a.depth - b.depth);
    
    // Draw points back to front
    pointsWithDepth.forEach(({ px, py, color, radius }) => {
        ctxTet.fillStyle = color;
        ctxTet.beginPath();
        ctxTet.arc(px, py, radius, 0, 2 * Math.PI);
        ctxTet.fill();
        
        // Add subtle border for clarity
        ctxTet.strokeStyle = 'rgba(0,0,0,0.3)';
        ctxTet.lineWidth = 0.5;
        ctxTet.stroke();
    });
    
    // Legend
    ctxTet.font = '12px Arial';
    const legendX = 20;
    let legendY = 30;
    
    const legend = [
        ['red', 'Vertices', 6],
        ['green', 'Edge nodes', 5],
        ['orange', 'Face nodes', 4],
        ['purple', 'Interior', 4]
    ];
    
    legend.forEach(([color, label, radius]) => {
        if (color === 'purple' && r < 4) return;
        if (color === 'orange' && r < 3) return;
        
        ctxTet.fillStyle = color;
        ctxTet.beginPath();
        ctxTet.arc(legendX + 5, legendY, radius, 0, 2*Math.PI);
        ctxTet.fill();
        ctxTet.fillStyle = 'black';
        ctxTet.fillText(label, legendX + 15, legendY + 4);
        legendY += 20;
    });
    
    container.querySelector('#tet-nodes').textContent = pointsWithDepth.length;
}

function updateFormulas(r) {
    const formulasDiv = container.querySelector('#formulas');
    const triNodes = countTriangleNodes(r);
    const tetNodes = countTetrahedronNodes(r);
    
    formulasDiv.innerHTML = `
        <p><strong>Triangle (d=2):</strong> N(r) = C(r+2, 2) = (r+1)(r+2)/2 = 
           <strong>${triNodes}</strong> nodes</p>
        <p><strong>Tetrahedron (d=3):</strong> N(r) = C(r+3, 3) = (r+1)(r+2)(r+3)/6 = 
           <strong>${tetNodes}</strong> nodes</p>
        <p><strong>General d-simplex:</strong> N(r) = C(r+d, d)</p>
    `;
}

function updateExplanation(r) {
    const explanationDiv = container.querySelector('#explanation');
    
    let desc = `For degree r=${r} Lagrange elements, nodes are placed at barycentric lattice points (ℓ/r, m/r, n/r, ...) 
                where ℓ + m + n + ... = r and all coordinates are non-negative integers.`;
    
    if (r === 1) {
        desc += `<br><br>For r=1 (linear elements), nodes are only at vertices.`;
    } else if (r === 2) {
        desc += `<br><br>For r=2 (quadratic elements), we get vertices plus edge midpoints.`;
    } else if (r === 3) {
        desc += `<br><br>For r=3 (cubic elements), we get vertices, two points on each edge, and face/interior points.`;
    } else {
        desc += `<br><br>For r≥4, the pattern continues with more nodes on edges, faces, and interior.`;
    }
    
    explanationDiv.innerHTML = desc;
}

function updateTable() {
    const tableBody = container.querySelector('#table-body');
    tableBody.innerHTML = '';
    
    const descriptions = [
        'Vertices only',
        'Vertices + edge midpoints',
        'Vertices + 2 edge nodes + face centers',
        'Vertices + 3 edge nodes + face nodes + volume center',
        'Vertices + 4 edge nodes + face nodes + volume nodes'
    ];
    
    for (let degree = 1; degree <= 5; degree++) {
        const row = document.createElement('tr');
        row.style.background = degree === parseInt(slider.value) ? '#ffffcc' : 'white';
        
        row.innerHTML = `
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center; color: #000;"><strong>${degree}</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center; color: #000;">${countTriangleNodes(degree)}</td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center; color: #000;">${countTetrahedronNodes(degree)}</td>
            <td style="padding: 8px; border: 1px solid #ddd; color: #000;">${descriptions[degree-1]}</td>
        `;
        
        tableBody.appendChild(row);
    }
}

function update() {
    const r = parseInt(slider.value);
    degreeValue.textContent = r;
    drawTriangle(r);
    drawTetrahedron(r);
    updateFormulas(r);
    updateExplanation(r);
    updateTable();
}

slider.oninput = update;

// Initial draw
update();
})();