// Exercise 4.x.5: Homogeneity and Scaling Arguments
(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for ex_5_homogeneity');
        return;
    }

    container.innerHTML = `
    <div style="max-width: 1000px; margin: 0 auto;">
        <h3 style="color: #00ff41;">Homogeneity: Scaling from Reference to Physical Elements</h3>
        <p style="color: #e4e4e4;">See how norms transform under affine mappings</p>
        
        <div style="margin: 20px 0;">
            <label for="scale-slider" style="color: #e4e4e4;">
                Scale Factor h: <strong id="scale-value">1.0</strong>
            </label>
            <input type="range" id="scale-slider" min="20" max="300" value="100" 
                   style="width: 400px; margin-left: 10px;">
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
            <div>
                <canvas id="canvas-scaling" width="500" height="500" 
                        style="border: 1px solid #ccc; background: white;"></canvas>
                <p style="text-align: center; margin-top: 10px; color: #e4e4e4;">
                    Reference (blue) vs Physical (red) Elements
                </p>
            </div>
            <div style="flex: 1; min-width: 350px;">
                <div style="background: #e3f2fd; padding: 20px; border-radius: 5px; margin-bottom: 15px; border: 2px solid #2196f3;">
                    <h4 style="color: #000;">Reference Element K̂</h4>
                    <div id="ref-metrics"></div>
                </div>
                
                <div style="background: #ffebee; padding: 20px; border-radius: 5px; margin-bottom: 15px; border: 2px solid #f44336;">
                    <h4 style="color: #000;">Physical Element K</h4>
                    <div id="phys-metrics"></div>
                </div>
                
                <div style="background: #fff3cd; padding: 20px; border-radius: 5px;">
                    <h4 style="color: #000;">Scaling Relations</h4>
                    <div id="scaling-formulas"></div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 20px; background: #f0f0f0; border-radius: 5px;">
            <h4 style="color: #000;">Key Scaling Formulas</h4>
            <div style="background: white; padding: 15px; border-radius: 5px; margin-top: 10px; font-family: 'Courier New', monospace; color: #000;">
                <div style="margin-bottom: 10px;"><strong>For affine map F<sub>K</sub>(x̂) = B<sub>K</sub>x̂ + b<sub>K</sub> with ||B<sub>K</sub>|| ~ h:</strong></div>
                <div style="padding-left: 20px; line-height: 2;">
                    • Jacobian: |J<sub>K</sub>| = det(B<sub>K</sub>) ~ h<sup>d</sup> (volume scaling)<br>
                    • L² norm: ||v||<sub>L²(K)</sub> ~ h<sup>d/2</sup> ||v̂||<sub>L²(K̂)</sub><br>
                    • H¹ seminorm: |v|<sub>H¹(K)</sub> ~ h<sup>(d/2)-1</sup> |v̂|<sub>H¹(K̂)</sub><br>
                    • H<sup>m</sup> seminorm: |v|<sub>H<sup>m</sup>(K)</sub> ~ h<sup>(d/2)-m</sup> |v̂|<sub>H<sup>m</sup>(K̂)</sub>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 5px;">
            <h4 style="color: #000;">Why This Matters</h4>
            <ul style="color: #000; line-height: 1.8;">
                <li><strong>Approximation theory:</strong> Interpolation error ||u - Π<sub>h</sub>u||<sub>H¹</sub> ≤ Ch||u||<sub>H²</sub></li>
                <li><strong>Inverse inequalities:</strong> |v|<sub>H<sup>m</sup></sub> ≤ Ch<sup>s-m</sup>|v|<sub>H<sup>s</sup></sub> (note h<sup>negative</sup>!)</li>
                <li><strong>Assembly:</strong> Local stiffness includes |J<sub>K</sub>| factor</li>
                <li><strong>Error estimates:</strong> All convergence rates come from these scaling laws</li>
            </ul>
        </div>
    </div>
`;

const canvas = container.querySelector('#canvas-scaling');
const ctx = canvas.getContext('2d');
const slider = container.querySelector('#scale-slider');
const scaleValue = container.querySelector('#scale-value');

// Reference triangle (unit size)
const refTriangle = [
    [0, 0],
    [1, 0],
    [0, 1]
];

function affineMap(point, scale) {
    // F_K(x̂) = scale * x̂ + translation
    return [
        point[0] * scale + 100,
        point[1] * scale + 400
    ];
}

function toCanvas(x, y, scale, isRef) {
    if (isRef) {
        // Reference element positioned on left side (fixed 100px size)
        const refScale = 100;
        return [
            80 + x * refScale,
            400 - y * refScale
        ];
    } else {
        // Physical element positioned on right side
        return [
            320 + x * scale,
            400 - y * scale
        ];
    }
}

function triangleArea(vertices) {
    const [v1, v2, v3] = vertices;
    return 0.5 * Math.abs(
        (v2[0] - v1[0]) * (v3[1] - v1[1]) - 
        (v3[0] - v1[0]) * (v2[1] - v1[1])
    );
}

function edgeLength(v1, v2) {
    return Math.sqrt((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2);
}

function drawTriangle(vertices, color, label, ctx) {
    // Fill
    ctx.fillStyle = color + '20';
    ctx.beginPath();
    ctx.moveTo(vertices[0][0], vertices[0][1]);
    ctx.lineTo(vertices[1][0], vertices[1][1]);
    ctx.lineTo(vertices[2][0], vertices[2][1]);
    ctx.closePath();
    ctx.fill();
    
    // Outline
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Label at centroid
    const cx = (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3;
    const cy = (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3;
    ctx.fillStyle = color;
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(label, cx, cy);
}

function draw() {
    const scale = parseFloat(slider.value);
    const h = scale / 100;
    scaleValue.textContent = h.toFixed(2);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw reference triangle (always same size)
    const refVerts = refTriangle.map(v => toCanvas(v[0], v[1], 100, true));
    drawTriangle(refVerts, '#2196f3', 'K̂', ctx);
    
    // Draw physical triangle (scaled)
    const physVerts = refTriangle.map(v => toCanvas(v[0], v[1], scale, false));
    drawTriangle(physVerts, '#f44336', 'K', ctx);
    
    // Draw scale indicator
    ctx.strokeStyle = '#ff9800';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    // Horizontal scale line below physical element
    const y = 450;
    const startX = 320; // Match physical element offset
    ctx.beginPath();
    ctx.moveTo(startX, y);
    ctx.lineTo(startX + scale, y);
    ctx.stroke();
    
    // Arrows
    ctx.setLineDash([]);
    ctx.fillStyle = '#ff9800';
    // Start arrow
    ctx.beginPath();
    ctx.moveTo(startX, y);
    ctx.lineTo(startX + 10, y-5);
    ctx.lineTo(startX + 10, y+5);
    ctx.fill();
    // End arrow
    ctx.beginPath();
    ctx.moveTo(startX + scale, y);
    ctx.lineTo(startX + scale - 10, y-5);
    ctx.lineTo(startX + scale - 10, y+5);
    ctx.fill();
    
    // Label
    ctx.fillStyle = '#ff9800';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`h = ${h.toFixed(2)}`, startX + scale/2, y + 20);
    
    updateMetrics(h);
}

function updateMetrics(h) {
    const d = 2; // dimension
    
    // Reference element
    const refDiv = container.querySelector('#ref-metrics');
    refDiv.innerHTML = `
        <table style="width: 100%; line-height: 1.8; color: #000;">
            <tr>
                <td><strong>Area |K̂|:</strong></td>
                <td style="text-align: right;">0.5</td>
            </tr>
            <tr>
                <td><strong>Diameter h<sub>K̂</sub>:</strong></td>
                <td style="text-align: right;">√2 ≈ 1.41</td>
            </tr>
            <tr>
                <td><strong>Sample ||v̂||<sub>L²</sub>:</strong></td>
                <td style="text-align: right;">1.0</td>
            </tr>
            <tr>
                <td><strong>Sample |v̂|<sub>H¹</sub>:</strong></td>
                <td style="text-align: right;">1.0</td>
            </tr>
        </table>
    `;
    
    // Physical element
    const area = 0.5 * h * h;
    const diameter = Math.sqrt(2) * h;
    const l2_norm = Math.sqrt(h * h) * 1.0; // h^{d/2} factor
    const h1_seminorm = Math.sqrt(h * h) / h * 1.0; // h^{d/2 - 1} factor
    
    const physDiv = container.querySelector('#phys-metrics');
    physDiv.innerHTML = `
        <table style="width: 100%; line-height: 1.8; color: #000;">
            <tr>
                <td><strong>Area |K|:</strong></td>
                <td style="text-align: right;">${area.toFixed(3)}</td>
            </tr>
            <tr>
                <td><strong>Diameter h<sub>K</sub>:</strong></td>
                <td style="text-align: right;">${diameter.toFixed(3)}</td>
            </tr>
            <tr>
                <td><strong>Sample ||v||<sub>L²</sub>:</strong></td>
                <td style="text-align: right;">${l2_norm.toFixed(3)}</td>
            </tr>
            <tr>
                <td><strong>Sample |v|<sub>H¹</sub>:</strong></td>
                <td style="text-align: right;">${h1_seminorm.toFixed(3)}</td>
            </tr>
        </table>
    `;
    
    // Scaling formulas
    const scalingDiv = container.querySelector('#scaling-formulas');
    scalingDiv.innerHTML = `
        <table style="width: 100%; line-height: 2; color: #000;">
            <tr style="background: #ffffcc;">
                <td><strong>|J<sub>K</sub>|:</strong></td>
                <td style="font-family: monospace; text-align: right;">
                    h² = ${(h*h).toFixed(3)}
                </td>
            </tr>
            <tr>
                <td><strong>||v||<sub>L²</sub> ratio:</strong></td>
                <td style="font-family: monospace; text-align: right;">
                    h<sup>d/2</sup> = h = ${h.toFixed(3)}
                </td>
            </tr>
            <tr>
                <td><strong>|v|<sub>H¹</sub> ratio:</strong></td>
                <td style="font-family: monospace; text-align: right;">
                    h<sup>(d/2)-1</sup> = h⁰ = 1.000
                </td>
            </tr>
            <tr style="background: #ffebee;">
                <td><strong>|v|<sub>H²</sub> ratio:</strong></td>
                <td style="font-family: monospace; text-align: right;">
                    h<sup>(d/2)-2</sup> = h<sup>-1</sup> = ${(1/h).toFixed(3)}
                </td>
            </tr>
        </table>
        <p style="color: #666; margin-top: 10px; font-size: 12px; font-style: italic;">
            Note: H² ratio has h⁻¹ → grows as element shrinks (inverse inequality!)
        </p>
    `;
}

slider.oninput = draw;

// Initial draw
draw();
})();