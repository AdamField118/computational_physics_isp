/**
 * N-Body Simulation Dashboard - JavaScript
 * Adam Field - Computational Physics ISP
 */

// ============================================
// Global State
// ============================================
let benchmarkData = null;
let scalingChart = null;
let speedupChart = null;
let energyChart = null;

// Three.js variables
let scene, camera, renderer, particles;
let animationId = null;
let simulationRunning = false;
let particleData = [];

// Color scheme matching CSS
const colors = {
    jax: '#00ff41',
    fortran: '#ff00ff',
    cpp: '#ff8800',
    c: '#00ffff',
    rust: '#ff6b35',
    julia: '#9558b2',
    python: '#ff0000'
};

// ============================================
// Data Loading & Processing
// ============================================

/**
 * Load benchmark data from JSON file
 */
async function loadBenchmarkData() {
    try {
        const response = await fetch('/data/benchmark_results.json');
        if (!response.ok) {
            console.warn('Benchmark data not found, using sample data');
            return generateSampleData();
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.warn('Error loading benchmark data:', error);
        //return generateSampleData();
    }
}

/**
 * Generate sample data if JSON file isn't available
 */
function generateSampleData() {
    return {
        timestamp: new Date().toISOString(),
        results: [
            { implementation: 'JAX (GPU)', n_particles: 10, n_steps: 1000, runtime: 0.084, time_per_step: 0.000084, energy_drift: 0.00175 },
            { implementation: 'JAX (GPU)', n_particles: 50, n_steps: 1000, runtime: 0.096, time_per_step: 0.000096, energy_drift: 0.027 },
            { implementation: 'JAX (GPU)', n_particles: 100, n_steps: 1000, runtime: 0.125, time_per_step: 0.000125, energy_drift: 0.046 },
            { implementation: 'JAX (GPU)', n_particles: 500, n_steps: 1000, runtime: 0.088, time_per_step: 0.000088, energy_drift: 0.139 },
            { implementation: 'JAX (GPU)', n_particles: 1000, n_steps: 1000, runtime: 0.080, time_per_step: 0.000080, energy_drift: 3.316 },
            
            { implementation: 'Fortran (OpenMP)', n_particles: 10, n_steps: 1000, runtime: 0.015, time_per_step: 0.000015, energy_drift: 0.000023 },
            { implementation: 'Fortran (OpenMP)', n_particles: 50, n_steps: 1000, runtime: 0.020, time_per_step: 0.000020, energy_drift: 0.0093 },
            { implementation: 'Fortran (OpenMP)', n_particles: 100, n_steps: 1000, runtime: 0.034, time_per_step: 0.000034, energy_drift: 0.022 },
            { implementation: 'Fortran (OpenMP)', n_particles: 500, n_steps: 1000, runtime: 0.394, time_per_step: 0.000394, energy_drift: 0.935 },
            { implementation: 'Fortran (OpenMP)', n_particles: 1000, n_steps: 1000, runtime: 1.373, time_per_step: 0.001373, energy_drift: 3.940 },
            
            { implementation: 'Rust', n_particles: 10, n_steps: 1000, runtime: 0.012, time_per_step: 0.000012, energy_drift: 0.000023 },
            { implementation: 'Rust', n_particles: 50, n_steps: 1000, runtime: 0.018, time_per_step: 0.000018, energy_drift: 0.0093 },
            { implementation: 'Rust', n_particles: 100, n_steps: 1000, runtime: 0.030, time_per_step: 0.000030, energy_drift: 0.022 },
            { implementation: 'Rust', n_particles: 500, n_steps: 1000, runtime: 0.350, time_per_step: 0.000350, energy_drift: 0.935 },
            { implementation: 'Rust', n_particles: 1000, n_steps: 1000, runtime: 1.250, time_per_step: 0.001250, energy_drift: 3.940 },
            
            { implementation: 'Julia', n_particles: 10, n_steps: 1000, runtime: 0.013, time_per_step: 0.000013, energy_drift: 0.000023 },
            { implementation: 'Julia', n_particles: 50, n_steps: 1000, runtime: 0.019, time_per_step: 0.000019, energy_drift: 0.0093 },
            { implementation: 'Julia', n_particles: 100, n_steps: 1000, runtime: 0.032, time_per_step: 0.000032, energy_drift: 0.022 },
            { implementation: 'Julia', n_particles: 500, n_steps: 1000, runtime: 0.370, time_per_step: 0.000370, energy_drift: 0.935 },
            { implementation: 'Julia', n_particles: 1000, n_steps: 1000, runtime: 1.300, time_per_step: 0.001300, energy_drift: 3.940 },
            
            { implementation: 'Python (NumPy)', n_particles: 10, n_steps: 1000, runtime: 0.035, time_per_step: 0.000035, energy_drift: 0.000023 },
            { implementation: 'Python (NumPy)', n_particles: 50, n_steps: 1000, runtime: 0.225, time_per_step: 0.000225, energy_drift: 0.0093 },
            { implementation: 'Python (NumPy)', n_particles: 100, n_steps: 1000, runtime: 1.257, time_per_step: 0.001257, energy_drift: 0.022 },
            { implementation: 'Python (NumPy)', n_particles: 500, n_steps: 1000, runtime: 27.46, time_per_step: 0.02746, energy_drift: 0.934 },
            { implementation: 'Python (NumPy)', n_particles: 1000, n_steps: 1000, runtime: 110.84, time_per_step: 0.11084, energy_drift: 4.331 },
        ]
    };
}

/**
 * Process benchmark data into organized structure
 */
function processBenchmarkData(data) {
    const implementations = {};
    
    data.results.forEach(result => {
        const impl = result.implementation;
        if (!implementations[impl]) {
            implementations[impl] = [];
        }
        implementations[impl].push(result);
    });
    
    // Sort by particle count
    Object.keys(implementations).forEach(impl => {
        implementations[impl].sort((a, b) => a.n_particles - b.n_particles);
    });
    
    return implementations;
}

// ============================================
// UI Updates
// ============================================

/**
 * Update statistics cards
 */
function updateStats(data) {
    const allResults = data.results;
    
    // Max particles
    const maxParticles = Math.max(...allResults.map(r => r.n_particles));
    document.getElementById('total-particles').textContent = maxParticles.toLocaleString();
    
    // Steps
    const steps = allResults[0]?.n_steps || 1000;
    document.getElementById('total-steps').textContent = steps.toLocaleString();
    
    // Energy conservation (average at N=50)
    const n50Results = allResults.filter(r => r.n_particles === 50);
    if (n50Results.length > 0) {
        const avgDrift = n50Results.reduce((sum, r) => sum + r.energy_drift, 0) / n50Results.length;
        document.getElementById('energy-conservation').textContent = `${avgDrift.toFixed(3)}%`;
    }
    
    // GPU speedup at N=1000
    const jaxResult = allResults.find(r => r.implementation.includes('JAX') && r.n_particles === 1000);
    const pythonResult = allResults.find(r => r.implementation.includes('Python') && r.n_particles === 1000);
    
    if (jaxResult && pythonResult) {
        const speedup = pythonResult.time_per_step / jaxResult.time_per_step;
        document.getElementById('gpu-speedup').textContent = `${speedup.toFixed(0)}×`;
    }
}

/**
 * Populate performance table
 */
function populatePerformanceTable(implementations) {
    const tbody = document.querySelector('#performanceTable tbody');
    tbody.innerHTML = '';
    
    const nValues = [10, 50, 100, 500, 1000];
    const implOrder = ['JAX (GPU)', 'Fortran (OpenMP)', 'Rust', 'Julia', 'C++', 'C', 'Python (NumPy)'];
    
    implOrder.forEach(implName => {
        if (!implementations[implName]) return;
        
        const row = document.createElement('tr');
        
        // Implementation name
        const nameCell = document.createElement('td');
        nameCell.textContent = implName;
        nameCell.style.fontWeight = 'bold';
        row.appendChild(nameCell);
        
        // Times for each N
        nValues.forEach(n => {
            const cell = document.createElement('td');
            const result = implementations[implName].find(r => r.n_particles === n);
            
            if (result) {
                const timeMs = result.time_per_step * 1000;
                cell.textContent = timeMs < 1 ? timeMs.toFixed(4) : timeMs.toFixed(2);
            } else {
                cell.textContent = '—';
            }
            
            row.appendChild(cell);
        });
        
        tbody.appendChild(row);
    });
}

// ============================================
// Chart Creation
// ============================================

/**
 * Create scaling chart (time vs N)
 */
function createScalingChart(implementations) {
    const ctx = document.getElementById('scalingChart').getContext('2d');
    
    const datasets = [];
    const implOrder = ['JAX (GPU)', 'Fortran (OpenMP)', 'Rust', 'Julia', 'C++', 'C', 'Python (NumPy)'];
    const colorMap = {
        'JAX (GPU)': colors.jax,
        'Fortran (OpenMP)': colors.fortran,
        'Rust': colors.rust,
        'Julia': colors.julia,
        'C++': colors.cpp,
        'C': colors.c,
        'Python (NumPy)': colors.python
    };
    
    implOrder.forEach(implName => {
        if (!implementations[implName]) return;
        
        const data = implementations[implName].map(r => ({
            x: r.n_particles,
            y: r.time_per_step * 1000  // Convert to ms
        }));
        
        datasets.push({
            label: implName,
            data: data,
            borderColor: colorMap[implName],
            backgroundColor: colorMap[implName] + '33',
            borderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
        });
    });
    
    if (scalingChart) scalingChart.destroy();
    
    scalingChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Number of Particles (N)',
                        color: '#fff',
                        font: { size: 14 }
                    },
                    ticks: { color: '#b0b0b0' },
                    grid: { color: '#333' }
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Time per Step (ms)',
                        color: '#fff',
                        font: { size: 14 }
                    },
                    ticks: { color: '#b0b0b0' },
                    grid: { color: '#333' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#fff', font: { size: 12 } }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(4)} ms`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create speedup chart (vs Python)
 */
function createSpeedupChart(implementations) {
    const ctx = document.getElementById('speedupChart').getContext('2d');
    
    const pythonData = implementations['Python (NumPy)'];
    if (!pythonData) return;
    
    // Create lookup for Python times
    const pythonTimes = {};
    pythonData.forEach(r => {
        pythonTimes[r.n_particles] = r.time_per_step;
    });
    
    const datasets = [];
    const implOrder = ['JAX (GPU)', 'Fortran (OpenMP)', 'Rust', 'Julia', 'C++', 'C'];
    const colorMap = {
        'JAX (GPU)': colors.jax,
        'Fortran (OpenMP)': colors.fortran,
        'Rust': colors.rust,
        'Julia': colors.julia,
        'C++': colors.cpp,
        'C': colors.c
    };
    
    implOrder.forEach(implName => {
        if (!implementations[implName]) return;
        
        const data = implementations[implName]
            .filter(r => pythonTimes[r.n_particles])
            .map(r => ({
                x: r.n_particles,
                y: pythonTimes[r.n_particles] / r.time_per_step
            }));
        
        datasets.push({
            label: implName,
            data: data,
            borderColor: colorMap[implName],
            backgroundColor: colorMap[implName] + '33',
            borderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
        });
    });
    
    if (speedupChart) speedupChart.destroy();
    
    speedupChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Number of Particles (N)',
                        color: '#fff',
                        font: { size: 14 }
                    },
                    ticks: { color: '#b0b0b0' },
                    grid: { color: '#333' }
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Speedup vs Python',
                        color: '#fff',
                        font: { size: 14 }
                    },
                    ticks: { color: '#b0b0b0' },
                    grid: { color: '#333' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#fff', font: { size: 12 } }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}× faster`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create energy conservation chart
 */
function createEnergyChart(implementations) {
    const ctx = document.getElementById('energyChart').getContext('2d');
    
    const datasets = [];
    const implOrder = ['JAX (GPU)', 'Fortran (OpenMP)', 'Rust', 'Julia', 'C++', 'C', 'Python (NumPy)'];
    const colorMap = {
        'JAX (GPU)': colors.jax,
        'Fortran (OpenMP)': colors.fortran,
        'Rust': colors.rust,
        'Julia': colors.julia,
        'C++': colors.cpp,
        'C': colors.c,
        'Python (NumPy)': colors.python
    };
    
    implOrder.forEach(implName => {
        if (!implementations[implName]) return;
        
        const data = implementations[implName].map(r => ({
            x: r.n_particles,
            y: r.energy_drift
        }));
        
        datasets.push({
            label: implName,
            data: data,
            borderColor: colorMap[implName],
            backgroundColor: colorMap[implName] + '33',
            borderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
        });
    });
    
    if (energyChart) energyChart.destroy();
    
    energyChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Number of Particles (N)',
                        color: '#fff',
                        font: { size: 14 }
                    },
                    ticks: { color: '#b0b0b0' },
                    grid: { color: '#333' }
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Energy Drift (%)',
                        color: '#fff',
                        font: { size: 14 }
                    },
                    ticks: { color: '#b0b0b0' },
                    grid: { color: '#333' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#fff', font: { size: 12 } }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(4)}%`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// 3D Visualization (Three.js)
// ============================================

/**
 * Initialize Three.js scene
 */
function initThreeJS() {
    const container = document.getElementById('threejs-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    
    // Camera
    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 30;
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);
    
    // Grid helper
    const gridHelper = new THREE.GridHelper(40, 40, 0x00ff41, 0x333333);
    scene.add(gridHelper);
    
    // Create particles
    createParticles(50);
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

/**
 * Create particle system
 */
function createParticles(count) {
    // Remove existing particles
    if (particles) {
        scene.remove(particles);
    }
    
    particleData = [];
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];
    const velocities = [];
    
    for (let i = 0; i < count; i++) {
        // Random position
        const x = (Math.random() - 0.5) * 20;
        const y = (Math.random() - 0.5) * 20;
        const z = (Math.random() - 0.5) * 20;
        positions.push(x, y, z);
        
        // Random velocity
        const vx = (Math.random() - 0.5) * 0.1;
        const vy = (Math.random() - 0.5) * 0.1;
        const vz = (Math.random() - 0.5) * 0.1;
        velocities.push(vx, vy, vz);
        
        // Color (green spectrum)
        const colorValue = Math.random();
        colors.push(0, 1, colorValue);
        
        particleData.push({
            position: new THREE.Vector3(x, y, z),
            velocity: new THREE.Vector3(vx, vy, vz),
            mass: 0.5 + Math.random() * 0.5
        });
    }
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
        size: 0.5,
        vertexColors: true,
        transparent: true,
        opacity: 0.9
    });
    
    particles = new THREE.Points(geometry, material);
    scene.add(particles);
}

/**
 * Update particles with simple gravity
 */
function updateParticles(speedMultiplier = 1.0) {
    if (!particles || !simulationRunning) return;
    
    const positions = particles.geometry.attributes.position.array;
    const G = 0.01;  // Simplified gravity constant
    const dt = 0.1 * speedMultiplier;
    
    // Simple O(N²) gravity simulation
    for (let i = 0; i < particleData.length; i++) {
        const p1 = particleData[i];
        let ax = 0, ay = 0, az = 0;
        
        for (let j = 0; j < particleData.length; j++) {
            if (i === j) continue;
            
            const p2 = particleData[j];
            const dx = p2.position.x - p1.position.x;
            const dy = p2.position.y - p1.position.y;
            const dz = p2.position.z - p1.position.z;
            
            const distSq = dx*dx + dy*dy + dz*dz + 0.1;  // Softening
            const dist = Math.sqrt(distSq);
            const force = G * p2.mass / (dist * distSq);
            
            ax += force * dx;
            ay += force * dy;
            az += force * dz;
        }
        
        // Update velocity
        p1.velocity.x += ax * dt;
        p1.velocity.y += ay * dt;
        p1.velocity.z += az * dt;
        
        // Update position
        p1.position.x += p1.velocity.x * dt;
        p1.position.y += p1.velocity.y * dt;
        p1.position.z += p1.velocity.z * dt;
        
        // Update geometry
        positions[i * 3] = p1.position.x;
        positions[i * 3 + 1] = p1.position.y;
        positions[i * 3 + 2] = p1.position.z;
    }
    
    particles.geometry.attributes.position.needsUpdate = true;
}

/**
 * Animation loop
 */
function animate() {
    animationId = requestAnimationFrame(animate);
    
    const speed = parseFloat(document.getElementById('speed-control').value);
    updateParticles(speed);
    
    // Rotate camera slowly
    camera.position.x = Math.sin(Date.now() * 0.0001) * 30;
    camera.position.z = Math.cos(Date.now() * 0.0001) * 30;
    camera.lookAt(scene.position);
    
    renderer.render(scene, camera);
}

/**
 * Handle window resize
 */
function onWindowResize() {
    const container = document.getElementById('threejs-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// ============================================
// Event Handlers
// ============================================

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            const targetId = link.getAttribute('href').slice(1);
            document.getElementById(targetId).scrollIntoView({ behavior: 'smooth' });
        });
    });
    
    // Particle count slider
    document.getElementById('particle-count').addEventListener('input', (e) => {
        const count = parseInt(e.target.value);
        document.getElementById('particle-count-value').textContent = count;
    });
    
    // Speed slider
    document.getElementById('speed-control').addEventListener('input', (e) => {
        const speed = parseFloat(e.target.value);
        document.getElementById('speed-value').textContent = speed.toFixed(1);
    });
    
    // Simulation controls
    document.getElementById('start-sim').addEventListener('click', () => {
        simulationRunning = true;
        if (!animationId) {
            animate();
        }
    });
    
    document.getElementById('pause-sim').addEventListener('click', () => {
        simulationRunning = false;
    });
    
    document.getElementById('reset-sim').addEventListener('click', () => {
        const count = parseInt(document.getElementById('particle-count').value);
        createParticles(count);
        simulationRunning = false;
    });
}

// ============================================
// Initialization
// ============================================

/**
 * Initialize dashboard
 */
async function init() {
    console.log('Initializing N-Body Dashboard...');
    
    // Load and process data
    benchmarkData = await loadBenchmarkData();
    const implementations = processBenchmarkData(benchmarkData);
    
    // Update UI
    updateStats(benchmarkData);
    populatePerformanceTable(implementations);
    
    // Create charts
    createScalingChart(implementations);
    createSpeedupChart(implementations);
    createEnergyChart(implementations);
    
    // Initialize 3D visualization
    initThreeJS();
    
    // Setup event listeners
    setupEventListeners();
    
    console.log('Dashboard initialized successfully!');
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);