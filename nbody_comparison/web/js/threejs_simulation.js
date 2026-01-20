/**
 * N-Body 3D Simulation - Code Container Version
 * Interactive Three.js visualization
 */

(function() {
    const container = window.currentCodeContainer;
    if (!container) {
        console.error('No container found for 3D simulation');
        return;
    }

    // Create HTML structure
    container.innerHTML = `
        <div class="threejs-simulation">
            <div id="threejs-canvas-container"></div>
            
            <div class="simulation-controls">
                <div class="control-group">
                    <label for="particle-count-slider">
                        Number of Particles: <span id="particle-count-value">50</span>
                    </label>
                    <input type="range" id="particle-count-slider" min="10" max="200" value="50" step="10">
                </div>
                
                <div class="control-group">
                    <label for="speed-slider">
                        Animation Speed: <span id="speed-value">1.0</span>×
                    </label>
                    <input type="range" id="speed-slider" min="0.1" max="3.0" value="1.0" step="0.1">
                </div>

                <div class="button-group">
                    <button id="start-btn" class="sim-btn sim-btn-primary">Start</button>
                    <button id="pause-btn" class="sim-btn sim-btn-secondary">Pause</button>
                    <button id="reset-btn" class="sim-btn sim-btn-secondary">Reset</button>
                </div>
            </div>

            <div class="simulation-note">
                <strong>⚠️ Browser Performance Note:</strong> This runs in JavaScript on your CPU. 
                For N > 200, performance degrades. For production simulations with N > 1000, 
                use the Python/JAX GPU implementation (1000× faster).
            </div>
        </div>

        <style>
            .threejs-simulation {
                width: 100%;
                max-width: 1200px;
                margin: 20px auto;
            }
            #threejs-canvas-container {
                width: 100%;
                height: 600px;
                background: #000;
                border-radius: 8px;
                margin-bottom: 20px;
                border: 2px solid #404040;
                position: relative;
            }
            .simulation-controls {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                padding: 20px;
                background: rgba(42, 42, 42, 0.5);
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .control-group label {
                color: #b0b0b0;
                font-size: 0.9rem;
                font-weight: 500;
            }
            .control-group input[type="range"] {
                width: 100%;
                height: 6px;
                border-radius: 3px;
                background: #404040;
                outline: none;
                -webkit-appearance: none;
            }
            .control-group input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #00ff41;
                cursor: pointer;
            }
            .control-group input[type="range"]::-moz-range-thumb {
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #00ff41;
                cursor: pointer;
                border: none;
            }
            .button-group {
                display: flex;
                gap: 10px;
                align-items: flex-end;
            }
            .sim-btn {
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .sim-btn-primary {
                background: #00ff41;
                color: #000;
            }
            .sim-btn-primary:hover {
                background: #00cc33;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 255, 65, 0.4);
            }
            .sim-btn-secondary {
                background: transparent;
                border: 2px solid #00ff41;
                color: #00ff41;
            }
            .sim-btn-secondary:hover {
                background: rgba(0, 255, 65, 0.1);
            }
            .simulation-note {
                margin-top: 20px;
                padding: 15px;
                background: rgba(255, 136, 0, 0.1);
                border-left: 4px solid #ff8800;
                border-radius: 4px;
                font-size: 0.9rem;
                color: #e4e4e4;
            }
            @media (max-width: 768px) {
                #threejs-canvas-container {
                    height: 400px;
                }
                .button-group {
                    flex-direction: column;
                }
                .sim-btn {
                    width: 100%;
                }
            }
        </style>
    `;

    // Three.js variables
    let scene, camera, renderer, particles;
    let animationId = null;
    let simulationRunning = false;
    let particleData = [];

    // Initialize Three.js scene
    function initThreeJS() {
        const canvasContainer = container.querySelector('#threejs-canvas-container');
        const width = canvasContainer.clientWidth;
        const height = canvasContainer.clientHeight;
        
        // Scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);
        
        // Camera
        camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        camera.position.z = 30;
        
        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        canvasContainer.appendChild(renderer.domElement);
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);
        
        const pointLight = new THREE.PointLight(0xffffff, 1, 100);
        pointLight.position.set(10, 10, 10);
        scene.add(pointLight);
        
        // Grid helper
        const gridHelper = new THREE.GridHelper(40, 40, 0x00ff41, 0x333333);
        scene.add(gridHelper);
        
        // Create initial particles
        createParticles(50);
        
        // Handle window resize
        window.addEventListener('resize', onWindowResize);
    }

    // Create particle system
    function createParticles(count) {
        if (particles) {
            scene.remove(particles);
        }
        
        particleData = [];
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        
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

    // Update particles with simple gravity
    function updateParticles(speedMultiplier) {
        if (!particles || !simulationRunning) return;
        
        const positions = particles.geometry.attributes.position.array;
        const G = 0.01;
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
                
                const distSq = dx*dx + dy*dy + dz*dz + 0.1;
                const dist = Math.sqrt(distSq);
                const force = G * p2.mass / (dist * distSq);
                
                ax += force * dx;
                ay += force * dy;
                az += force * dz;
            }
            
            // Update velocity (Velocity Verlet simplified)
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

    // Animation loop
    function animate() {
        animationId = requestAnimationFrame(animate);
        
        const speed = parseFloat(container.querySelector('#speed-slider').value);
        updateParticles(speed);
        
        // Rotate camera slowly
        camera.position.x = Math.sin(Date.now() * 0.0001) * 30;
        camera.position.z = Math.cos(Date.now() * 0.0001) * 30;
        camera.lookAt(scene.position);
        
        renderer.render(scene, camera);
    }

    // Handle window resize
    function onWindowResize() {
        const canvasContainer = container.querySelector('#threejs-canvas-container');
        camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    }

    // Setup event listeners
    function setupEventListeners() {
        // Particle count slider
        container.querySelector('#particle-count-slider').addEventListener('input', (e) => {
            const count = parseInt(e.target.value);
            container.querySelector('#particle-count-value').textContent = count;
        });
        
        // Speed slider
        container.querySelector('#speed-slider').addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            container.querySelector('#speed-value').textContent = speed.toFixed(1);
        });
        
        // Start button
        container.querySelector('#start-btn').addEventListener('click', () => {
            simulationRunning = true;
            if (!animationId) {
                animate();
            }
        });
        
        // Pause button
        container.querySelector('#pause-btn').addEventListener('click', () => {
            simulationRunning = false;
        });
        
        // Reset button
        container.querySelector('#reset-btn').addEventListener('click', () => {
            const count = parseInt(container.querySelector('#particle-count-slider').value);
            createParticles(count);
            simulationRunning = false;
        });
    }

    // Initialize simulation
    function init() {
        console.log('Initializing 3D N-body simulation...');
        
        // Check if Three.js is loaded
        if (typeof THREE === 'undefined') {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
            script.onload = () => {
                initThreeJS();
                setupEventListeners();
                console.log('3D simulation initialized successfully!');
            };
            document.head.appendChild(script);
        } else {
            initThreeJS();
            setupEventListeners();
            console.log('3D simulation initialized successfully!');
        }
    }

    init();
})();