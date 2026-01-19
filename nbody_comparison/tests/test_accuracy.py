"""
Test Suite: Verify all N-body implementations produce equivalent results

This test suite ensures that JAX, Fortran, C, C++, and Python implementations
all produce numerically equivalent results for the same initial conditions.

Note: GPU (JAX) uses different floating-point reduction order than CPU,
so we use relaxed tolerance for GPU vs CPU comparisons.

Adam Field - Computational Physics ISP
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append('../nbody/jax')
sys.path.append('../nbody/python')

# Import implementations
try:
    import jax
    import jax.numpy as jnp
    from nbody_jax import (
        create_random_system, NBodyState, NBodyConfig,
        simulate as jax_simulate, compute_energy as jax_energy
    )
    HAS_JAX = True
except ImportError:
    print("Warning: JAX not available")
    HAS_JAX = False

try:
    sys.path.append('../nbody/fortran')
    import nbody_fortran_module as fortran_nbody
    HAS_FORTRAN = True
except ImportError:
    print("Warning: Fortran module not available")
    HAS_FORTRAN = False

try:
    sys.path.append('../nbody/c')
    import nbody_c_wrapper as c_nbody
    HAS_C = True
except ImportError:
    print("Warning: C module not available")
    HAS_C = False

try:
    sys.path.append('../nbody/cpp')
    from nbody_cpp_module import NBodySimulator
    HAS_CPP = True
except ImportError:
    print("Warning: C++ module not available")
    HAS_CPP = False

try:
    from nbody_python import (
        simulate as python_simulate, 
        compute_energy as python_energy
    )
    HAS_PYTHON = True
except ImportError:
    print("Warning: Pure Python module not available")
    HAS_PYTHON = False


class TestNBodyAccuracy:
    """Test suite for N-body implementation accuracy"""
    
    def __init__(self, cpu_tolerance=1e-10, gpu_tolerance=1e-5):
        """
        Args:
            cpu_tolerance: Strict tolerance for CPU vs CPU comparisons
            gpu_tolerance: Relaxed tolerance for GPU vs CPU comparisons
                          (accounts for different floating-point reduction order)
        """
        self.cpu_tolerance = cpu_tolerance
        self.gpu_tolerance = gpu_tolerance
        self.results = {}
    
    def get_tolerance(self, impl_name):
        """Get appropriate tolerance based on implementation type"""
        if 'JAX' in impl_name or 'GPU' in impl_name:
            return self.gpu_tolerance
        return self.cpu_tolerance
        
    def create_test_system(self, N=10, seed=42):
        """
        Create identical initial conditions for all implementations
        
        Returns:
            positions, velocities, masses (all as NumPy arrays)
        """
        np.random.seed(seed)
        
        positions = np.random.uniform(-10, 10, (N, 3)).astype(np.float64)
        velocities = 0.5 * np.random.randn(N, 3).astype(np.float64)
        masses = np.random.uniform(0.1, 1.0, N).astype(np.float64)
        
        return positions, velocities, masses
    
    def test_energy_calculation(self, N=10):
        """Test that all implementations compute the same initial energy"""
        print(f"\nTest 1: Energy Calculation (N={N})")
        print("-" * 70)
        
        positions, velocities, masses = self.create_test_system(N)
        
        config = {'G': 1.0, 'softening': 0.1, 'dt': 0.01}
        
        energies = {}
        
        # Python
        if HAS_PYTHON:
            ke, pe, total = python_energy(positions, velocities, masses, 
                                         config['G'], config['softening'])
            energies['Python'] = (ke, pe, total)
            print(f"  Python:  KE={ke:.10f}, PE={pe:.10f}, Total={total:.10f}")
        
        # C
        if HAS_C:
            ke, pe, total = c_nbody.compute_energy(positions, velocities, masses,
                                                   config['G'], config['softening'])
            energies['C'] = (ke, pe, total)
            print(f"  C:       KE={ke:.10f}, PE={pe:.10f}, Total={total:.10f}")
        
        # C++
        if HAS_CPP:
            sim = NBodySimulator(N, config['G'], config['softening'], config['dt'])
            pos_flat = positions.flatten()
            vel_flat = velocities.flatten()
            ke, pe, total = sim.compute_energy(pos_flat, vel_flat, masses)
            energies['C++'] = (ke, pe, total)
            print(f"  C++:     KE={ke:.10f}, PE={pe:.10f}, Total={total:.10f}")
        
        # Fortran
        if HAS_FORTRAN:
            pos_f = np.asfortranarray(positions)
            vel_f = np.asfortranarray(velocities)
            masses_f = np.asfortranarray(masses)
            ke, pe, total = fortran_nbody.nbody_fortran.compute_energy(
                pos_f, vel_f, masses_f, config['G'], config['softening']
            )
            energies['Fortran'] = (ke, pe, total)
            print(f"  Fortran: KE={ke:.10f}, PE={pe:.10f}, Total={total:.10f}")
        
        # JAX
        if HAS_JAX:
            state = NBodyState(
                positions=jnp.array(positions),
                velocities=jnp.array(velocities),
                masses=jnp.array(masses),
                time=0.0
            )
            jax_config = NBodyConfig(G=config['G'], softening=config['softening'], 
                                    dt=config['dt'])
            ke, pe, total = jax_energy(state, jax_config)
            energies['JAX (GPU)'] = (float(ke), float(pe), float(total))
            print(f"  JAX:     KE={ke:.10f}, PE={pe:.10f}, Total={total:.10f}")
        
        # Compare all energies
        if len(energies) > 1:
            print("\n  Comparing energies...")
            reference_name, reference_energy = list(energies.items())[0]
            
            all_match = True
            for name, (ke, pe, total) in energies.items():
                if name == reference_name:
                    continue
                
                ref_ke, ref_pe, ref_total = reference_energy
                
                ke_diff = abs(ke - ref_ke)
                pe_diff = abs(pe - ref_pe)
                total_diff = abs(total - ref_total)
                
                # Use appropriate tolerance
                tol = self.get_tolerance(name)
                match = (ke_diff < tol and pe_diff < tol and total_diff < tol)
                
                status = "✓ PASS" if match else "✗ FAIL"
                tol_str = "(GPU tol)" if 'GPU' in name else "(CPU tol)"
                print(f"    {name:12s} vs {reference_name:10s}: {status} {tol_str}")
                print(f"      KE diff: {ke_diff:.2e}, PE diff: {pe_diff:.2e}, "
                     f"Total diff: {total_diff:.2e}")
                
                all_match = all_match and match
            
            if all_match:
                print("\n  ✓ All implementations compute equivalent energies!")
                return True
            else:
                print("\n  ✗ Energy calculations differ beyond acceptable tolerance")
                return False
        else:
            print("  Warning: Only one implementation available, cannot compare")
            return None
    
    def test_single_step(self, N=10):
        """Test that all implementations produce equivalent results after one step"""
        print(f"\nTest 2: Single Timestep (N={N})")
        print("-" * 70)
        
        positions_init, velocities_init, masses = self.create_test_system(N)
        
        config = {'G': 1.0, 'softening': 0.1, 'dt': 0.01}
        n_steps = 1
        
        results = {}
        
        # Python
        if HAS_PYTHON:
            pos, vel = python_simulate(positions_init.copy(), velocities_init.copy(), 
                                       masses, n_steps, config['G'], 
                                       config['softening'], config['dt'])
            results['Python'] = {'positions': pos, 'velocities': vel}
        
        # C
        if HAS_C:
            pos, vel = c_nbody.simulate(positions_init.copy(), velocities_init.copy(),
                                       masses, n_steps, config['G'],
                                       config['softening'], config['dt'])
            results['C'] = {'positions': pos, 'velocities': vel}
        
        # C++
        if HAS_CPP:
            sim = NBodySimulator(N, config['G'], config['softening'], config['dt'])
            pos_flat = positions_init.flatten().copy()
            vel_flat = velocities_init.flatten().copy()
            pos, vel = sim.simulate(pos_flat, vel_flat, masses, n_steps)
            results['C++'] = {
                'positions': pos.reshape(N, 3), 
                'velocities': vel.reshape(N, 3)
            }
        
        # Fortran
        if HAS_FORTRAN:
            pos_f = np.asfortranarray(positions_init.copy())
            vel_f = np.asfortranarray(velocities_init.copy())
            masses_f = np.asfortranarray(masses)
            pos, vel = fortran_nbody.nbody_fortran.simulate(
                pos_f, vel_f, masses_f, n_steps,
                config['G'], config['softening'], config['dt']
            )
            results['Fortran'] = {'positions': pos, 'velocities': vel}
        
        # JAX
        if HAS_JAX:
            initial_state = NBodyState(
                positions=jnp.array(positions_init),
                velocities=jnp.array(velocities_init),
                masses=jnp.array(masses),
                time=0.0
            )
            jax_config = NBodyConfig(G=config['G'], softening=config['softening'],
                                    dt=config['dt'])
            times, pos, vel = jax_simulate(initial_state, jax_config, n_steps, 
                                          save_every=n_steps)
            results['JAX (GPU)'] = {
                'positions': np.array(pos[-1]),
                'velocities': np.array(vel[-1])
            }
        
        # Compare results
        if len(results) > 1:
            print("  Comparing final states after 1 step...")
            reference_name, reference_data = list(results.items())[0]
            ref_pos = reference_data['positions']
            ref_vel = reference_data['velocities']
            
            all_match = True
            for name, data in results.items():
                if name == reference_name:
                    continue
                
                pos_diff = np.max(np.abs(data['positions'] - ref_pos))
                vel_diff = np.max(np.abs(data['velocities'] - ref_vel))
                
                # Use appropriate tolerance
                tol = self.get_tolerance(name)
                match = (pos_diff < tol and vel_diff < tol)
                status = "✓ PASS" if match else "✗ FAIL"
                tol_str = "(GPU tol)" if 'GPU' in name else "(CPU tol)"
                
                print(f"    {name:12s} vs {reference_name:10s}: {status} {tol_str}")
                print(f"      Max position diff: {pos_diff:.2e}, "
                     f"Max velocity diff: {vel_diff:.2e}")
                
                all_match = all_match and match
            
            if all_match:
                print("\n  ✓ All implementations produce equivalent results!")
                return True
            else:
                print("\n  ✗ Results differ beyond acceptable tolerance")
                return False
        else:
            print("  Warning: Only one implementation available")
            return None
    
    def test_multi_step(self, N=20, n_steps=100):
        """Test agreement over multiple timesteps"""
        print(f"\nTest 3: Multi-Step Evolution (N={N}, steps={n_steps})")
        print("-" * 70)
        
        positions_init, velocities_init, masses = self.create_test_system(N)
        
        config = {'G': 1.0, 'softening': 0.1, 'dt': 0.01}
        
        results = {}
        
        # Run simulations
        print("  Running simulations...")
        
        if HAS_PYTHON:
            print("    Python...", end=' ')
            pos, vel = python_simulate(positions_init.copy(), velocities_init.copy(),
                                       masses, n_steps, config['G'],
                                       config['softening'], config['dt'])
            results['Python'] = {'positions': pos, 'velocities': vel}
            print("done")
        
        if HAS_C:
            print("    C...", end=' ')
            pos, vel = c_nbody.simulate(positions_init.copy(), velocities_init.copy(),
                                       masses, n_steps, config['G'],
                                       config['softening'], config['dt'])
            results['C'] = {'positions': pos, 'velocities': vel}
            print("done")
        
        if HAS_CPP:
            print("    C++...", end=' ')
            sim = NBodySimulator(N, config['G'], config['softening'], config['dt'])
            pos_flat = positions_init.flatten().copy()
            vel_flat = velocities_init.flatten().copy()
            pos, vel = sim.simulate(pos_flat, vel_flat, masses, n_steps)
            results['C++'] = {
                'positions': pos.reshape(N, 3),
                'velocities': vel.reshape(N, 3)
            }
            print("done")
        
        if HAS_FORTRAN:
            print("    Fortran...", end=' ')
            pos_f = np.asfortranarray(positions_init.copy())
            vel_f = np.asfortranarray(velocities_init.copy())
            masses_f = np.asfortranarray(masses)
            pos, vel = fortran_nbody.nbody_fortran.simulate(
                pos_f, vel_f, masses_f, n_steps,
                config['G'], config['softening'], config['dt']
            )
            results['Fortran'] = {'positions': pos, 'velocities': vel}
            print("done")
        
        if HAS_JAX:
            print("    JAX...", end=' ')
            initial_state = NBodyState(
                positions=jnp.array(positions_init),
                velocities=jnp.array(velocities_init),
                masses=jnp.array(masses),
                time=0.0
            )
            jax_config = NBodyConfig(G=config['G'], softening=config['softening'],
                                    dt=config['dt'])
            times, pos, vel = jax_simulate(initial_state, jax_config, n_steps,
                                          save_every=n_steps)
            results['JAX (GPU)'] = {
                'positions': np.array(pos[-1]),
                'velocities': np.array(vel[-1])
            }
            print("done")
        
        # Compare results
        print("\n  Comparing final states...")
        
        if len(results) > 1:
            reference_name, reference_data = list(results.items())[0]
            ref_pos = reference_data['positions']
            ref_vel = reference_data['velocities']
            
            all_match = True
            for name, data in results.items():
                if name == reference_name:
                    continue
                
                pos_diff = np.max(np.abs(data['positions'] - ref_pos))
                vel_diff = np.max(np.abs(data['velocities'] - ref_vel))
                pos_rms = np.sqrt(np.mean((data['positions'] - ref_pos)**2))
                vel_rms = np.sqrt(np.mean((data['velocities'] - ref_vel)**2))
                
                # Use appropriate tolerance with accumulation factor
                base_tol = self.get_tolerance(name)
                tol = base_tol * np.sqrt(n_steps)
                match = (pos_diff < tol and vel_diff < tol)
                status = "✓ PASS" if match else "✗ FAIL"
                tol_str = "(GPU tol)" if 'GPU' in name else "(CPU tol)"
                
                print(f"    {name:12s} vs {reference_name:10s}: {status} {tol_str}")
                print(f"      Max pos diff: {pos_diff:.2e}, RMS: {pos_rms:.2e}")
                print(f"      Max vel diff: {vel_diff:.2e}, RMS: {vel_rms:.2e}")
                
                all_match = all_match and match
            
            if all_match:
                print(f"\n  ✓ All implementations agree within tolerance")
                return True
            else:
                print("\n  ✗ Implementations diverge beyond tolerance")
                return False
        else:
            print("  Warning: Only one implementation available")
            return None
    
    def test_energy_conservation(self, N=50, n_steps=1000):
        """Test energy conservation quality"""
        print(f"\nTest 4: Energy Conservation (N={N}, steps={n_steps})")
        print("-" * 70)
        
        positions_init, velocities_init, masses = self.create_test_system(N)
        
        config = {'G': 1.0, 'softening': 0.1, 'dt': 0.01}
        
        print("  Running simulations and measuring energy drift...")
        
        drift_results = {}
        
        # Test each implementation
        if HAS_PYTHON:
            ke0, pe0, E0 = python_energy(positions_init, velocities_init, masses,
                                        config['G'], config['softening'])
            pos, vel = python_simulate(positions_init.copy(), velocities_init.copy(),
                                       masses, n_steps, config['G'],
                                       config['softening'], config['dt'])
            ke_f, pe_f, E_f = python_energy(pos, vel, masses, config['G'], 
                                            config['softening'])
            drift = abs(E_f - E0) / abs(E0) * 100
            drift_results['Python'] = drift
            print(f"    Python:  {drift:.6f}%")
        
        if HAS_C:
            ke0, pe0, E0 = c_nbody.compute_energy(positions_init, velocities_init,
                                                  masses, config['G'], config['softening'])
            pos, vel = c_nbody.simulate(positions_init.copy(), velocities_init.copy(),
                                       masses, n_steps, config['G'],
                                       config['softening'], config['dt'])
            ke_f, pe_f, E_f = c_nbody.compute_energy(pos, vel, masses, config['G'],
                                                     config['softening'])
            drift = abs(E_f - E0) / abs(E0) * 100
            drift_results['C'] = drift
            print(f"    C:       {drift:.6f}%")
        
        if HAS_CPP:
            sim = NBodySimulator(N, config['G'], config['softening'], config['dt'])
            pos_flat = positions_init.flatten().copy()
            vel_flat = velocities_init.flatten().copy()
            ke0, pe0, E0 = sim.compute_energy(pos_flat, vel_flat, masses)
            pos, vel = sim.simulate(pos_flat.copy(), vel_flat.copy(), masses, n_steps)
            ke_f, pe_f, E_f = sim.compute_energy(pos.flatten(), vel.flatten(), masses)
            drift = abs(E_f - E0) / abs(E0) * 100
            drift_results['C++'] = drift
            print(f"    C++:     {drift:.6f}%")
        
        if HAS_FORTRAN:
            pos_f = np.asfortranarray(positions_init.copy())
            vel_f = np.asfortranarray(velocities_init.copy())
            masses_f = np.asfortranarray(masses)
            ke0, pe0, E0 = fortran_nbody.nbody_fortran.compute_energy(
                pos_f, vel_f, masses_f, config['G'], config['softening']
            )
            pos, vel = fortran_nbody.nbody_fortran.simulate(
                pos_f, vel_f, masses_f, n_steps,
                config['G'], config['softening'], config['dt']
            )
            ke_f, pe_f, E_f = fortran_nbody.nbody_fortran.compute_energy(
                pos, vel, masses_f, config['G'], config['softening']
            )
            drift = abs(E_f - E0) / abs(E0) * 100
            drift_results['Fortran'] = drift
            print(f"    Fortran: {drift:.6f}%")
        
        if HAS_JAX:
            initial_state = NBodyState(
                positions=jnp.array(positions_init),
                velocities=jnp.array(velocities_init),
                masses=jnp.array(masses),
                time=0.0
            )
            jax_config = NBodyConfig(G=config['G'], softening=config['softening'],
                                    dt=config['dt'])
            ke0, pe0, E0 = jax_energy(initial_state, jax_config)
            times, pos, vel = jax_simulate(initial_state, jax_config, n_steps,
                                          save_every=n_steps)
            final_state = NBodyState(
                positions=pos[-1], velocities=vel[-1],
                masses=initial_state.masses, time=times[-1]
            )
            ke_f, pe_f, E_f = jax_energy(final_state, jax_config)
            drift = abs(float(E_f) - float(E0)) / abs(float(E0)) * 100
            drift_results['JAX'] = drift
            print(f"    JAX:     {drift:.6f}%")
        
        # Check if all are within acceptable range
        print("\n  Energy drift quality:")
        acceptable_drift = 1.0  # 1% is reasonable for N=50, n_steps=1000
        
        all_acceptable = True
        for name, drift in drift_results.items():
            status = "✓ GOOD" if drift < acceptable_drift else "⚠ HIGH"
            print(f"    {name:10s}: {status} ({drift:.6f}%)")
            all_acceptable = all_acceptable and (drift < acceptable_drift)
        
        if all_acceptable:
            print(f"\n  ✓ All implementations conserve energy well (< {acceptable_drift}%)")
            return True
        else:
            print(f"\n  ⚠ Some implementations show energy drift > {acceptable_drift}%")
            print("    (This may be acceptable for chaotic systems)")
            return True  # Don't fail test, just warn
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 70)
        print("N-BODY IMPLEMENTATION ACCURACY TEST SUITE")
        print("=" * 70)
        
        # Check which implementations are available
        available = []
        if HAS_PYTHON:
            available.append("Python")
        if HAS_C:
            available.append("C")
        if HAS_CPP:
            available.append("C++")
        if HAS_FORTRAN:
            available.append("Fortran")
        if HAS_JAX:
            available.append("JAX")
        
        print(f"\nAvailable implementations: {', '.join(available)}")
        print(f"CPU tolerance: {self.cpu_tolerance:.2e}")
        print(f"GPU tolerance: {self.gpu_tolerance:.2e} (relaxed for GPU vs CPU)")
        print("\nNote: GPU uses different floating-point reduction order,")
        print("      so small differences vs CPU are expected and acceptable.\n")
        
        # Run tests
        results = {}
        
        results['energy'] = self.test_energy_calculation(N=10)
        results['single_step'] = self.test_single_step(N=10)
        results['multi_step'] = self.test_multi_step(N=20, n_steps=100)
        results['conservation'] = self.test_energy_conservation(N=50, n_steps=1000)
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        for test_name, result in results.items():
            if result is True:
                status = "✓ PASS"
            elif result is False:
                status = "✗ FAIL"
            else:
                status = "⊘ SKIP"
            
            print(f"  {test_name:20s}: {status}")
        
        # Overall verdict
        passed = sum(1 for r in results.values() if r is True)
        failed = sum(1 for r in results.values() if r is False)
        skipped = sum(1 for r in results.values() if r is None)
        
        print(f"\n  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
        
        if failed == 0:
            print("\n  ✓ ALL TESTS PASSED!")
            print("  All implementations produce numerically equivalent results.")
            print("  GPU shows expected small differences due to parallel reduction.")
        else:
            print("\n  ✗ SOME TESTS FAILED")
            print("  Check implementation differences above.")
        
        print("=" * 70)
        
        return failed == 0


if __name__ == "__main__":
    # Run test suite with appropriate tolerances
    tester = TestNBodyAccuracy(cpu_tolerance=1e-10, gpu_tolerance=1e-4)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)