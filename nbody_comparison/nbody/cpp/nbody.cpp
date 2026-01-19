/*
 * N-Body Gravitational Simulation - C++ Implementation
 * Adam Field - Computational Physics ISP
 */

#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class NBodySimulator {
private:
    int n_particles;
    double G;
    double softening;
    double dt;
    
    std::vector<double> positions;
    std::vector<double> velocities;
    std::vector<double> masses;
    std::vector<double> accelerations;
    
public:
    NBodySimulator(int n, double g = 1.0, double soft = 0.1, double timestep = 0.01)
        : n_particles(n), G(g), softening(soft), dt(timestep) {
        positions.resize(n * 3);
        velocities.resize(n * 3);
        masses.resize(n);
        accelerations.resize(n * 3);
    }
    
    void computeAcceleration() {
        // Zero out accelerations
        std::fill(accelerations.begin(), accelerations.end(), 0.0);
        
        // PARALLEL O(N²) pairwise force calculation
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < n_particles; i++) {
            double ax = 0.0, ay = 0.0, az = 0.0;
            
            // Cache particle i's position
            double px_i = positions[i*3 + 0];
            double py_i = positions[i*3 + 1];
            double pz_i = positions[i*3 + 2];
            
            for (int j = 0; j < n_particles; j++) {
                if (i != j) {
                    // Displacement vector
                    double dx = positions[j*3 + 0] - px_i;
                    double dy = positions[j*3 + 1] - py_i;
                    double dz = positions[j*3 + 2] - pz_i;
                    
                    // Distance with softening
                    double dist_sq = dx*dx + dy*dy + dz*dz + softening*softening;
                    double dist = std::sqrt(dist_sq);
                    
                    // 1/r³ term
                    double inv_dist_cubed = 1.0 / (dist * dist_sq);
                    
                    // Accumulate acceleration
                    double factor = G * masses[j] * inv_dist_cubed;
                    ax += factor * dx;
                    ay += factor * dy;
                    az += factor * dz;
                }
            }
            
            accelerations[i*3 + 0] = ax;
            accelerations[i*3 + 1] = ay;
            accelerations[i*3 + 2] = az;
        }
    }
    
    void velocityVerletStep() {
        std::vector<double> acc_current = accelerations;
        
        // Compute current acceleration
        computeAcceleration();
        acc_current = accelerations;
        
        // Update positions
        for (int i = 0; i < n_particles * 3; i++) {
            positions[i] += velocities[i] * dt + 0.5 * acc_current[i] * dt * dt;
        }
        
        // Compute new acceleration
        computeAcceleration();
        
        // Update velocities
        for (int i = 0; i < n_particles * 3; i++) {
            velocities[i] += 0.5 * (acc_current[i] + accelerations[i]) * dt;
        }
    }
    
    std::pair<py::array_t<double>, py::array_t<double>> 
    simulate(py::array_t<double> pos_init, 
             py::array_t<double> vel_init,
             py::array_t<double> mass_arr,
             int n_steps) {
        
        // Copy initial conditions
        auto pos_buf = pos_init.request();
        auto vel_buf = vel_init.request();
        auto mass_buf = mass_arr.request();
        
        double* pos_ptr = (double*) pos_buf.ptr;
        double* vel_ptr = (double*) vel_buf.ptr;
        double* mass_ptr = (double*) mass_buf.ptr;
        
        for (int i = 0; i < n_particles * 3; i++) {
            positions[i] = pos_ptr[i];
            velocities[i] = vel_ptr[i];
        }
        for (int i = 0; i < n_particles; i++) {
            masses[i] = mass_ptr[i];
        }
        
        // Time integration
        for (int step = 0; step < n_steps; step++) {
            velocityVerletStep();
        }
        
        // Return final state
        py::array_t<double> pos_out({n_particles, 3});
        py::array_t<double> vel_out({n_particles, 3});
        
        auto pos_out_buf = pos_out.request();
        auto vel_out_buf = vel_out.request();
        
        double* pos_out_ptr = (double*) pos_out_buf.ptr;
        double* vel_out_ptr = (double*) vel_out_buf.ptr;
        
        for (int i = 0; i < n_particles * 3; i++) {
            pos_out_ptr[i] = positions[i];
            vel_out_ptr[i] = velocities[i];
        }
        
        return std::make_pair(pos_out, vel_out);
    }
    
    std::tuple<double, double, double>
    computeEnergy(py::array_t<double> pos_arr,
                  py::array_t<double> vel_arr,
                  py::array_t<double> mass_arr) {
        
        auto pos_buf = pos_arr.request();
        auto vel_buf = vel_arr.request();
        auto mass_buf = mass_arr.request();
        
        double* pos_ptr = (double*) pos_buf.ptr;
        double* vel_ptr = (double*) vel_buf.ptr;
        double* mass_ptr = (double*) mass_buf.ptr;
        
        // Kinetic energy
        double ke = 0.0;
        for (int i = 0; i < n_particles; i++) {
            double v_sq = vel_ptr[i*3 + 0] * vel_ptr[i*3 + 0] +
                         vel_ptr[i*3 + 1] * vel_ptr[i*3 + 1] +
                         vel_ptr[i*3 + 2] * vel_ptr[i*3 + 2];
            ke += 0.5 * mass_ptr[i] * v_sq;
        }
        
        // Potential energy
        double pe = 0.0;
        for (int i = 0; i < n_particles; i++) {
            for (int j = i + 1; j < n_particles; j++) {
                double dx = pos_ptr[j*3 + 0] - pos_ptr[i*3 + 0];
                double dy = pos_ptr[j*3 + 1] - pos_ptr[i*3 + 1];
                double dz = pos_ptr[j*3 + 2] - pos_ptr[i*3 + 2];
                
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz + softening*softening);
                
                pe -= G * mass_ptr[i] * mass_ptr[j] / dist;
            }
        }
        
        return std::make_tuple(ke, pe, ke + pe);
    }
};

PYBIND11_MODULE(nbody_cpp_module, m) {
    py::class_<NBodySimulator>(m, "NBodySimulator")
        .def(py::init<int, double, double, double>(),
             py::arg("n_particles"),
             py::arg("G") = 1.0,
             py::arg("softening") = 0.1,
             py::arg("dt") = 0.01)
        .def("simulate", &NBodySimulator::simulate)
        .def("compute_energy", &NBodySimulator::computeEnergy);
}