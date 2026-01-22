/*
 * 1D FEM Assembly in C++ with OpenMP parallelization
 * 
 * Compile with pybind11:
 *   c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp \
 *       $(python3 -m pybind11 --includes) \
 *       fem_assembly.cpp -o fem_cpp$(python3-config --extension-suffix) -lgomp
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <omp.h>

namespace py = pybind11;

py::tuple assemble_system(int n, py::array_t<double> f_vals_array) {
    // Get array info
    auto f_vals_buf = f_vals_array.request();
    double* f_vals = static_cast<double*>(f_vals_buf.ptr);
    
    // Allocate output arrays
    auto K = py::array_t<double>({n, n});
    auto F = py::array_t<double>(n);
    
    auto K_buf = K.request();
    auto F_buf = F.request();
    double* K_ptr = static_cast<double*>(K_buf.ptr);
    double* F_ptr = static_cast<double*>(F_buf.ptr);
    
    // Initialize
    std::memset(K_ptr, 0, n * n * sizeof(double));
    std::memset(F_ptr, 0, n * sizeof(double));
    
    const double h = 1.0 / n;
    const double k_local = 1.0 / h;
    
    // Assemble stiffness matrix with OpenMP
    #pragma omp parallel for
    for (int e = 1; e <= n; e++) {
        int i_left = e - 1;
        int i_right = e;
        
        if (i_left > 0) {
            int idx_left = i_left - 1;
            int idx_right = i_right - 1;
            
            #pragma omp atomic
            K_ptr[idx_left * n + idx_left] += k_local;
            
            #pragma omp atomic
            K_ptr[idx_left * n + idx_right] += -k_local;
            
            #pragma omp atomic
            K_ptr[idx_right * n + idx_left] += -k_local;
        }
        
        int idx_right = i_right - 1;
        #pragma omp atomic
        K_ptr[idx_right * n + idx_right] += k_local;
    }
    
    // Assemble load vector with OpenMP
    #pragma omp parallel for
    for (int i = 1; i < n; i++) {
        F_ptr[i - 1] = (h / 2.0) * (f_vals[i - 1] + f_vals[i + 1]);
    }
    
    F_ptr[n - 1] = (h / 2.0) * f_vals[n - 1];
    
    return py::make_tuple(K, F);
}

py::tuple assemble_system_serial(int n, py::array_t<double> f_vals_array) {
    auto f_vals_buf = f_vals_array.request();
    double* f_vals = static_cast<double*>(f_vals_buf.ptr);
    
    auto K = py::array_t<double>({n, n});
    auto F = py::array_t<double>(n);
    
    auto K_buf = K.request();
    auto F_buf = F.request();
    double* K_ptr = static_cast<double*>(K_buf.ptr);
    double* F_ptr = static_cast<double*>(F_buf.ptr);
    
    std::memset(K_ptr, 0, n * n * sizeof(double));
    std::memset(F_ptr, 0, n * sizeof(double));
    
    const double h = 1.0 / n;
    const double k_local = 1.0 / h;
    
    // Serial assembly
    for (int e = 1; e <= n; e++) {
        int i_left = e - 1;
        int i_right = e;
        
        if (i_left > 0) {
            int idx_left = i_left - 1;
            int idx_right = i_right - 1;
            
            K_ptr[idx_left * n + idx_left] += k_local;
            K_ptr[idx_left * n + idx_right] += -k_local;
            K_ptr[idx_right * n + idx_left] += -k_local;
        }
        
        int idx_right = i_right - 1;
        K_ptr[idx_right * n + idx_right] += k_local;
    }
    
    for (int i = 1; i < n; i++) {
        F_ptr[i - 1] = (h / 2.0) * (f_vals[i - 1] + f_vals[i + 1]);
    }
    F_ptr[n - 1] = (h / 2.0) * f_vals[n - 1];
    
    return py::make_tuple(K, F);
}

PYBIND11_MODULE(fem_cpp, m) {
    m.doc() = "1D FEM assembly with C++ and OpenMP";
    m.def("assemble_system", &assemble_system, 
          "Assemble stiffness matrix and load vector (parallel)");
    m.def("assemble_system_serial", &assemble_system_serial,
          "Assemble stiffness matrix and load vector (serial)");
}