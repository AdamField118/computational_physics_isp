/*
 * 1D FEM Assembly in C++ (receives pre-allocated arrays like C)
 * 
 * CRITICAL: Takes pre-allocated numpy arrays as input (like C does)
 * Arrays are already zeroed by NumPy - just write values!
 * 
 * Compile with pybind11:
 *   c++ -O3 -Wall -shared -std=c++11 -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       fem_assembly.cpp -o fem_cpp$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void assemble_system(
    int n,
    py::array_t<double> f_vals_array,
    py::array_t<double> K_array,
    py::array_t<double> F_array
) {
    // Get pointers to pre-allocated arrays (already zeroed by NumPy!)
    auto f_vals_buf = f_vals_array.request();
    auto K_buf = K_array.request();
    auto F_buf = F_array.request();
    
    double* f_vals = static_cast<double*>(f_vals_buf.ptr);
    double* K_ptr = static_cast<double*>(K_buf.ptr);
    double* F_ptr = static_cast<double*>(F_buf.ptr);
    
    const double h = 1.0 / n;
    const double k_local = 1.0 / h;
    
    // K and F are ALREADY ZEROED by NumPy!
    // Just write values directly - NO ZEROING NEEDED!
    
    // Assemble load vector
    for (int i = 1; i < n; i++) {
        F_ptr[i-1] = (h / 2.0) * (f_vals[i-1] + f_vals[i+1]);
    }
    F_ptr[n-1] = (h / 2.0) * f_vals[n-1];
    
    // Assemble stiffness matrix using pointer arithmetic
    K_ptr[0] = k_local;
    
    double* Kprev = K_ptr;
    double* Kcur = K_ptr + n;
    
    for (int e = 2; e <= n; e++) {
        int i = e - 2;
        
        Kprev[i]   += k_local;
        Kprev[i+1] -= k_local;
        Kcur[i]    -= k_local;
        Kcur[i+1]  += k_local;
        
        Kprev = Kcur;
        Kcur += n;
    }
}

PYBIND11_MODULE(fem_cpp, m) {
    m.doc() = "1D FEM assembly (receives pre-allocated arrays like C)";
    m.def("assemble_system", &assemble_system,
          "Assemble into pre-allocated K and F arrays");
}