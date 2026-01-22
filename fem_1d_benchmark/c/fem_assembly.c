/*
 * 1D FEM Assembly in C
 * 
 * Compile as shared library:
 *   gcc -O3 -fPIC -shared -o fem_c.so fem_assembly.c
 * 
 * Use in Python with ctypes:
 *   from ctypes import CDLL, c_int, POINTER, c_double
 *   lib = CDLL('./fem_c.so')
 *   lib.assemble_system.argtypes = [c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Assemble stiffness matrix and load vector for 1D FEM
 * 
 * @param n       Number of elements
 * @param f_vals  Source function values at nodes (length n+1)
 * @param K       Output stiffness matrix (n x n, row-major)
 * @param F       Output load vector (length n)
 */

void assemble_system(int n, const double* f_vals, double* K, double* F) {
    const double h = 1.0 / n;
    const double k_local = 1.0 / h;
    
    // Initialize arrays
    memset(K, 0, n * n * sizeof(double));
    memset(F, 0, n * sizeof(double));
    
    // Assemble stiffness matrix element by element
    // Element e connects nodes e-1 and e (0-based indexing)
    for (int e = 1; e <= n; e++) {
        int i_left = e - 1;
        int i_right = e;
        
        // Map to reduced system (excluding u_0 = 0)
        if (i_left > 0) {
            int idx_left = i_left - 1;   // 0-based index in reduced system
            int idx_right = i_right - 1;
            
            // K[idx_left, idx_left] += k_local
            K[idx_left * n + idx_left] += k_local;
            
            // K[idx_left, idx_right] += -k_local
            K[idx_left * n + idx_right] += -k_local;
            
            // K[idx_right, idx_left] += -k_local
            K[idx_right * n + idx_left] += -k_local;
        }
        
        // K[idx_right, idx_right] += k_local
        int idx_right = i_right - 1;
        K[idx_right * n + idx_right] += k_local;
    }
    
    // Assemble load vector using trapezoidal rule
    // Interior nodes
    for (int i = 1; i < n; i++) {
        F[i - 1] = (h / 2.0) * (f_vals[i - 1] + f_vals[i + 1]);
    }
    
    // Last node
    F[n - 1] = (h / 2.0) * f_vals[n - 1];
}


/*
 * Python wrapper-friendly version that allocates memory
 * 
 * Returns pointers that Python can read, but Python is responsible
 * for freeing them (or use the simpler version above with preallocated arrays)
 */
void assemble_system_alloc(int n, const double* f_vals, double** K_out, double** F_out) {
    // Allocate
    double* K = (double*)malloc(n * n * sizeof(double));
    double* F = (double*)malloc(n * sizeof(double));
    
    if (!K || !F) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    // Call the assembly routine
    assemble_system(n, f_vals, K, F);
    
    // Return pointers
    *K_out = K;
    *F_out = F;
}


/*
 * Helper: Free allocated memory
 */
void free_arrays(double* K, double* F) {
    if (K) free(K);
    if (F) free(F);
}


/* Example main for standalone testing */
#ifdef TEST_MAIN
int main() {
    int n = 10;
    double h = 1.0 / n;
    
    // Allocate arrays
    double* f_vals = (double*)malloc((n + 1) * sizeof(double));
    double* K = (double*)malloc(n * n * sizeof(double));
    double* F = (double*)malloc(n * sizeof(double));
    
    // Set up source term: f(x) = 2 - 6x
    for (int i = 0; i <= n; i++) {
        double x = i * h;
        f_vals[i] = 2.0 - 6.0 * x;
    }
    
    // Assemble
    assemble_system(n, f_vals, K, F);
    
    // Print a few entries
    printf("K[0,0] = %f (expected: %f)\n", K[0], 2.0/h);
    printf("K[0,1] = %f (expected: %f)\n", K[1], -1.0/h);
    printf("F[0] = %f\n", F[0]);
    
    // Cleanup
    free(f_vals);
    free(K);
    free(F);
    
    return 0;
}
#endif