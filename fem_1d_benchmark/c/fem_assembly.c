/*
 * 1D FEM Assembly in C with OpenMP parallelization
 * 
 * Compile as shared library:
 *   gcc -O3 -fPIC -shared -fopenmp -o fem_c.so fem_assembly.c -lgomp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void assemble_system(int n, const double* f_vals, double* K, double* F) {
    const double h = 1.0 / n;
    const double k_local = 1.0 / h;
    
    // Initialize arrays
    memset(K, 0, n * n * sizeof(double));
    memset(F, 0, n * sizeof(double));
    
    // Parallel assembly of stiffness matrix
    #pragma omp parallel for
    for (int e = 1; e <= n; e++) {
        int i_left = e - 1;
        int i_right = e;
        
        if (i_left > 0) {
            int idx_left = i_left - 1;
            int idx_right = i_right - 1;
            
            #pragma omp atomic
            K[idx_left * n + idx_left] += k_local;
            
            #pragma omp atomic
            K[idx_left * n + idx_right] += -k_local;
            
            #pragma omp atomic
            K[idx_right * n + idx_left] += -k_local;
        }
        
        int idx_right = i_right - 1;
        #pragma omp atomic
        K[idx_right * n + idx_right] += k_local;
    }
    
    // Parallel assembly of load vector
    #pragma omp parallel for
    for (int i = 1; i < n; i++) {
        F[i - 1] = (h / 2.0) * (f_vals[i - 1] + f_vals[i + 1]);
    }
    
    F[n - 1] = (h / 2.0) * f_vals[n - 1];
}

void assemble_system_serial(int n, const double* f_vals, double* K, double* F) {
    const double h = 1.0 / n;
    const double k_local = 1.0 / h;
    
    memset(K, 0, n * n * sizeof(double));
    memset(F, 0, n * sizeof(double));
    
    // Serial assembly
    for (int e = 1; e <= n; e++) {
        int i_left = e - 1;
        int i_right = e;
        
        if (i_left > 0) {
            int idx_left = i_left - 1;
            int idx_right = i_right - 1;
            
            K[idx_left * n + idx_left] += k_local;
            K[idx_left * n + idx_right] += -k_local;
            K[idx_right * n + idx_left] += -k_local;
        }
        
        int idx_right = i_right - 1;
        K[idx_right * n + idx_right] += k_local;
    }
    
    for (int i = 1; i < n; i++) {
        F[i - 1] = (h / 2.0) * (f_vals[i - 1] + f_vals[i + 1]);
    }
    F[n - 1] = (h / 2.0) * f_vals[n - 1];
}

#ifdef TEST_MAIN
int main() {
    int n = 10;
    double h = 1.0 / n;
    
    double* f_vals = (double*)malloc((n + 1) * sizeof(double));
    double* K = (double*)malloc(n * n * sizeof(double));
    double* F = (double*)malloc(n * sizeof(double));
    
    for (int i = 0; i <= n; i++) {
        double x = i * h;
        f_vals[i] = 2.0 - 6.0 * x;
    }
    
    assemble_system(n, f_vals, K, F);
    
    printf("K[0,0] = %f (expected: %f)\n", K[0], 2.0/h);
    printf("K[0,1] = %f (expected: %f)\n", K[1], -1.0/h);
    printf("F[0] = %f\n", F[0]);
    
    free(f_vals);
    free(K);
    free(F);
    
    return 0;
}
#endif