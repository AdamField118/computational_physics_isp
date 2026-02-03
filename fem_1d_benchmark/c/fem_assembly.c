/*
 * 1D FEM Assembly in C (Optimized - trust pre-zeroed arrays)
 * 
 * Key insight: NumPy already zeroes K and F with np.zeros()
 * NumPy uses calloc-like optimization (OS lazy zero pages)
 * So we DON'T need to zero again - just write the non-zero entries!
 * 
 * Compile as shared library:
 *   gcc -O3 -fPIC -shared -o fem_c.so fem_assembly.c
 */

#include <stdio.h>
#include <stdlib.h>

void assemble_system(int n, const double* f_vals, double* K, double* F) {
    const double h = 1.0 / n;
    const double k_local = 1.0 / h;
    
    // OPTIMIZATION: K and F are already zeroed by NumPy!
    // NumPy's np.zeros() uses OS-optimized allocation (like calloc)
    // Don't waste time re-zeroing - just write the values we need
    
    // Assemble load vector (direct assignment - F is already zero)
    for (int i = 1; i < n; i++) {
        F[i-1] = (h / 2.0) * (f_vals[i-1] + f_vals[i+1]);
    }
    F[n-1] = (h / 2.0) * f_vals[n-1];
    
    // Assemble stiffness matrix using pointer arithmetic
    // Handle first element
    K[0] = k_local;
    
    // Main loop with pointer arithmetic (no index calculations)
    double* Kprev = K;           // Points to row (i-1)
    double* Kcur = K + n;        // Points to row i
    
    for (int e = 2; e <= n; e++) {
        int i = e - 2;
        
        // Write directly to matrix (it's already zero)
        // Use += since these might be touched by multiple elements
        Kprev[i]   += k_local;   // K[i,i]
        Kprev[i+1] += -k_local;  // K[i,i+1]  (could use -= but showing it clearly)
        Kcur[i]    += -k_local;  // K[i+1,i]
        Kcur[i+1]  += k_local;   // K[i+1,i+1]
        
        // Advance row pointers (one addition vs 4 multiplications!)
        Kprev = Kcur;
        Kcur += n;
    }
}

#ifdef TEST_MAIN
int main() {
    int n = 10;
    double h = 1.0 / n;
    
    double* f_vals = (double*)malloc((n + 1) * sizeof(double));
    
    // Use calloc for standalone C code!
    double* K = (double*)calloc(n * n, sizeof(double));  // OS-optimized zeroing
    double* F = (double*)calloc(n, sizeof(double));      // OS-optimized zeroing
    
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