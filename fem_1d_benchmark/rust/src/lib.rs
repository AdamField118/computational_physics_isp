use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Mutex;

/// Assemble stiffness matrix and load vector (parallel version with Rayon)
#[pyfunction]
fn assemble_system<'py>(
    py: Python<'py>,
    n: usize,
    f_vals: PyReadonlyArray1<f64>,
) -> (&'py PyArray2<f64>, &'py PyArray1<f64>) {
    let f_vals = f_vals.as_slice().unwrap();
    
    let h = 1.0 / n as f64;
    let k_local = 1.0 / h;
    
    // Allocate output arrays
    let mut k_data = vec![0.0; n * n];
    let mut f_data = vec![0.0; n];
    
    // Use Mutex for thread-safe updates
    let k_mutex = Mutex::new(&mut k_data);
    
    // Parallel assembly of stiffness matrix
    (1..=n).into_par_iter().for_each(|e| {
        let i_left = e - 1;
        let i_right = e;
        
        let mut updates = Vec::new();
        
        if i_left > 0 {
            let idx_left = i_left - 1;
            let idx_right = i_right - 1;
            
            updates.push((idx_left * n + idx_left, k_local));
            updates.push((idx_left * n + idx_right, -k_local));
            updates.push((idx_right * n + idx_left, -k_local));
        }
        
        let idx_right = i_right - 1;
        updates.push((idx_right * n + idx_right, k_local));
        
        // Apply updates atomically
        let mut k_guard = k_mutex.lock().unwrap();
        for (idx, val) in updates {
            k_guard[idx] += val;
        }
    });
    
    // Parallel assembly of load vector
    f_data.par_iter_mut().enumerate().for_each(|(i, f)| {
        if i < n - 1 {
            *f = (h / 2.0) * (f_vals[i] + f_vals[i + 2]);
        } else {
            // Last node uses only left contribution (Neumann BC on right)
            *f = (h / 2.0) * f_vals[n - 1];
        }
    });
    
    // Convert to numpy arrays
    let k_array = PyArray2::from_vec2(py, &vec_to_matrix(k_data, n)).unwrap();
    let f_array = PyArray1::from_vec(py, f_data);
    
    (k_array, f_array)
}

/// Serial version for comparison
#[pyfunction]
fn assemble_system_serial<'py>(
    py: Python<'py>,
    n: usize,
    f_vals: PyReadonlyArray1<f64>,
) -> (&'py PyArray2<f64>, &'py PyArray1<f64>) {
    let f_vals = f_vals.as_slice().unwrap();
    
    let h = 1.0 / n as f64;
    let k_local = 1.0 / h;
    
    let mut k_data = vec![0.0; n * n];
    let mut f_data = vec![0.0; n];
    
    // Serial assembly
    for e in 1..=n {
        let i_left = e - 1;
        let i_right = e;
        
        if i_left > 0 {
            let idx_left = i_left - 1;
            let idx_right = i_right - 1;
            
            k_data[idx_left * n + idx_left] += k_local;
            k_data[idx_left * n + idx_right] += -k_local;
            k_data[idx_right * n + idx_left] += -k_local;
        }
        
        let idx_right = i_right - 1;
        k_data[idx_right * n + idx_right] += k_local;
    }
    
    for i in 0..(n - 1) {
        f_data[i] = (h / 2.0) * (f_vals[i] + f_vals[i + 2]);
    }
    // Last node uses only left contribution (Neumann BC on right)
    f_data[n - 1] = (h / 2.0) * f_vals[n - 1];
    
    let k_array = PyArray2::from_vec2(py, &vec_to_matrix(k_data, n)).unwrap();
    let f_array = PyArray1::from_vec(py, f_data);
    
    (k_array, f_array)
}

/// Optimized parallel version using partitioning
#[pyfunction]
fn assemble_system_optimized<'py>(
    py: Python<'py>,
    n: usize,
    f_vals: PyReadonlyArray1<f64>,
) -> (&'py PyArray2<f64>, &'py PyArray1<f64>) {
    let f_vals = f_vals.as_slice().unwrap();
    
    let h = 1.0 / n as f64;
    let k_local = 1.0 / h;
    
    // Parallel assembly using reduce pattern
    let k_data: Vec<f64> = (0..n * n)
        .into_par_iter()
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            
            // Compute contribution to K[i][j]
            let mut val = 0.0;
            
            // Element e contributes to K[i][j] if it connects nodes i and j
            // Element e connects nodes e-1 and e (in 0-indexed system, after removing u_0)
            
            if i == j {
                // Diagonal: contributions from elements connecting to node i+1
                if i + 1 >= 1 {
                    val += k_local; // Right element
                }
                if i + 2 <= n {
                    val += k_local; // Left element
                }
            } else if j == i + 1 || i == j + 1 {
                // Off-diagonal: adjacent nodes
                val = -k_local;
            }
            
            val
        })
        .collect();
    
    // Parallel assembly of load vector
    let f_data: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            if i < n - 1 {
                (h / 2.0) * (f_vals[i] + f_vals[i + 2])
            } else {
                // Last node uses only left contribution (Neumann BC on right)
                (h / 2.0) * f_vals[n - 1]
            }
        })
        .collect();
    
    let k_array = PyArray2::from_vec2(py, &vec_to_matrix(k_data, n)).unwrap();
    let f_array = PyArray1::from_vec(py, f_data);
    
    (k_array, f_array)
}

fn vec_to_matrix(vec: Vec<f64>, n: usize) -> Vec<Vec<f64>> {
    vec.chunks(n).map(|chunk| chunk.to_vec()).collect()
}

#[pymodule]
fn fem_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(assemble_system, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_system_serial, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_system_optimized, m)?)?;
    Ok(())
}