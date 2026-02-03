use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;

/// Assemble into pre-allocated arrays (like C does)
/// 
/// Arrays K and F are already zeroed by NumPy - just write values!
#[pyfunction]
fn assemble_system(
    n: usize,
    f_vals: PyReadonlyArray1<f64>,
    mut k_array: PyReadwriteArray2<f64>,
    mut f_array: PyReadwriteArray1<f64>,
) {
    let f_vals = f_vals.as_slice().unwrap();
    
    let h = 1.0 / n as f64;
    let k_local = 1.0 / h;
    
    // Get mutable access to pre-allocated arrays (already zeroed by NumPy!)
    let k_data = k_array.as_slice_mut().unwrap();
    let f_data = f_array.as_slice_mut().unwrap();
    
    // K and F are ALREADY ZEROED by NumPy!
    // Just write values directly - NO ZEROING NEEDED!
    
    // Assemble load vector
    for i in 0..(n - 1) {
        f_data[i] = (h / 2.0) * (f_vals[i] + f_vals[i + 2]);
    }
    f_data[n - 1] = (h / 2.0) * f_vals[n - 1];
    
    // Assemble stiffness matrix
    k_data[0] = k_local;
    
    for e in 2..=n {
        let i = e - 2;
        let row_i = i * n;
        let row_i_plus_1 = (i + 1) * n;
        
        k_data[row_i + i]         += k_local;
        k_data[row_i + (i + 1)]   -= k_local;
        k_data[row_i_plus_1 + i]  -= k_local;
        k_data[row_i_plus_1 + (i + 1)] += k_local;
    }
}

#[pymodule]
fn fem_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(assemble_system, m)?)?;
    Ok(())
}