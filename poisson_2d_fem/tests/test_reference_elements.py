# tests/test_reference_element.py
def test_reference_stiffness():
    """Verify reference stiffness matrix"""
    K_ref = fem_fortran.compute_reference_stiffness()
    
    # Expected from textbook
    K_expected = 0.5 * np.array([
        [ 2, -1, -1],
        [-1,  1,  0],
        [-1,  0,  1]
    ])
    
    np.testing.assert_allclose(K_ref, K_expected, rtol=1e-12)

def test_basis_partition_unity():
    """Verify sum of basis functions = 1"""
    xi, eta = 0.3, 0.4
    sum_phi = sum(fem_fortran.phi_ref(i, xi, eta) for i in [1,2,3])
    assert abs(sum_phi - 1.0) < 1e-14