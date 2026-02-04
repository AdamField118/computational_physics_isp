module fem_assembly
    !f2py skip ::  ! Don't wrap this module
    use mesh_types
    use reference_element
    implicit none
    
contains
    
    !======================================================================
    ! Compute affine transformation F_K: ref -> physical element K
    ! F_K(xi_hat) = B_K * xi_hat + b_vec
    !======================================================================
    subroutine compute_affine_map(mesh, elem_id, B_K, b_vec, det_B)
        type(mesh_t), intent(in) :: mesh
        integer, intent(in) :: elem_id
        real(dp), intent(out) :: B_K(2, 2), b_vec(2), det_B
        
        integer :: v1, v2, v3
        real(dp) :: x1(2), x2(2), x3(2)
        
        ! Get vertices of element (1-indexed)
        v1 = mesh%elements(elem_id, 1)
        v2 = mesh%elements(elem_id, 2)
        v3 = mesh%elements(elem_id, 3)
        
        x1 = mesh%nodes(v1, :)
        x2 = mesh%nodes(v2, :)
        x3 = mesh%nodes(v3, :)
        
        ! B_K = [x2-x1, x3-x1] (column vectors)
        B_K(:, 1) = x2 - x1
        B_K(:, 2) = x3 - x1
        
        b_vec = x1
        
        ! Jacobian determinant = |det(B_K)| = 2 * Area(K)
        det_B = B_K(1,1)*B_K(2,2) - B_K(1,2)*B_K(2,1)
    end subroutine compute_affine_map
    
    !======================================================================
    ! Assemble global stiffness matrix (sparse format)
    ! K(i,j) = sum_K int_K grad_phi_i \cdot grad_phi_j dx
    !======================================================================
    subroutine assemble_stiffness(mesh, K_global)
        type(mesh_t), intent(in) :: mesh
        real(dp), intent(out) :: K_global(mesh%n_nodes, mesh%n_nodes)
        
        integer :: elem, i, j, loc_i, loc_j, glob_i, glob_j
        real(dp) :: B_K(2, 2), b_vec(2), det_B, B_inv_T(2, 2)
        real(dp) :: K_ref(3, 3), K_elem(3, 3)
        real(dp) :: grad_i_ref(2), grad_j_ref(2)
        real(dp) :: grad_i_phys(2), grad_j_phys(2)
        real(dp) :: G_K(2, 2)
        
        ! Initialize
        K_global = 0.0_dp
        
        ! Compute reference stiffness (same for all elements)
        call compute_reference_stiffness(K_ref)
        
        ! Loop over elements
        do elem = 1, mesh%n_elements
            
            ! Affine map: ref -> physical
            call compute_affine_map(mesh, elem, B_K, b_vec, det_B)
            
            ! B_inv_T = (B_K^{-1})^T (for gradient transformation)
            call invert_2x2(B_K, B_inv_T)
            B_inv_T = transpose(B_inv_T)
            
            ! Metric tensor: G_K = B_K^{-1} * (B_K^{-1})^T
            G_K = matmul(B_inv_T, transpose(B_inv_T))
            
            ! Local stiffness: K_elem(i,j) = |det_B| * grad_i^T * G_K * grad_j
            do loc_i = 1, 3
                grad_i_ref = grad_phi_ref(loc_i)
                do loc_j = 1, 3
                    grad_j_ref = grad_phi_ref(loc_j)
                    
                    ! Transform gradients: grad_phys = B_inv_T * grad_ref
                    grad_i_phys = matmul(B_inv_T, grad_i_ref)
                    grad_j_phys = matmul(B_inv_T, grad_j_ref)
                    
                    ! K_elem = |det_B| * grad_i \cdot grad_j
                    K_elem(loc_i, loc_j) = abs(det_B) * dot_product(grad_i_phys, grad_j_phys)
                end do
            end do
            
            ! Add to global matrix (assembly process)
            do loc_i = 1, 3
                glob_i = mesh%elements(elem, loc_i)
                do loc_j = 1, 3
                    glob_j = mesh%elements(elem, loc_j)
                    K_global(glob_i, glob_j) = K_global(glob_i, glob_j) + K_elem(loc_i, loc_j)
                end do
            end do
            
        end do
        
    end subroutine assemble_stiffness
    
    !======================================================================
    ! Assemble load vector F(i) = int_Omega f * phi_i dx
    !======================================================================
    subroutine assemble_load(mesh, f_func, F_global)
        type(mesh_t), intent(in) :: mesh
        real(dp), intent(out) :: F_global(mesh%n_nodes)
        
        interface
            function f_func(x, y) result(val)
                import :: dp
                real(dp), intent(in) :: x, y
                real(dp) :: val
            end function f_func
        end interface
        
        integer :: elem, loc_i, glob_i, q
        real(dp) :: B_K(2, 2), b_vec(2), det_B
        real(dp) :: xi_q, eta_q, w_q, f_val, phi_val
        real(dp) :: x_q, y_q, xi_hat(2), x_phys(2)
        real(dp), allocatable :: xi(:), eta(:), weights(:)
        integer :: n_quad
        
        ! Initialize
        F_global = 0.0_dp
        
        ! Quadrature rule (order 2 for f(x,y) up to quadratic)
        call gauss_points_triangle(2, xi, eta, weights, n_quad)
        
        ! Loop over elements
        do elem = 1, mesh%n_elements
            
            call compute_affine_map(mesh, elem, B_K, b_vec, det_B)
            
            ! Numerical integration over reference element
            do q = 1, n_quad
                xi_q = xi(q)
                eta_q = eta(q)
                w_q = weights(q)
                
                ! Map quadrature point to physical element
                xi_hat = [xi_q, eta_q]
                x_phys = matmul(B_K, xi_hat) + b_vec
                x_q = x_phys(1)
                y_q = x_phys(2)
                
                ! Evaluate f at quadrature point
                f_val = f_func(x_q, y_q)
                
                ! Add contribution to each basis function
                do loc_i = 1, 3
                    phi_val = phi_ref(loc_i, xi_q, eta_q)
                    glob_i = mesh%elements(elem, loc_i)
                    
                    F_global(glob_i) = F_global(glob_i) + &
                        abs(det_B) * w_q * f_val * phi_val
                end do
            end do
            
        end do
        
        deallocate(xi, eta, weights)
        
    end subroutine assemble_load
    
    !======================================================================
    ! Helper: Invert 2x2 matrix
    !======================================================================
    pure subroutine invert_2x2(A, A_inv)
        real(dp), intent(in) :: A(2, 2)
        real(dp), intent(out) :: A_inv(2, 2)
        real(dp) :: det
        
        det = A(1,1)*A(2,2) - A(1,2)*A(2,1)
        
        A_inv(1,1) = A(2,2) / det
        A_inv(1,2) = -A(1,2) / det
        A_inv(2,1) = -A(2,1) / det
        A_inv(2,2) = A(1,1) / det
    end subroutine invert_2x2
    
end module fem_assembly