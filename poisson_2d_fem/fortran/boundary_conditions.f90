module boundary_conditions
    use mesh_types
    implicit none
    
contains
    
    !======================================================================
    ! Apply homogeneous Dirichlet BC: u = 0 on boundary
    ! Method: Zero out rows/columns of K, set diagonal to 1, set F to 0
    !======================================================================
    subroutine apply_dirichlet_zero(mesh, K, F)
        type(mesh_t), intent(in) :: mesh
        real(dp), intent(inout) :: K(:,:), F(:)
        
        integer :: i, node
        
        do i = 1, mesh%n_boundary
            node = mesh%boundary(i)
            
            ! Zero out row and column
            K(node, :) = 0.0_dp
            K(:, node) = 0.0_dp
            
            ! Set diagonal to 1
            K(node, node) = 1.0_dp
            
            ! Set RHS to 0
            F(node) = 0.0_dp
        end do
        
    end subroutine apply_dirichlet_zero
    
    !======================================================================
    ! Apply non-homogeneous Dirichlet BC: u = g(x,y) on boundary
    !======================================================================
    subroutine apply_dirichlet_nonzero(mesh, K, F, g_func)
        type(mesh_t), intent(in) :: mesh
        real(dp), intent(inout) :: K(:,:), F(:)
        
        interface
            function g_func(x, y) result(val)
                import :: dp
                real(dp), intent(in) :: x, y
                real(dp) :: val
            end function g_func
        end interface
        
        integer :: i, node
        real(dp) :: x, y, g_val
        
        do i = 1, mesh%n_boundary
            node = mesh%boundary(i)
            
            x = mesh%nodes(node, 1)
            y = mesh%nodes(node, 2)
            g_val = g_func(x, y)
            
            ! Modify RHS for non-zero BC
            F = F - K(:, node) * g_val
            
            ! Zero out row and column
            K(node, :) = 0.0_dp
            K(:, node) = 0.0_dp
            
            ! Set diagonal and RHS
            K(node, node) = 1.0_dp
            F(node) = g_val
        end do
        
    end subroutine apply_dirichlet_nonzero
    
end module boundary_conditions