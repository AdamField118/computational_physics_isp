module reference_element
    use mesh_types, only: dp
    implicit none
    
    ! Reference triangle: vertices (0,0), (1,0), (0,1)
    
contains
    
    !======================================================================
    ! P1 basis functions in barycentric coordinates
    !======================================================================
    pure function phi_ref(node, xi, eta) result(val)
        integer, intent(in) :: node
        real(dp), intent(in) :: xi, eta
        real(dp) :: val
        
        select case(node)
        case(1)
            val = 1.0_dp - xi - eta   ! lambda_1
        case(2)
            val = xi                   ! lambda_2
        case(3)
            val = eta                  ! lambda_3
        end select
    end function phi_ref
    
    !======================================================================
    ! Gradient of P1 basis functions (constant on reference element)
    !======================================================================
    pure function grad_phi_ref(node) result(grad)
        integer, intent(in) :: node
        real(dp) :: grad(2)
        
        select case(node)
        case(1)
            grad = [-1.0_dp, -1.0_dp]
        case(2)
            grad = [1.0_dp, 0.0_dp]
        case(3)
            grad = [0.0_dp, 1.0_dp]
        end select
    end function grad_phi_ref
    
    !======================================================================
    ! Reference stiffness matrix (constant for all P1 triangles)
    ! K_ref(i,j) = int_ref grad_phi_i · grad_phi_j dA
    !======================================================================
    subroutine compute_reference_stiffness(K_ref)
        real(dp), intent(out) :: K_ref(3, 3)
        integer :: i, j
        real(dp) :: grad_i(2), grad_j(2)
        
        ! Area of reference triangle = 1/2
        do i = 1, 3
            grad_i = grad_phi_ref(i)
            do j = 1, 3
                grad_j = grad_phi_ref(j)
                K_ref(i, j) = 0.5_dp * dot_product(grad_i, grad_j)
            end do
        end do
        
        ! Result:
        ! K_ref = 0.5 * [[ 2, -1, -1],
        !                [-1,  1,  0],
        !                [-1,  0,  1]]
    end subroutine compute_reference_stiffness
    
    !======================================================================
    ! Gauss quadrature on reference triangle
    ! 1-point: centroid, weight = 1/2 (exact for degree ≤ 1)
    ! 3-point: degree ≤ 2 (needed for f(x,y) non-constant)
    !======================================================================
    subroutine gauss_points_triangle(order, xi, eta, weights, n_quad)
        integer, intent(in) :: order
        real(dp), allocatable, intent(out) :: xi(:), eta(:), weights(:)
        integer, intent(out) :: n_quad
        
        if (order == 1) then
            ! 1-point rule (centroid)
            n_quad = 1
            allocate(xi(1), eta(1), weights(1))
            xi = [1.0_dp/3.0_dp]
            eta = [1.0_dp/3.0_dp]
            weights = [0.5_dp]  ! Area = 1/2
            
        else if (order == 2) then
            ! 3-point rule (midpoints of edges)
            n_quad = 3
            allocate(xi(3), eta(3), weights(3))
            xi = [0.5_dp, 0.5_dp, 0.0_dp]
            eta = [0.0_dp, 0.5_dp, 0.5_dp]
            weights = [1.0_dp/6.0_dp, 1.0_dp/6.0_dp, 1.0_dp/6.0_dp]
        end if
    end subroutine gauss_points_triangle
    
end module reference_element