module python_interface
    use mesh_types
    use fem_assembly
    use boundary_conditions
    use linear_solver
    implicit none

contains

    ! Default source function f(x,y) = -2*pi^2*sin(pi*x)*sin(pi*y)
    ! This gives exact solution u(x,y) = sin(pi*x)*sin(pi*y)
    function default_source(x, y) result(val)
        real(dp), intent(in) :: x, y
        real(dp) :: val
        real(dp), parameter :: pi = 3.141592653589793_dp
        
        val = -2.0_dp * pi**2 * sin(pi*x) * sin(pi*y)
    end function default_source

    subroutine solve_poisson_2d(nodes, elements, boundary, n_nodes, n_elements, n_boundary, u_solution)
        !f2py intent(in) :: nodes, elements, boundary, n_nodes, n_elements, n_boundary
        !f2py intent(out) :: u_solution
        
        real(dp), intent(in) :: nodes(n_nodes, 2)
        integer, intent(in) :: elements(n_elements, 3)
        integer, intent(in) :: boundary(n_boundary)
        integer, intent(in) :: n_nodes, n_elements, n_boundary
        real(dp), intent(out) :: u_solution(n_nodes)
        
        type(mesh_t) :: mesh
        real(dp), allocatable :: K(:,:), F(:)
        
        ! Initialize mesh from arrays
        mesh%n_nodes = n_nodes
        mesh%n_elements = n_elements
        mesh%n_boundary = n_boundary
        allocate(mesh%nodes(n_nodes, 2))
        allocate(mesh%elements(n_elements, 3))
        allocate(mesh%boundary(n_boundary))
        mesh%nodes = nodes
        mesh%elements = elements
        mesh%boundary = boundary
        
        ! Assemble and solve
        allocate(K(n_nodes, n_nodes), F(n_nodes))
        call assemble_stiffness(mesh, K)
        call assemble_load(mesh, default_source, F)  ! Fixed: added f_func
        call apply_dirichlet_zero(mesh, K, F)
        call solve_system(K, F, u_solution)
        
        ! Cleanup
        deallocate(K, F)
        call mesh_cleanup(mesh)
        
    end subroutine solve_poisson_2d

end module python_interface