module mesh_types
    !f2py skip ::  ! Don't wrap this module
    implicit none
    integer, parameter :: dp = selected_real_kind(15, 307)
    
    type :: mesh_t
        integer :: n_nodes                       ! Number of nodes
        integer :: n_elements                    ! Number of triangles
        integer :: n_boundary                    ! Number of boundary nodes
        real(dp), allocatable :: nodes(:,:)      ! (n_nodes, 2) - coordinates
        integer, allocatable :: elements(:,:)    ! (n_elements, 3) - connectivity
        integer, allocatable :: boundary(:)      ! (n_boundary) - boundary node IDs
    end type mesh_t
    
contains
    
    subroutine mesh_init(mesh, n_nodes, n_elements, n_boundary)
        type(mesh_t), intent(out) :: mesh
        integer, intent(in) :: n_nodes, n_elements, n_boundary
        
        mesh%n_nodes = n_nodes
        mesh%n_elements = n_elements
        mesh%n_boundary = n_boundary
        
        allocate(mesh%nodes(n_nodes, 2))
        allocate(mesh%elements(n_elements, 3))
        allocate(mesh%boundary(n_boundary))
    end subroutine mesh_init
    
    subroutine mesh_cleanup(mesh)
        type(mesh_t), intent(inout) :: mesh
        
        if (allocated(mesh%nodes)) deallocate(mesh%nodes)
        if (allocated(mesh%elements)) deallocate(mesh%elements)
        if (allocated(mesh%boundary)) deallocate(mesh%boundary)
    end subroutine mesh_cleanup
    
end module mesh_types