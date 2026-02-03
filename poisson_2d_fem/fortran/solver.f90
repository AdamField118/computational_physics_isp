module linear_solver
    use mesh_types, only: dp
    implicit none
    
contains
    
    !======================================================================
    ! Solve KU = F using LAPACK (LU factorization)
    ! For symmetric positive definite: use DPOSV (Cholesky)
    !======================================================================
    subroutine solve_system(K, F, U)
        real(dp), intent(in) :: K(:,:)
        real(dp), intent(in) :: F(:)
        real(dp), intent(out) :: U(:)
        
        integer :: n, info
        real(dp), allocatable :: K_copy(:,:), F_copy(:)
        
        n = size(F)
        allocate(K_copy(n, n), F_copy(n))
        
        ! Copy because LAPACK overwrites
        K_copy = K
        F_copy = F
        
        ! DPOSV: Cholesky factorization + solve for SPD matrix
        ! K_copy is overwritten with factorization
        ! F_copy is overwritten with solution
        call DPOSV('U', n, 1, K_copy, n, F_copy, n, info)
        
        if (info /= 0) then
            print *, 'ERROR: DPOSV failed with info = ', info
            stop
        end if
        
        U = F_copy
        
        deallocate(K_copy, F_copy)
        
    end subroutine solve_system
    
end module linear_solver