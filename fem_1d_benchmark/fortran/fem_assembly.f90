! 1D FEM Assembly in Fortran (receives pre-allocated arrays like C)
!
! CRITICAL: Takes pre-allocated arrays as input (like C does)
! Arrays are already zeroed by NumPy - just write values!
!
! Compile with: f2py -c -m fem_fortran fem_assembly.f90 -O3

subroutine assemble_system(n, f_vals, K, F)
    implicit none

    integer, intent(in) :: n
    real(8), intent(in)  :: f_vals(n+1)
    real(8), intent(inout) :: K(n,n)  ! INOUT not OUT - array pre-allocated!
    real(8), intent(inout) :: F(n)    ! INOUT not OUT - array pre-allocated!
!f2py integer intent(in) :: n
!f2py real(8) dimension(n+1),intent(in) :: f_vals
!f2py real(8) dimension(n,n),intent(inout) :: K
!f2py real(8) dimension(n),intent(inout) :: F

    integer :: e, i
    real(8) :: h, k_local
    
    h = 1.0d0 / dble(n)
    k_local = 1.0d0 / h

    ! K and F are ALREADY ZEROED by NumPy!
    ! Just write values directly - NO ZEROING NEEDED!
    
    ! Assemble load vector
    do i = 1, n-1
        F(i) = (h / 2.0d0) * (f_vals(i) + f_vals(i+2))
    enddo
    F(n) = (h / 2.0d0) * f_vals(n)

    ! Assemble stiffness matrix
    K(1, 1) = k_local
    
    do e = 2, n
        i = e - 1
        K(i, i)     = K(i, i) + k_local
        K(i+1, i)   = K(i+1, i) - k_local
        K(i, i+1)   = K(i, i+1) - k_local
        K(i+1, i+1) = K(i+1, i+1) + k_local
    enddo

end subroutine assemble_system