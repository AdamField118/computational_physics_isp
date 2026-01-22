! 1D FEM Assembly in Fortran with OpenMP parallelization
! Compile with: f2py -c -m fem_fortran fem_assembly.f90 --f90flags="-fopenmp" -lgomp

subroutine assemble_system(n, f_vals, K, F)
    use omp_lib
    implicit none

    integer, intent(in) :: n
    real(8), intent(in)  :: f_vals(n+1)
    real(8), intent(out) :: K(n,n)
    real(8), intent(out) :: F(n)
!f2py integer intent(in) :: n
!f2py real(8) dimension(n+1),intent(in) :: f_vals
!f2py real(8) dimension(n,n),intent(out) :: K
!f2py real(8) dimension(n),intent(out) :: F

    integer :: e, i, nodeL, nodeR, iL, iR
    real(8) :: h, k_local
    
    ! Initialize outputs
    K = 0.0d0
    F = 0.0d0

    h = 1.0d0 / dble(n)
    k_local = 1.0d0 / h

    ! Parallel assembly of stiffness matrix
    !$OMP PARALLEL DO PRIVATE(e, nodeL, nodeR, iL, iR) REDUCTION(+:K)
    do e = 1, n
        nodeL = e
        nodeR = e + 1

        if (nodeL >= 2) then
            iL = nodeL - 1
            iR = nodeR - 1
            !$OMP ATOMIC
            K(iL, iL) = K(iL, iL) + k_local
            !$OMP ATOMIC
            K(iL, iR) = K(iL, iR) - k_local
            !$OMP ATOMIC
            K(iR, iL) = K(iR, iL) - k_local
        endif

        iR = nodeR - 1
        !$OMP ATOMIC
        K(iR, iR) = K(iR, iR) + k_local
    enddo
    !$OMP END PARALLEL DO

    ! Parallel assembly of load vector
    !$OMP PARALLEL DO PRIVATE(i)
    do i = 1, n-1
        F(i) = (h / 2.0d0) * (f_vals(i) + f_vals(i+2))
    enddo
    !$OMP END PARALLEL DO
    
    ! Last node uses only left contribution (Neumann BC on right)
    F(n) = (h / 2.0d0) * f_vals(n)

end subroutine assemble_system


! Serial version for comparison
subroutine assemble_system_serial(n, f_vals, K, F)
    implicit none

    integer, intent(in) :: n
    real(8), intent(in)  :: f_vals(n+1)
    real(8), intent(out) :: K(n,n)
    real(8), intent(out) :: F(n)
!f2py integer intent(in) :: n
!f2py real(8) dimension(n+1),intent(in) :: f_vals
!f2py real(8) dimension(n,n),intent(out) :: K
!f2py real(8) dimension(n),intent(out) :: F

    integer :: e, i, nodeL, nodeR, iL, iR
    real(8) :: h, k_local
    
    K = 0.0d0
    F = 0.0d0

    h = 1.0d0 / dble(n)
    k_local = 1.0d0 / h

    do e = 1, n
        nodeL = e
        nodeR = e + 1

        if (nodeL >= 2) then
            iL = nodeL - 1
            iR = nodeR - 1
            K(iL, iL) = K(iL, iL) + k_local
            K(iL, iR) = K(iL, iR) - k_local
            K(iR, iL) = K(iR, iL) - k_local
        endif

        iR = nodeR - 1
        K(iR, iR) = K(iR, iR) + k_local
    enddo

    do i = 1, n-1
        F(i) = (h / 2.0d0) * (f_vals(i) + f_vals(i+2))
    enddo
    ! Last node uses only left contribution (Neumann BC on right)
    F(n) = (h / 2.0d0) * f_vals(n)

end subroutine assemble_system_serial