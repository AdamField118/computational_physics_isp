! 1D FEM Assembly in Fortran
! Compile with: f2py -c -m fem_fortran fem_assembly.f90

subroutine assemble_system(n, f_vals, K, F)
    implicit none

    integer, intent(in) :: n
    real(8), intent(in)  :: f_vals(:)       ! expects length n+1
    real(8), intent(out) :: K(:,:)          ! caller must allocate (n,n)
    real(8), intent(out) :: F(:)            ! caller must allocate (n,)

    integer :: e, nodeL, nodeR, iL, iR, i, nv
    real(8) :: h, k_local

    nv = size(f_vals)
    if (nv /= n + 1) then
        write(*,*) 'ERROR in assemble_system: size(f_vals)=', nv, ' but expected n+1=', n+1
        stop 1
    endif

    ! Do NOT allocate K/F here. Caller provides memory.
    ! initialize outputs
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
        F(i) = (h / 2.0d0) * ( f_vals(i) + f_vals(i + 2) )
    enddo
    F(n) = (h / 2.0d0) * ( f_vals(n) + f_vals(n + 1) )

end subroutine assemble_system