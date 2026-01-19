! N-Body Gravitational Simulation - Fortran Implementation
! Designed to be wrapped with f2py for Python interoperability
! Adam Field - Computational Physics ISP

module nbody_fortran
    implicit none
    
    ! Use double precision throughout
    integer, parameter :: dp = selected_real_kind(15, 307)
    
contains

    ! ========================================================================
    ! Compute gravitational accelerations on all particles
    ! ========================================================================
    subroutine compute_acceleration(positions, masses, n_particles, G, &
                                   softening, accelerations)
        implicit none
        
        ! Input/output declarations
        integer, intent(in) :: n_particles
        real(dp), intent(in) :: positions(n_particles, 3)
        real(dp), intent(in) :: masses(n_particles)
        real(dp), intent(in) :: G, softening
        real(dp), intent(out) :: accelerations(n_particles, 3)
        
        ! Local variables
        integer :: i, j
        real(dp) :: dx, dy, dz, dist_sq, dist, inv_dist_cubed
        real(dp) :: fx, fy, fz
        
        ! Initialize accelerations to zero
        accelerations = 0.0_dp
        
        ! O(N²) pairwise force calculation
        !$OMP PARALLEL DO PRIVATE(j, dx, dy, dz, dist_sq, dist, &
        !$OMP                      inv_dist_cubed, fx, fy, fz) &
        !$OMP             REDUCTION(+:accelerations)
        do i = 1, n_particles
            do j = 1, n_particles
                if (i /= j) then
                    ! Displacement vector from i to j
                    dx = positions(j, 1) - positions(i, 1)
                    dy = positions(j, 2) - positions(i, 2)
                    dz = positions(j, 3) - positions(i, 3)
                    
                    ! Distance with softening
                    dist_sq = dx*dx + dy*dy + dz*dz + softening*softening
                    dist = sqrt(dist_sq)
                    
                    ! 1/r³ term
                    inv_dist_cubed = 1.0_dp / (dist * dist_sq)
                    
                    ! Force contribution (F/m_i = G*m_j/r² in direction of j)
                    fx = G * masses(j) * dx * inv_dist_cubed
                    fy = G * masses(j) * dy * inv_dist_cubed
                    fz = G * masses(j) * dz * inv_dist_cubed
                    
                    ! Accumulate acceleration
                    accelerations(i, 1) = accelerations(i, 1) + fx
                    accelerations(i, 2) = accelerations(i, 2) + fy
                    accelerations(i, 3) = accelerations(i, 3) + fz
                end if
            end do
        end do
        !$OMP END PARALLEL DO
        
    end subroutine compute_acceleration
    
    
    ! ========================================================================
    ! Single timestep using Velocity Verlet integration
    ! ========================================================================
    subroutine velocity_verlet_step(positions, velocities, masses, &
                                   n_particles, G, softening, dt)
        implicit none
        
        ! Input/output
        integer, intent(in) :: n_particles
        real(dp), intent(inout) :: positions(n_particles, 3)
        real(dp), intent(inout) :: velocities(n_particles, 3)
        real(dp), intent(in) :: masses(n_particles)
        real(dp), intent(in) :: G, softening, dt
        
        ! Local variables
        real(dp) :: acc_current(n_particles, 3)
        real(dp) :: acc_new(n_particles, 3)
        real(dp) :: dt_sq
        
        dt_sq = dt * dt
        
        ! Compute current acceleration
        call compute_acceleration(positions, masses, n_particles, G, &
                                 softening, acc_current)
        
        ! Update positions: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt²
        positions = positions + velocities * dt + 0.5_dp * acc_current * dt_sq
        
        ! Compute new acceleration at updated positions
        call compute_acceleration(positions, masses, n_particles, G, &
                                 softening, acc_new)
        
        ! Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        velocities = velocities + 0.5_dp * (acc_current + acc_new) * dt
        
    end subroutine velocity_verlet_step
    
    
    ! ========================================================================
    ! Run full simulation for n_steps
    ! ========================================================================
    subroutine simulate(positions_init, velocities_init, masses, &
                       n_particles, n_steps, G, softening, dt, &
                       positions_out, velocities_out)
        implicit none
        
        ! Input
        integer, intent(in) :: n_particles, n_steps
        real(dp), intent(in) :: positions_init(n_particles, 3)
        real(dp), intent(in) :: velocities_init(n_particles, 3)
        real(dp), intent(in) :: masses(n_particles)
        real(dp), intent(in) :: G, softening, dt
        
        ! Output (final state only)
        real(dp), intent(out) :: positions_out(n_particles, 3)
        real(dp), intent(out) :: velocities_out(n_particles, 3)
        
        ! Working variables
        real(dp) :: positions(n_particles, 3)
        real(dp) :: velocities(n_particles, 3)
        integer :: step
        
        ! Initialize with input conditions
        positions = positions_init
        velocities = velocities_init
        
        ! Time integration loop
        do step = 1, n_steps
            call velocity_verlet_step(positions, velocities, masses, &
                                     n_particles, G, softening, dt)
        end do
        
        ! Copy final state to output
        positions_out = positions
        velocities_out = velocities
        
    end subroutine simulate
    
    
    ! ========================================================================
    ! Compute total energy (for verification)
    ! ========================================================================
    subroutine compute_energy(positions, velocities, masses, n_particles, &
                             G, softening, kinetic, potential, total)
        implicit none
        
        ! Input
        integer, intent(in) :: n_particles
        real(dp), intent(in) :: positions(n_particles, 3)
        real(dp), intent(in) :: velocities(n_particles, 3)
        real(dp), intent(in) :: masses(n_particles)
        real(dp), intent(in) :: G, softening
        
        ! Output
        real(dp), intent(out) :: kinetic, potential, total
        
        ! Local variables
        integer :: i, j
        real(dp) :: v_sq, dx, dy, dz, dist
        
        ! Kinetic energy: KE = 0.5 * m * v²
        kinetic = 0.0_dp
        do i = 1, n_particles
            v_sq = velocities(i,1)**2 + velocities(i,2)**2 + velocities(i,3)**2
            kinetic = kinetic + 0.5_dp * masses(i) * v_sq
        end do
        
        ! Potential energy: PE = -G * m_i * m_j / r_ij (sum over pairs)
        potential = 0.0_dp
        do i = 1, n_particles
            do j = i+1, n_particles
                dx = positions(j,1) - positions(i,1)
                dy = positions(j,2) - positions(i,2)
                dz = positions(j,3) - positions(i,3)
                
                dist = sqrt(dx*dx + dy*dy + dz*dz + softening*softening)
                
                potential = potential - G * masses(i) * masses(j) / dist
            end do
        end do
        
        total = kinetic + potential
        
    end subroutine compute_energy

end module nbody_fortran