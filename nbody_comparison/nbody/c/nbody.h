/*
 * N-Body Gravitational Simulation - C Header
 */

#ifndef NBODY_H
#define NBODY_H

void compute_acceleration(double* positions, double* masses, int n_particles,
                         double G, double softening, double* accelerations);

void velocity_verlet_step(double* positions, double* velocities, double* masses,
                         int n_particles, double G, double softening, double dt);

void simulate(double* positions_init, double* velocities_init, double* masses,
             int n_particles, int n_steps, double G, double softening, double dt,
             double* positions_out, double* velocities_out);

void compute_energy(double* positions, double* velocities, double* masses,
                   int n_particles, double G, double softening,
                   double* kinetic, double* potential, double* total);

#endif /* NBODY_H */