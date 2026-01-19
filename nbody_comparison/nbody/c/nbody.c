/*
 * N-Body Gravitational Simulation - C Implementation
 * Adam Field - Computational Physics ISP
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "nbody.h"

/* Compute gravitational accelerations on all particles - PARALLEL */
void compute_acceleration(double* positions, double* masses, int n_particles,
                         double G, double softening, double* accelerations) {
    int i, j;
    double dx, dy, dz, dist_sq, dist, inv_dist_cubed;
    double soft_sq = softening * softening;
    
    /* Initialize accelerations to zero */
    for (i = 0; i < n_particles * 3; i++) {
        accelerations[i] = 0.0;
    }
    
    /* PARALLEL O(N²) pairwise force calculation */
    #pragma omp parallel for private(j, dx, dy, dz, dist_sq, dist, inv_dist_cubed) schedule(dynamic, 32)
    for (i = 0; i < n_particles; i++) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        
        /* Cache particle i's position (better cache locality) */
        double px_i = positions[i*3 + 0];
        double py_i = positions[i*3 + 1];
        double pz_i = positions[i*3 + 2];
        
        for (j = 0; j < n_particles; j++) {
            if (i != j) {
                /* Displacement vector from i to j */
                dx = positions[j*3 + 0] - px_i;
                dy = positions[j*3 + 1] - py_i;
                dz = positions[j*3 + 2] - pz_i;
                
                /* Distance with softening */
                dist_sq = dx*dx + dy*dy + dz*dz + soft_sq;
                dist = sqrt(dist_sq);
                
                /* 1/r³ term */
                inv_dist_cubed = 1.0 / (dist * dist_sq);
                
                /* Force contribution */
                double factor = G * masses[j] * inv_dist_cubed;
                ax += factor * dx;
                ay += factor * dy;
                az += factor * dz;
            }
        }
        
        /* Write accumulated result */
        accelerations[i*3 + 0] = ax;
        accelerations[i*3 + 1] = ay;
        accelerations[i*3 + 2] = az;
    }
}

/* Single timestep using Velocity Verlet integration */
void velocity_verlet_step(double* positions, double* velocities, double* masses,
                         int n_particles, double G, double softening, double dt) {
    int i;
    double* acc_current = (double*)malloc(n_particles * 3 * sizeof(double));
    double* acc_new = (double*)malloc(n_particles * 3 * sizeof(double));
    double dt_sq = dt * dt;
    
    /* Compute current acceleration */
    compute_acceleration(positions, masses, n_particles, G, softening, acc_current);
    
    /* Update positions: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt² */
    for (i = 0; i < n_particles * 3; i++) {
        positions[i] = positions[i] + velocities[i] * dt + 0.5 * acc_current[i] * dt_sq;
    }
    
    /* Compute new acceleration */
    compute_acceleration(positions, masses, n_particles, G, softening, acc_new);
    
    /* Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt */
    for (i = 0; i < n_particles * 3; i++) {
        velocities[i] = velocities[i] + 0.5 * (acc_current[i] + acc_new[i]) * dt;
    }
    
    free(acc_current);
    free(acc_new);
}

/* Run full simulation for n_steps */
void simulate(double* positions_init, double* velocities_init, double* masses,
             int n_particles, int n_steps, double G, double softening, double dt,
             double* positions_out, double* velocities_out) {
    int i, step;
    
    /* Copy initial conditions */
    for (i = 0; i < n_particles * 3; i++) {
        positions_out[i] = positions_init[i];
        velocities_out[i] = velocities_init[i];
    }
    
    /* Time integration loop */
    for (step = 0; step < n_steps; step++) {
        velocity_verlet_step(positions_out, velocities_out, masses,
                           n_particles, G, softening, dt);
    }
}

/* Compute total energy */
void compute_energy(double* positions, double* velocities, double* masses,
                   int n_particles, double G, double softening,
                   double* kinetic, double* potential, double* total) {
    int i, j;
    double v_sq, dx, dy, dz, dist;
    double soft_sq = softening * softening;
    
    /* Kinetic energy: KE = 0.5 * m * v² */
    *kinetic = 0.0;
    for (i = 0; i < n_particles; i++) {
        v_sq = velocities[i*3 + 0] * velocities[i*3 + 0] +
               velocities[i*3 + 1] * velocities[i*3 + 1] +
               velocities[i*3 + 2] * velocities[i*3 + 2];
        *kinetic += 0.5 * masses[i] * v_sq;
    }
    
    /* Potential energy: PE = -G * m_i * m_j / r_ij */
    *potential = 0.0;
    for (i = 0; i < n_particles; i++) {
        for (j = i + 1; j < n_particles; j++) {
            dx = positions[j*3 + 0] - positions[i*3 + 0];
            dy = positions[j*3 + 1] - positions[i*3 + 1];
            dz = positions[j*3 + 2] - positions[i*3 + 2];
            
            dist = sqrt(dx*dx + dy*dy + dz*dz + soft_sq);
            
            *potential -= G * masses[i] * masses[j] / dist;
        }
    }
    
    *total = *kinetic + *potential;
}