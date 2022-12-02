#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

#define N_THREADS 2
#define N_BLOCKS 1

/*** GPU functions ***/
// Maybe use for initial velociites, masses, and positions
__global__ void init_rand_kernel(curandState *state) {
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 curand_init(0, idx, 0, &state[idx]);
}

// Get accelerations
__global__ void get_acc_kernel(float *p, float *m, float G, int N, float *a, curandState *state) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Get acceleration for one planet (each thread handles one)
  float x = 0;
  float y = 0;
  float z = 0;

  // iterate through all other planets
  for (int i = 0; i < N; i++){
    if(i != tid){
      // get difference in position of neighboring planet
      // calculate inverse
      // update acceleration
    }
  }

  // Adjust with Newton's Gravitational constant

  // assign new x,y,z accelerations to "a"
}

// update velocity of singular planet (used each half kick)
__global__ void get_vel_kernel(float *p, float *v, float td, curandState *state) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // new velocity = velocity + acceleration * (td / 2.0)
}

// update position of singular planet (drift)
__global__ void get_pos_kernel(float *p, float *v, float td, curandState *state) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // new position = position + velocity * td
}


/*** CPU functions ***/
curandState* init_rand() {
  curandState *d_state;
  cudaMalloc(&d_state, N_BLOCKS * N_THREADS * sizeof(curandState));
  init_rand_kernel<<<N_BLOCKS, N_THREADS>>>(d_state);
  return d_state;
}

// returns data from N-body simulation
float* n_body(int N, int G, float td, int timesteps) {
  // N x 3 matrix of random starting positions of planets (N x (x,y,z))
  float* planet_pos;
  float* d_planet_pos;
  // N x 3 matrix of random velocities of planets
  float* planet_vel;
  float* d_planet_vel;
  // N x 1 vector of random masses of planets
  float* planet_mass;
  float* d_planet_mass;
  // N x 1 vector of random masses of planets
  float* planet_acc;
  float* d_planet_acc;
  // N x 3 x # timesteps matrix of positions of planets over all timesteps
  float* data;
  float* d_data;

  // Allocate memory

  // Initialize pos, vel, mass here???
  // Create starting postions --> random
  // Add positions to data

  // Create starting velocities --> random

  // Create masses of planets --> keep the same or random?

  // Copy variables host to device

  // Get acceleration of planets --> call GPU kernel here

  // Copy new accelerations device to host

  // Loop for number of timesteps --> timestep 0 already complete
  for(int i = 1; i < timesteps; i++){
    // Have to call multiple kernels and use cudaDeviceSynchronize()

    // Use leapfrog integration
    // 1) First half kick --> update velocities
      // get_vel kernel

    // 2) Drift --> update positions
      // get_pos kernal

    // 3) update acceleration with new positions
      // get_acc kernal
    
    // 4) Second half od kick --> update velocities again
      // get_vel kernel
    
    // 5) Add new positions to data
  }

  // Copy varibles device to host

  // Free memory

  // Return data

}


int main(int argc, char** argv) {
  // Number of planets 
  int N = 2;
  // Newton's Gravitational Constant
  float G = pow(6.67 * 10, -11);
  // Start time of simulation
  auto t_start = std::chrono::high_resolution_clock::now();

  //  Set number of timesteps (number of interations for simulation)
  float td = 0.01;
  int timesteps = 50;

  // Positions over all timesteps that will be written to output file
  // call CPU function here!!
  float* data;

  // Write to output file

  // End time of simulation
  auto t_end = std::chrono::high_resolution_clock::now();

  // Computer duration --> need to look up how... I don't remember --> print duration
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

  return 0;
}
