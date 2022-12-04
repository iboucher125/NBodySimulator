#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <random>
#include <iostream>
#include <bits/stdc++.h>
#include <math.h>

#define N_THREADS 2
#define N_BLOCKS 1

/*** GPU functions ***/
/*** Get accelerations ***/
__global__ void get_acc_kernel(float **p, float m, float **a, float G, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(int i=0; i<N; i++){
    *a = new float[3];
    for(int j=0; j<3; j++){
      for(int k=0; k<N; k++){
        // // Get acceleration for one planet (each thread handles one)
        // float x = 0;
        // float y = 0;
        // float z = 0;
        // // iterate through all other planets
        // for (int z = 0; z < N; z++){
        //   if(z != tid){
        //     // get difference in position of neighboring planet
        //     float dx = p[z][0] - p[k][0];
        //     float dy = p[z][1] - p[k][1];
        //     float dz = p[z][2] - p[k][2];
        //     // calculate inversepmlanet_mass
        //     x = x + (m[z] * dx / inv);
        //     y = y + (m[z] * dy / inv);
        //     z = z + (m[z] * dz / inv);
          }
        }
        // x = x * G;
        // y = y * G;
        // z = z * G;

        // new_acc[k][0] = x;
        // new_acc[k][1] = y;
        // new_acc[k][2] = z;
      }
    }
  }

  // Adjust with Newton's Gravitational constant

  // Assign new x,y,z accelerations to "a"
}

// Update velocity of singular planet (used each half kick)
__global__ void get_vel_kernel(float *p, float *v, float td, curandState *state) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // new velocity = velocity + acceleration * (td / 2.0)
}

// Update position of singular planet (drift)
__global__ void get_pos_kernel(float *p, float *v, float td, curandState *state) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // new position = position + velocity * td
}


/*** CPU functions ***/

// Returns data from N-body simulation
float* n_body(int N, int G, float td, int timesteps) {
  // N x 3 matrix of random starting positions of planets (N x (x,y,z))
  float** planet_pos = new float*[N];
  float** d_planet_pos;
  // N x 3 matrix of random velocities of planets
  float** planet_vel = new float*[N];
  float** d_planet_vel;
  // N x 1 vector of random masses of planets
  float planet_mass[N];
  float d_planet_mass;
  // N x 1 vector of random masses of planets
  float** planet_acc[N];
  float** d_planet_acc;
  // N x 3 x # timesteps matrix of positions of planets over all timesteps
  float*** data = new float**[timesteps];
  float** d_data;

  // Allocate memory
  cudaMalloc(&d_planet_mass, planet_mass.size() * sizeof(float));
  cudaMalloc(&d_planet_pos, planet_pos.size() * sizeof(float));
  cudaMalloc(&d_planet_vel, planet_vel.size() * sizeof(float));
  cudaMalloc(&d_planet_acc, planet_acc.size() * sizeof(float));

  // Create N x 1 array of masses of planets
  for(int i=0; i<N; i++){
    planet_mass[i] = rand()/float(RAND_MAX)*1.f+0.f;
    // std::cout << planet_mass[i] << std::endl;
  }

  // Create N x 3 matrix of random starting velocities & positions for each planet
  for(int i=0; i<N; i++){
    planet_vel[i] = new float[3];
    planet_pos[i] = new float[3];
    for(int j=0; j<3; j++){
      planet_vel[i][j] = rand()/float(RAND_MAX)*1.f+0.f;
      planet_pos[i][j] = rand()/float(RAND_MAX)*1.f+0.f;
    }
  }

  std::cout << " " << std::endl; 
  // Add matrix of positions to data
  for(int i=0; i<1; i++){
    data[i] = new float*[N];
    for(int j=0; j<N; j++){
      data[i][j] = new float[3];
      for (int k=0; k<3; k++){
        data[i][j][k] = planet_pos[j][k];
        std::cout << data[i][j][k] << std::endl;
      }
    }
  }

  // Copy variables host to device
  cudaMemcpy(d_planet_mass, planet_mass, planet_mass.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_planet_pos, planet_pos, planet_pos.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_planet_vel, planet_vel, planet_vel.size() * sizeof(float), cudaMemcpyHostToDevice);

  // Get acceleration of planets --> call GPU kernel here
  get_acc_kernel<<<N_BLOCKS, N_THREADS>>>(planet_pos, planet_mass, planet_acc, G, N);


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

  // debug
  n_body(N, G, td, timesteps);

  // Write to output file

  // End time of simulation
  auto t_end = std::chrono::high_resolution_clock::now();

  // Computer duration --> need to look up how... I don't remember --> print duration
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

  return 0;
}
