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
__global__ void get_acc_kernel(float *p, float *m, float *a, float G, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Accleration (x, y, z) for plaent with id tid
  float x = 0;
  float y = 0;
  float z = 0;// Loop for number of timesteps --> timestep 0 already complete

  for(int i=0; i<N; i++){
    if(i != tid){
      // get difference in position of neighboring planet
      float dx = p[0 + i * 3] - p[0 + tid * 3];
      float dy = p[1 + i * 3] - p[1 + tid * 3];
      float dz = p[2 + i * 3] - p[2 + tid * 3];

      // Calculate inverse with softening length (0.1) -- Part to account for particles close to eachother
      float inv = pow(pow(dx, 2) + pow(dy, 2) + pow(dz, 2) + pow(0.1, 2), -1.5);

      // calculate inversepmlanet_mass
      x = x + (m[i] * dx / inv);
      y = y + (m[i] * dy / inv);
      z = z + (m[i] * dz / inv);
    }
  }      

  // Adjust with Newton's Gravitational constant
  x = x * G;
  y = y * G;
  z = z * G;

  // Assign new x,y,z accelerations to "a"
  a[0 + tid * 3] = x;
  a[1 + tid * 3] = y;
  a[2 + tid * 3] = z;
  printf("%d ", tid);
  printf("%f ", a[0 + tid * 3]);
  printf("%f ", a[1 + tid * 3]);
  printf("%f ", a[2 + tid * 3]);
  printf("here");
}

// Update velocity of singular planet (used each half kick)
__global__ void get_vel_kernel(float *v, float *a, float td) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // new velocity = velocity + acceleration * (td / 2.0)
  v[0 + tid * 3] = v[0 + tid * 3] + (a[0 + tid * 3] * td / 2.0);
  v[1 + tid * 3] = v[1 + tid * 3] + (a[1 + tid * 3] * td / 2.0);
  v[2 + tid * 3] = v[2 + tid * 3] + (a[2 + tid * 3] * td / 2.0);
}

// Update position of singular planet (drift)
__global__ void get_pos_kernel(float *p, float *v, float *data, float td, float timesteps, int i) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // new position = position + velocity * td
  p[0 + tid * 3] = p[0 + tid * 3] + (v[0 + tid * 3] * td);
  p[1 + tid * 3] = p[1 + tid * 3] + (v[1 + tid * 3] * td);
  p[2 + tid * 3] = p[2 + tid * 3] + (v[2 + tid * 3] * td);

  // // idx probably wrongs here!!! --> also error "expression must have integral or unscoped enum type"
  // data[0 + (tid * 3 * timesteps) + (i * 3)] = p[0 + tid * 3];
  // data[1 + (tid * 3 * timesteps) + (i * 3)] = p[1 + tid * 3];
  // data[2 + (tid * 3 * timesteps) + (i * 3)] = p[2 + tid * 3];
}


/*** CPU functions ***/

// Returns data from N-body simulation
float* n_body(int N, int G, float td, int timesteps) {
  // N x 3 matrix of random starting positions of planets (N x (x,y,z))
  float* planet_pos = new float[N*3];
  float* d_planet_pos;
  // N x 3 matrix of random velocities of planets
  float* planet_vel = new float[N*3];
  float* d_planet_vel;
  // N x 1 vector of random masses of planets
  float* planet_mass = new float[N];
  float* d_planet_mass;
  // N x 1 vector of random masses of planets
  float* planet_acc;
  float* d_planet_acc;
  // N x 3 x # timesteps matrix of positions of planets over all timesteps
  float* data = new float[N * 3 * timesteps];
  float* d_data;

  // Allocate memory
  planet_acc = (float*)malloc((N*3)* sizeof(float));
  planet_pos = (float*)malloc((N*3)* sizeof(float));
  planet_vel = (float*)malloc((N*3)* sizeof(float));
  planet_mass = (float*)malloc(N * sizeof(float));
  data = (float*)malloc(N * 3 * timesteps * sizeof(float));

  cudaMalloc(&d_planet_mass, N * sizeof(float));
  cudaMalloc(&d_planet_pos, N * 3 * sizeof(float));
  cudaMalloc(&d_planet_vel, N * 3 * sizeof(float));
  cudaMalloc(&d_planet_acc, N * 3 * sizeof(float));
  cudaMalloc(&d_data, N * 3 * timesteps * sizeof(float));

  // Create N x 1 array of masses of planets
  for(int i=0; i<N; i++){
    planet_mass[i] = rand()/float(RAND_MAX)*1.f+0.f;
  }

  // Create N x 3 matrix of random starting velocities & positions for each planet
  for(int i= 0; i< N; i++){
    planet_pos[0 + 3*i] = rand()/float(RAND_MAX)*1.f+0.f;
    planet_pos[1 + 3*i] = rand()/float(RAND_MAX)*1.f+0.f;
    planet_pos[2 + 3*i] = rand()/float(RAND_MAX)*1.f+0.f;
    planet_vel[0 + 3*i] = rand()/float(RAND_MAX)*1.f+0.f;
    planet_vel[1 + 3*i] = rand()/float(RAND_MAX)*1.f+0.f;
    planet_vel[2 + 3*i] = rand()/float(RAND_MAX)*1.f+0.f;
    // Note sure if this idx is correct!
    data[0 + (i * 3 * timesteps)] = planet_pos[0 + 3*i];
    data[1 + (i * 3 * timesteps)] = planet_pos[1 + 3*i];
    data[2 + (i * 3 * timesteps)] = planet_pos[2 + 3*i];
  }

  // OLD METHOD:
  // std::cout << " " << std::endl; 
  // // Add matrix of positions to data
  // for(int i=0; i<1; i++){
  //   data[i] = new float*[N];
  //   for(int j=0; j<N; j++){
  //     data[i][j] = new float[3];
  //     for (int k=0; k<3; k++){
  //       data[i][j][k] = planet_pos[k + 3*i];
  //       // std::cout << data[i][j][k] << std::endl;
  //     }
  //   }
  // }

  // Copy variables from host to device
  cudaMemcpy(d_planet_mass, planet_mass, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_planet_pos, planet_pos, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_planet_vel, planet_vel, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, data, N * 3 * timesteps * sizeof(float), cudaMemcpyHostToDevice);

  // Get acceleration of planets --> call GPU kernel here
  get_acc_kernel<<<N_BLOCKS, N_THREADS>>>(planet_pos, planet_mass, planet_acc, G, N);

  // Copy new accelerations device to host
  cudaMemcpy(planet_acc, d_planet_acc, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);

  // // Debugging acceleration:
  // for(int i=0; i< (N * 3); i++){
  //   // std::cout << planet_acc << std::endl;
  // }

  // Loop for number of timesteps --> timestep 0 already complete
  for(int i = 1; i < timesteps; i++){
    // Have to call multiple kernels and use cudaDeviceSynchronize()
    // Use leapfrog integration
    // 1) First half kick --> update velocities
    get_vel_kernel<<<N_BLOCKS, N_THREADS>>>(planet_vel, planet_acc, td);
    cudaDeviceSynchronize();

    // 2) Drift --> update positions
    get_pos_kernel<<<N_BLOCKS, N_THREADS>>>(planet_pos, planet_vel, data, td, timesteps, i);
    cudaDeviceSynchronize();

    // 3) update acceleration with new positions
    get_acc_kernel<<<N_BLOCKS, N_THREADS>>>(planet_pos, planet_mass, planet_acc, G, N);
    cudaDeviceSynchronize();
    
    // 4) Second half od kick --> update velocities again
    get_vel_kernel<<<N_BLOCKS, N_THREADS>>>(planet_vel, planet_acc, td);
    cudaDeviceSynchronize();
    
    // 5) Add new positions to data --> not sure if we should copy memory back over here and the recopy back
    cudaMemcpy(planet_pos, d_planet_pos, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
  }

  // Copy varibles device to host --> maybe just need 
  cudaMemcpy(planet_pos, d_planet_pos, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);

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
