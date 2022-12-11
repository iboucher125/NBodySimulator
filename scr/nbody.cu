#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <random>
#include <iostream>
#include <bits/stdc++.h>
#include <math.h>
#include <fstream>

// #define N_THREADS 
#define N_BLOCKS 1

/*** GPU functions ***/

// Update acceleration of particles
__device__ void get_acc_kernel(double *p, double *m, double *a, double G, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Accleration (x, y, z) for plaent with id tid
  double x = 0;
  double y = 0;
  double z = 0;

  for(int i=0; i<N; i++){
    if(i != tid){
      // Get difference in position of neighboring particle
      double dx = p[0 + i * 3] - p[0 + tid * 3];
      double dy = p[1 + i * 3] - p[1 + tid * 3];
      double dz = p[2 + i * 3] - p[2 + tid * 3];
      // printf("p1-pj1=dx: %e - %e = %e; p2-pj2=dy: %e - %e = %e; p3-pj3=dz: %e - %e = %e\n", p[0 + i * 3], p[0 + tid * 3], dx, p[1 + i * 3], p[1 + tid * 3], dy, p[2 + i * 3],p[2 + tid * 3],dz);

      // Calculate inverse with softening length (0.1) -- Part to account for particles close to eachother
      double inv = pow(pow(dx, 2) + pow(dy, 2) + pow(dz, 2) + pow(0.1, 2), -1.5);
      // printf("inv: %e\n", inv);

      x = x + (m[i] * dx / inv);
      y = y + (m[i] * dy / inv);
      z = z + (m[i] * dz / inv);
      // printf("x: %e; y: %e; z: %e\n", x, y, z);
    }
  }      

  // Adjust with Newton's Gravitational constant
  x = x * G;
  y = y * G;
  z = z * G;
  // printf("x: %e; y: %e; z: %e\n", x, y, z);

  // Assign new x,y,z accelerations to "a"
  a[0 + tid * 3] = x;
  a[1 + tid * 3] = y;
  a[2 + tid * 3] = z;
}

// Update velocity of singular particle (used each half kick)
__device__ void get_vel_kernel(double *v, double *a, double td) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  v[0 + tid * 3] = v[0 + tid * 3] + (a[0 + tid * 3] * td / 2.0);
  v[1 + tid * 3] = v[1 + tid * 3] + (a[1 + tid * 3] * td / 2.0);
  v[2 + tid * 3] = v[2 + tid * 3] + (a[2 + tid * 3] * td / 2.0);

  // printf("v = v + a(td/2): %e = %e + %e(%e/2)\n",v[0 + tid * 3] , v[0 + tid * 3] , a[0 + tid * 3] , td );
  // printf("v = v + a(td/2): %e = %e + %e(%e/2)\n",v[1 + tid * 3] , v[1 + tid * 3] , a[1 + tid * 3] , td );
  // printf("v = v + a(td/2): %e = %e + %e(%e/2)\n",vnum[2 + tid * 3] , v[2 + tid * 3] , a[2 + tid * 3] , td );
}

  // Update position of singular particle (drift)
__device__ void get_pos_kernel(double *p, double *v, double *data, double td, int N, int i) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // new position = position + velocity * td
  p[0 + tid * 3] = p[0 + tid * 3] + (v[0 + tid * 3] * td);
  p[1 + tid * 3] = p[1 + tid * 3] + (v[1 + tid * 3] * td);
  p[2 + tid * 3] = p[2 + tid * 3] + (v[2 + tid * 3] * td);

  // printf("p=p+v*td: %e = %e+%e*%e\n", p[0 + tid * 3], p[0 + tid * 3], v[0 + tid * 3], td);
  // printf("p1=p1+v*td: %e = %e+%e*%e\n", p[1 + tid * 3], p[1 + tid * 3], v[1 + tid * 3], td);
  // printf("p2=p2+v*td: %e = %e+%e*%e\n", p[2 + tid * 3], p[2 + tid * 3], v[2 + tid * 3], td);

  data[N + (i * N * 3 + 3 * tid + 0)] = p[0 + tid * 3];
  data[N + (i * N * 3 + 3 * tid + 1)] = p[1 + tid * 3];
  data[N + (i * N * 3 + 3 * tid + 2)] = p[2 + tid * 3];
  // printf("d0: %e; d1: %e; d2: %e\n", data[N + (i * N * 3 + 3 * tid + 0)], data[N + (i * N * 3 + 3 * tid + 1)], data[N + (i * N * 3 + 3 * tid + 2)]);
  // printf("p0: %e; p1: %e; p2: %e\n", p[0 + tid * 3], p[1 + tid * 3], p[2 + tid * 3]);

  // printf("d0[%d] = p0[%d]; d1[%d] = p1[%d]; d2[%d] = p2[%d]\n", N +(i * N * 3 + 3 * tid + 0), (0 + tid * 3), N + (i * N * 3 + 3 * tid + 1), (1 + tid * 3), N + (i * N * 3 + 3 * tid + 2), (2 + tid * 3));
}

// Run N-body simulation
__global__ void generate_data_kernel(double *p, double * v, double *m, double *a, double *data, int timesteps, double td, int G, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Get acceleration of particles
  get_acc_kernel(p, m, a, G, N);
  __syncthreads();

  // Loop for number of timesteps --> timestep 0 already complete
  for(int i = 1; i < timesteps; i++){
    // Use leapfrog integration
    // 1) First half kick --> update velocities
    get_vel_kernel(v, a, td);
    __syncthreads();

    // 2) Drift --> update positions
    get_pos_kernel(p, v, data, td, N, i);
    __syncthreads();

    // 3) update acceleration with new positions
    get_acc_kernel(p, m, a, G, N);
    __syncthreads();
    
    // 4) Second half od kick --> update velocities again
    get_vel_kernel(v, a, td);
    __syncthreads();
  }
}

/*** CPU functions ***/

// Returns data (mass and all positions) from N-body simulation
double* n_body(int N, double G, double td, int timesteps) {
  // Array of random starting positions of particles (N x (x,y,z))
  double* particle_pos = new double[N*3];
  double* d_particle_pos;
  // Array of random velocities of particles
  double* particle_vel = new double[N*3];
  double* d_particle_vel;
  // Array of random masses of particles
  double* particle_mass = new double[N];
  double* d_particle_mass;
  // Array of random masses of particles
  double* particle_acc;
  double* d_particle_acc;
  // Array of positions of particles over all timesteps
  double* data = new double[N * 3 * timesteps + N];
  double* d_data;

  // Allocate memory
  particle_acc = (double*)malloc((N*3)* sizeof(double));
  particle_pos = (double*)malloc((N*3)* sizeof(double));
  particle_vel = (double*)malloc((N*3)* sizeof(double));
  particle_mass = (double*)malloc(N * sizeof(double));
  data = (double*)malloc((N * 3 * timesteps + N) * sizeof(double));

  cudaMalloc(&d_particle_mass, N * sizeof(double));
  cudaMalloc(&d_particle_pos, N * 3 * sizeof(double));
  cudaMalloc(&d_particle_vel, N * 3 * sizeof(double));
  cudaMalloc(&d_particle_acc, N * 3 * sizeof(double));
  cudaMalloc(&d_data, (N * 3 * timesteps + N) * sizeof(double));

  // Fill array of masses of particles
  for(int i=0; i<N; i++){
    particle_mass[i] = rand()/double(RAND_MAX)*1.f+0.f;
    data[i] = particle_mass[i];
  }

  // Fill array of random starting velocities & positions for each particle
  for(int i= 0; i<N; i++){
    if (i % 3 == 1){
      particle_pos[0 + 3 * i] = rand()/double(RAND_MAX)*3.f-2.f;
      particle_pos[1 + 3 * i] = rand()/double(RAND_MAX)*2.f-2.f;
      particle_pos[2 + 3 * i] = rand()/double(RAND_MAX)*2.f-2.f;
    }
    else if (i % 3 == 2){
      particle_pos[0 + 3 * i] = rand()/double(RAND_MAX)*0.f-2.f;
      particle_pos[1 + 3 * i] = rand()/double(RAND_MAX)*2.f-2.f;
      particle_pos[2 + 3 * i] = rand()/double(RAND_MAX)*2.f-2.f;
    }
    else{
      particle_pos[0 + 3 * i] = rand()/double(RAND_MAX)*2.f-2.f;
      particle_pos[1 + 3 * i] = rand()/double(RAND_MAX)*2.f+0.f;
      particle_pos[2 + 3 * i] = rand()/double(RAND_MAX)*2.f-2.f;
    }
    particle_vel[0 + 3 * i] = rand()/double(RAND_MAX)*1.f+0.f;
    particle_vel[1 + 3 * i] = rand()/double(RAND_MAX)*1.f+0.f;
    particle_vel[2 + 3 * i] = rand()/double(RAND_MAX)*1.f+0.f;
    // std::cout <<  " start pos : " << particle_pos[0 + 3*i] << ", " << particle_pos[1 + 3 * i] << ", " << particle_pos[2 + 3 * i] <<std::endl;
    
    // Save initial particle positions
    data[(0 + 3 * i) + N] = particle_pos[0 + 3*i];
    data[(1 + 3 * i) + N] = particle_pos[1 + 3*i];
    data[(2 + 3 * i) + N] = particle_pos[2 + 3*i];
    // std::cout << (0 + 3 * i) + N << " = " << particle_pos[0 + 3*i] <<std::endl;
  }

  // Copy variables from host to device
  cudaMemcpy(d_particle_mass, particle_mass, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_particle_pos, particle_pos, N * 3 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_particle_vel, particle_vel, N * 3 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, data, N * 3 * timesteps * sizeof(double), cudaMemcpyHostToDevice);

  // Call GPU kernel to run simulation
  generate_data_kernel<<<N_BLOCKS, N>>>(d_particle_pos, d_particle_vel, d_particle_mass, d_particle_acc, d_data,timesteps, td, G, N);

  // Copy varibles device to host --> maybe just need 
  cudaMemcpy(data, d_data, (N * 3 * timesteps +N) * sizeof(double), cudaMemcpyDeviceToHost);

  // Free memory
  cudaFree(d_particle_pos);
  cudaFree(d_particle_vel);
  cudaFree(d_particle_acc);
  cudaFree(d_particle_mass);
  cudaFree(d_data);
  free(particle_pos);
  free(particle_vel);
  free(particle_acc);
  free(particle_mass); 

  // Return particles masses and all positions --> data
  return data;
}


int main(int argc, char** argv) {
  // Number of particles 
  int N = atoi(argv[1]);
  // Newton's Gravitational Constant
  double G = pow(6.67 * 10, -11);
  
  // Start time of simulation
  auto t_start = std::chrono::high_resolution_clock::now();

  //  Set number of timesteps (number of interations for simulation)
  double td = 0.01;
  int timesteps = atoi(argv[2]);

  // Run N-body simulation
  double* data = n_body(  N, G, td, timesteps);

  // End time of simulation
  auto t_end = std::chrono::high_resolution_clock::now();
  // Runtime duration in seconds
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start) * 0.000001;
  std::cout << "Computation Duration: " << total_time.count() << std::endl;

  // Write to output file
  std::ofstream output_file;
  output_file.open("../data/output_cu.txt");
  output_file << "Positions of " << N << " particles over " << timesteps <<" timesteps: \n";
  
  // Write masses
  for(int i=0; i < N; i++){
    if (i == N-1) {
      output_file << data[i] << "\n";
    } else {
      output_file << data[i] << ", ";
    }
  }

  // Write runtime duration
  output_file << total_time.count() << "\n";

  // Write positions
  int curr_step = 0;
  for(int i=N; i < timesteps * N * 3 + N; i++){
    if (curr_step == (N*3)-1) {
      // std::cout << "end i : " << i << std::endl;
      output_file << data[i] << "\n";
      curr_step = 0;
    } else {
      curr_step++;
      // std::cout << "comma i : " << data[i] << std::endl;
      output_file << data[i] << ", ";
    }
  }
  output_file.close();

  // Free data
  free(data);

  return 0;
}
