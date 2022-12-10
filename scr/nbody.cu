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

#define N_THREADS 2
#define N_BLOCKS 1

/*** GPU functions ***/
/*** Get accelerations ***/
// Run N-body simulation
__global__ void generate_data_kernel(double *v, double *a, double td) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  //TODO: move leap-frog integration from CPU function to here
  // change other kernels to __device__
}

// Update acceleration of planets
__global__ void get_acc_kernel(double *p, double *m, double *a, double G, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Accleration (x, y, z) for plaent with id tid
  double x = 0;
  double y = 0;
  double z = 0;

  for(int i=0; i<N; i++){
    if(i != tid){
      // Get difference in position of neighboring planet
      double dx = p[0 + i * 3] - p[0 + tid * 3];
      double dy = p[1 + i * 3] - p[1 + tid * 3];
      double dz = p[2 + i * 3] - p[2 + tid * 3];
      // printf("p1-pj1=dx: %e - %e = %e; p2-pj2=dy: %e - %e = %e; p3-pj3=dz: %e - %e = %e\n", p[0 + i * 3], p[0 + tid * 3], dx, p[1 + i * 3], p[1 + tid * 3], dy, p[2 + i * 3],p[2 + tid * 3],dz);

      // Calculate inverse with softening length (0.1) -- Part to account for particles close to eachother
      double inv = pow(pow(dx, 2) + pow(dy, 2) + pow(dz, 2) + pow(0.1, 2), -1.5);
      // printf("inv: %e\n", inv);
      // printf("m: %e\n", m[i]);

      // calculate inversepmlanet_mass
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

// Update velocity of singular planet (used each half kick)
__global__ void get_vel_kernel(double *v, double *a, double td) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // new velocity = velocity + acceleration * (td / 2.0)
  
  v[0 + tid * 3] = v[0 + tid * 3] + (a[0 + tid * 3] * td / 2.0);
  v[1 + tid * 3] = v[1 + tid * 3] + (a[1 + tid * 3] * td / 2.0);
  v[2 + tid * 3] = v[2 + tid * 3] + (a[2 + tid * 3] * td / 2.0);

  // printf("v = v + a(td/2): %e = %e + %e(%e/2)\n",v[0 + tid * 3] , v[0 + tid * 3] , a[0 + tid * 3] , td );
  // printf("v = v + a(td/2): %e = %e + %e(%e/2)\n",v[1 + tid * 3] , v[1 + tid * 3] , a[1 + tid * 3] , td );
  // printf("v = v + a(td/2): %e = %e + %e(%e/2)\n",v[2 + tid * 3] , v[2 + tid * 3] , a[2 + tid * 3] , td );
}

  // Update position of singular planet (drift)
__global__ void get_pos_kernel(double *p, double *v, double *data, double td, int N, int timesteps, int i) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // new position = position + velocity * td
  p[0 + tid * 3] = p[0 + tid * 3] + (v[0 + tid * 3] * td);
  p[1 + tid * 3] = p[1 + tid * 3] + (v[1 + tid * 3] * td);
  p[2 + tid * 3] = p[2 + tid * 3] + (v[2 + tid * 3] * td);

  // printf("p=p+v*td: %e = %e+%e*%e\n", p[0 + tid * 3], p[0 + tid * 3], v[0 + tid * 3], td);
  // printf("p1=p1+v*td: %e = %e+%e*%e\n", p[1 + tid * 3], p[1 + tid * 3], v[1 + tid * 3], td);
  // printf("p2=p2+v*td: %e = %e+%e*%e\n", p[2 + tid * 3], p[2 + tid * 3], v[2 + tid * 3], td);

  // idx probably wrongs here!!! --> also error "expression must have integral or unscoped enum type"
  // printf("tid (planet): %d; i (ts): %d\n", tid, i);
  // printf("time: %d\n", timesteps);
  // data[0 + (tid * 3 * i) + (i * 3)] = p[0 + tid * 3];
  // data[1 + (tid * 3 * i) + (i * 3)] = p[1 + tid * 3];
  // data[2 + (tid * 3 * i) + (i * 3)] = p[2 + tid * 3];

  data[i * N * 3 + 3 * tid + 0] = p[0 + tid * 3];
  data[i * N * 3 + 3 * tid + 1] = p[1 + tid * 3];
  data[i * N * 3 + 3 * tid + 2] = p[2 + tid * 3];

  printf("d0[%e] = p0[%e]; d1[%e] = p1[%e]; d2[%e] = p2[%e]\n", data[(i * N * 3 + 3 * tid + 0)], p[(0 + tid * 3)], data[i * N * 3 + 3 * tid + 1], p[(1 + tid * 3)], data[i * N * 3 + 3 * tid + 2], p[(2 + tid * 3)]);
  // printf("data[%d] = p[%d]\n", 1 + (tid * 3 * timesteps) + (i * 3), 1 + tid * 3);
  // printf("data[%d] = p[%d]\n", 2 + (tid * 3 * timesteps) + (i * 3), 2 + tid * 3);
}


/*** CPU functions ***/

// Returns data from N-body simulation
double* n_body(double *planet_mass, int N, double G, double td, int timesteps) {
  // N x 3 matrix of random starting positions of planets (N x (x,y,z))
  double* planet_pos = new double[N*3];
  double* d_planet_pos;
  // N x 3 matrix of random velocities of planets
  double* planet_vel = new double[N*3];
  double* d_planet_vel;
  // N x 1 vector of random masses of planets
  // double* planet_mass = new double[N];
  double* d_planet_mass;
  // N x 1 vector of random masses of planets
  double* planet_acc;
  double* d_planet_acc;
  // N x 3 x # timesteps matrix of positions of planets over all timesteps
  double* data = new double[N * 3 * timesteps];
  double* d_data;

  // Allocate memory
  planet_acc = (double*)malloc((N*3)* sizeof(double));
  planet_pos = (double*)malloc((N*3)* sizeof(double));
  planet_vel = (double*)malloc((N*3)* sizeof(double));
  planet_mass = (double*)malloc(N * sizeof(double));
  data = (double*)malloc(N * 3 * timesteps * sizeof(double));

  cudaMalloc(&d_planet_mass, N * sizeof(double));
  cudaMalloc(&d_planet_pos, N * 3 * sizeof(double));
  cudaMalloc(&d_planet_vel, N * 3 * sizeof(double));
  cudaMalloc(&d_planet_acc, N * 3 * sizeof(double));
  cudaMalloc(&d_data, N * 3 * timesteps * sizeof(double));

  // Create N x 1 array of masses of planets
  for(int i=0; i<N; i++){
    planet_mass[i] = rand()/double(RAND_MAX)*1.f+0.f;
    // std::cout << planet_mass[i] <<std::endl;
  }

  // Create N x 3 matrix of random starting velocities & positions for each planet
  for(int i= 0; i<N; i++){
    planet_pos[0 + 3 * i] = rand()/double(RAND_MAX)*1.f+0.f;
    planet_pos[1 + 3 * i] = rand()/double(RAND_MAX)*1.f+0.f;
    planet_pos[2 + 3 * i] = rand()/double(RAND_MAX)*1.f+0.f;
    planet_vel[0 + 3 * i] = rand()/double(RAND_MAX)*1.f+0.f;
    planet_vel[1 + 3 * i] = rand()/double(RAND_MAX)*1.f+0.f;
    planet_vel[2 + 3 * i] = rand()/double(RAND_MAX)*1.f+0.f;
    // Note sure if this idx is correct!
    data[0 + 3 * i] = planet_pos[0 + 3*i];
    data[1 + 3 * i] = planet_pos[1 + 3*i];
    data[2 + 3 * i] = planet_pos[2 + 3*i];
  }

  // Copy variables from host to device
  cudaMemcpy(d_planet_mass, planet_mass, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_planet_pos, planet_pos, N * 3 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_planet_vel, planet_vel, N * 3 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, data, N * 3 * timesteps * sizeof(double), cudaMemcpyHostToDevice);

  // Get acceleration of planets --> call GPU kernel here
  get_acc_kernel<<<N_BLOCKS, N_THREADS>>>(d_planet_pos, d_planet_mass, d_planet_acc, G, N);

  // Copy new accelerations device to host
  cudaMemcpy(planet_acc, d_planet_acc, N * 3 * sizeof(double), cudaMemcpyDeviceToHost);

  // Loop for number of timesteps --> timestep 0 already complete
  for(int i = 1; i <= timesteps; i++){
    // Have to call multiple kernels and use cudaDeviceSynchronize()
    // Use leapfrog integration
    // 1) First half kick --> update velocities
    get_vel_kernel<<<N_BLOCKS, N_THREADS>>>(d_planet_vel, d_planet_acc, td);
    cudaDeviceSynchronize();

    // 2) Drift --> update positions
    get_pos_kernel<<<N_BLOCKS, N_THREADS>>>(d_planet_pos, d_planet_vel, d_data, td, N, timesteps, i);
    cudaDeviceSynchronize();

    // 3) update acceleration with new positions
    get_acc_kernel<<<N_BLOCKS, N_THREADS>>>(d_planet_pos, d_planet_mass, d_planet_acc, G, N);
    cudaDeviceSynchronize();
    
    // 4) Second half od kick --> update velocities again
    get_vel_kernel<<<N_BLOCKS, N_THREADS>>>(d_planet_vel, d_planet_acc, td);
  }

  // Copy varibles device to host --> maybe just need 
  cudaMemcpy(data, d_data, (N * 3 * timesteps) * sizeof(double), cudaMemcpyDeviceToHost);

  // Free memory
  cudaFree(d_planet_pos);
  cudaFree(d_planet_vel);
  cudaFree(d_planet_acc);
  cudaFree(d_planet_mass);
  cudaFree(d_data);
  free(planet_pos);
  free(planet_vel);
  free(planet_acc);
  free(planet_mass); 

  // Return all positions --> data
  return data;
}


int main(int argc, char** argv) {
  // Number of planets 
  int N = 5;
  // Newton's Gravitational Constant
  double G = pow(6.67 * 10, -11);
  
  // Start time of simulation
  auto t_start = std::chrono::high_resolution_clock::now();

  //  Set number of timesteps (number of interations for simulation)
  double td = 0.01;
  int timesteps = 5;

  // Create N x 1 array of masses of planets
  double *planet_mass;
  planet_mass = (double *)malloc(N);
  // for(int i=0; i<N; i++){
  //   planet_mass[i] = rand()/double(RAND_MAX)*1.f+0.f;
  // }
  // call CPU function here!!
  // for(int i=0; i < N; i++){
  //   std::cout << planet_mass[i] << std::endl;
  // }
  double* data = n_body(planet_mass, N, G, td, timesteps);
  // for(int i=0; i < N; i++){
  //   std::cout << planet_mass[i] << std::endl;
  // }

  // Write to output file
  std::ofstream output_file;
  output_file.open("../data/output_cu.txt");
  output_file << "Positions of " << N << " planets over " << timesteps <<" timesteps: \n";
  // Write masses
  for(int i=0; i < N; i++){
    if (i == N -1){
      output_file << planet_mass[i] << "\n";
    }
    else{
      output_file << planet_mass[i] << ", ";
    }
  }

  // Write positions
  int curr_step = 0;
  for(int i=0; i < timesteps; i++){
    for(int j=0; j < N * 3; j++){
      if (j == N * 3 -1){
        output_file << data[j + (N * 3 * i)] << "\n";
      }
      else{
        output_file << data[j + (N * 3 * i)] << ", ";
      }
    }
  }
  output_file.close();

  // Free positions
  free(data);

  // End time of simulation
  auto t_end = std::chrono::high_resolution_clock::now();

  // Computer duration --> need to look up how... I don't remember --> print duration
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
  std::cout << "Computation Duration: " << total_time.count() << std::endl;

  return 0;
}
