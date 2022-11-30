#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include "utils.h"
#include <vector>
#include <tuple>

#define N_THREADS 1024
#define N_BLOCKS 16

/*** GPU functions ***/
__global__ void init_get_acc_kernel(curandState *state) {
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void get_acc_kernel(float *map, int rows, int cols, int* bx, int* by,
                                   int steps, curandState *state) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  //TODO: 
}

/*** CPU functions ***/
curandState* init_rand() {
  curandState *d_state;
  cudaMalloc(&d_state, N_BLOCKS * N_THREADS * sizeof(curandState));
  init_rand_kernel<<<N_BLOCKS, N_THREADS>>>(d_state);
  return d_state;
}

float random_walk(float* map, int rows, int cols, int steps) {
//   
}


int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <map_file> \n", argv[0]);
    return 1;
  }

  float *map;
  int rows, cols;
  read_bin(argv[1], &map, &rows, &cols);

  printf("%d %d\n", rows, cols);

  // As a starting point, try to get it working with a single steps value
  int steps = 256;

  float max_val = random_walk(map, rows, cols, steps);
  printf("Random walk max value: %f\n", max_val);
  
  float max_local = local_max(map, rows, cols, steps);
  printf("Local max value: %f\n", max_local);

  float max_local_res = local_max_restart(map, rows, cols, steps);
  printf("Local restart max value: %f\n", max_local_res);
  

  return 0;
}
