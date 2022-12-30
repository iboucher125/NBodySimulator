# NBodySimulator
## Summary
This project simulates how particles move through space using the N-body problem. The project consists of a Python based serial implementation and a C++/Cuda based parallel implementation.

## Steps To Run N-Body Simulation:
1. Run nbody.py OR nbody.cu to generate data from the N-body simulation.
2. Run simulator.py to visualize the simulation.
3. (Optional) Run evaluate.py to visualize the performance evaluation of the serial and parallel implementations.
4. (Optional) Run cuda_eval.py to visualize the performance evaluation of nbody_not_opt.cu and nbody.cu.

## Components
**Python**
* nbody.py - Generates positions of N particles over a given number of timesteps. This program is located in the python_code/ directory. N and timesteps are command line arguments. This program generates an output file called output_py.txt.

This is an example of running this program for N=100 and timesteps=50:

```
python3 nbody.py 100 50
 ```

* simulator.py - Plots the movement of particle postions generated from nbody.py and nbody.cu. The input file used in this program is specified as a command line argument.

This is an example of running the simulation with an input file called output_py.txt:

```
python3 simulator.py output_py.txt

```

* evaluate.py - Plots the runtime of both nbody.py and nbody.cu as the number of bodies increases in powers of two and the number of timestesp is 150. The number of iterations (times number of bodies increases) is specified as a command line argument.

This is an example of running the evaluation with iterations=5 (N=[2,4,8,16,32]):

```
python3 evaluate.py 5

```

* cuda_eval.py - Plots the runtime of both nbody.cu and nbody_not_opt.cu as the number of bodies increases in powers of two and the number of timestesp is 150. The number of iterations (times number of bodies increases) is specified as a command line argument.

This is an example of running the evaluation with iterations=5 (N=[2,4,8,16,32]):

```
python3 cuda_eval.py 5

```

**C++/Cuda**
* nbody.cu - Generates positions of N particles over a given number of timesteps. This program is located in the src/ directory.

To run this program you must set up a `build/` directory for CMake:

```
mkdir build
cd build
```

Inside the build directory, run CMake to generate a Makefile for the project.

```
cmake ..
```

After compliling, this program can be run using N and timesteps as command line arguments. This program generates an output file called output_cu.txt.

This is an example of running this program for N=100 and timesteps=50:

```
./nbody 100 50
```

* nbody_not_opt.cu - This is an earlier version of our GPU based n-body simulation where the time integration is implemnted in a CPU fucntion rather than a GPU kernel. This program Generates positions of N particles over a given number of timesteps. This program is located in the src/ directory.

```
./nbody_not_opt 100 50
```

**Additional Directories**
* data/ - Contains the output files generated by nbody.py, nbody.cu, and nbody_not_opt.

* figures/ - Contains examples of figures generated by evalute.py and cuda_eval.py.
