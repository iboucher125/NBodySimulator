# NBodySimulator
## Summary
This project simulates how particles move through space using the N-body problem. The project consists of a Python based serial implementation and a C++/Cuda based parallel implementation.

## Steps To Run N-Body Simulation:
1. Run nbody.py OR nbody.cu to generate data from the N-body simulation.
2. Run simulator.py to visulize the simulation.
3. (Optional) Run performance evaluation of Python and Cuda versions (not implemented yet...).

## Components
**Python**
* nbody.py - Generates positions of N particles over given timesteps. This program is located in the python_code/ directory. N and timesteps are command line arguments. This program generates an output file called output_py.txt.

This is an example of running this program for N=100 and timesteps=50:

```
python3 nbody.py 100 50
 ```

* simulator.py - Plots the movement of particle postions generated from nbody.py and nbody.cu. The input file used in this program is specified as a command line argument.

This is an example of running the simulation with an input file called output_py.txt:

```
python3 simulator.py output_py.txt

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
nbody.cu 100 50
```
