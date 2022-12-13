import sys
import os
import matplotlib.pyplot as plt

'''
* This program generates a graph comparing runtime of serial and parallel 
implemetation of the N-body simulation.
* N increases in powers of 2
* Number of timsteps is constant (150).
'''

def runSim(N, iterations):
    cpu_runtimes = []
    gpu_runtimes = []
    n_vals = []
    for i in range(iterations):
        n_vals.append(N)
        print("Running simulation with N = " + str(N))

        # Get runtime of version with time integration in CPU function
        os.chdir("../build")
        os.system("./nbody_not_opt " + str(N) + " 150")
        f_cpu = open("../data/output_cu.txt", 'r')
        content_cpu = f_cpu.readlines()
        cpu_runtimes.append(float(content_cpu[2]))
        f_cpu.close()

        # Get runtime of version with time integration in GPU kernel
        os.system("./nbody " + str(N) + " 150")
        f_gpu = open("../data/output_cu.txt", 'r')
        content_gpu = f_gpu.readlines()
        gpu_runtimes.append(float(content_gpu[2]))
        f_gpu.close()

        os.chdir("../python_code")

        # Increment number of N (powers of 2)
        N *= 2
        
    return cpu_runtimes, gpu_runtimes, n_vals

def graph(cpu, gpu, n_vals):
    plt.xscale("log")
    plt.title(label="Runtime Comparision", fontsize=20, color='black')
    plt.plot(n_vals, cpu, label="nbody_not_opt.cu")
    plt.plot(n_vals, gpu, label="nbody.cu")
    plt.xlabel('Number of Particles (log scale)')
    plt.ylabel('Runtime (sec)')
    plt.legend()
    plt.savefig("../figures/Cuda_evaluate_" + str(len(n_vals)) + "_iter.png")

def main():
    ''' Run Evaluation of original parallel version vs final parallel version '''

    iterations = int(sys.argv[1])
    cpu, gpu, n_vals= runSim(2, iterations)
    graph(cpu, gpu, n_vals)

main()