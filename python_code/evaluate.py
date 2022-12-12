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
    py_runtimes = []
    cu_runtimes = []
    n_vals = []
    for i in range(iterations):
        n_vals.append(N)
        print("Running simulation with N = " + str(N))
        # Get serial nbody runtime
        os.system("python3 nbody.py " + str(N) + " 150")
        f_py = open("../data/output_py.txt", 'r')
        content_py = f_py.readlines()
        py_runtimes.append(float(content_py[2]))
        f_py.close()

        # Get parallel nbody runtime
        os.chdir("../build")
        os.system("./nbody " + str(N) + " 150")
        os.chdir("../python_code")
        f_cu = open("../data/output_cu.txt", 'r')
        content_cu = f_cu.readlines()
        cu_runtimes.append(float(content_cu[2]))
        f_cu.close()
        
        # Increment number of N (powers of 2)
        N *= 2
        
    return py_runtimes, cu_runtimes, n_vals

def graph(py, cu, n_vals):
    plt.title(label="Runtime Comparision", fontsize=20, color='black')
    plt.plot(n_vals, py, label="Serial Implementation")
    plt.plot(n_vals, cu, label="Parallel Implementation")
    plt.xlabel('# of Particles')
    plt.ylabel('Runtime (sec)')
    plt.show()

def main():
    ''' Run Evaluation of Serial vs Parallel Implementaiton '''

    iterations = int(sys.argv[1])
    py, cu, n_vals= runSim(2, iterations)
    graph(py, cu, n_vals)

main()