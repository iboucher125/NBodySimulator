import sys
import os
import matplotlib.pyplot as plt

'''
* This program generates a graph compaing runtime of serial and parallel 
implemetation of the N-body simulation.
* N (# of bodies) increases in powers of 2
* Number of timsteps is constant (150).
'''

def runSim(bodies, iterations):
    py_runtimes = []
    cu_runtimes = []
    for i in range(iterations):
        # Get serial nbody runtime
        os.system("python3 nbody.py " + str(bodies) + " 150")
        f_py = open("../data/output_py.txt", 'r')
        content_py = f_py.readlines()
        py_runtimes.append(float(content_py[2]))
        f_py.close()

        # Get parallel nbody runtime
        os.system("./nbody " + str(bodies) + " 150")
        f_cu = open("../data/output_cu.txt", 'r')
        content_cu = f_cu.readlines()
        cu_runtimes.append(float(content_cu[2]))
        f_cu.close()
        
        # Increment number of bodies (powers of 2)
        bodies *= 2
        
    return py_runtimes, cu_runtimes

def graph(py, cu, iterations):
    N = 2
    x_values = []
    for i in range(iterations):
        x_values.append(N)
        N *= 2
    plt.title(label="Runtime Comparision", fontsize=20, color='black')
    plt.plot(x_values, py, label="Serial Implementation")
    plt.plot(x_values, cu, label="Parallel Implementation")
    plt.xlabel('Number of Bodies')
    plt.ylabel('Runtime (sec)')
    plt.show()

def main():
    ''' Run Evaluation of Serial vs Parallel Implementaiton '''

    iterations = int(sys.argv[1])
    py, cu = runSim(2, iterations)
    graph(py, cu, 5)

main()