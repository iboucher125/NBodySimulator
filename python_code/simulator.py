import numpy as np
import matplotlib.pyplot as plt 

'''
* Visulizes the N-body simulation from a dataset
'''

def main():
    # Path for output file
    #f = open("../data/output_py.txt", 'r')
    f = open("../data/output_cu.txt", 'r')
    content = f.readlines()

    # Recreate N x 1 array of planet masses
    masses = content[1].replace("[", "").replace("]", "").replace(",", "").split()
    planet_mass = np.array(masses).astype(float)
    # print(planet_mass)
    
    # Visulization setup
    grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(grid[0:2,0])
    ax1.set_facecolor("blue")

    # Parse file to create N x 3 matrix of current planet positions THEN plot
    for line in content[2:]:
        # Get planet positions
        planet_pos = np.array(line.replace("[", "").replace("]", "").replace(",", "").split()).astype(float)
        N = int(len(planet_pos) / 3) # each planet has (x, y, z)
        planet_pos = planet_pos.reshape(N, 3)

        # Plot
        plt.sca(ax1)
        plt.cla()
        plt.scatter(planet_pos[:, 0], planet_pos[:, 1], s=100*planet_mass, color = 'lime', edgecolor='purple')
        ax1.set(xlim=(-2, 2), ylim=(-2, 2))
        ax1.set_aspect('equal', 'box')
        plt.title(label= "N-Body Simulation", fontsize=20, color='black')
        plt.pause(0.001)

    f.close()

main()