import numpy as np
import matplotlib.pyplot as plt 
import sys

'''
* This program visulizes the N-body simulation from a dataset
'''

def main():
    # Path for output file
    f = open("../data/" + sys.argv[1], 'r')
    content = f.readlines()

    # Recreate N x 1 array of particle masses
    masses = content[1].replace("[", "").replace("]", "").replace(",", "").split()
    particle_mass = np.array(masses).astype(float)
    
    # Visulization setup
    grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(grid[0:2,0])
    ax1.set_facecolor("blue")

    # Parse file to create N x 3 matrix of current particle positions THEN plot
    for line in content[3:]:
        # Get particle positions
        particle_pos = np.array(line.replace("[", "").replace("]", "").replace(",", "").split()).astype(float)
        N = int(len(particle_pos) / 3) # each particle has (x, y, z)
        particle_pos = particle_pos.reshape(N, 3)

        # Plot
        plt.sca(ax1)
        plt.cla()
        plt.scatter(particle_pos[:, 0], particle_pos[:, 1], s=100*particle_mass, color = 'lime', edgecolor='purple')
        ax1.set(xlim=(-3, 3), ylim=(-3, 3))
        ax1.set_aspect('equal', 'box')
        plt.title(label= "N-Body Simulation", fontsize=20, color='black')
        plt.pause(0.001)

    f.close()

main()