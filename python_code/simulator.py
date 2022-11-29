import numpy as np
import matplotlib.pyplot as plt 

'''
Visulizes the N-body simulation from a dataset
'''
def main():
    f = open("output.txt", 'r')

    # Parse file to create N x 3 matrix of current planet positions
    planet_pos = []
    for line in f:
        curr_pos = line.split()
        N = len(curr_pos) / 3 # each planet has (x,y, z)
        idx = 0
        for i in range(N):
            pos = [curr_pos[idx],curr_pos[idx + 1], curr_pos[idx + 2]]
            planet_pos.append(pos)

    # Visulization setup
    grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(grid[0:2,0])

    # Create N x 1 array of masses of planets
    planet_mass = np.random.rand(100,1)

    # Timsteps
    for i in range(len(f)):
        plt.sca(ax1)
        plt.cla()
        plt.scatter(planet_pos[:, 0], planet_pos[:, 1], s=50*planet_mass, color = 'lime', edgecolor='purple')
        ax1.set(xlim=(-8, 8), ylim=(-8, 8))
        ax1.set_aspect('equal', 'box')
        ax1.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        ax1.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        plt.pause(0.001)

    f.close()

main()