import numpy as np
import matplotlib.pyplot as plt 

'''
Visulizes the N-body simulation from a dataset
'''
def main():
    f = open("output.txt", 'r')
    
    # Visulization setup
    grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(grid[0:2,0])
    ax1.set_facecolor("blue")

    # Create N x 1 array of masses of planets
    planet_mass = np.random.rand(100,1)

    # Parse file to create N x 3 matrix of current planet positions
    for line in f:
        curr_pos = line.replace("[", "").replace("]", "").replace(",", "").split()
        N = len(curr_pos) / 3 # each planet has (x,y, z)

        # converting all items to float
        for i in range(len(curr_pos)):
            curr_pos[i] = float(curr_pos[i])
        
        planet_pos = np.array(curr_pos)
        planet_pos = planet_pos.reshape(int(N), 3)

        plt.sca(ax1)
        plt.cla()
        plt.scatter(planet_pos[:, 0], planet_pos[:, 1], s=100*planet_mass, color = 'lime', edgecolor='purple')
        ax1.set(xlim=(-3, 3), ylim=(-3, 3))
        ax1.set_aspect('equal', 'box')
        plt.title(label= "N-Body Simulation", fontsize=20, color='black')
        plt.pause(0.01)

    f.close()

main()