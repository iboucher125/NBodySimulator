import numpy as np
import math
import time
import matplotlib.pyplot as plt 

''' 

* Simulation of the N-body problem. Includes visulaization of the moemvents
  of planetsin  the simulation. 
* Should also write each planet postion (x,y,z) for each timestep to output file
* Then we can write a python simulator that reads an output file and does the visualization

'''

def getAcc(p, m, G, N):
    # new_acc is N x 3 matrix of updated accelerations (x, y, z for each planet)
    new_acc = np.zeros((len(p), 3))
    
    # get accleration for each planet
    for i in range(N):
        x = 0
        y = 0
        z = 0
        # Get force exerted on each particel THEN sum
        for j in range(N):
            if j != i:
                # Need x, y, z of current neighboring planet [x, y, z] --> 0, 1, 2
                dx = p[j][0]
                dy = p[j][1]
                dz = p[j][2]

                # Calculate inverse with softening length (0.1) -- Prart to account for particles close to eachother
                inv = (math.sqrt((dx**2 + dy**2 + dz**2 + 0.1**2)))**(-1.5)

                # Update acceleration (x, y, z)
                x += m[j] * dx / inv
                y += m[j] * dy / inv
                z += m[j] * dy / inv
        
        # Ajust with Newton's gravitational constant
        x *= G
        y *= G
        z *= G

        # Update with new acceleration
        new_acc[i][0] = x
        new_acc[i][1] = y
        new_acc[i][2] = z

    return new_acc

def format(data):
    print("This function returns the updated data array/matrix")

def generateOutput(data, output_file):
    print("This function writes data to output file")


def main():
    ''' Run N-body Simulation '''
    # Number of particles (start with 2)
    N = 100
    # Newton's Gravitational Constant
    G = 6.67 * 10**-11
    # Random number generator seed
    np.random.seed(811)
    # Start timer -> for performance comparision
    t_start = time.time()

    # Create N x 3 matrix of random starting postion of planets (size N) -> each partile has x,y,z corrdinate
    planet_pos = np.random.randn(N, 3)
    # print("positions: ", planet_pos)

    # could make helper funttion that creates a  

    # Write planet positions to output file --- Not Working!!!!
    f = open("output.csv", "a")
    f.write(str(t_start) + "\n")
    f.write(planet_pos)

    # Create N x 3 array of random starting velocities for each planet
    planet_vel = np.random.randn(N, 3)
    # print("velocities: ", planet_vel)

    # Create N x 1 array of masses of planets --> currently all the same. Make it random?
    planet_mass = np.random.rand(100,1)
    # print(planet_mass)

    # Get starting accelerations of planets (N x 3 matrix)
    planet_acc = getAcc(planet_pos, planet_mass, G, N)

    # Set number of timesteps (number of interations for simulation)
    td = 0.01 # Timestep duration
    # timesteps = int(1/td) # not sure what to put here
    timesteps = 100

    # Visulization setup
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])

    # Loop for number of timesetps
    for i in range(timesteps): # change 5 to timesteps
        # Leapfrog integration
        # 1) first half kick
        planet_vel += planet_acc * (td/2.0)

        # 2) Drift --> Update positions of all planets
        planet_pos += planet_vel * td

        # 3) Get new accleration for each planet
        planet_acc = getAcc(planet_pos, planet_mass, G, N)

        # 4) Second half of kick --> update velocities
        planet_vel += planet_acc * (td/2.0)

        # 5) Plot
        plt.sca(ax1)
        plt.cla()
        plt.scatter(planet_pos[:, 0], planet_pos[:, 1], s=50*planet_mass, color = 'green')
        ax1.set(xlim=(-8, 8), ylim=(-8, 8))
        ax1.set_aspect('equal', 'box')
        ax1.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        ax1.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        plt.pause(0.001)

        # 6) Write to output file
        f.write(str(time.time()) + "\n")
        f.write(planet_pos + "\n")
    
    # Get end time of simuation
    t_end = time.time()
    print("Simulation duration: ", t_end - t_start)
    f.close()

main()