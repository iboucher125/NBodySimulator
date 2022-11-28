import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt 

def getAcc(p, m, G, N):
    # softening is the softening length
    # new_acc is N x 3 matrix of updated accelerations
    new_acc = np.zeros((len(p), 3))
    
    # get accleration for each planet
    for i in range(N):
        x = 0
        y = 0
        z = 0
        # Get force exerted on each particel THEN sum
        for j in range(N):
            if j != i:
                # [x, y, z] --> 0, 1, 2
                dx = p[j][0]
                dy = p[j][1]
                dz = p[j][2]

                # Calculate inverse with softening length (0.1) -- account for particles close to eachother
                inv = (math.sqrt((dx**2 + dy**2 + dz**2 + 0.1**2)))**3

                # Update acceleration (x, y, z)
                x += m[j] * (dx / inv) # HERE!!
                y += m[j] * (dy / inv)
                z += m[j] * (dy / inv)
        
        # Ajust with Newton's gravitational constant
        x *= G
        y += G
        z *= G
        # x = x * G
        # y = y * G
        # z = z * G

        # Update with new acceleration
        new_acc[i][0] = x
        new_acc[i][1] = y
        new_acc[i][2] = z

    return new_acc


def main():
    ''' Run N-body Simulation '''
    # Number of particles (start with 2)
    N = 100
    # Newton's Gravitational Constant
    G = 6.67 * 10**-11 # Google
    # End time
    t_end = 5
    # Random number generator seed
    np.random.seed(811)

    # Start timer
    t_start = time.time()

    # Create matrix of random starting postion of planets (size N) -> each partile has x,y,z corrdinate
    # pos is an N x 3 matrix of positions
    planet_pos = np.random.randn(N, 3)
    # print("positions: ", planet_pos)

    # Write planet positions to output file
    with open("output.csv","w") as f:
        f.write(str(t_start) + "\n")

    # reopen and append to it
    with open("output.csv","a") as f:
        np.savetxt(f, planet_pos, fmt="%d")
        f.write("\n")

    # Set random starting velocities for each planet
    # vel is an N x 3 matrix of velocities
    planet_vel = np.random.randn(N, 3)
    # print("velocities: ", planet_vel)

    # Set mass of planets --> all the same or make it random?
    # mass is an N x 1 vector of masses
    planet_mass = 30 * np.ones((N, 1))/N # total mass is 30
    # print(planet_mass)

    # Get starting accelerations of planets
    planet_acc = getAcc(planet_pos, planet_mass, G, N)

    # Set number of timesteps (number of interations for simulation)
    dt = 0.01 # Timestep duration
    timesteps = int(t_end/dt)

    # Visulization setup
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])

    # Loop for number of timesetps
    for i in range(5):
        # Leapfrog integration
        # 1) first half kick
        planet_vel += planet_acc * (dt/2.0)

        # 2) Drift --> Update positions of all planets
        planet_pos += planet_vel * dt

        # 3) Get new accleration for each planet
        planet_acc = getAcc(planet_pos, planet_mass, G, N)

        # 4) Second half of kick --> update velocities
        planet_vel += planet_acc * (dt/2.0)

        # 5) Plot
        plt.sca(ax1)
        plt.cla()
        plt.scatter(planet_pos[1:, 0], planet_pos[1:, 1], s=10, color = 'green')
        ax1.set(xlim=(-8, 8), ylim=(-8, 8))
        ax1.set_aspect('equal', 'box')
        ax1.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        ax1.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
        plt.pause(0.001)

        # 6) Write to output file
        with open("output.csv","a") as f:
            f.write(str(time.time()) + "\n")
            np.savetxt(f, planet_pos, fmt="%d")
            f.write("\n")
    
    # Get end time of simuation
    t_end = time.time()
    print("Simulation duration: ", t_end - t_start)
    f.close()

main()