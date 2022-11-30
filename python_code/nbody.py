import numpy as np
import math
import time
import matplotlib.pyplot as plt 

''' 
* Generate momvent of planets using the N-body problem. 
* Produces an output file with planets' positions over number of timesteps.
'''

# Return accelertaions (x, y, z) for each planet
# p: Array of planet postions (x, y, z)
# m: Array of planet masses
# G: Newton's Gravitational Constant
# N: Number of planets
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

# Return list with postions of planets for each timestep
# data: list of postions of planets for each timestep
# p: Planet postions (x, y, z)
def format(data, p):
    all_pos = []
    for i in range(len(p)):
        for j in range(3):
            all_pos.append(p[i][j])
    data.append(all_pos)
    
    return data

# Write data to output file
def generateOutput(data, output_file):
    f = open(output_file, 'a')
    for i in range(len(data)):
        f.write(str(data[i]) + "\n")
    f.close()


def main():
    ''' Generate N-body Simulation Data '''

    # Number of particles (start with 2)
    N = 100
    # Newton's Gravitational Constant
    G = 6.67 * 10**-11
    # Random number generator seed
    np.random.seed(811)
    # Start timer -> for performance comparision
    t_start = time.time()
    # Name of output file
    output = "output.txt"

    # Create N x 3 matrix of random starting postion of planets (size N) -> each partile has x,y,z corrdinate
    planet_pos = np.random.randn(N, 3)

    # Data that will be outputed
    data = []
    data = format(data, planet_pos)

    # Create N x 3 matrix of random starting velocities for each planet
    planet_vel = np.random.randn(N, 3)

    # Create N x 1 array of masses of planets
    planet_mass = np.random.rand(N,1)

    # Get starting accelerations of planets (N x 3 matrix)
    planet_acc = getAcc(planet_pos, planet_mass, G, N)

    # Set number of timesteps (number of interations for simulation)
    td = 0.01 # Timestep duration
    timesteps = 50

    f = open(output, 'w+')
    f.write("Positions of " + str(N) + " planets over " + str(timesteps) + " timesteps: \n")
    f.write(str(planet_mass.tolist()) + "\n")
    f.close()

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

        # 6) Append to data that will be outputed
        data = format(data, planet_pos)
    
    generateOutput(data, output)

    # Get end time of simuation
    t_end = time.time()
    print("Computation duration: ", t_end - t_start)

main()