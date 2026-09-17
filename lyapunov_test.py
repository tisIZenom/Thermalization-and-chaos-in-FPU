## this is to test whether the system is still thermal or not after the


import numpy as np
import matplotlib.pyplot as plt

from CreateChain import create_chain
from LangevinDynamcis import Langevin
from SpringSystem import springsystem
from velocity_verlet_thermalized import velver
from benettin_local_fixed import benettin


# Simulation parameters
N = 600

temperature = 10
dt = 0.01
total_time = 25
gamma = 0.5


# create system langeivn

members, neighbours = create_chain(N)

system_langevin = springsystem(members, neighbours, 0, 0.7)


# now do langevin LangevinDynamcis
#
system_langevined, kinetic, potential, total = Langevin(
    system_langevin, total_time, dt, gamma, temperature
)

lyapunov, local_lyap, delta, kineticmean, potentialmean, totalmean = benettin(
    system_langevined, 300, 10, 0.01, 0.001, 2
)


print(lyapunov)
print(local_lyap)
plt.plot(totalmean)

plt.show()
