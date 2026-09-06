## This is one of the crucial parts for the simulation
# need to loop over all the points and make sure that the system can really thermalize
# Using the euler method.

from potential_differentiated import potential_dif
import numpy as np
from energy_measurement import energy


def Langevin(system, total_time, dt, gamma, Temperature):

    time_step = int(total_time / dt)
    N = len(system.members)

    # Expectation value of <pi sr> = T
    total_energy = np.zeros(time_step)
    kinetic_energy = np.zeros(time_step)
    potential_energy = np.zeros(time_step)

    # first the noise:

    # now the master equation:

    for i in range(time_step):
        force = potential_dif(system)

        noise = np.sqrt(2 * gamma * Temperature * dt) * np.random.randn(N)

        # just as a test I will map how the kinetic_energy of a particle goes
        kinetic_energy[i], potential_energy[i], total_energy[i] = energy(system)

        # Technically python supports vector addition so I dont need this loop but I want to visualize the
        # dynamics
        for j in range(N):
            # momentum update
            system.momentum[j] = (
                system.momentum[j]
                + (force[j] * dt)
                - (gamma * system.momentum[j] * dt)
                + noise[j]
            )

            # position update
            system.displacement[j] = system.momentum[j] * dt

            # make sure that the end points dont update:
            system.displacement[0] = 0
            system.momentum[0] = 0
            system.displacement[-1] = 0
            system.momentum[-1] = 0

    return system, kinetic_energy, potential_energy, total_energy
