## This case we will try to get the distribution using the montecarlo method

# Randomly th momenta are generated accordint to a normal distribution.
#

import random
import numpy as np
from potential import potential
from energy_measurement import energy


def metropolis(system, temperature, sweeps):

    total_energy = np.zeros(sweeps)
    kinetic_average = np.zeros(sweeps)
    potential_average = np.zeros(sweeps)
    acceptance = 0

    for i in range(sweeps):
        # One sweep is the change in the system for one whole configuration

        # compute the kinetic energy of the middle guy
        kinetic_average[i], potential_average[i], total_energy[i] = energy(system)

        for attempt in range(1, len(system.members) - 1):
            old_potential = potential(system)
            old_kinetic = 0.5 * (system.momentum[attempt] ** 2)

            old_config_displacement = system.trial_move(attempt, 0.1)
            old_config_momentum = system.trial_momentum(attempt, 0.1)

            new_potential = potential(system)
            new_kinetic = 0.5 * (system.momentum[attempt] ** 2)

            delta_potential = new_potential - old_potential
            delta_kinetic = new_kinetic - old_kinetic

            total_energy_dif = delta_kinetic + delta_potential

            if total_energy_dif <= 0:
                acceptance += 1
                pass

            else:
                probability = np.exp((-1 * total_energy_dif) / temperature)

                if random.random() < probability:
                    acceptance += 1
                    pass

                else:
                    system.displacement[attempt] = old_config_displacement
                    system.momentum[attempt] = old_config_momentum

            # make sure that the edge particles are not moved.
            system.displacement[0] = 0.0
            system.displacement[-1] = 0.0

            system.momentum[0] = 0
            system.momentum[-1] = 0

        # this is for the momentum of the system. Simply needs to follow the normal
        #
        system.momentum[0] = 0
        system.momentum[-1] = 0
    print(acceptance / (sweeps * len(system.members)))
    return system, kinetic_average, potential_average, total_energy
