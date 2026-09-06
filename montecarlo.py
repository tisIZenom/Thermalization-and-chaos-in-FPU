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

    for i in range(sweeps):
        # One sweep is the change in the system for one whole configuration

        # compute the kinetic energy of the middle guy
        kinetic_average[i], potential_average[i], total_energy[i] = energy(system)

        for attempt in range(len(system.members)):
            target = np.random.randint(len(system.members))

            old_potential = potential(system)

            old_config = system.trial_move(target, 0.01)

            new_potential = potential(system)

            delta_potential = new_potential - old_potential

            if delta_potential <= 0:
                pass

            else:
                probability = np.exp((-1 * delta_potential) / temperature)

                if random.random() < probability:
                    pass

                else:
                    system.displacement[target] = old_config
            # make sure that the edge particles are not moved.
            system.displacement[0] = 0.0
            system.displacement[-1] = 0.0

            system.momentum[0] = 0
            system.momentum[-1] = 0

        # this is for the momentum of the system. Simply needs to follow the normal
        #
        system.momentum = np.random.normal(0, np.sqrt(temperature), len(system.members))

    return system, kinetic_average, potential_average, total_energy
