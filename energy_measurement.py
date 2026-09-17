## This is for measuring the energy of the system.

import numpy as np


def energy(system):
    N = len(system.members)
    q = system.displacement[1 : N - 1]
    p = system.momentum[1 : N - 1]

    kinetic = 0.5 * np.sum(p**2)

    potential = 0.0

    for i in range(len(q) - 1):
        r = q[i + 1] - q[i]

        potential += 0.5 * (r**2)
        potential += (system.beta / 4) * (r**4)

    total = (kinetic + potential) / len(q)

    momentum_average = (2 * kinetic) / len(q)

    potential_average = potential / len(q)

    # Need to make a per site energy measuring devise.
    #
    # energy per site :
    Energy_per_site = np.zeros(len(q))

    for i in range(len(q) - 1):
        Energy_per_site[i] = (
            0.5 * (p[i] ** 2)
            + 0.5 * ((q[i + 1] - q[i]) ** 2)
            + system.beta * 0.25 * ((q[i + 1] - q[i]) ** 4)
        )

    return momentum_average, potential_average, total
