## This is a complicated function for the potential at any given point. I think it needs two differnet values

import numpy as np


def potential(system):

    q = system.displacement
    N = len(system.members)
    beta = system.beta
    potential = 0.0

    for i in range(1, N - 1):
        # compute the distances first
        distance = q[i + 1] - q[i]

        # compute the potential at all the points in the system
        potential += ((distance**2) / 2) + ((beta / 4) * (distance**4))

    return potential
