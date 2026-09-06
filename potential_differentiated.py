## using the equation that I have differentiated we getL

import numpy as np

# Returns a force value associated with each member in the system


def potential_dif(system):
    beta = system.beta
    q = system.displacement
    N = len(system.members)
    force = np.zeros(N)

    for i in range(1, N - 1):
        # these are the distances for both left and right
        right = q[i + 1] - q[i]
        left = q[i] - q[i - 1]

        force[i] = -right - beta * (right**3) + left + beta * (left**3)

        force[i] = -force[i]
        # fixed boundary:
        force[0] = 0.0
        force[-1] = 0.0

    return force
