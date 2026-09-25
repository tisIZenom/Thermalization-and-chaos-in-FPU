# This is just to measure the distance bw two trajectories
#
# Purely the lyapunov exponent no normalization to see what tau should be:
#


import copy
import numpy as np
from Velocity_verlet import velver


def lyapunov(system, target, dt, epsilon, tau):

    original = copy.deepcopy(system)
    perturbed = copy.deepcopy(system)

    timesteps = int(tau / dt)

    # --------------------------------------------------
    # Create initial perturbation
    # --------------------------------------------------

    N = len(system.members)

    dq = np.zeros(N)
    dp = np.zeros(N)

    dq[target] = 1.0
    dp[target] = 1.0

    norm = np.sqrt(np.sum(dq**2) + np.sum(dp**2))

    dq *= epsilon / norm
    dp *= epsilon / norm

    distance = np.zeros(timesteps)

    perturbed.displacement[target] += dq[target]
    perturbed.momentum[target] += dp[target]

    for step in range(timesteps):
        original_evolving, kin, pot, tot = velver(original, dt, dt)

        perturbed_evolving, kin, pot, tot = velver(perturbed, dt, dt)

        dq = perturbed_evolving.displacement - original_evolving.displacement
        dp = perturbed_evolving.momentum - original_evolving.momentum

        distance[step] = np.sqrt(np.sum(dq**2) + np.sum(dp**2))

    return distance
