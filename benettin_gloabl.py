# this is for the global lyapunov exponent of the system as a whole
#


import copy
import numpy as np
from Thermalization.Frustrated_Spins.Extra_code.montecarlo_graph import energy
from Velocity_verlet import velver
from SpringSystem import springsystem


def benetin(system, M, dt, epsilon, tau):

    original = copy.deepcopy(system)
    perturbed = copy.deepcopy(system)

    kineticmean = []
    potentialmean = []
    totalmean = []

    delt = np.zeros(M)
    # perturb the whole system
    for target in range(len(perturbed.members)):
        perturbed.perturbation_drift(target, epsilon)
        perturbed.perturbation_kick(target, epsilon)

    lyapcount = 0.0

    for interval in range(M):
        delq = np.zeros(len(perturbed.members))
        delp = np.zeros(len(perturbed.members))
        delcounter = 0.0

        # system A evolving
        original_evolving, kin, pot, tot = velver(original, tau, dt)
        # making system B

        # evolving system B
        perturbed_evolving, kinp, potp, totp = velver(perturbed, tau, dt)

        # finding the difference

        for j in range(len(perturbed.members)):
            delq[j] = (
                perturbed_evolving.displacement[j] - original_evolving.displacement[j]
            )
            delp[j] = perturbed_evolving.momentum[j] - original_evolving.momentum[j]

            delcounter += delq[j] ** 2 + delp[j] ** 2

        # gives us the delta X
        #
        delt[interval] = np.sqrt(delcounter)

        lyapcount += np.log(delt[interval] / epsilon)

        # reset values for the next iteration
        for i in range(len(perturbed.members)):
            perturbed.momentum[i] = perturbed.momentum[i] - epsilon * (
                original_evolving.momentum[i] - perturbed_evolving.momentum[i]
            ) / np.abs(original_evolving.momentum[i] - perturbed_evolving.momentum[i])

            perturbed.displacement[i] = perturbed.displacement[i] + epsilon * (
                original_evolving.displacement[i] - perturbed_evolving.displacement[i]
            ) / np.abs(
                original_evolving.displacement[i] - perturbed_evolving.displacement[i]
            )
        original = copy.deepcopy(original_evolving)

        # this gives us del

    return (lyapcount / (M * tau)), delt
