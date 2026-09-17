# this is for the global lyapunov exponent of the system as a whole
#


import copy
import numpy as np
from Velocity_verlet import velver
from SpringSystem import springsystem
from momentum_statistics import momentum_check


def benetin(system, M, dt, epsilon, tau, temperature):

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

    lyapcount = np.zeros(M)

    for interval in range(M):
        delq = np.zeros(len(perturbed.members))
        delp = np.zeros(len(perturbed.members))
        delcounter = 0.0

        # system A evolving
        original_evolving, kin, pot, tot = velver(original, tau, dt)
        # making system B

        # evolving system B
        perturbed_evolving, kinp, potp, totp = velver(perturbed, tau, dt)

        kineticmean.append(np.mean(kinp))
        potentialmean.append(np.mean(potp))
        totalmean.append(np.mean(totp))

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

        lyapcount[interval] = np.log(delt[interval] / epsilon)

        delta = delt[interval]

        # reset values for the next iteration
        for i in range(len(perturbed.members)):
            perturbed.displacement[i] = (
                original_evolving.displacement[i] + scale * delq[i]
            )

            perturbed.momentum[i] = original_evolving.momentum[i] + scale * delp[i]

        original = copy.deepcopy(original_evolving)

        # Make sure the system remains thermal while we do this

        if interval % 100 == 0:
            statistics_org, p_value_org, meanorg, varorg = momentum_check(
                original_evolving, temperature
            )
            statistics_per, p_value_per, meanper, varper = momentum_check(
                perturbed, temperature
            )

            if p_value_org or p_value_per <= 0.7:
                print("system lost thermalization", interval)

    return lyapcount, delt, kineticmean, potentialmean, totalmean
