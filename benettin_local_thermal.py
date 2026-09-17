## the way we extract the lyapunov exponent in this case is by using the
# benetin algorithm
#
import copy
import numpy as np
from velocity_verlet_thermalized import velver
from SpringSystem import springsystem


def benetin(system, target, M, dt, epsilon, tau, temperature):

    original = copy.deepcopy(system)
    perturbed = copy.deepcopy(system)

    kineticmean = np.zeros(M)
    potentialmean = np.zeros(M)
    totalmean = np.zeros(M)

    delt = np.zeros(M)

    perturbed.perturbation_drift(target, epsilon)
    perturbed.perturbation_kick(target, epsilon)

    lyapcount = 0.0

    for interval in range(M):
        # system A evolving
        (
            original_evolving,
            kin,
            pot,
            tot,
            normalized,
            temperaturedistance,
            statistic,
            pvalue,
            corr_mom,
            corr_kin,
            corr_mode,
            thermalized,
        ) = velver(original, tau, dt, temperature)

        # making system B

        # evolving system B
        (
            perturbed_evolving,
            kinp,
            potp,
            totp,
            normalizedp,
            temperaturedistancep,
            statisticp,
            pvaluep,
            corr_momp,
            corr_kinp,
            corr_modep,
            thermalizedp,
        ) = velver(perturbed, tau, dt, temperature)

        kineticmean[interval] = np.mean(kinp)
        potentialmean[interval] = np.mean(potp)
        totalmean[interval] = np.mean(totp)

        if not thermalized:
            print("Original system went out of thermlization: ", interval)

        if not thermalizedp:
            print("The perturbed system went out of thermlization: ", interval)
        # finding the difference
        delq = (
            perturbed_evolving.displacement[target]
            - original_evolving.displacement[target]
        )
        delp = perturbed_evolving.momentum[target] - original_evolving.momentum[target]

        delt[interval] = np.sqrt(delq**2 + delp**2)
        lyapcount += np.log(np.sqrt(delq**2 + delp**2) / epsilon)

        # reset values for the next iteration
        for i in range(len(perturbed.members)):
            perturbed.momentum[i] = perturbed.momentum[i] + epsilon * (
                perturbed.momentum[i] - perturbed_evolving.momentum[i]
            ) / (perturbed.momentum[i] - perturbed_evolving.momentum[i])

            perturbed.displacement[i] = perturbed.displacement[i] + epsilon * (
                perturbed.displacement[i] - perturbed_evolving.displacement[i]
            ) / (perturbed.displacement[i] - perturbed_evolving.displacement[i])
        original = copy.deepcopy(original_evolving)

        # this gives us del

    return (lyapcount / (M * tau)), delt, kineticmean, potentialmean, totalmean
