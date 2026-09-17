import copy
import numpy as np
from Velocity_verlet import velver


def benettin(system, target, M, dt, epsilon, tau):

    original = copy.deepcopy(system)
    perturbed = copy.deepcopy(system)

    kineticmean = np.zeros(M)
    potentialmean = np.zeros(M)
    totalmean = np.zeros(M)

    delt = np.zeros(M)
    local_lyap = np.zeros(M)

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

    perturbed.displacement[target] += dq[target]
    perturbed.momentum[target] += dp[target]

    lyapcount = 0.0

    # --------------------------------------------------
    # Benettin loop
    # --------------------------------------------------

    for interval in range(M):
        # Evolve both trajectories
        original_evolving, kin, pot, tot = velver(original, tau, dt)

        perturbed_evolving, kinp, potp, totp = velver(perturbed, tau, dt)

        # --------------------------------------------------
        # Calculate full phase-space separation
        # --------------------------------------------------

        dq = perturbed_evolving.displacement - original_evolving.displacement

        dp = perturbed_evolving.momentum - original_evolving.momentum

        delta = np.sqrt(np.sum(dq**2) + np.sum(dp**2))

        delt[interval] = delta

        # --------------------------------------------------
        # Local/finite-time Lyapunov exponent
        # --------------------------------------------------

        local_lyap[interval] = np.log(delta / epsilon) / tau

        lyapcount += np.log(delta / epsilon)

        # --------------------------------------------------
        # Renormalize perturbation
        # --------------------------------------------------

        dq *= epsilon / delta
        dp *= epsilon / delta

        perturbed = copy.deepcopy(original_evolving)

        perturbed.displacement += dq
        perturbed.momentum += dp

        # Original trajectory continues normally
        original = copy.deepcopy(original_evolving)

        # Diagnostics
        kineticmean[interval] = np.mean(kinp)
        potentialmean[interval] = np.mean(potp)
        totalmean[interval] = np.mean(totp)

    # --------------------------------------------------
    # Global maximal Lyapunov exponent
    # --------------------------------------------------

    lambda_max = lyapcount / (M * tau)

    return (lambda_max, local_lyap, delt, kineticmean, potentialmean, totalmean)
