# ============================================================
# Measure perturbation growth in an FPU chain
#
# Purpose:
#   Determine an appropriate characteristic time / tau for
#   the Benettin Lyapunov exponent calculation.
#
#   NO renormalization is performed here.
#
#   We simply measure
#
#       d(t) = ||X_perturbed(t) - X_original(t)||
#
#   as a function of time.
# ============================================================

import copy
import numpy as np
import matplotlib.pyplot as plt

from CreateChain import create_chain
from Langevin_thermalized import Langevin
from SpringSystem import springsystem
from Velocity_verlet import velver


# ============================================================
# Simulation parameters
# ============================================================

N = 500

dt = 0.1

# Time for Langevin thermalization
thermalization_time = 20

gamma = 0.7

# Temperatures to investigate
temperatures = np.logspace(-1, 2, 10)

# Initial perturbation size
epsilon = 1e-4

# Site at which perturbation is introduced
target = 250

# Time for which we allow the perturbation to grow
#
# IMPORTANT:
# This should be considerably larger than the tau values
# you expect to use in the Benettin calculation.
#
measurement_time = 20.0


# ============================================================
# Measure distance between two trajectories
# ============================================================


def measure_distance(system, target, dt, epsilon, measurement_time):
    """
    Measure the distance between two initially nearby
    trajectories without any renormalization.

    Parameters
    ----------
    system : springsystem
        Already thermalized system.

    target : int
        Site at which the perturbation is introduced.

    dt : float
        Integration timestep.

    epsilon : float
        Initial perturbation magnitude.

    measurement_time : float
        Total time for which the perturbation is allowed
        to evolve.

    Returns
    -------
    times : numpy array
        Time values.

    distance : numpy array
        Phase-space distance between the two trajectories.
    """

    # --------------------------------------------------------
    # Make two identical copies
    # --------------------------------------------------------

    original = copy.deepcopy(system)
    perturbed = copy.deepcopy(system)

    # --------------------------------------------------------
    # Number of particles
    # --------------------------------------------------------

    N = len(system.members)

    # --------------------------------------------------------
    # Construct initial perturbation
    #
    # We perturb both q and p, exactly as in your existing
    # lyapunov() function.
    # --------------------------------------------------------

    dq = np.zeros(N)
    dp = np.zeros(N)

    dq[target] = 1.0
    dp[target] = 1.0

    # Normalize perturbation
    norm = np.sqrt(np.sum(dq**2) + np.sum(dp**2))

    dq *= epsilon / norm
    dp *= epsilon / norm

    # --------------------------------------------------------
    # Apply perturbation
    # --------------------------------------------------------

    perturbed.displacement[target] += dq[target]
    perturbed.momentum[target] += dp[target]

    # --------------------------------------------------------
    # Number of integration steps
    # --------------------------------------------------------

    timesteps = int(measurement_time / dt)

    # --------------------------------------------------------
    # Arrays for storing results
    # --------------------------------------------------------

    times = np.arange(1, timesteps + 1) * dt

    distance = np.zeros(timesteps)

    # --------------------------------------------------------
    # Evolution
    # --------------------------------------------------------

    for step in range(timesteps):
        # Evolve original trajectory
        original_evolving, kin, pot, tot = velver(original, dt, dt)

        # Evolve perturbed trajectory
        perturbed_evolving, kin, pot, tot = velver(perturbed, dt, dt)

        # ----------------------------------------------------
        # Difference in phase space
        # ----------------------------------------------------

        dq = perturbed_evolving.displacement - original_evolving.displacement

        dp = perturbed_evolving.momentum - original_evolving.momentum

        # ----------------------------------------------------
        # Phase-space distance
        # ----------------------------------------------------

        distance[step] = np.sqrt(np.sum(dq**2) + np.sum(dp**2))

    return times, distance


# ============================================================
# Run simulations
# ============================================================

all_times = []
all_distances = []

for temperature in temperatures:
    print()
    print("=" * 60)
    print(f"Temperature = {temperature:.5g}")
    print("=" * 60)

    # --------------------------------------------------------
    # Create a NEW chain for every temperature
    # --------------------------------------------------------

    members, neighbours = create_chain(N)

    system = springsystem(members, neighbours, 0, 0.7)

    # --------------------------------------------------------
    # Langevin thermalization
    # --------------------------------------------------------

    print("Thermalizing...")

    system_thermalized, kinetic, potential, total, *_ = Langevin(
        system, 20, 0.1, gamma, temperature
    )

    print("Thermalization complete.")

    # --------------------------------------------------------
    # Measure perturbation growth
    # --------------------------------------------------------

    print("Measuring perturbation growth...")

    times, distance = measure_distance(
        system_thermalized, target, 0.01, epsilon, measurement_time
    )

    all_times.append(times)
    all_distances.append(distance)

    print("Measurement complete.")


# ============================================================
# Plot 1:
# Distance vs time
# ============================================================

plt.figure(figsize=(9, 6))

for temperature, times, distance in zip(temperatures, all_times, all_distances):
    plt.semilogy(times, distance, label=f"T = {temperature:.3g}")

plt.xlabel("Time")
plt.ylabel(r"Phase-space distance $d(t)$")

plt.title(
    r"Perturbation growth: "
    r"$d(t)=||\delta X(t)||$"
)

plt.legend(fontsize=9, ncol=2)

plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 2:
# Logarithmic perturbation growth
#
# If d(t) ~ d(0) exp(lambda t), then
#
# log[d(t)/d(0)] = lambda t
#
# Therefore the exponential regime appears as a straight line.
# ============================================================

plt.figure(figsize=(9, 6))

for temperature, times, distance in zip(temperatures, all_times, all_distances):
    log_distance = np.log(distance / distance[0])

    plt.plot(times, log_distance, label=f"T = {temperature:.3g}")

plt.xlabel("Time")
plt.ylabel(r"$\ln[d(t)/d(0)]$")

plt.title("Logarithmic perturbation growth")

plt.legend(fontsize=9, ncol=2)

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
