# ============================================================
# Ensemble averaged perturbation growth in an FPU chain
#
# For each temperature:
#
#   1. Create an independent FPU chain
#   2. Thermalize it with Langevin dynamics
#   3. Create original + perturbed trajectories
#   4. Measure phase-space distance d(t)
#   5. Repeat for many ensembles
#   6. Calculate:
#
#          <d(t)>
#          SEM[d(t)]
#
#   7. Plot ensemble-averaged perturbation growth
#
# NO renormalization is performed.
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

dt = 0.01

# Langevin thermalization
thermalization_time = 20.0
thermalization_dt = 0.1

gamma = 0.7

# Temperatures
temperatures = np.logspace(-1, 2, 10)

# Number of independent realizations
n_ensembles = 50

# Initial perturbation
epsilon = 1e-4

# Site at which perturbation is introduced
target = 250

# Perturbation measurement time
measurement_time = 20.0


# ============================================================
# Measure perturbation growth for ONE realization
# ============================================================


def measure_distance(system, target, dt, epsilon, measurement_time):
    """
    Measure phase-space distance between two nearby
    trajectories without renormalization.

    Parameters
    ----------
    system : springsystem
        Already thermalized system.

    target : int
        Site at which perturbation is introduced.

    dt : float
        Integration timestep.

    epsilon : float
        Initial perturbation magnitude.

    measurement_time : float
        Total measurement time.

    Returns
    -------
    times : ndarray
        Time array.

    distance : ndarray
        Phase-space distance d(t).
    """

    # --------------------------------------------------------
    # Make two identical copies
    # --------------------------------------------------------

    original = copy.deepcopy(system)
    perturbed = copy.deepcopy(system)

    # --------------------------------------------------------
    # Number of particles
    # --------------------------------------------------------

    N_particles = len(system.members)

    # --------------------------------------------------------
    # Construct perturbation
    # --------------------------------------------------------

    dq = np.zeros(N_particles)
    dp = np.zeros(N_particles)

    dq[target] = 1.0
    dp[target] = 1.0

    # Normalize perturbation
    norm = np.sqrt(np.sum(dq**2) + np.sum(dp**2))

    dq *= epsilon / norm
    dp *= epsilon / norm

    # --------------------------------------------------------
    # Apply perturbation
    # --------------------------------------------------------

    perturbed.displacement += dq
    perturbed.momentum += dp

    # --------------------------------------------------------
    # Number of integration steps
    # --------------------------------------------------------

    timesteps = int(measurement_time / dt)

    times = np.arange(1, timesteps + 1) * dt

    distance = np.zeros(timesteps)

    # --------------------------------------------------------
    # Evolution
    # --------------------------------------------------------

    for step in range(timesteps):
        # Evolve original
        original_evolving, kin, pot, tot = velver(original, dt, dt)

        # Evolve perturbed
        perturbed_evolving, kin, pot, tot = velver(perturbed, dt, dt)

        # ----------------------------------------------------
        # Phase-space difference
        # ----------------------------------------------------

        dq = perturbed_evolving.displacement - original_evolving.displacement

        dp = perturbed_evolving.momentum - original_evolving.momentum

        # ----------------------------------------------------
        # Euclidean phase-space distance
        # ----------------------------------------------------

        distance[step] = np.sqrt(np.sum(dq**2) + np.sum(dp**2))

        # Update systems
        original = original_evolving
        perturbed = perturbed_evolving

    return times, distance


# ============================================================
# Ensemble statistics
# ============================================================


def calculate_ensemble_statistics(all_distances):
    """
    Calculate ensemble mean and standard error.

    Parameters
    ----------
    all_distances : ndarray
        Shape:

            (n_ensembles, n_timesteps)

    Returns
    -------
    mean_distance : ndarray
        Ensemble average.

    sem_distance : ndarray
        Standard error of the mean.
    """

    mean_distance = np.mean(all_distances, axis=0)

    std_distance = np.std(all_distances, axis=0, ddof=1)

    sem_distance = std_distance / np.sqrt(all_distances.shape[0])

    return mean_distance, sem_distance


# ============================================================
# Main simulation
# ============================================================

ensemble_results = {}

for temperature in temperatures:
    print()
    print("=" * 70)
    print(f"Temperature = {temperature:.5g}")
    print("=" * 70)

    all_distances = []

    # --------------------------------------------------------
    # Ensemble loop
    # --------------------------------------------------------

    for ensemble in range(n_ensembles):
        print(f"Ensemble {ensemble + 1}/{n_ensembles}", end="\r")

        # ----------------------------------------------------
        # Create a completely new chain
        # ----------------------------------------------------

        members, neighbours = create_chain(N)

        system = springsystem(members, neighbours, 0, 0.7)

        # ----------------------------------------------------
        # Langevin thermalization
        # ----------------------------------------------------

        system_thermalized, kinetic, potential, total, *_ = Langevin(
            system, thermalization_time, thermalization_dt, gamma, temperature
        )

        # ----------------------------------------------------
        # Measure perturbation growth
        # ----------------------------------------------------

        times, distance = measure_distance(
            system_thermalized, target, dt, epsilon, measurement_time
        )

        all_distances.append(distance)

    print()

    # --------------------------------------------------------
    # Convert to NumPy array
    #
    # Shape:
    #
    #   (n_ensembles, n_timesteps)
    #
    # --------------------------------------------------------

    all_distances = np.asarray(all_distances)

    # --------------------------------------------------------
    # Calculate ensemble statistics
    # --------------------------------------------------------

    mean_distance, sem_distance = calculate_ensemble_statistics(all_distances)

    # --------------------------------------------------------
    # Store everything
    # --------------------------------------------------------

    ensemble_results[temperature] = {
        "times": times,
        "all_distances": all_distances,
        "mean_distance": mean_distance,
        "sem_distance": sem_distance,
    }

    print(f"Completed T = {temperature:.5g}")


# ============================================================
# Plot 1:
# Ensemble averaged distance
# ============================================================

plt.figure(figsize=(9, 6))

for temperature in temperatures:
    data = ensemble_results[temperature]

    times = data["times"]
    mean_distance = data["mean_distance"]
    sem_distance = data["sem_distance"]

    plt.semilogy(times, mean_distance, label=f"T = {temperature:.3g}")

    # SEM region
    plt.fill_between(
        times,
        np.maximum(mean_distance - sem_distance, 1e-20),
        mean_distance + sem_distance,
        alpha=0.15,
    )


plt.xlabel("Time")
plt.ylabel(r"Ensemble averaged distance $\langle d(t) \rangle$")

plt.title(r"Ensemble-averaged perturbation growth")

plt.legend(fontsize=9, ncol=2)

plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 2:
# Logarithmic ensemble-averaged growth
#
#     ln(<d(t)> / <d(0)>)
#
# ============================================================

plt.figure(figsize=(9, 6))

for temperature in temperatures:
    data = ensemble_results[temperature]

    times = data["times"]
    mean_distance = data["mean_distance"]

    log_distance = np.log(mean_distance / mean_distance[0])

    plt.plot(times, log_distance, label=f"T = {temperature:.3g}")


plt.xlabel("Time")
plt.ylabel(r"$\ln[\langle d(t)\rangle / \langle d(0)\rangle]$")

plt.title("Logarithmic ensemble-averaged perturbation growth")

plt.legend(fontsize=9, ncol=2)

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
