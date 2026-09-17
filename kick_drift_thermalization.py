import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle


from SpringSystem import springsystem
from CreateChain import create_chain
from Langevin_thermalized import Langevin


# =========================================================
# Parameters
# =========================================================

N = 500

NUMBER_OF_ENSEMBLES = 50
NUMBER_OF_WORKERS = 10

temperature = 1.0

total_time = 20
dt = 0.01
gamma = 0.7

beta = 0.8

PERTURBATION_SITE = 250
PERTURBATION_SIZE = 1.0


# =========================================================
# Run one ensemble
# =========================================================


def run_ensemble(args):
    """
    Run one independent realization of the system.

    mode = 1 --> perturbation drift
    mode = 2 --> perturbation kick

    seed --> independent random seed
    """

    mode, seed = args

    # Give this ensemble its own random number generator
    np.random.seed(seed)

    # -----------------------------------------------------
    # Create a completely new FPU chain
    # -----------------------------------------------------

    members, neighbours = create_chain(N)

    system = springsystem(members, neighbours, 0, beta)

    # -----------------------------------------------------
    # Apply the appropriate perturbation
    # -----------------------------------------------------

    if mode == 1:
        # Set 1
        system.perturbation_drift(PERTURBATION_SITE, PERTURBATION_SIZE)

    elif mode == 2:
        # Set 2
        system.perturbation_kick(PERTURBATION_SITE, PERTURBATION_SIZE)

    else:
        raise ValueError("mode must be either 1 or 2")

    # -----------------------------------------------------
    # Langevin thermalization
    # -----------------------------------------------------

    results = Langevin(system, total_time, dt, gamma, temperature)

    return results


# =========================================================
# Convert ensemble results into averages
# =========================================================


def average_ensembles(results):
    """
    Average each observable over all ensembles.

    results is a list containing the output of Langevin()
    from every realization.
    """

    # Number of quantities returned by Langevin
    number_of_quantities = len(results[0])

    averaged = []

    for quantity in range(number_of_quantities):
        data = [np.asarray(result[quantity]) for result in results]

        # Stack along ensemble axis
        data = np.stack(data, axis=0)

        # Average over ensemble axis
        mean = np.mean(data, axis=0)

        # Standard error of the mean
        sem = np.std(data, axis=0, ddof=1) / np.sqrt(len(data))

        averaged.append((mean, sem))

    return averaged


# =========================================================
# Plot ensemble averages
# =========================================================


def plot_comparison(avg_set1, avg_set2):

    names = [
        "Kinetic Energy",
        "Potential Energy",
        "Total Energy",
        "Normalized Spectral Entropy",
        "Temperature Difference",
        "Statistic",
        "KS p-value",
        "Momentum Correlator",
        "Kinetic Correlator",
        "Mode Correlator",
    ]

    # -----------------------------------------------------
    # Plot each observable separately
    # -----------------------------------------------------

    for i, name in enumerate(names):
        mean1, sem1 = avg_set1[i]
        mean2, sem2 = avg_set2[i]

        plt.figure(figsize=(9, 5))

        x1 = np.arange(len(mean1))
        x2 = np.arange(len(mean2))

        plt.plot(x1, mean1, label="Set 1: perturbation drift")

        plt.plot(x2, mean2, label="Set 2: perturbation kick")

        # Optional uncertainty bands
        plt.fill_between(x1, mean1 - sem1, mean1 + sem1, alpha=0.2)

        plt.fill_between(x2, mean2 - sem2, mean2 + sem2, alpha=0.2)

        plt.xlabel("Time / Lag")
        plt.ylabel(name)

        plt.title(
            f"Ensemble averaged {name}\n{NUMBER_OF_ENSEMBLES} realizations per set"
        )

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.show()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FPU ENSEMBLE SIMULATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Generate independent seeds
    # -----------------------------------------------------

    rng = np.random.default_rng(12345)

    seeds_set1 = rng.integers(0, 2**32 - 1, size=NUMBER_OF_ENSEMBLES)

    seeds_set2 = rng.integers(0, 2**32 - 1, size=NUMBER_OF_ENSEMBLES)

    # -----------------------------------------------------
    # Arguments for each ensemble
    # -----------------------------------------------------

    jobs_set1 = [(1, seed) for seed in seeds_set1]

    jobs_set2 = [(2, seed) for seed in seeds_set2]

    # -----------------------------------------------------
    # Create 10 parallel workers
    # -----------------------------------------------------

    with Pool(processes=NUMBER_OF_WORKERS) as pool:
        # =================================================
        # SET 1
        # =================================================

        print("\nRunning SET 1...")
        print("Perturbation: drift")

        results_set1 = pool.map(run_ensemble, jobs_set1)

        print("Set 1 complete.")

        # =================================================
        # SET 2
        # =================================================

        print("\nRunning SET 2...")
        print("Perturbation: kick")

        results_set2 = pool.map(run_ensemble, jobs_set2)

        print("Set 2 complete.")

    # -----------------------------------------------------
    # Ensemble averaging
    # -----------------------------------------------------

    print("\nCalculating ensemble averages...")

    avg_set1 = average_ensembles(results_set1)
    avg_set2 = average_ensembles(results_set2)

    print("Averaging complete.")

    with open("averaged set 1.pkl", "wb") as file:
        pickle.dump(avg_set1, file)

    with open("averaged set 2.pkl", "wb") as file:
        pickle.dump(avg_set2, file)

    # -----------------------------------------------------
    # Plot comparison
    # -----------------------------------------------------

    plot_comparison(avg_set1, avg_set2)

    print("\nSimulation finished.")
