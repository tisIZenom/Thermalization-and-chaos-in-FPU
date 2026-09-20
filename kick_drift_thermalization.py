import os

# ---------------------------------------------------------
# Limit numerical libraries to ONE thread per process
# ---------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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

NUMBER_OF_ENSEMBLES = 10
NUMBER_OF_WORKERS = 5

temperature = 1.0

total_time = 10
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

    kinetic = np.stack([r[1] for r in results], axis=0)

    potential = np.stack([r[2] for r in results], axis=0)

    total = np.stack([r[3] for r in results], axis=0)

    spectral = np.stack([r[4] for r in results], axis=0)

    temp_diff = np.stack([r[5] for r in results], axis=0)

    statistic = np.stack([r[6] for r in results], axis=0)

    p_value = np.stack([r[7] for r in results], axis=0)

    momentum_corr = np.stack([r[8] for r in results], axis=0)

    kinetic_corr = np.stack([r[9] for r in results], axis=0)

    mode_corr = np.stack([r[10] for r in results], axis=0)

    # -----------------------------------------------------
    # Ensemble mean
    # -----------------------------------------------------

    means = {
        "kinetic": np.mean(kinetic, axis=0),
        "potential": np.mean(potential, axis=0),
        "total": np.mean(total, axis=0),
        "spectral": np.mean(spectral, axis=0),
        "temp_diff": np.mean(temp_diff, axis=0),
        "statistic": np.mean(statistic, axis=0),
        "p_value": np.mean(p_value, axis=0),
        "momentum_corr": np.mean(momentum_corr, axis=0),
        "kinetic_corr": np.mean(kinetic_corr, axis=0),
        "mode_corr": np.mean(mode_corr, axis=0),
    }

    # -----------------------------------------------------
    # Standard error
    # -----------------------------------------------------

    sem = {
        "kinetic": np.std(kinetic, axis=0, ddof=1) / np.sqrt(len(results)),
        "potential": np.std(potential, axis=0, ddof=1) / np.sqrt(len(results)),
        "total": np.std(total, axis=0, ddof=1) / np.sqrt(len(results)),
        "spectral": np.std(spectral, axis=0, ddof=1) / np.sqrt(len(results)),
        "temp_diff": np.std(temp_diff, axis=0, ddof=1) / np.sqrt(len(results)),
        "statistic": np.std(statistic, axis=0, ddof=1) / np.sqrt(len(results)),
        "p_value": np.std(p_value, axis=0, ddof=1) / np.sqrt(len(results)),
        "momentum_corr": np.std(momentum_corr, axis=0, ddof=1) / np.sqrt(len(results)),
        "kinetic_corr": np.std(kinetic_corr, axis=0, ddof=1) / np.sqrt(len(results)),
        "mode_corr": np.std(mode_corr, axis=0, ddof=1) / np.sqrt(len(results)),
    }

    return means, sem


# =========================================================
# Plot ensemble averages
# =========================================================


def plot_comparison(mean_set1, sem_set1, mean_set2, sem_set2):

    quantities = [
        ("kinetic", "Kinetic Energy"),
        ("potential", "Potential Energy"),
        ("total", "Total Energy"),
        ("spectral", "Normalized Spectral Entropy"),
        ("temp_diff", "Temperature Difference"),
        ("statistic", "KS Statistic"),
        ("p_value", "KS p-value"),
        ("momentum_corr", "Momentum Correlator"),
        ("kinetic_corr", "Kinetic Correlator"),
        ("mode_corr", "Mode Correlator"),
    ]

    for key, title in quantities:
        mean1 = mean_set1[key]
        error1 = sem_set1[key]

        mean2 = mean_set2[key]
        error2 = sem_set2[key]

        # -------------------------------------------------
        # x-axis
        # -------------------------------------------------

        x1 = np.arange(len(mean1)) * dt
        x2 = np.arange(len(mean2)) * dt

        # -------------------------------------------------
        # Plot
        # -------------------------------------------------

        plt.figure(figsize=(9, 5))

        plt.plot(x1, mean1, label="Set 1: Drift")

        plt.plot(x2, mean2, label="Set 2: Kick")

        # -------------------------------------------------
        # SEM
        # -------------------------------------------------

        plt.fill_between(x1, mean1 - error1, mean1 + error1, alpha=0.2)

        plt.fill_between(x2, mean2 - error2, mean2 + error2, alpha=0.2)

        plt.xlabel("Time")
        plt.ylabel(title)

        plt.title(f"{title}\n{NUMBER_OF_ENSEMBLES} ensembles per set")

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
    print("\nCalculating ensemble averages...")

    mean_set1, sem_set1 = average_ensembles(results_set1)
    mean_set2, sem_set2 = average_ensembles(results_set2)

    print("Averaging complete.")

    plot_comparison(mean_set1, sem_set1, mean_set2, sem_set2)

    print("\nSimulation finished.")
