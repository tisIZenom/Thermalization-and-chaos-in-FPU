"""
Parallel ensemble automation for FPU thermalization.

Runs:
    10 temperatures
    100 ensembles per temperature
    1000 simulations total

Uses:
    10 parallel processes

Results are saved dynamically as each ensemble finishes.

Directory structure:

results/
    thermalization/
        T_0.100000/
            ensemble_0000.pkl
            ensemble_0001.pkl
            ...
            average.pkl
            progress.pkl

        T_0.215443/
            ...

        summary.pkl
        averages.pkl
"""

import os
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed

from SpringSystem import springsystem
from CreateChain import create_chain
from Langevin_thermalized import Langevin


# ============================================================
# PARAMETERS
# ============================================================

N = 1000

N_TEMPERATURES = 10
N_ENSEMBLES = 100

T_MIN = 0.1
T_MAX = 100.0

N_WORKERS = 10

TOTAL_TIME = 100
DT = 0.01
GAMMA = 0.7

BASE_DIR = "results/thermalization"


# ============================================================
# TEMPERATURES
# ============================================================

temperatures = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_TEMPERATURES)


# ============================================================
# DIRECTORY SETUP
# ============================================================

os.makedirs(BASE_DIR, exist_ok=True)

for temperature in temperatures:
    temperature_dir = os.path.join(BASE_DIR, f"T_{temperature:.6f}")

    os.makedirs(temperature_dir, exist_ok=True)


# ============================================================
# WORKER FUNCTION
# ============================================================


def run_single_ensemble(args):
    """
    Run one FPU ensemble.

    This function is executed by one of the worker processes.
    """

    temperature, ensemble_number, N = args

    # --------------------------------------------------------
    # Create a unique random number generator
    # --------------------------------------------------------

    # Gives each ensemble an independent random sequence.
    seed = int(temperature * 1_000_000) + ensemble_number * 1009 + os.getpid()

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Create system
    # --------------------------------------------------------

    members, neighbours = create_chain(N)

    system = springsystem(members, neighbours, 0, 1)

    # --------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------

    system.momentum = rng.uniform(-1, 1, size=len(system.members))

    system.displacement = rng.uniform(-1, 1, size=len(system.members))

    # --------------------------------------------------------
    # Langevin thermalization
    # --------------------------------------------------------

    result = Langevin(system, TOTAL_TIME, DT, GAMMA, temperature)

    (
        system,
        kinetic_energy,
        potential_energy,
        total_energy,
        normalized_spectral,
        temp_diff,
        statistic,
        p_value,
        correlator_momentum,
        correlator_kinetic,
        correlator_mode,
    ) = result

    # --------------------------------------------------------
    # Return everything needed by parent process
    # --------------------------------------------------------

    return {
        "temperature": temperature,
        "ensemble": ensemble_number,
        "system": system,
        "kinetic_energy": np.asarray(kinetic_energy),
        "potential_energy": np.asarray(potential_energy),
        "total_energy": np.asarray(total_energy),
        "normalized_spectral": np.asarray(normalized_spectral),
        "temp_diff": np.asarray(temp_diff),
        "statistic": np.asarray(statistic),
        "p_value": np.asarray(p_value),
        "correlator_momentum": np.asarray(correlator_momentum),
        "correlator_kinetic": np.asarray(correlator_kinetic),
        "correlator_mode": np.asarray(correlator_mode),
    }


# ============================================================
# PICKLE HELPER
# ============================================================


def save_pickle(data, filename):
    """
    Save data safely using a temporary file first.

    This prevents partially-written pickle files if the
    program is interrupted during writing.
    """

    temporary_file = filename + ".tmp"

    with open(temporary_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(temporary_file, filename)


# ============================================================
# CREATE JOB LIST
# ============================================================

jobs = []

for temperature in temperatures:
    temperature_dir = os.path.join(BASE_DIR, f"T_{temperature:.6f}")

    for ensemble in range(N_ENSEMBLES):
        filename = os.path.join(temperature_dir, f"ensemble_{ensemble:04d}.pkl")

        # ----------------------------------------------------
        # Skip simulations that already exist.
        #
        # This allows the program to be safely restarted.
        # ----------------------------------------------------

        if os.path.exists(filename):
            print(f"Skipping existing: T={temperature:.6f}, ensemble={ensemble}")
            continue

        jobs.append((temperature, ensemble, N))


print()
print("=" * 70)
print("FPU THERMALIZATION AUTOMATION")
print("=" * 70)
print(f"System size       : {N}")
print(f"Temperatures      : {N_TEMPERATURES}")
print(f"Ensembles/temp    : {N_ENSEMBLES}")
print(f"Total simulations : {N_TEMPERATURES * N_ENSEMBLES}")
print(f"Workers           : {N_WORKERS}")
print(f"Temperature range : {T_MIN} -> {T_MAX}")
print(f"Remaining jobs    : {len(jobs)}")
print("=" * 70)
print()


# ============================================================
# STORAGE FOR RUNNING ENSEMBLE SUMS
# ============================================================

sums = {}

counts = {}


def initialize_temperature(temperature, result):
    """
    Create running sums for a temperature.
    """

    sums[temperature] = {}

    sums[temperature]["kinetic_energy"] = np.zeros_like(
        result["kinetic_energy"], dtype=float
    )

    sums[temperature]["potential_energy"] = np.zeros_like(
        result["potential_energy"], dtype=float
    )

    sums[temperature]["total_energy"] = np.zeros_like(
        result["total_energy"], dtype=float
    )

    sums[temperature]["normalized_spectral"] = np.zeros_like(
        result["normalized_spectral"], dtype=float
    )

    sums[temperature]["temp_diff"] = np.zeros_like(result["temp_diff"], dtype=float)

    sums[temperature]["statistic"] = np.zeros_like(result["statistic"], dtype=float)

    sums[temperature]["p_value"] = np.zeros_like(result["p_value"], dtype=float)

    sums[temperature]["correlator_momentum"] = np.zeros_like(
        result["correlator_momentum"], dtype=float
    )

    sums[temperature]["correlator_kinetic"] = np.zeros_like(
        result["correlator_kinetic"], dtype=float
    )

    sums[temperature]["correlator_mode"] = np.zeros_like(
        result["correlator_mode"], dtype=float
    )

    counts[temperature] = 0


# ============================================================
# UPDATE ENSEMBLE AVERAGE
# ============================================================


def update_average(result):
    """
    Add one completed ensemble to the running average.
    """

    temperature = result["temperature"]

    if temperature not in sums:
        initialize_temperature(temperature, result)

    # --------------------------------------------------------
    # Add current ensemble to running sums
    # --------------------------------------------------------

    for key in sums[temperature]:
        sums[temperature][key] += result[key]

    counts[temperature] += 1

    # --------------------------------------------------------
    # Calculate current averages
    # --------------------------------------------------------

    average = {}

    for key in sums[temperature]:
        average[key] = sums[temperature][key] / counts[temperature]

    return average


# ============================================================
# SAVE CURRENT AVERAGE
# ============================================================


def save_current_average(temperature, average):

    temperature_dir = os.path.join(BASE_DIR, f"T_{temperature:.6f}")

    average_file = os.path.join(temperature_dir, "average.pkl")

    average_data = {
        "temperature": temperature,
        "ensembles_completed": counts[temperature],
        "average": average,
    }

    save_pickle(average_data, average_file)


# ============================================================
# MAIN PARALLEL EXECUTION
# ============================================================

if __name__ == "__main__":
    completed = 0

    total_jobs = len(jobs)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # ----------------------------------------------------
        # Submit all jobs
        # ----------------------------------------------------

        futures = {executor.submit(run_single_ensemble, job): job for job in jobs}

        # ----------------------------------------------------
        # Process results as soon as they finish
        # ----------------------------------------------------

        for future in as_completed(futures):
            temperature, ensemble, _ = futures[future]

            try:
                result = future.result()

                completed += 1

                # ------------------------------------------------
                # Save individual ensemble immediately
                # ------------------------------------------------

                temperature_dir = os.path.join(BASE_DIR, f"T_{temperature:.6f}")

                ensemble_file = os.path.join(
                    temperature_dir, f"ensemble_{ensemble:04d}.pkl"
                )

                save_pickle(result, ensemble_file)

                # ------------------------------------------------
                # Update ensemble average
                # ------------------------------------------------

                average = update_average(result)

                # ------------------------------------------------
                # Save current average immediately
                # ------------------------------------------------

                save_current_average(temperature, average)

                # ------------------------------------------------
                # Save progress information
                # ------------------------------------------------

                progress_file = os.path.join(temperature_dir, "progress.pkl")

                progress = {
                    "temperature": temperature,
                    "ensembles_completed": counts[temperature],
                    "total_ensembles": N_ENSEMBLES,
                }

                save_pickle(progress, progress_file)

                print(
                    f"[{completed}/{total_jobs}] "
                    f"T={temperature:.6f} | "
                    f"ensemble={ensemble:04d} | "
                    f"completed at this T="
                    f"{counts[temperature]}/{N_ENSEMBLES}"
                )

            except Exception as error:
                print(f"ERROR: T={temperature:.6f}, ensemble={ensemble:04d}")

                print(error)

    # ========================================================
    # SAVE FINAL AVERAGES
    # ========================================================

    final_averages = {}

    for temperature in sums:
        average = {}

        for key in sums[temperature]:
            average[key] = sums[temperature][key] / counts[temperature]

        final_averages[temperature] = average

    save_pickle(final_averages, os.path.join(BASE_DIR, "averages.pkl"))

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {
        "N": N,
        "temperatures": temperatures,
        "ensembles_per_temperature": N_ENSEMBLES,
        "workers": N_WORKERS,
        "total_time": TOTAL_TIME,
        "dt": DT,
        "gamma": GAMMA,
        "completed_counts": counts,
    }

    save_pickle(summary, os.path.join(BASE_DIR, "summary.pkl"))

    print()
    print("=" * 70)
    print("ALL SIMULATIONS COMPLETE")
    print("=" * 70)
