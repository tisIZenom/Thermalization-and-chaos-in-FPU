# ============================================================
# FPU THERMALIZATION PARALLEL AUTOMATION
#
# Runs:
#     N_TEMPERATURES temperatures
#     N_ENSEMBLES ensembles per temperature
#
# Design:
#     - Bounded number of in-flight processes
#     - One native numerical thread per worker
#     - Worker saves its own large result pickle
#     - Parent only receives small metadata
#     - Incremental/running averages
#     - Safe restart
#     - Automatic recovery after interrupted runs
#     - Ctrl+C terminates worker processes
#
# Directory structure:
#
# results/
#     thermalization/
#         T_0.100000/
#             ensemble_0000.pkl
#             ensemble_0001.pkl
#             ...
#             average.pkl
#             progress.pkl
#
#         T_0.359381/
#             ...
#
#         averages.pkl
#         summary.pkl
#
# ============================================================


# ============================================================
# IMPORTANT:
# SET THESE BEFORE IMPORTING NUMPY / SCIPY
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


# ============================================================
# IMPORTS
# ============================================================

import pickle
from pathlib import Path
import multiprocessing as mp

import numpy as np

from concurrent.futures import (
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED,
)

from SpringSystem import springsystem
from CreateChain import create_chain
from Langevin_thermalized import Langevin


# ============================================================
# PARAMETERS
# ============================================================

N = 500

N_TEMPERATURES = 10
N_ENSEMBLES = 5

T_MIN = 0.1
T_MAX = 100.0

# ------------------------------------------------------------
# START CONSERVATIVELY.
#
# Increase to 3 or 4 only after checking CPU temperature.
# ------------------------------------------------------------

N_WORKERS = 5

TOTAL_TIME = 100.0
DT = 0.01
GAMMA = 0.7

BETA = 1.0
ALPHA = 0.0

# ------------------------------------------------------------
# Restart worker processes periodically.
#
# This is useful for very long simulations in case some
# library/code path slowly accumulates memory.
#
# Set to None to disable.
# ------------------------------------------------------------

WORKER_TASKS_BEFORE_RESTART = 20

# ------------------------------------------------------------
# Master random seed.
#
# Each temperature/ensemble receives a deterministic,
# independent SeedSequence derived from this.
# ------------------------------------------------------------

MASTER_SEED = 20260917

# ------------------------------------------------------------
# Base output directory
# ------------------------------------------------------------

BASE_DIR = Path("results/thermalization")


# ============================================================
# DATA THAT IS AVERAGED
# ============================================================

AVERAGE_KEYS = (
    "kinetic_energy",
    "potential_energy",
    "total_energy",
    "normalized_spectral",
    "temp_diff",
    "statistic",
    "p_value",
    "correlator_momentum",
    "correlator_kinetic",
    "correlator_mode",
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def temperature_directory(temperature):
    """
    Return directory corresponding to one temperature.
    """

    return BASE_DIR / f"T_{temperature:.6f}"


def ensemble_filename(temperature, ensemble_number):
    """
    Return path to one ensemble result.
    """

    return temperature_directory(temperature) / f"ensemble_{ensemble_number:04d}.pkl"


def save_pickle_atomic(data, filename):
    """
    Save pickle safely.

    The data are first written to a temporary file and then
    atomically renamed into place.

    Therefore a killed/interrupted process should not leave
    a half-written .pkl file.
    """

    filename = Path(filename)

    filename.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = filename.with_name(filename.name + ".tmp")

    with open(temporary_file, "wb") as f:
        pickle.dump(
            data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(
        temporary_file,
        filename,
    )


def load_pickle(filename):
    """
    Load a pickle file.
    """

    with open(filename, "rb") as f:
        return pickle.load(f)


# ============================================================
# WORKER FUNCTION
# ============================================================


def run_single_ensemble(job):
    """
    Run one complete FPU ensemble.

    IMPORTANT:
    The worker saves the large result dictionary directly
    to disk. Only small metadata are returned to the parent.
    """

    (
        temperature_index,
        temperature,
        ensemble_number,
        system_size,
    ) = job

    # --------------------------------------------------------
    # Deterministic but independent random seed
    # --------------------------------------------------------

    seed_sequence = np.random.SeedSequence(
        [
            MASTER_SEED,
            temperature_index,
            ensemble_number,
        ]
    )

    rng = np.random.default_rng(seed_sequence)

    # --------------------------------------------------------
    # Create FPU chain
    # --------------------------------------------------------

    members, neighbours = create_chain(system_size)

    system = springsystem(
        members,
        neighbours,
        ALPHA,
        BETA,
    )

    # --------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------

    system.momentum = rng.uniform(
        -1.0,
        1.0,
        size=len(system.members),
    )

    system.displacement = rng.uniform(
        -1.0,
        1.0,
        size=len(system.members),
    )

    # --------------------------------------------------------
    # Langevin thermalization
    # --------------------------------------------------------

    result = Langevin(
        system,
        TOTAL_TIME,
        DT,
        GAMMA,
        temperature,
    )

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
    # Construct complete ensemble result
    # --------------------------------------------------------

    result_data = {
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

    # --------------------------------------------------------
    # Worker writes the large result directly to disk
    # --------------------------------------------------------

    output_file = ensemble_filename(
        temperature,
        ensemble_number,
    )

    save_pickle_atomic(
        result_data,
        output_file,
    )

    # --------------------------------------------------------
    # Only small metadata cross the process boundary
    # --------------------------------------------------------

    return {
        "temperature_index": temperature_index,
        "temperature": temperature,
        "ensemble": ensemble_number,
        "filename": str(output_file),
    }


# ============================================================
# RUNNING MEAN
# ============================================================


def update_running_average(
    current_average,
    count,
    result,
):
    """
    Update an existing running mean with one new ensemble.

    If count ensembles have already been included:

        new_mean =
            old_mean + (new_value - old_mean)/(count + 1)

    This avoids maintaining potentially enormous running sums.
    """

    new_count = count + 1

    if current_average is None:
        average = {}

        for key in AVERAGE_KEYS:
            average[key] = np.asarray(
                result[key],
                dtype=float,
            ).copy()

        return average

    # --------------------------------------------------------
    # Update mean
    # --------------------------------------------------------

    for key in AVERAGE_KEYS:
        new_value = np.asarray(
            result[key],
            dtype=float,
        )

        current_average[key] += (new_value - current_average[key]) / new_count

    return current_average


# ============================================================
# DISCOVER COMPLETED ENSEMBLES
# ============================================================


def discover_completed_ensembles(
    temperature,
    n_ensembles,
):
    """
    Find ensemble files that already exist.

    Returns a set of ensemble indices.
    """

    directory = temperature_directory(temperature)

    completed = set()

    if not directory.exists():
        return completed

    for ensemble_number in range(n_ensembles):
        filename = ensemble_filename(
            temperature,
            ensemble_number,
        )

        if filename.exists():
            completed.add(ensemble_number)

    return completed


# ============================================================
# REBUILD AVERAGE FROM EXISTING FILES
# ============================================================


def rebuild_temperature_average(
    temperature,
    completed_ensembles,
):
    """
    Reconstruct the running average by reading the ensemble
    files from disk.

    This is only called when the saved average/progress
    information is missing or inconsistent.
    """

    print(
        f"Repairing average for T={temperature:.6f} "
        f"from {len(completed_ensembles)} ensemble files..."
    )

    average = None
    count = 0

    for ensemble_number in sorted(completed_ensembles):
        filename = ensemble_filename(
            temperature,
            ensemble_number,
        )

        try:
            result = load_pickle(filename)

            average = update_running_average(
                average,
                count,
                result,
            )

            count += 1

        except Exception as error:
            print(f"WARNING: could not read {filename}")

            print(f"       {error}")

    return average, count


# ============================================================
# LOAD OR REBUILD EXISTING STATE
# ============================================================


def load_temperature_state(
    temperature,
    completed_ensembles,
):
    """
    Load saved average/progress when they are consistent.

    Otherwise rebuild the state from ensemble files.
    """

    count_from_files = len(completed_ensembles)

    if count_from_files == 0:
        return {
            "average": None,
            "count": 0,
        }

    average_file = temperature_directory(temperature) / "average.pkl"

    progress_file = temperature_directory(temperature) / "progress.pkl"

    # --------------------------------------------------------
    # Try loading saved state
    # --------------------------------------------------------

    try:
        average_data = load_pickle(average_file)

        progress_data = load_pickle(progress_file)

        saved_count = int(average_data["ensembles_completed"])

        progress_count = int(progress_data["ensembles_completed"])

        saved_average = average_data["average"]

        # ----------------------------------------------------
        # Check consistency
        # ----------------------------------------------------

        if (
            saved_count == count_from_files
            and progress_count == count_from_files
            and all(key in saved_average for key in AVERAGE_KEYS)
        ):
            print(
                f"Recovered saved state: T={temperature:.6f}, {saved_count} ensembles"
            )

            return {
                "average": saved_average,
                "count": saved_count,
            }

    except Exception:
        pass

    # --------------------------------------------------------
    # Saved state is missing/inconsistent.
    # Rebuild from individual ensemble files.
    # --------------------------------------------------------

    average, count = rebuild_temperature_average(
        temperature,
        completed_ensembles,
    )

    return {
        "average": average,
        "count": count,
    }


# ============================================================
# SAVE TEMPERATURE AVERAGE
# ============================================================


def save_temperature_average(
    temperature,
    average,
    count,
):
    """
    Save current running average.
    """

    temperature_dir = temperature_directory(temperature)

    filename = temperature_dir / "average.pkl"

    data = {
        "temperature": temperature,
        "ensembles_completed": count,
        "average": average,
    }

    save_pickle_atomic(
        data,
        filename,
    )


# ============================================================
# SAVE PROGRESS
# ============================================================


def save_temperature_progress(
    temperature,
    count,
):
    """
    Save progress information.
    """

    temperature_dir = temperature_directory(temperature)

    filename = temperature_dir / "progress.pkl"

    data = {
        "temperature": temperature,
        "ensembles_completed": count,
        "total_ensembles": N_ENSEMBLES,
    }

    save_pickle_atomic(
        data,
        filename,
    )


# ============================================================
# SAVE FINAL AVERAGES
# ============================================================


def save_final_averages(state):
    """
    Save average for every temperature.
    """

    final_averages = {}

    for temperature_index, temp_state in state.items():
        temperature = temperatures[temperature_index]

        if temp_state["average"] is not None:
            final_averages[float(temperature)] = temp_state["average"]

    save_pickle_atomic(
        final_averages,
        BASE_DIR / "averages.pkl",
    )


# ============================================================
# SAVE SUMMARY
# ============================================================


def save_summary(state):
    """
    Save global simulation information.
    """

    completed_counts = {}

    for temperature_index, temp_state in state.items():
        temperature = temperatures[temperature_index]

        completed_counts[float(temperature)] = temp_state["count"]

    summary = {
        "N": N,
        "temperatures": temperatures,
        "ensembles_per_temperature": N_ENSEMBLES,
        "workers": N_WORKERS,
        "total_time": TOTAL_TIME,
        "dt": DT,
        "gamma": GAMMA,
        "alpha": ALPHA,
        "beta": BETA,
        "master_seed": MASTER_SEED,
        "completed_counts": completed_counts,
    }

    save_pickle_atomic(
        summary,
        BASE_DIR / "summary.pkl",
    )


# ============================================================
# TERMINATE EXECUTOR
# ============================================================


def terminate_executor(executor):
    """
    Aggressively terminate worker processes.

    ProcessPoolExecutor does not immediately kill currently
    running jobs when cancel_futures=True is used.

    Therefore, when Ctrl+C is received, explicitly terminate
    the child processes as well.
    """

    try:
        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # CPython ProcessPoolExecutor exposes its worker process
    # objects internally.
    #
    # This is intentionally used only for emergency shutdown.
    # --------------------------------------------------------

    processes = getattr(
        executor,
        "_processes",
        {},
    )

    if processes is None:
        processes = {}

    processes = list(processes.values())

    # --------------------------------------------------------
    # First attempt graceful termination
    # --------------------------------------------------------

    for process in processes:
        try:
            if process.is_alive():
                process.terminate()

        except Exception:
            pass

    # --------------------------------------------------------
    # Give them a short opportunity to exit
    # --------------------------------------------------------

    for process in processes:
        try:
            process.join(timeout=2.0)

        except Exception:
            pass

    # --------------------------------------------------------
    # Kill anything that remains
    # --------------------------------------------------------

    for process in processes:
        try:
            if process.is_alive():
                process.kill()

        except Exception:
            pass

    for process in processes:
        try:
            process.join(timeout=1.0)

        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Multiprocessing context
    #
    # Spawn starts clean worker interpreters and avoids many
    # problems associated with forking NumPy-heavy processes.
    # --------------------------------------------------------

    ctx = mp.get_context("spawn")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    BASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Temperature grid
    # --------------------------------------------------------

    temperatures = np.logspace(
        np.log10(T_MIN),
        np.log10(T_MAX),
        N_TEMPERATURES,
    )

    # --------------------------------------------------------
    # State for every temperature
    # --------------------------------------------------------

    state = {}

    # --------------------------------------------------------
    # Jobs still needing to be run
    # --------------------------------------------------------

    jobs = []

    print()
    print("=" * 72)
    print("FPU THERMALIZATION AUTOMATION")
    print("=" * 72)

    print(f"System size        : {N}")

    print(f"Temperatures       : {N_TEMPERATURES}")

    print(f"Ensembles/temp     : {N_ENSEMBLES}")

    print(f"Total simulations  : {N_TEMPERATURES * N_ENSEMBLES}")

    print(f"Workers            : {N_WORKERS}")

    print(f"Total time         : {TOTAL_TIME}")

    print(f"dt                 : {DT}")

    print(f"Gamma              : {GAMMA}")

    print(f"Temperature range  : {T_MIN} -> {T_MAX}")

    print(f"Native threads/work: 1")

    print(f"Worker recycle     : {WORKER_TASKS_BEFORE_RESTART}")

    print("=" * 72)
    print()

    # ========================================================
    # PREPARE ALL TEMPERATURE STATES
    # ========================================================

    for temperature_index, temperature in enumerate(temperatures):
        temperature_dir = temperature_directory(temperature)

        temperature_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Find existing results
        # ----------------------------------------------------

        completed_ensembles = discover_completed_ensembles(
            temperature,
            N_ENSEMBLES,
        )

        # ----------------------------------------------------
        # Load/rebuild average state
        # ----------------------------------------------------

        state[temperature_index] = load_temperature_state(
            temperature,
            completed_ensembles,
        )

        # ----------------------------------------------------
        # Add missing jobs
        # ----------------------------------------------------

        for ensemble_number in range(N_ENSEMBLES):
            if ensemble_number not in (completed_ensembles):
                jobs.append(
                    (
                        temperature_index,
                        float(temperature),
                        ensemble_number,
                        N,
                    )
                )

        print(
            f"T={temperature:.6f} | "
            f"completed="
            f"{state[temperature_index]['count']}/"
            f"{N_ENSEMBLES} | "
            f"remaining="
            f"{N_ENSEMBLES - len(completed_ensembles)}"
        )

    print()
    print(f"Remaining jobs: {len(jobs)}")
    print()

    # ========================================================
    # NOTHING LEFT TO DO
    # ========================================================

    if len(jobs) == 0:
        print("=" * 72)
        print("ALL EXISTING SIMULATIONS ARE COMPLETE")
        print("=" * 72)

        save_final_averages(state)
        save_summary(state)

        raise SystemExit(0)

    # ========================================================
    # PARALLEL EXECUTION
    # ========================================================

    executor = ProcessPoolExecutor(
        max_workers=N_WORKERS,
        mp_context=ctx,
        max_tasks_per_child=WORKER_TASKS_BEFORE_RESTART,
    )

    pending = {}

    next_job_index = 0

    completed_this_run = 0

    failed_jobs = []

    try:
        # ====================================================
        # SUBMIT ONLY N_WORKERS JOBS INITIALLY
        # ====================================================

        while len(pending) < N_WORKERS and next_job_index < len(jobs):
            job = jobs[next_job_index]

            future = executor.submit(
                run_single_ensemble,
                job,
            )

            pending[future] = job

            next_job_index += 1

        # ====================================================
        # MAIN COMPLETION LOOP
        # ====================================================

        while pending:
            # ------------------------------------------------
            # Wait until at least one job finishes
            # ------------------------------------------------

            done, _ = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            # ------------------------------------------------
            # Process completed jobs
            # ------------------------------------------------

            for future in done:
                job = pending.pop(future)

                (
                    temperature_index,
                    temperature,
                    ensemble_number,
                    system_size,
                ) = job

                try:
                    metadata = future.result()

                    # ----------------------------------------
                    # Load result saved by worker
                    # ----------------------------------------

                    result = load_pickle(metadata["filename"])

                    # ----------------------------------------
                    # Update running mean
                    # ----------------------------------------

                    current_state = state[temperature_index]

                    current_state["average"] = update_running_average(
                        current_state["average"],
                        current_state["count"],
                        result,
                    )

                    current_state["count"] += 1

                    # ----------------------------------------
                    # Save updated state
                    # ----------------------------------------

                    save_temperature_average(
                        temperature,
                        current_state["average"],
                        current_state["count"],
                    )

                    save_temperature_progress(
                        temperature,
                        current_state["count"],
                    )

                    completed_this_run += 1

                    # ----------------------------------------
                    # Progress information
                    # ----------------------------------------

                    total_completed = sum(
                        temp_state["count"] for temp_state in state.values()
                    )

                    total_target = N_TEMPERATURES * N_ENSEMBLES

                    print(
                        f"[{total_completed}/"
                        f"{total_target}] "
                        f"T={temperature:.6f} | "
                        f"ensemble="
                        f"{ensemble_number:04d} | "
                        f"T progress="
                        f"{current_state['count']}/"
                        f"{N_ENSEMBLES}",
                        flush=True,
                    )

                except Exception as error:
                    print()
                    print("ERROR while processing:")

                    print(f"    T={temperature:.6f}")

                    print(f"    ensemble={ensemble_number:04d}")

                    print(f"    error={error}")

                    print()

                    failed_jobs.append(
                        {
                            "temperature": temperature,
                            "ensemble": ensemble_number,
                            "error": repr(error),
                        }
                    )

                # --------------------------------------------
                # Keep the worker pool full
                # --------------------------------------------

                if next_job_index < len(jobs):
                    next_job = jobs[next_job_index]

                    new_future = executor.submit(
                        run_single_ensemble,
                        next_job,
                    )

                    pending[new_future] = next_job

                    next_job_index += 1

    except KeyboardInterrupt:
        # ====================================================
        # USER REQUESTED STOP
        # ====================================================

        print()
        print()
        print("=" * 72)
        print("INTERRUPT RECEIVED")
        print("Stopping worker processes...")
        print("=" * 72)

        # ----------------------------------------------------
        # Cancel futures that have not started.
        # ----------------------------------------------------

        for future in pending:
            future.cancel()

        # ----------------------------------------------------
        # Terminate workers that are currently running.
        # ----------------------------------------------------

        terminate_executor(executor)

        # ----------------------------------------------------
        # Save whatever state is already complete.
        # ----------------------------------------------------

        save_final_averages(state)

        save_summary(state)

        if failed_jobs:
            save_pickle_atomic(
                failed_jobs,
                BASE_DIR / "failed_jobs.pkl",
            )

        print()
        print("Already-completed ensemble files are safe.")

        print("Run the script again to resume.")

        print()

        raise SystemExit(130)

    except Exception:
        # ----------------------------------------------------
        # Unexpected error
        # ----------------------------------------------------

        terminate_executor(executor)

        save_final_averages(state)

        save_summary(state)

        raise

    else:
        # ====================================================
        # NORMAL SHUTDOWN
        # ====================================================

        executor.shutdown(wait=True)

    # ========================================================
    # SAVE FINAL DATA
    # ========================================================

    save_final_averages(state)

    save_summary(state)

    if failed_jobs:
        save_pickle_atomic(
            failed_jobs,
            BASE_DIR / "failed_jobs.pkl",
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 72)
    print("FPU THERMALIZATION RUN COMPLETE")
    print("=" * 72)

    print(f"Completed this run : {completed_this_run}")

    print(f"Failed this run    : {len(failed_jobs)}")

    print()

    for temperature_index, temp_state in state.items():
        temperature = temperatures[temperature_index]

        print(f"T={temperature:.6f} : {temp_state['count']}/{N_ENSEMBLES}")

    print("=" * 72)
