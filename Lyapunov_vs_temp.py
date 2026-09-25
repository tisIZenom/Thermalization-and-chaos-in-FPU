# ============================================================
# FPU-beta chain:
#
# Measure the maximal/local Lyapunov exponent as a function
# of temperature.
#
# For each temperature:
#   1. Create fresh FPU systems
#   2. Thermalize each system using Langevin dynamics
#   3. Calculate the Lyapunov exponent using Benettin
#   4. Average over independent realizations
#   5. Calculate the standard error
#
# Finally:
#   - Plot lambda_max vs temperature
#   - Plot energy vs temperature
#   - Perform a linear fit
#   - Perform a power-law fit
#
# Power law:
#
#       lambda_max ~ A T^alpha
#
# ============================================================


import numpy as np
import matplotlib.pyplot as plt

from SpringSystem import springsystem
from CreateChain import create_chain
from benettin_local_fixed import benettin
from Langevin_thermalized import Langevin


# ============================================================
# Parameters
# ============================================================

# ------------------------------------------------------------
# FPU system
# ------------------------------------------------------------

N = 500

alpha = 0
beta = 100


# ------------------------------------------------------------
# Benettin parameters
# ------------------------------------------------------------

target = 250

num_steps = 100

dt = 0.01

epsilon = 0.0001


# ------------------------------------------------------------
# Ensemble parameters
# ------------------------------------------------------------

# Number of independent systems at each temperature
number_of_systems = 1


# ------------------------------------------------------------
# Langevin thermalization
# ------------------------------------------------------------

thermalization_time = 20

gamma = 0.7


# ------------------------------------------------------------
# Temperature range
# ------------------------------------------------------------

number_of_temperatures = 10

temperatures = np.logspace(np.log10(0.1), np.log10(100), number_of_temperatures)


# ============================================================
# Print parameters
# ============================================================

print("=" * 70)
print("FPU LYAPUNOV EXPONENT VS TEMPERATURE")
print("=" * 70)

print(f"N                    = {N}")
print(f"alpha                = {alpha}")
print(f"beta                 = {beta}")
print(f"target               = {target}")
print(f"Benettin steps       = {num_steps}")
print(f"dt                   = {dt}")
print(f"epsilon              = {epsilon}")
print(f"thermalization time  = {thermalization_time}")
print(f"gamma                = {gamma}")
print(f"number of systems    = {number_of_systems}")

print("\nTemperatures:")
print(temperatures)

print("=" * 70)


# ============================================================
# Storage
# ============================================================

num_temperatures = len(temperatures)


# Lyapunov exponent for every realization
#
# Shape:
#
#     [temperature, realization]
#
lambda_all = np.zeros((num_temperatures, number_of_systems))


# Energy diagnostics
kinetic_all = np.zeros((num_temperatures, number_of_systems))

potential_all = np.zeros((num_temperatures, number_of_systems))

total_all = np.zeros((num_temperatures, number_of_systems))


# ============================================================
# Main temperature loop
# ============================================================

for i, temperature in enumerate(temperatures):
    print("\n")
    print("=" * 70)
    print(f"Temperature {i + 1}/{num_temperatures}")
    print(f"T = {temperature:.6g}")
    print("=" * 70)

    # --------------------------------------------------------
    # Ensemble loop
    # --------------------------------------------------------

    for j in range(number_of_systems):
        print(f"  System {j + 1}/{number_of_systems}", end="\r")

        # ====================================================
        # Create a fresh system
        # ====================================================

        members, positions = create_chain(N)

        system = springsystem(members, positions, alpha, beta)

        # ====================================================
        # Langevin thermalization
        # ====================================================

        system, *_ = Langevin(system, thermalization_time, dt, gamma, temperature)

        # ====================================================
        # Benettin Lyapunov calculation
        # ====================================================

        (lambda_max, local_lyap, delt, kineticmean, potentialmean, totalmean) = (
            benettin(system, target, num_steps, dt, epsilon, 0.5)
        )

        # ====================================================
        # Store results
        # ====================================================
        lambda_all[i, j] = lambda_max

        kinetic_all[i, j] = np.mean(kineticmean)

        potential_all[i, j] = np.mean(potentialmean)

        total_all[i, j] = np.mean(totalmean)

    # --------------------------------------------------------
    # Print result for this temperature
    # --------------------------------------------------------

    mean_lambda = np.mean(lambda_all[i])

    std_lambda = np.std(lambda_all[i], ddof=1)

    sem_lambda = std_lambda / np.sqrt(number_of_systems)

    print()

    print(f"T = {temperature:.6g} | lambda = {mean_lambda:.6e} +/- {sem_lambda:.6e}")


# ============================================================
# Ensemble averages
# ============================================================

lambda_mean = np.mean(lambda_all, axis=1)

lambda_std = np.std(lambda_all, axis=1, ddof=1)

lambda_sem = lambda_std / np.sqrt(number_of_systems)


# ------------------------------------------------------------
# Energy averages
# ------------------------------------------------------------

kinetic_mean = np.mean(kinetic_all, axis=1)

potential_mean = np.mean(potential_all, axis=1)

total_mean = np.mean(total_all, axis=1)


# ============================================================
# Print final results
# ============================================================

print("\n\n")

print("=" * 100)
print("FINAL RESULTS")
print("=" * 100)

print(
    f"{'T':>12}"
    f"{'Lambda':>18}"
    f"{'Std':>18}"
    f"{'SEM':>18}"
    f"{'Kinetic':>18}"
    f"{'Potential':>18}"
    f"{'Total':>18}"
)

print("-" * 100)


for i, T in enumerate(temperatures):
    print(
        f"{T:12.5g}"
        f"{lambda_mean[i]:18.8e}"
        f"{lambda_std[i]:18.8e}"
        f"{lambda_sem[i]:18.8e}"
        f"{kinetic_mean[i]:18.8e}"
        f"{potential_mean[i]:18.8e}"
        f"{total_mean[i]:18.8e}"
    )


# ============================================================
# Plot 1:
# Lyapunov exponent vs temperature
# ============================================================

plt.figure(figsize=(8, 6))

plt.errorbar(
    temperatures,
    lambda_mean,
    yerr=lambda_sem,
    fmt="o-",
    capsize=4,
    linewidth=1.5,
    label="Simulation",
)

# Temperature spans several orders of magnitude
plt.xscale("log")

plt.xlabel("Temperature $T$")

plt.ylabel(r"Lyapunov exponent $\lambda_{\max}$")

plt.title("Lyapunov Exponent vs Temperature")

plt.grid(True, which="both", alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# Plot 2:
# Energy vs temperature
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(temperatures, kinetic_mean, "o-", label="Kinetic")

plt.plot(temperatures, potential_mean, "o-", label="Potential")

plt.plot(temperatures, total_mean, "o-", label="Total")

plt.xscale("log")

plt.xlabel("Temperature $T$")

plt.ylabel("Mean Energy")

plt.title("Energy vs Temperature")

plt.grid(True, which="both", alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# LINEAR FIT
#
# lambda = a*T + b
#
# ============================================================

print("\n")
print("=" * 70)
print("LINEAR FIT")
print("=" * 70)


linear_coefficients, linear_covariance = np.polyfit(
    temperatures, lambda_mean, 1, cov=True
)


slope = linear_coefficients[0]

intercept = linear_coefficients[1]


slope_error = np.sqrt(linear_covariance[0, 0])

intercept_error = np.sqrt(linear_covariance[1, 1])


print(f"Slope     = {slope:.8e} +/- {slope_error:.8e}")

print(f"Intercept = {intercept:.8e} +/- {intercept_error:.8e}")


# ------------------------------------------------------------
# Generate linear fit
# ------------------------------------------------------------

T_linear_fit = np.linspace(temperatures.min(), temperatures.max(), 300)


lambda_linear_fit = slope * T_linear_fit + intercept


# ------------------------------------------------------------
# Plot linear fit
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.errorbar(
    temperatures, lambda_mean, yerr=lambda_sem, fmt="o", capsize=4, label="Simulation"
)

plt.plot(
    T_linear_fit,
    lambda_linear_fit,
    "--",
    linewidth=1.5,
    label=(
        rf"Linear fit: "
        rf"$\lambda = ({slope:.3e}"
        rf"\pm{slope_error:.1e})T"
        rf" + {intercept:.3e}$"
    ),
)

plt.xscale("log")

plt.xlabel("Temperature $T$")

plt.ylabel(r"Lyapunov exponent $\lambda_{\max}$")

plt.title("Linear Fit of Lyapunov Exponent")

plt.grid(True, which="both", alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# POWER-LAW FIT
#
# lambda = A*T^alpha
#
# Taking logarithms:
#
# log(lambda) = alpha*log(T) + log(A)
#
# ============================================================

print("\n")
print("=" * 70)
print("POWER-LAW FIT")
print("=" * 70)


# ------------------------------------------------------------
# Select only positive Lyapunov exponents
# ------------------------------------------------------------

positive_mask = lambda_mean > 0


positive_temperatures = temperatures[positive_mask]

positive_lambda = lambda_mean[positive_mask]


# ------------------------------------------------------------
# Check that enough points exist
# ------------------------------------------------------------

if len(positive_lambda) < 2:
    print("Not enough positive Lyapunov exponents for a power-law fit.")

else:
    # --------------------------------------------------------
    # Transform to logarithmic variables
    # --------------------------------------------------------

    log_T = np.log(positive_temperatures)

    log_lambda = np.log(positive_lambda)

    # --------------------------------------------------------
    # Fit:
    #
    # log(lambda) = alpha log(T) + log(A)
    # --------------------------------------------------------

    power_coefficients, power_covariance = np.polyfit(log_T, log_lambda, 1, cov=True)

    alpha_fit = power_coefficients[0]

    log_A = power_coefficients[1]

    # --------------------------------------------------------
    # Uncertainties
    # --------------------------------------------------------

    alpha_error = np.sqrt(power_covariance[0, 0])

    log_A_error = np.sqrt(power_covariance[1, 1])

    # --------------------------------------------------------
    # Convert log(A) -> A
    # --------------------------------------------------------

    A = np.exp(log_A)

    # Error propagation:
    #
    # A = exp(log(A))
    #
    # sigma_A = A * sigma_logA
    #

    A_error = A * log_A_error

    # --------------------------------------------------------
    # Print power-law result
    # --------------------------------------------------------

    print("Model:")

    print(r"    lambda = A*T^alpha")

    print()

    print(f"alpha = {alpha_fit:.8f} +/- {alpha_error:.8f}")

    print(f"A     = {A:.8e} +/- {A_error:.8e}")

    # --------------------------------------------------------
    # Generate power-law curve
    # --------------------------------------------------------

    T_power_fit = np.logspace(
        np.log10(positive_temperatures.min()),
        np.log10(positive_temperatures.max()),
        300,
    )

    lambda_power_fit = A * T_power_fit**alpha_fit

    # ========================================================
    # Plot power-law fit
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        temperatures,
        lambda_mean,
        yerr=lambda_sem,
        fmt="o",
        capsize=4,
        label="Simulation",
    )

    plt.plot(
        T_power_fit,
        lambda_power_fit,
        "--",
        linewidth=1.5,
        label=(
            rf"Power law: "
            rf"$\lambda \propto "
            rf"T^{{{alpha_fit:.3f}"
            rf"\pm{alpha_error:.3f}}}$"
        ),
    )

    plt.xscale("log")

    plt.yscale("log")

    plt.xlabel("Temperature $T$")

    plt.ylabel(r"Lyapunov exponent $\lambda_{\max}$")

    plt.title("Power-Law Scaling of Lyapunov Exponent")

    plt.grid(True, which="both", alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# END
# ============================================================
