# ============================================================
# Generate FPU systems with different energies and calculate
# the maximal Lyapunov exponent for each system.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from SpringSystem import springsystem
from CreateChain import create_chain
from benettin_local_fixed import benettin


# ============================================================
# Parameters
# ============================================================

N = 500

# Benettin parameters
target = 250
num_steps = 1000
dt = 0.01
epsilon = 0.0001

number_of_systems = 25

# FPU parameters
alpha = 0
beta = 1


# ============================================================
# Storage
# ============================================================

# Total energy of each realization
system_energies = np.zeros(number_of_systems)

# Maximal Lyapunov exponent of each realization
lambda_values = np.zeros(number_of_systems)

# Other quantities returned by Benettin
lyap_values = []
delt_values = []
kinetic_values = []
potential_values = []
total_values = []


# ============================================================
# Create the chain
# ============================================================

members, positions = create_chain(N)


# ============================================================
# Generate systems and calculate Lyapunov exponents
# ============================================================

for i in range(number_of_systems):
    print(f"Running system {i + 1}/{number_of_systems}")

    # --------------------------------------------------------
    # Create a fresh system
    # --------------------------------------------------------

    system = springsystem(members, positions, alpha, beta)

    # --------------------------------------------------------
    # Random initial conditions
    #
    # q and p are independently sampled from [-1, 1].
    # --------------------------------------------------------

    p = np.random.uniform(-1, 1, size=N)
    q = np.random.uniform(-1, 1, size=N)

    # --------------------------------------------------------
    # Apply initial conditions
    # --------------------------------------------------------

    system.displacement = q
    system.momentum = p

    # --------------------------------------------------------
    # Calculate the initial total energy
    # --------------------------------------------------------

    system_energies[i] = system.total_energy()

    # --------------------------------------------------------
    # Run Benettin algorithm
    # --------------------------------------------------------

    (lambda_max, local_lyap, delt, kineticmean, potentialmean, totalmean) = benettin(
        system, target, num_steps, dt, epsilon, 1
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    lambda_values[i] = lambda_max

    lyap_values.append(np.asarray(local_lyap))
    delt_values.append(np.asarray(delt))
    kinetic_values.append(np.asarray(kineticmean))
    potential_values.append(np.asarray(potentialmean))
    total_values.append(np.asarray(totalmean))


# ============================================================
# Sort everything according to system energy
# ============================================================

sort_indices = np.argsort(system_energies)

sorted_energies = system_energies[sort_indices]
sorted_lambda = lambda_values[sort_indices]


# ============================================================
# Print results
# ============================================================

print("\nResults:")
print("-" * 50)

for energy, lyap in zip(sorted_energies, sorted_lambda):
    print(f"Energy = {energy:.6e}    Lambda_max = {lyap:.6e}")


# ============================================================
# Plot 1:
# Maximal Lyapunov exponent vs total energy
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(sorted_energies, sorted_lambda, s=50)

plt.plot(sorted_energies, sorted_lambda, alpha=0.5)

plt.xlabel("Total Energy")
plt.ylabel(r"$\lambda_{\max}$")
plt.title("Maximal Lyapunov Exponent vs Total Energy")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# Plot 2:
# Energy per particle vs Lyapunov exponent
# ============================================================

energy_per_particle = sorted_energies / N

plt.figure(figsize=(8, 6))

plt.scatter(energy_per_particle, sorted_lambda, s=50)

plt.plot(energy_per_particle, sorted_lambda, alpha=0.5)

plt.xlabel("Energy per particle")
plt.ylabel(r"$\lambda_{\max}$")
plt.title("Maximal Lyapunov Exponent vs Energy per Particle")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# Plot 3:
# Local Lyapunov exponent convergence for all systems
# ============================================================

plt.figure(figsize=(9, 6))

for i in range(number_of_systems):
    plt.plot(lyap_values[i], alpha=0.5)

plt.xlabel("Benettin iteration")
plt.ylabel(r"Local $\lambda$")
plt.title("Lyapunov Exponent Convergence")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# Plot 4:
# Distance between trajectories
# ============================================================

plt.figure(figsize=(9, 6))

for i in range(number_of_systems):
    plt.plot(delt_values[i], alpha=0.5)

plt.xlabel("Benettin iteration")
plt.ylabel(r"$\delta$")
plt.title("Perturbation Distance")

plt.yscale("log")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# Plot 5:
# Mean kinetic, potential and total energies
# ============================================================

plt.figure(figsize=(9, 6))

for i in range(number_of_systems):
    plt.plot(kinetic_values[i], alpha=0.3, label="Kinetic" if i == 0 else None)

    plt.plot(potential_values[i], alpha=0.3, label="Potential" if i == 0 else None)

    plt.plot(total_values[i], alpha=0.3, label="Total" if i == 0 else None)

plt.xlabel("Benettin iteration")
plt.ylabel("Energy")
plt.title("Energy During Benettin Evolution")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
