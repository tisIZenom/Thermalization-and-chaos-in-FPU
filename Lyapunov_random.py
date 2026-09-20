## Choose a random configuration of the system and compute
## the maximal Lyapunov exponent for different beta values.

import numpy as np
import matplotlib.pyplot as plt

from SpringSystem import springsystem
from CreateChain import create_chain
from benettin_local_fixed import benettin


# ============================================================
# Parameters
# ============================================================

N = 500

# Beta values: 1, 10, 100, 1000
# Change this if you want a denser sweep.
beta_values = np.logspace(0, 3, 4)

# Benettin parameters
target = 250
num_steps = 1000
dt = 0.01
epsilon = 0.0001
renormalization = 1


# ============================================================
# Create the chain
# ============================================================

members, positions = create_chain(N)


# ============================================================
# Generate ONE initial configuration
# ============================================================

# Uniform random initial conditions
q = np.random.uniform(-1, 1, N)
p = np.random.uniform(-1, 1, N)


# ============================================================
# Storage
# ============================================================

lambda_max_values = []

# These will contain averages over the Benettin trajectory
local_lyap_mean = []
delt_mean = []
kinetic_mean = []
potential_mean = []
total_mean = []

# Initial energy of each system
initial_energy = []


# ============================================================
# Run beta sweep
# ============================================================

for beta in beta_values:
    print("\n" + "=" * 60)
    print(f"Running beta = {beta}")
    print("=" * 60)

    # Create a NEW system for this beta
    system = springsystem(
        members,
        positions,
        0,  # alpha
        beta,
    )

    # Copy the SAME initial configuration
    # so beta is the parameter being changed.
    system.displacement = q.copy()
    system.momentum = p.copy()

    # Initial energy
    E0 = system.total_energy()
    initial_energy.append(E0)

    print(f"Initial energy = {E0}")

    # --------------------------------------------------------
    # Benettin algorithm
    # --------------------------------------------------------

    (lambda_max, local_lyap, delt, kineticmean, potentialmean, totalmean) = benettin(
        system, target, num_steps, dt, epsilon, renormalization
    )

    # --------------------------------------------------------
    # Store maximal Lyapunov exponent
    # --------------------------------------------------------

    lambda_max_values.append(lambda_max)

    # --------------------------------------------------------
    # Average the other quantities
    #
    # Ignore NaN values if they occur during the simulation.
    # --------------------------------------------------------

    local_lyap_mean.append(np.nanmean(local_lyap))

    delt_mean.append(np.nanmean(delt))

    kinetic_mean.append(np.nanmean(kineticmean))

    potential_mean.append(np.nanmean(potentialmean))

    total_mean.append(np.nanmean(totalmean))

    print(f"Lambda_max          = {lambda_max}")
    print(f"Mean local Lyapunov  = {local_lyap_mean[-1]}")
    print(f"Mean delta           = {delt_mean[-1]}")
    print(f"Mean kinetic energy  = {kinetic_mean[-1]}")
    print(f"Mean potential energy= {potential_mean[-1]}")
    print(f"Mean total energy    = {total_mean[-1]}")


# ============================================================
# Convert to numpy arrays
# ============================================================

beta_values = np.asarray(beta_values)

lambda_max_values = np.asarray(lambda_max_values)
local_lyap_mean = np.asarray(local_lyap_mean)
delt_mean = np.asarray(delt_mean)
kinetic_mean = np.asarray(kinetic_mean)
potential_mean = np.asarray(potential_mean)
total_mean = np.asarray(total_mean)
initial_energy = np.asarray(initial_energy)


# ============================================================
# Print summary table
# ============================================================

print("\n\nFinal results")
print("=" * 100)

print(
    f"{'Beta':>10}"
    f"{'Lambda_max':>15}"
    f"{'<Local Lyap>':>15}"
    f"{'<Delta>':>15}"
    f"{'<K>':>15}"
    f"{'<V>':>15}"
    f"{'<E>':>15}"
)

print("-" * 100)

for i in range(len(beta_values)):
    print(
        f"{beta_values[i]:>10.3g}"
        f"{lambda_max_values[i]:>15.6g}"
        f"{local_lyap_mean[i]:>15.6g}"
        f"{delt_mean[i]:>15.6g}"
        f"{kinetic_mean[i]:>15.6g}"
        f"{potential_mean[i]:>15.6g}"
        f"{total_mean[i]:>15.6g}"
    )


# ============================================================
# Plot 1: Maximal Lyapunov exponent vs beta
# ============================================================

plt.figure(figsize=(8, 6))

plt.semilogx(beta_values, lambda_max_values, "o-")

plt.xlabel(r"$\beta$")
plt.ylabel(r"$\lambda_{\max}$")
plt.title("Maximal Lyapunov Exponent vs Beta")

plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 2: Mean local Lyapunov exponent vs beta
# ============================================================

plt.figure(figsize=(8, 6))

plt.semilogx(beta_values, local_lyap_mean, "o-")

plt.xlabel(r"$\beta$")
plt.ylabel(r"$\langle \lambda_{\mathrm{local}} \rangle$")
plt.title("Mean Local Lyapunov Exponent vs Beta")

plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 3: Mean delta vs beta
# ============================================================

plt.figure(figsize=(8, 6))

plt.loglog(beta_values, delt_mean, "o-")

plt.xlabel(r"$\beta$")
plt.ylabel(r"$\langle \delta \rangle$")
plt.title("Mean Perturbation Separation vs Beta")

plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 4: Energies vs beta
# ============================================================

plt.figure(figsize=(8, 6))

plt.semilogx(beta_values, kinetic_mean, "o-", label="Kinetic")

plt.semilogx(beta_values, potential_mean, "o-", label="Potential")

plt.semilogx(beta_values, total_mean, "o-", label="Total")

plt.xlabel(r"$\beta$")
plt.ylabel("Energy")

plt.title("Mean Energies vs Beta")

plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 5: Initial energy vs beta
# ============================================================

plt.figure(figsize=(8, 6))

plt.semilogx(beta_values, initial_energy, "o-")

plt.xlabel(r"$\beta$")
plt.ylabel(r"$E_0$")
plt.title("Initial Energy vs Beta")

plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
