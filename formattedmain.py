# Checking if everything works as expected.
#
# Comparing Langevin dynamics and Metropolis Monte Carlo
# for the FPU-beta chain.

from CreateChain import create_chain
from LangevinDynamcis import Langevin
from SpringSystem import springsystem
from montecarlo import metropolis
from Velocity_verlet import velver


import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Simulation parameters
# ============================================================

N = 100

temperature = 10
dt = 0.01

# Langevin parameters
total_time = 1000
gamma = 0.7

# Monte Carlo parameters
mc_temperature = temperature
mc_sweeps = 10000


# ============================================================
# Create the system
# ============================================================

members, neighbours = create_chain(N)

system_langevin = springsystem(members, neighbours, 0, 0.5)
system_montecarlo = springsystem(members, neighbours, 0, 00.5)


# ============================================================
# Langevin dynamics
# ============================================================

print(f"Temperature of the system is {temperature}")
print("Starting Langevin dynamics...")

system_langevin, kinetic_averageL, potential_averageL, total_energyL = Langevin(
    system_langevin, total_time, dt, gamma, temperature
)


# ============================================================
# Metropolis Monte Carlo
# ============================================================

print("Now moving onto the Monte Carlo method")

system_montecarlo, kinetic_average, potential_average, total_energy = metropolis(
    system_montecarlo, mc_temperature, mc_sweeps
)


# ============================================================
# Langevin energy plot
# ============================================================

time_L = np.arange(len(kinetic_averageL)) * dt

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time_L, kinetic_averageL, label="Kinetic Energy")
ax.plot(time_L, potential_averageL, label="Potential Energy")
ax.plot(time_L, total_energyL, label="Total Energy")

ax.set_xlabel("Time")
ax.set_ylabel("Energy")
ax.set_title(f"Langevin Thermalization: T = {temperature}")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# Langevin energy averages

print("The mean kinetic, potential and total energies for Langevin dynamics are:")

print("Mean kinetic energy:", np.mean(kinetic_averageL))
print("Mean potential energy:", np.mean(potential_averageL))
print("Mean total energy:", np.mean(total_energyL))


# ============================================================
# Monte Carlo energy plot
# ============================================================

sweeps = np.arange(len(kinetic_average))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sweeps, kinetic_average, label="Kinetic Energy")
ax.plot(sweeps, potential_average, label="Potential Energy")
ax.plot(sweeps, total_energy, label="Total Energy")

ax.set_xlabel("Monte Carlo Sweep")
ax.set_ylabel("Energy")
ax.set_title(f"Metropolis Thermalization: T = {temperature}")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# Monte Carlo energy averages

print("The mean kinetic, potential and total energies for Monte Carlo are:")

print("Mean kinetic energy:", np.mean(kinetic_average))
print("Mean potential energy:", np.mean(potential_average))
print("Mean total energy:", np.mean(total_energy))


# ============================================================
# Theoretical Gaussian momentum distribution
# ============================================================

# Theoretical distribution for m = 1:
#
# P(p) = 1/sqrt(2*pi*T) * exp(-p^2/(2*T))

p_theory = np.linspace(-5 * np.sqrt(temperature), 5 * np.sqrt(temperature), 500)

gaussian = (
    1 / np.sqrt(2 * np.pi * temperature) * np.exp(-(p_theory**2) / (2 * temperature))
)


# ============================================================
# Langevin momentum distribution
# ============================================================

p = system_langevin.momentum[1:-1]

fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(p, bins=50, density=True, alpha=0.7, label="Langevin")

ax.plot(p_theory, gaussian, linewidth=2, label=rf"Theory: $T={temperature}$")

ax.set_xlabel(r"Momentum $p$")
ax.set_ylabel(r"$P(p)$")
ax.set_title("Langevin Momentum Distribution")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ============================================================
# Monte Carlo momentum distribution
# ============================================================

p = system_montecarlo.momentum[1:-1]

fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(p, bins=50, density=True, alpha=0.7, label="Monte Carlo")

ax.plot(p_theory, gaussian, linewidth=2, label=rf"Theory: $T={temperature}$")

ax.set_xlabel(r"Momentum $p$")
ax.set_ylabel(r"$P(p)$")
ax.set_title("Monte Carlo Momentum Distribution")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()
