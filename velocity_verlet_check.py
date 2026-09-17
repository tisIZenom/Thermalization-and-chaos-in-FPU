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

# Velocity Verlet parameters
verlet_time = 10
verlet_dt = 0.001


# ============================================================
# Create the systems
# ============================================================

members, neighbours = create_chain(N)

system_langevin = springsystem(members, neighbours, 0, 0.5)
system_montecarlo = springsystem(members, neighbours, 0, 0.5)


# ============================================================
# Langevin thermalization
# ============================================================

print(f"Temperature of the system is {temperature}")
print("Starting Langevin dynamics...")

system_langevin, kinetic_averageL, potential_averageL, total_energyL = Langevin(
    system_langevin, total_time, dt, gamma, temperature
)


# ============================================================
# Monte Carlo thermalization
# ============================================================

print("Now moving onto the Monte Carlo method")

system_montecarlo, kinetic_average, potential_average, total_energy = metropolis(
    system_montecarlo, mc_temperature, mc_sweeps
)


# ============================================================
# Velocity Verlet
# ============================================================

print("Starting velocity Verlet from Langevin state...")

system_velocity_langevin, kinetic, potential, total = velver(
    system_langevin, verlet_time, verlet_dt
)


print("Starting velocity Verlet from Monte Carlo state...")

system_velocity_montecarlo, kineticm, potentialm, totalm = velver(
    system_montecarlo, verlet_time, verlet_dt
)


# ============================================================
# Time arrays
# ============================================================

time_verlet = np.arange(len(total)) * verlet_dt


# ============================================================
# 1. Langevin -> Velocity Verlet
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time_verlet, kinetic, label="Kinetic Energy")
ax.plot(time_verlet, potential, label="Potential Energy")
ax.plot(time_verlet, total, label="Total Energy")

ax.set_xlabel("Time")
ax.set_ylabel("Energy")
ax.set_title("Velocity Verlet: Starting from Langevin Thermalized State")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ============================================================
# 2. Monte Carlo -> Velocity Verlet
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time_verlet, kineticm, label="Kinetic Energy")
ax.plot(time_verlet, potentialm, label="Potential Energy")
ax.plot(time_verlet, totalm, label="Total Energy")

ax.set_xlabel("Time")
ax.set_ylabel("Energy")
ax.set_title("Velocity Verlet: Starting from Monte Carlo State")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ============================================================
# 3. Compare total energy directly
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time_verlet, total, label="Langevin → Verlet")

ax.plot(time_verlet, totalm, label="Monte Carlo → Verlet")

ax.set_xlabel("Time")
ax.set_ylabel("Total Energy")
ax.set_title("Total Energy Conservation During Velocity Verlet")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ============================================================
# 4. Energy deviation from initial value
# ============================================================

initial_energy_L = total[0]
initial_energy_M = totalm[0]

energy_error_L = total - initial_energy_L
energy_error_M = totalm - initial_energy_M

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time_verlet, energy_error_L, label="Langevin → Verlet")

ax.plot(time_verlet, energy_error_M, label="Monte Carlo → Verlet")

ax.axhline(0, linewidth=1)

ax.set_xlabel("Time")
ax.set_ylabel(r"$E(t)-E(0)$")
ax.set_title("Energy Error During Velocity Verlet")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ============================================================
# 5. Print quantitative energy conservation
# ============================================================

print("\n==========================================")
print("Velocity Verlet energy conservation")
print("==========================================")

print("\nLangevin → Verlet:")
print("Initial energy:", total[0])
print("Final energy:", total[-1])
print("Maximum |E(t)-E(0)|:", np.max(np.abs(energy_error_L)))

print("\nMonte Carlo → Verlet:")
print("Initial energy:", totalm[0])
print("Final energy:", totalm[-1])
print("Maximum |E(t)-E(0)|:", np.max(np.abs(energy_error_M)))


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
