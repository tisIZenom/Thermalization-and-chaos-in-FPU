# Checking if everything works as expected.
#
from CreateChain import create_chain
from LangevinDynamcis import Langevin
from SpringSystem import springsystem
from potential_differentiated import potential_dif
from montecarlo import metropolis


import numpy as np
import matplotlib.pyplot as plt

members, neighbours = create_chain(108)

system_langevin = springsystem(members, neighbours, 0, 1)
system_montecarlo = springsystem(members, neighbours, 0, 1)


system_langevin, kinetic_averageL, potential_averageL, total_energyL = Langevin(
    system_langevin, 1000, 0.01, 0.5, 1
)

print("temperature of the system is 1")

print("now moving onto the monte carlo method")

system_montecarlo, kinetic_average, potential_average, total_energy = metropolis(
    system_montecarlo, 1, 10000
)


## ### the plotting parts
# the LangevinDynamcis plot
#
dt = 0.01

time_L = np.arange(len(kinetic_averageL)) * dt

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time_L, kinetic_averageL, label="Kinetic Energy")
ax.plot(time_L, potential_averageL, label="Potential Energy")
ax.plot(time_L, total_energyL, label="Total Energy")

ax.set_xlabel("Time")
ax.set_ylabel("Energy")
ax.set_title("Langevin Thermalization: T = 10")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()

print(
    "the values of the kinetic, potential and total energy mean of the LangevinDynamcis is: "
)
print(np.mean((kinetic_averageL)))
print(np.mean(potential_averageL))
print(np.mean(total_energyL))


## Now the montecarlo part

sweeps = np.arange(len(kinetic_average))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sweeps, kinetic_average, label="Kinetic Energy")
ax.plot(sweeps, potential_average, label="Potential Energy")
ax.plot(sweeps, total_energy, label="Total Energy")

ax.set_xlabel("Monte Carlo Sweep")
ax.set_ylabel("Energy")
ax.set_title("Metropolis Thermalization: T = 10")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()

print("This is for the montecarlo kinetic, potential and total means: ")
print(np.mean(kinetic_average))
print(np.mean(potential_average))
print(np.mean(total_energy))


## Also plotting the gaussian of the system.

p = system_montecarlo.momentum[1:-1]

fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(p, bins=100, density=True, alpha=0.7, label="Monte Carlo")

p_theory = np.linspace(-12, 12, 500)

gaussian = 1 / np.sqrt(2 * np.pi * 10) * np.exp(-(p_theory**2) / (2 * 10))

ax.plot(p_theory, gaussian, linewidth=2, label=r"Theory: $T=10$")

ax.set_xlabel(r"Momentum $p$")
ax.set_ylabel(r"$P(p)$")
ax.set_title("Momentum Distribution")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()
