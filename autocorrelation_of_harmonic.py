import numpy as np
from velocity_verlet_thermalized import velver
from CreateChain import create_chain
from SpringSystem import springsystem
from Langevin_thermalized import Langevin

import matplotlib.pyplot as plt


members, neighbours = create_chain(500)

system = springsystem(members, neighbours, 0, 1)

system, *_ = Langevin(system, 10, 0.01, 1, 10)

(
    system,
    kinetic_average,
    potential_average,
    total_energy,
    normalized_spectral,
    temperature_distance,
    statistic,
    p_value,
    Correlator_momentum,
    Correlator_kinetic,
    Correlator_mode,
    Thermalized,
) = velver(system, 20, 0.01, 10)

# plot all the values from the velocity verlet. Need to check if there is complete autocorrelation since this is an integrable system.
#
#
# ---------------------------------------------------------
# Plot diagnostics from Velocity Verlet
# ---------------------------------------------------------

dt = 0.01
total_time = 10

# Construct time axis
n = len(kinetic_average)
time = np.arange(n) * dt

# ---------------------------------------------------------
# 1. Energy
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(time, kinetic_average, label="Kinetic Energy")
plt.plot(time, potential_average, label="Potential Energy")
plt.plot(time, total_energy, label="Total Energy")

plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy vs Time")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 2. Normalized spectral entropy
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(time, normalized_spectral)

plt.xlabel("Time")
plt.ylabel("Normalized Spectral Entropy")
plt.title("Spectral Entropy vs Time")
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 3. Temperature distance
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(time, temperature_distance)

plt.xlabel("Time")
plt.ylabel("Temperature Distance")
plt.title("Temperature Distance vs Time")
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 4. KS statistic and p-value
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(time, statistic, label="KS Statistic")
plt.plot(time, p_value, label="p-value")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("KS Test Diagnostics")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 5. Momentum autocorrelation
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(time, Correlator_momentum)

plt.xlabel("Time")
plt.ylabel("C(t)")
plt.title("Momentum Autocorrelation")
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 6. Kinetic-energy autocorrelation
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(time, Correlator_kinetic)

plt.xlabel("Time")
plt.ylabel("C(t)")
plt.title("Kinetic Energy Autocorrelation")
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 7. Mode autocorrelation
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(time, Correlator_mode)

plt.xlabel("Time")
plt.ylabel("C(t)")
plt.title("Mode Autocorrelation")
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 8. Thermalization indicator
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(time, Thermalized)

plt.xlabel("Time")
plt.ylabel("Thermalized")
plt.title("Thermalization Indicator")
plt.grid(True)

plt.tight_layout()
plt.show()
