import numpy as np
import matplotlib.pyplot as plt

from SpringSystem import springsystem
from CreateChain import create_chain
from Langevin_thermalized import Langevin


# ---------------------------------------------------------
# Create system
# ---------------------------------------------------------

members, neighbours = create_chain(500)

system = springsystem(members, neighbours, 0, 1)


# ---------------------------------------------------------
# Run Langevin thermalization
# ---------------------------------------------------------

(
    system_lange,
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
) = Langevin(
    system,
    10,  # total time
    0.01,  # dt
    0.7,  # gamma
    10,  # temperature
)


# ---------------------------------------------------------
# Time array
# ---------------------------------------------------------

dt = 0.01
time = np.arange(len(kinetic_energy)) * dt


# ---------------------------------------------------------
# 1. Kinetic energy
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(time, kinetic_energy)
plt.xlabel("Time")
plt.ylabel("Kinetic energy")
plt.title("Kinetic Energy vs Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 2. Potential energy
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(time, potential_energy)
plt.xlabel("Time")
plt.ylabel("Potential energy")
plt.title("Potential Energy vs Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 3. Total energy
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(time, total_energy)
plt.xlabel("Time")
plt.ylabel("Total energy")
plt.title("Total Energy vs Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 4. Spectral entropy
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(time, normalized_spectral)
plt.xlabel("Time")
plt.ylabel("Normalized spectral entropy")
plt.title("Spectral Entropy vs Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 5. Temperature difference
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(time, temp_diff)
plt.xlabel("Time")
plt.ylabel("Temperature difference")
plt.title("Temperature Difference vs Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 6. Statistical quantity
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(time, statistic)
plt.xlabel("Time")
plt.ylabel("Statistic")
plt.title("Statistical Diagnostic vs Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 7. KS-test p-value
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(time, p_value)
plt.axhline(0.05, linestyle="--", label="p = 0.05")

plt.xlabel("Time")
plt.ylabel("KS-test p-value")
plt.title("Momentum Gaussianity: KS-test")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 8. Momentum autocorrelation
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(time, correlator_momentum)
plt.xlabel("Time")
plt.ylabel("Momentum autocorrelation")
plt.title("Momentum Autocorrelation vs Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 9. Kinetic-energy autocorrelation
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(time, correlator_kinetic)
plt.xlabel("Time")
plt.ylabel("Kinetic-energy autocorrelation")
plt.title("Kinetic Energy Autocorrelation vs Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 10. Mode-energy autocorrelation
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(time, correlator_mode)
plt.xlabel("Time")
plt.ylabel("Mode-energy autocorrelation")
plt.title("Mode Energy Autocorrelation vs Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# FINAL MOMENTUM DISTRIBUTION
# =========================================================

# Extract final momenta
p_final = np.asarray(system_lange.momentum)

# Remove fixed-boundary momenta if necessary
# Uncomment if your boundary sites are included:
#
# p_final = p_final[1:-1]


# Gaussian parameters
mean_p = np.mean(p_final)
std_p = np.std(p_final, ddof=1)

x = np.linspace(mean_p - 4 * std_p, mean_p + 4 * std_p, 500)

gaussian = 1 / (std_p * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean_p) / std_p) ** 2)


# Histogram
plt.figure(figsize=(8, 5))

plt.hist(p_final, bins=30, density=True, alpha=0.6, label="Momentum distribution")

plt.plot(x, gaussian, linewidth=2, label="Gaussian fit")

plt.xlabel("Momentum $p$")
plt.ylabel("Probability density")
plt.title("Final Momentum Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


### Now compute lyapunov exponent
