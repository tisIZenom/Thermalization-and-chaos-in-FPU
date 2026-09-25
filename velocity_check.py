import numpy as np
import matplotlib.pyplot as plt

from SpringSystem import springsystem
from CreateChain import create_chain
from Langevin_thermalized import Langevin
from velocity_verlet_thermalized import velver

# ---------------------------------------------------------
# Create system
# ---------------------------------------------------------

members, neighbours = create_chain(500)

system = springsystem(members, neighbours, 0, 1)


# ---------------------------------------------------------
# Run Langevin thermalization
# ---------------------------------------------------------

(system_lange, *_) = Langevin(
    system,
    20,  # total time
    0.01,  # dt
    0.7,  # gamma
    10,  # temperature
)


(
    system,
    kinetic_average,
    potential_average,
    total_energy,
    normalized_spectral,
    temperaturediff,
    statistic,
    pvalue,
    Corr_mom,
    Corr_kin,
    Corr_mode,
    thermalized,
) = velver(system_lange, 10, 0.01, 10)


# ---------------------------------------------------------
# Plot correlations from the Velocity Verlet evolution
# ---------------------------------------------------------

dt = 0.01

# Number of correlation points
n_corr = len(Corr_mom)

# Time corresponding to each correlation lag
corr_time = np.arange(n_corr) * dt

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# ---------------------------------------------------------
# Momentum autocorrelation
# ---------------------------------------------------------

axes[0].plot(corr_time, Corr_mom, linewidth=1.5)

axes[0].axhline(0, linestyle="--", linewidth=1)

axes[0].set_ylabel(r"$C_p(t)$")
axes[0].set_title("Momentum Autocorrelation")
axes[0].grid(True, alpha=0.3)


# ---------------------------------------------------------
# Kinetic-energy autocorrelation
# ---------------------------------------------------------

axes[1].plot(corr_time, Corr_kin, linewidth=1.5)

axes[1].axhline(0, linestyle="--", linewidth=1)

axes[1].set_ylabel(r"$C_K(t)$")
axes[1].set_title("Kinetic Energy Autocorrelation")
axes[1].grid(True, alpha=0.3)


# ---------------------------------------------------------
# Mode autocorrelation
# ---------------------------------------------------------

axes[2].plot(corr_time, Corr_mode, linewidth=1.5)

axes[2].axhline(0, linestyle="--", linewidth=1)

axes[2].set_ylabel(r"$C_{\mathrm{mode}}(t)$")
axes[2].set_xlabel("Lag time")
axes[2].set_title("Mode Autocorrelation")
axes[2].grid(True, alpha=0.3)


plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Plot all diagnostics from the Velocity Verlet evolution
# ---------------------------------------------------------

dt = 0.01
n_steps = len(kinetic_average)
time = np.arange(n_steps) * dt

fig, axes = plt.subplots(4, 2, figsize=(14, 14))

# ---------------------------------------------------------
# 1. Kinetic energy
# ---------------------------------------------------------

axes[0, 0].plot(time, kinetic_average, linewidth=1.5)
axes[0, 0].set_ylabel(r"$\langle K\rangle$")
axes[0, 0].set_title("Kinetic Energy")
axes[0, 0].grid(True, alpha=0.3)


# ---------------------------------------------------------
# 2. Potential energy
# ---------------------------------------------------------

axes[0, 1].plot(time, potential_average, linewidth=1.5)
axes[0, 1].set_ylabel(r"$\langle V\rangle$")
axes[0, 1].set_title("Potential Energy")
axes[0, 1].grid(True, alpha=0.3)


# ---------------------------------------------------------
# 3. Total energy
# ---------------------------------------------------------

axes[1, 0].plot(time, total_energy, linewidth=1.5)
axes[1, 0].set_ylabel(r"$E = K+V$")
axes[1, 0].set_title("Total Energy")
axes[1, 0].grid(True, alpha=0.3)


# ---------------------------------------------------------
# 4. Temperature difference
# ---------------------------------------------------------

axes[1, 1].plot(time, temperaturediff, linewidth=1.5)
axes[1, 1].axhline(0, linestyle="--", linewidth=1)

axes[1, 1].set_ylabel(r"$\Delta T$")
axes[1, 1].set_title("Temperature Difference")
axes[1, 1].grid(True, alpha=0.3)


# ---------------------------------------------------------
# 5. Spectral entropy
# ---------------------------------------------------------

axes[2, 0].plot(time, normalized_spectral, linewidth=1.5)
axes[2, 0].set_ylabel("Normalized spectral entropy")
axes[2, 0].set_title("Spectral Entropy")
axes[2, 0].grid(True, alpha=0.3)


# ---------------------------------------------------------
# 6. KS statistic
# ---------------------------------------------------------

axes[2, 1].plot(time, statistic, linewidth=1.5)
axes[2, 1].set_ylabel("KS statistic")
axes[2, 1].set_title("KS Statistic")
axes[2, 1].grid(True, alpha=0.3)


# ---------------------------------------------------------
# 7. KS p-value
# ---------------------------------------------------------

axes[3, 0].plot(time, pvalue, linewidth=1.5)
axes[3, 0].axhline(0.05, linestyle="--", linewidth=1, label="p = 0.05")

axes[3, 0].set_ylabel("p-value")
axes[3, 0].set_xlabel("Time")
axes[3, 0].set_title("KS Test p-value")
axes[3, 0].legend()
axes[3, 0].grid(True, alpha=0.3)


# ---------------------------------------------------------
# 8. Thermalized flag
# ---------------------------------------------------------

plt.tight_layout()
plt.show()
