"""
Plot ensemble-averaged FPU thermalization data
for a single temperature.

Usage:

    python plot_temperature.py 10

or:

    python plot_temperature.py 0.1
"""

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS
# ============================================================

BASE_DIR = "results/thermalization"

# Number of points in the Langevin simulation
# This is only used to construct the time axis.
TOTAL_TIME = 100
DT = 0.01


# ============================================================
# GET TEMPERATURE FROM COMMAND LINE
# ============================================================

if len(sys.argv) != 2:
    print(
        "Usage:\n"
        "    python plot_temperature.py <temperature>\n\n"
        "Example:\n"
        "    python plot_temperature.py 10"
    )

    sys.exit(1)


temperature = float(sys.argv[1])


# ============================================================
# FIND TEMPERATURE DIRECTORY
# ============================================================

temperature_dir = os.path.join(BASE_DIR, f"T_{temperature:.6f}")

average_file = os.path.join(temperature_dir, "average.pkl")


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(average_file):
    print(f"\nCould not find:\n{average_file}\n")

    print(
        "Make sure the thermalization simulation has "
        "finished at least one ensemble for this temperature."
    )

    sys.exit(1)


# ============================================================
# LOAD AVERAGE
# ============================================================

with open(average_file, "rb") as f:
    data = pickle.load(f)


# The automation script stores the number of completed
# ensembles separately from the actual average arrays.

ensembles_completed = data["ensembles_completed"]

average = data["average"]


print()
print("=" * 60)
print("FPU THERMALIZATION AVERAGE")
print("=" * 60)
print(f"Temperature        : {temperature}")
print(f"Ensembles averaged : {ensembles_completed}")
print("=" * 60)
print()


# ============================================================
# EXTRACT DATA
# ============================================================

kinetic_energy = np.asarray(average["kinetic_energy"])

potential_energy = np.asarray(average["potential_energy"])

total_energy = np.asarray(average["total_energy"])

normalized_spectral = np.asarray(average["normalized_spectral"])

temp_diff = np.asarray(average["temp_diff"])

statistic = np.asarray(average["statistic"])

p_value = np.asarray(average["p_value"])

correlator_momentum = np.asarray(average["correlator_momentum"])

correlator_kinetic = np.asarray(average["correlator_kinetic"])

correlator_mode = np.asarray(average["correlator_mode"])


# ============================================================
# MAKE TIME AXIS
# ============================================================


def make_time_axis(data):

    return np.arange(len(data)) * DT


time_energy = make_time_axis(kinetic_energy)

time_spectral = make_time_axis(normalized_spectral)

time_temp = make_time_axis(temp_diff)

time_statistics = make_time_axis(statistic)

time_pvalue = make_time_axis(p_value)

time_momentum_corr = make_time_axis(correlator_momentum)

time_kinetic_corr = make_time_axis(correlator_kinetic)

time_mode_corr = make_time_axis(correlator_mode)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

plot_dir = os.path.join(temperature_dir, "plots")

os.makedirs(plot_dir, exist_ok=True)


# ============================================================
# 1. KINETIC ENERGY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_energy, kinetic_energy, label="Kinetic energy")

plt.xlabel("Time")
plt.ylabel("Kinetic energy")

plt.title(f"Ensemble-averaged kinetic energy — T = {temperature:g}")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "kinetic_energy.png"), dpi=300)

plt.show()


# ============================================================
# 2. POTENTIAL ENERGY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_energy, potential_energy, label="Potential energy")

plt.xlabel("Time")
plt.ylabel("Potential energy")

plt.title(f"Ensemble-averaged potential energy — T = {temperature:g}")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "potential_energy.png"), dpi=300)

plt.show()


# ============================================================
# 3. TOTAL ENERGY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_energy, total_energy, label="Total energy")

plt.xlabel("Time")
plt.ylabel("Total energy")

plt.title(f"Ensemble-averaged total energy — T = {temperature:g}")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "total_energy.png"), dpi=300)

plt.show()


# ============================================================
# 4. KINETIC + POTENTIAL + TOTAL ENERGY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_energy, kinetic_energy, label="Kinetic")

plt.plot(time_energy, potential_energy, label="Potential")

plt.plot(time_energy, total_energy, label="Total")

plt.xlabel("Time")
plt.ylabel("Energy")

plt.title(f"Energy evolution — T = {temperature:g}")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "all_energies.png"), dpi=300)

plt.show()


# ============================================================
# 5. NORMALIZED SPECTRAL ENTROPY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_spectral, normalized_spectral, label="Normalized spectral entropy")

plt.xlabel("Time")
plt.ylabel("Normalized spectral entropy")

plt.title(f"Spectral entropy — T = {temperature:g}")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "spectral_entropy.png"), dpi=300)

plt.show()


# ============================================================
# 6. TEMPERATURE DIFFERENCE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_temp, temp_diff, label="Temperature difference")

plt.xlabel("Time")
plt.ylabel(r"$|T_{\mathrm{system}} - T_{\mathrm{target}}|$")

plt.title(f"Temperature difference — T = {temperature:g}")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "temperature_difference.png"), dpi=300)

plt.show()


# ============================================================
# 7. KS STATISTIC
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_statistics, statistic, label="KS statistic")

plt.xlabel("Time")
plt.ylabel("KS statistic")

plt.title(f"KS statistic — T = {temperature:g}")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "ks_statistic.png"), dpi=300)

plt.show()


# ============================================================
# 8. KS P-VALUE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_pvalue, p_value, label="KS p-value")

plt.xlabel("Time")
plt.ylabel("p-value")

plt.title(f"KS test p-value — T = {temperature:g}")

plt.axhline(0.05, linestyle="--", label="p = 0.05")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "ks_p_value.png"), dpi=300)

plt.show()


# ============================================================
# 9. MOMENTUM AUTOCORRELATION
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_momentum_corr, correlator_momentum, label="Momentum autocorrelation")

plt.xlabel("Time lag")
plt.ylabel("Autocorrelation")

plt.title(f"Momentum autocorrelation — T = {temperature:g}")

plt.axhline(0, linestyle="--")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "momentum_autocorrelation.png"), dpi=300)

plt.show()


# ============================================================
# 10. KINETIC ENERGY AUTOCORRELATION
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_kinetic_corr, correlator_kinetic, label="Kinetic-energy autocorrelation")

plt.xlabel("Time lag")
plt.ylabel("Autocorrelation")

plt.title(f"Kinetic-energy autocorrelation — T = {temperature:g}")

plt.axhline(0, linestyle="--")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "kinetic_autocorrelation.png"), dpi=300)

plt.show()


# ============================================================
# 11. MODE ENERGY AUTOCORRELATION
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_mode_corr, correlator_mode, label="Mode-energy autocorrelation")

plt.xlabel("Time lag")
plt.ylabel("Autocorrelation")

plt.title(f"Mode-energy autocorrelation — T = {temperature:g}")

plt.axhline(0, linestyle="--")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "mode_autocorrelation.png"), dpi=300)

plt.show()


# ============================================================
# 12. THERMALIZATION DIAGNOSTICS TOGETHER
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(time_temp, temp_diff, label="Temperature difference")

plt.plot(time_spectral, normalized_spectral, label="Spectral entropy")

plt.xlabel("Time")
plt.ylabel("Diagnostic value")

plt.title(f"Thermalization diagnostics — T = {temperature:g}")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "thermalization_diagnostics.png"), dpi=300)

plt.show()


print()
print("=" * 60)
print("PLOTS SAVED")
print("=" * 60)
print(f"Directory: {plot_dir}")
print("=" * 60)
