import numpy as np
import matplotlib.pyplot as plt

from CreateChain import create_chain
from LangevinDynamcis import Langevin
from SpringSystem import springsystem
from montecarlo import metropolis
from Velocity_verlet import velver
from momentum_statistics import momentum_check
from benettin_gloabl import benetin


# ============================================================
# Simulation parameters
# ============================================================

N = 60
temperature = 10
dt = 0.01
total_time = 25
gamma = 0.5

epsilon = 0.0001
tau = 1.5
M = 1000


# ============================================================
# Create FPU chain
# ============================================================

members, neighbours = create_chain(N)

system_langevin = springsystem(members, neighbours, 0, 0.7)


# ============================================================
# Thermalize using Langevin dynamics
# ============================================================

system_langevined, kinetic, potential, total = Langevin(
    system_langevin, total_time, dt, gamma, temperature
)


# ============================================================
# Benettin calculation
# ============================================================

lyapunov, delt, kin, po, tot = benetin(
    system_langevined, M, dt, epsilon, tau, temperature
)
print("Lyapunov:")
print(type(lyapunov))
print("Length:", len(lyapunov))
print("First 10:", lyapunov[:10])

print("\nDelta:")
print(type(delt))
print("Length:", len(delt))
print("First 10:", delt[:10])

print("\nEnergy:")
print("tot first 10:", tot[:10])

# ============================================================
# Construct physical time axes
# ============================================================

time_lyap = np.arange(1, len(lyapunov) + 1) * tau

# If delt and energy are recorded at every Benettin step:
time_benettin = np.arange(1, len(delt) + 1) * tau


# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)


# ------------------------------------------------------------
# 1. Lyapunov exponent convergence
# ------------------------------------------------------------

axes[0].plot(time_lyap, lyapunov, linewidth=2)

axes[0].set_xlabel("Time")
axes[0].set_ylabel(r"Lyapunov exponent $\lambda$")
axes[0].set_title("Convergence of the Global Lyapunov Exponent")

axes[0].grid(True, alpha=0.3)


# ------------------------------------------------------------
# 2. Perturbation distance
# ------------------------------------------------------------

axes[1].plot(time_benettin, delt, linewidth=1.5)

axes[1].set_xlabel("Time")
axes[1].set_ylabel(r"Perturbation distance $\delta$")
axes[1].set_title("Perturbation Distance During Benettin Evolution")

axes[1].grid(True, alpha=0.3)


# ------------------------------------------------------------
# 3. Energy conservation / drift
# ------------------------------------------------------------

axes[2].plot(time_benettin, kin, label="Kinetic energy", linewidth=1.5)

axes[2].plot(time_benettin, po, label="Potential energy", linewidth=1.5)

axes[2].plot(time_benettin, tot, label="Total energy", linewidth=2)

axes[2].set_xlabel("Time")
axes[2].set_ylabel("Energy")
axes[2].set_title("Energy Evolution of the Perturbed Trajectory")

axes[2].legend()
axes[2].grid(True, alpha=0.3)


# ============================================================
# Overall figure
# ============================================================

fig.suptitle("Benettin Algorithm Diagnostics", fontsize=16, fontweight="bold")

plt.show()
