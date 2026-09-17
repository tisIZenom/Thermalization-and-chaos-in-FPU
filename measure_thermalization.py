## this is where we do 4 things

# measure the auto correlation for momentum
# The energy
# the mode energy variance
# KS statistic
# from the system
#
#

from scipy.stats import kstest
import numpy as np


def momentum_check(system, Temperature):

    mean = np.mean(system.momentum)
    variance = np.var(system.momentum)

    sigma = np.sqrt(Temperature)

    statistic, p_value = kstest(system.momentum, "norm", args=(0, sigma))

    return statistic, p_value, mean, variance


def Normal_modes(system):
    N = len(system.members)
    q = system.displacement[1:N]
    p = system.momentum[1:N]

    # normal mode indices:
    #
    k = np.arange(1, N)

    # This is for the sine transformation matrix
    j = np.arange(1, N)

    mode_matrix = np.sin(np.pi * np.outer(k, j) / (N + 1))

    Q = np.sqrt(2 / (N + 1)) * mode_matrix @ q
    P = np.sqrt(2 / (N + 1)) * mode_matrix @ p

    # now harmonic normal mode frequencies
    omega = 2 * np.sin(np.pi * k / (2 * (N + 1)))

    # now the enregy per mode
    E_mode = 0.5 * (P**2) + 0.5 * (omega**2) * (Q**2)

    E_mode = np.maximum(E_mode, 0)

    return E_mode


def spectral_entropy(E_mode):
    total_energy = np.sum(E_mode)
    N = len(E_mode)

    if total_energy == 0:
        return 0.0, 0.0

    probabilities = E_mode / total_energy

    # spectral_entropy
    Spec = -np.sum(probabilities * np.log(probabilities + 1e-16))

    Normalized = Spec / np.log(N)

    return Spec, Normalized


def Thermalized_or_not(system, Temperature):
    Thermalized = False
    N = len(system.members)
    q = system.displacement[1 : N - 1]
    p = system.momentum[1 : N - 1]

    # this is to track the momentum_average and the distance
    momentum_average = np.mean(p**2)
    momentum_variance = np.var(p**2)

    Temperature_distance = momentum_average - Temperature

    # now finding the energy average
    E_mode = Normal_modes(system)

    Spec, normalized = spectral_entropy(E_mode)

    # Momentum gaussian
    statisitc, p_value, mean, variance = momentum_check(system, Temperature)

    if (
        (normalized >= 0.8)
        and (Temperature_distance <= 0.5)
        and (statisitc <= 0.1)
        and (p_value > 0.2)
    ):
        Thermalized = True

    else:
        Thermalized = False
        print("Normalized spectral_entropy = ", normalized)
        print("Temperature_distance = ", Temperature_distance)
        print("gaussian deviattion, Statistic, p_value", statisitc, p_value)

    return normalized, Temperature_distance, statisitc, p_value, Thermalized
