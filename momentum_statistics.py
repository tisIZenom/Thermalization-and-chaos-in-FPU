# Kurtosis measures how close to a gasussian
# This is to see how the momentum of the system evolves. we need it to be a gaussian
#
#

from scipy.stats import kstest
import numpy as np


def momentum_check(system, Temperature):
    mean_tol = 0.1
    variance_tol = 0.1
    kurtosis_tol = 0.3

    mean = np.mean(system.momentum)
    variance = np.var(system.momentum)

    sigma = np.sqrt(Temperature)

    statistic, p_value = kstest(system.momentum, "norm", args=(0, sigma))

    return statistic, p_value, mean, variance
