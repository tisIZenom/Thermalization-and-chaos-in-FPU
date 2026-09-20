import numpy as np
from collections import deque


class OnlineAutocorrelation:
    def __init__(self, window_size, N):
        """
        Parameters
        ----------
        window_size : int
            Number of time points kept in the rolling window.

        N : int
            Number of particles in the FPU system.
        """

        self.window_size = window_size
        self.N = N

        # Store recent momenta
        self.momentum_history = deque(maxlen=window_size)

        # Store recent kinetic and potential energies
        self.kinetic_history = deque(maxlen=window_size)
        self.mode_history = deque(maxlen=window_size)

    def update(self, momenta, kinetic_energy, mode_energy):
        """
        Add the current state of the FPU system.

        Parameters
        ----------
        momenta : ndarray
            Momentum array of shape (N,).

        kinetic_energy : float
            Total kinetic energy at the current time.

        potential_energy : float
            Total potential energy at the current time.
        """

        self.momentum_history.append(np.copy(momenta))

        self.kinetic_history.append(kinetic_energy)
        self.mode_history.append(mode_energy)

    def autocorrelation(self, x):
        """
        Calculate normalized autocorrelation of a 1D time series.
        """

        x = np.asarray(x)

        if len(x) < 2:
            return None

        # Remove mean
        x = x - np.mean(x)

        # Variance
        variance = np.mean(x**2)

        if variance == 0:
            return None

        n = len(x)

        C = np.zeros(n)

        for lag in range(n):
            C[lag] = np.mean(x[: n - lag] * x[lag:]) / variance

        return C

    def momentum_correlation(self):
        """
        Calculate momentum autocorrelation averaged over all particles.
        """

        if len(self.momentum_history) < 2:
            return None

        data = np.asarray(self.momentum_history)

        # data shape = (time, particles)

        C_total = np.zeros(len(data))

        for i in range(self.N):
            p = data[:, i]

            C = self.autocorrelation(p)

            if C is not None:
                C_total += C

        C_total /= self.N

        return C_total

    def kinetic_energy_correlation(self):
        """
        Calculate autocorrelation of kinetic energy.
        """

        if len(self.kinetic_history) < 2:
            return None

        return self.autocorrelation(self.kinetic_history)

    def mode_energy_correlation(self):
        if len(self.momentum_history) < 2:
            return None

        data = np.asarray(self.momentum_history)

        # data shape = (time, particles)

        C_total = np.zeros(len(data))

        for i in range(self.N):
            p = data[:, i]

            C = self.autocorrelation(p)

            if C is not None:
                C_total += C

        C_total /= self.N

        return C_total

    def momentum_correlation_at_lag(self, lag):

        C = self.momentum_correlation()

        if C is None:
            return None

        if lag >= len(C):
            return None

        return C[lag]

    def kinetic_correlation_at_lag(self, lag):

        C = self.kinetic_energy_correlation()

        if C is None:
            return None

        if lag >= len(C):
            return None

        return C[lag]

    def mode_correlation_at_lag(self, lag):

        C = self.mode_energy_correlation()

        if C is None:
            return None

        if lag >= len(C):
            return None

        return C[lag]
