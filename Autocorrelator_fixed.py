import numpy as np
from collections import deque


class OnlineAutocorrelation:
    def __init__(self, window_size, N):
        self.window_size = window_size
        self.N = N

        self.momentum_history = deque(maxlen=window_size)
        self.kinetic_history = deque(maxlen=window_size)
        self.mode_history = deque(maxlen=window_size)

    def update(self, momenta, kinetic_energy, mode_energy):
        self.momentum_history.append(np.copy(momenta))
        self.kinetic_history.append(
            np.copy(kinetic_energy) if np.ndim(kinetic_energy) > 0 else kinetic_energy
        )
        self.mode_history.append(
            np.copy(mode_energy) if np.ndim(mode_energy) > 0 else mode_energy
        )

    def autocorrelation(self, x):
        x = np.asarray(x, dtype=float)

        if len(x) < 2:
            return None

        # Subtract mean along the time axis
        x = x - np.mean(x)

        # Denominator using np.sum to guarantee C(lag) <= 1
        denom = np.sum(x**2)

        if denom <= 0:
            return None

        n = len(x)
        C = np.empty(n)

        for lag in range(n):
            C[lag] = np.sum(x[: n - lag] * x[lag:]) / denom

        return C

    def momentum_correlation(self):
        if len(self.momentum_history) < 2:
            return None

        data = np.asarray(self.momentum_history)
        C_total = np.zeros(len(data))

        for i in range(data.shape[1]):
            C = self.autocorrelation(data[:, i])
            if C is not None:
                C_total += C

        C_total /= data.shape[1]
        return C_total

    def kinetic_energy_correlation(self):
        if len(self.kinetic_history) < 2:
            return None

        data = np.asarray(self.kinetic_history)

        # Handle scalar per time-step vs per-particle array
        if data.ndim == 1:
            return self.autocorrelation(data)

        C_total = np.zeros(len(data))
        for i in range(data.shape[1]):
            C = self.autocorrelation(data[:, i])
            if C is not None:
                C_total += C

        C_total /= data.shape[1]
        return C_total

    def mode_energy_correlation(self):
        if len(self.mode_history) < 2:
            return None

        data = np.asarray(self.mode_history)

        if data.ndim == 1:
            return self.autocorrelation(data)

        C_total = np.zeros(len(data))
        for mode in range(data.shape[1]):
            C = self.autocorrelation(data[:, mode])
            if C is not None:
                C_total += C

        C_total /= data.shape[1]
        return C_total

    def momentum_correlation_at_lag(self, lag):
        C = self.momentum_correlation()
        return None if C is None or lag >= len(C) else C[lag]

    def kinetic_correlation_at_lag(self, lag):
        C = self.kinetic_energy_correlation()
        return None if C is None or lag >= len(C) else C[lag]

    def mode_correlation_at_lag(self, lag):
        C = self.mode_energy_correlation()
        return None if C is None or lag >= len(C) else C[lag]
