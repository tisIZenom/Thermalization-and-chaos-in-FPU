# need to loop over all the points and make sure that the system can really thermalize
# Using the euler method.

from measure_thermalization import Thermalized_or_not
from potential_differentiated import potential_dif
import numpy as np
from energy_measurement import energy
from momentum_statistics import momentum_check
from measure_thermalization import Normal_modes
from Autocorrelator import OnlineAutocorrelation


def Langevin(system, total_time, dt, gamma, Temperature):

    time_step = int(total_time / dt)
    N = len(system.members)

    # plotting Autocorrelator
    Corr_mode = np.zeros(int(time_step))
    Corr_kin = np.zeros(int(time_step))
    Corr_mom = np.zeros(int(time_step))

    # Plotting thermlization
    normalized_Spectral = np.zeros(time_step)
    Temperature_distance = np.zeros(time_step)
    statistic = np.zeros(time_step)
    p_value = np.zeros(time_step)

    # Expectation value of <pi sr> = T
    total_energy = np.zeros(time_step)
    kinetic_energy = np.zeros(time_step)
    potential_energy = np.zeros(time_step)

    # initialize Autocorrelator
    AC = OnlineAutocorrelation(1000, N)

    for i in range(time_step):
        # Calculate the kinetic_energy of the system Autocorrelator stuff
        #
        kinetic_energy_AC = 0.5 * np.sum(system.momentum**2)

        Mode_energy = Normal_modes(system)

        AC.update(system.momentum, kinetic_energy_AC, Mode_energy)

        if i % 10 == 0:
            Corr_mom[i] = AC.momentum_correlation_at_lag(100)

            Corr_kin[i] = AC.kinetic_correlation_at_lag(100)
            Corr_mode[i] = AC.mode_correlation_at_lag(100)

        # Check if thermalized
        (
            normalized_Spectral[i],
            Temperature_distance[i],
            statistic[i],
            p_value[i],
            thermalized,
        ) = Thermalized_or_not(system, Temperature)

        if thermalized:
            print("System has thermalized at time: ", i)

        ## Ends here Langevin starts from here

        force = potential_dif(system)

        noise = np.sqrt(2 * gamma * Temperature * dt) * np.random.randn(N)

        # just as a test I will map how the kinetic_energy of a particle goes
        kinetic_energy[i], potential_energy[i], total_energy[i] = energy(system)

        # Technically python supports vector addition so I dont need this loop but I want to visualize the
        # dynamics
        for j in range(N):
            # momentum update
            system.momentum[j] = (
                system.momentum[j]
                + (force[j] * dt)
                - (gamma * system.momentum[j] * dt)
                + noise[j]
            )

            # position update
            system.displacement[j] += system.momentum[j] * dt

        # make sure that the end points dont update:
        system.displacement[0] = 0
        system.momentum[0] = 0
        system.displacement[-1] = 0
        system.momentum[-1] = 0

        # Check the momentum distribution and report whether the system remains gaussian.

    return (
        system,
        kinetic_energy,
        potential_energy,
        total_energy,
        normalized_Spectral,
        Temperature_distance,
        statistic,
        p_value,
        Corr_mom,
        Corr_kin,
        Corr_mode,
    )
