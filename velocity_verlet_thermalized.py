# This is the main integrator of the system.
# Make sure that the system remains thermal after disconnection from the bath
#
import numpy as np
from energy_measurement import energy
from potential_differentiated import potential_dif
from measure_thermalization import Thermalized_or_not
from measure_thermalization import Normal_modes
from Autocorrelator_fixed import OnlineAutocorrelation


def velver(system, total_time, dt, temperature):
    time_steps = int(total_time / dt)
    kinetic_average = np.zeros(time_steps)
    potential_average = np.zeros(time_steps)
    total_energy = np.zeros(time_steps)
    N = len(system.members)

    Corr_mode = np.zeros(time_steps)
    Corr_kin = np.zeros(time_steps)
    Corr_mom = np.zeros(time_steps)

    normalized_spectral = np.zeros(time_steps)
    temperaturediff = np.zeros(time_steps)
    statistic = np.zeros(time_steps)
    pvalue = np.zeros(time_steps)

    AC = OnlineAutocorrelation(1000, N)

    # time loop

    for step in range(time_steps):
        kinetic_average[step], potential_average[step], total_energy[step] = energy(
            system
        )

        ## This is for the AC
        kinetic_energy_AC = 0.5 * np.sum(system.momentum**2)
        Mode_energy = Normal_modes(system)

        AC.update(system.momentum, kinetic_energy_AC, Mode_energy)

        if step % 100 == 0:
            print("time: ", step)
        Corr_mom[step] = AC.momentum_correlation_at_lag(100)

        Corr_kin[step] = AC.kinetic_correlation_at_lag(100)
        Corr_mode[step] = AC.mode_correlation_at_lag(100)

        # Check if thermalized
        (
            normalized_spectral[step],
            temperaturediff[step],
            statistic[step],
            pvalue[step],
            thermalized,
        ) = Thermalized_or_not(system, temperature)

        if not thermalized:
            print("System has not thermalized at time: ", step)

        # compute force for the updtae
        force = potential_dif(system)

        # Step 1 momentum update
        for i in range(1, N - 1):
            system.momentum[i] = system.momentum[i] + (dt * 0.5 * force[i])

        # step 2 displacement updtae
        for i in range(1, N - 1):
            system.displacement[i] = system.displacement[i] + (dt * system.momentum[i])

        # step 3 updtae foces
        # update forces for the new positions
        updated_force = potential_dif(system)

        # final momentum update
        for i in range(1, N - 1):
            system.momentum[i] = system.momentum[i] + (dt * 0.5 * updated_force[i])

    return (
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
    )
