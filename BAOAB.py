## This is supposed to be a better version of the langevin dynamics which has a
# step by step integrator.
#
from statistics import variance

import numpy as np
from potential_differentiated import potential_dif
from energy_measurement import energy


def BAOAB(system, temperature, total_time, dt, gamma):
    time_steps = int(total_time / dt)
    kinetic_average = np.zeros(time_steps)
    potential_average = np.zeros(time_steps)
    total_energy = np.zeros(time_steps)
    N = len(system.members)

    # the loop for time
    for step in range(time_steps):
        kinetic_average[step], potential_average[step], total_energy[step] = energy(
            system
        )

        # Just measuring some diagnosticsL
        if step % 1000 == 0:
            print(
                "step =",
                step,
                "max q =",
                np.max(np.abs(system.displacement)),
                "max p =",
                np.max(np.abs(system.momentum)),
                "energy =",
                total_energy[step],
            )

        # compute the noise for each timestep

        constant = np.exp(-gamma * dt)
        sigma = np.sqrt(temperature * (1 - constant**2))

        force = potential_dif(system)

        # now comes the internal loop for the members:
        for i in range(1, N - 1):
            # first compute the force in the system

            # Step 1 B
            system.momentum[i] = system.momentum[i] + (dt * 0.5 * force[i])

        for i in range(1, N - 1):
            # step 2 A
            system.displacement[i] = system.displacement[i] + (
                dt * 0.5 * system.momentum[i]
            )

        for i in range(1, N - 1):
            # step 3 O the noise
            system.momentum[i] = constant * system.momentum[i] + (
                sigma * np.random.normal(loc=0, scale=1.0)
            )

        for i in range(1, N - 1):
            # step 4 A
            system.displacement[i] = system.displacement[i] + (
                0.5 * dt * system.momentum[i]
            )

        # update the force here and then use it to all the membersL
        updated_force = potential_dif(system)

        for i in range(1, N - 1):
            # Step 5 finall B
            # need to calculate the potential of the system and the force once again here
            system.momentum[i] = system.momentum[i] + (dt * 0.5 * updated_force[i])

        system.displacement[0] = 0.0
        system.displacement[-1] = 0.0

        system.momentum[0] = 0.0
        system.momentum[-1] = 0.0
        # this completes the baoab

    return system, kinetic_average, potential_average, total_energy
