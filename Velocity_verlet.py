# This is the main integrator of the system.
# Make sure that the system remains thermal after disconnection from the bath
#
import numpy as np
from energy_measurement import energy
from potential_differentiated import potential_dif


def velver(system, total_time, dt):
    time_steps = int(total_time / dt)
    kinetic_average = np.zeros(time_steps)
    potential_average = np.zeros(time_steps)
    total_energy = np.zeros(time_steps)
    N = len(system.members)

    # time loop

    for step in range(time_steps):
        kinetic_average[step], potential_average[step], total_energy[step] = energy(
            system
        )

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

    return system, kinetic_average, potential_average, total_energy
