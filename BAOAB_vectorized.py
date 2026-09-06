## This is the baoab thing but vectorized so that things are faster and maybe I dont make another stupid mistake.

import numpy as np
from potential_differentiated import potential_dif
from energy_measurement import energy


def BAOAB(system, temperature, total_time, dt, gamma):

    time_steps = int(total_time / dt)

    kinetic_average = np.zeros(time_steps)
    potential_average = np.zeros(time_steps)
    total_energy = np.zeros(time_steps)

    N = len(system.members)

    # OU constants
    constant = np.exp(-gamma * dt)
    sigma = np.sqrt(temperature * (1 - constant**2))

    for step in range(time_steps):
        # Measure energy

        kinetic_average[step], potential_average[step], total_energy[step] = energy(
            system
        )

        # try to see where the blow up is happening
        if not np.all(np.isfinite(system.displacement)):
            print("q became non-finite at step", step)
            break

        if not np.all(np.isfinite(system.momentum)):
            print("p became non-finite at step", step)
            break

        # B: half kick

        force = potential_dif(system)

        system.momentum[1:-1] += 0.5 * dt * force[1:-1]

        # A: half drift

        system.displacement[1:-1] += 0.5 * dt * system.momentum[1:-1]

        # O: Ornstein-Uhlenbeck thermostat
        noise = np.random.normal(0.0, 1.0, size=N - 2)

        system.momentum[1:-1] = constant * system.momentum[1:-1] + (sigma * noise)

        # A: half drift
        system.displacement[1:-1] += 0.5 * dt * system.momentum[1:-1]

        # B: half kick
        updated_force = potential_dif(system)
        system.momentum[1:-1] += 0.5 * dt * updated_force[1:-1]

        # Fixed boundaries

        system.displacement[0] = 0.0
        system.displacement[-1] = 0.0

        system.momentum[0] = 0.0
        system.momentum[-1] = 0.0

    return system, kinetic_average, potential_average, total_energy
