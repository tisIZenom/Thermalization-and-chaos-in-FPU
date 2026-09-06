import numpy as np
from potential_differentiated import potential_dif


def BAOAB_test(system, temperature, dt, gamma):

    N = len(system.members)

    constant = np.exp(-gamma * dt)
    sigma = np.sqrt(temperature * (1 - constant**2))

    print("\nINITIAL")
    print("q =", system.displacement)
    print("p =", system.momentum)

    # ============================================
    # B
    # ============================================

    force = potential_dif(system)

    print("\nFORCE")
    print(force)
    print("max force =", np.max(np.abs(force)))

    system.momentum[1:-1] += 0.5 * dt * force[1:-1]

    print("\nAFTER B")
    print("max q =", np.max(np.abs(system.displacement)))
    print("max p =", np.max(np.abs(system.momentum)))

    if not np.all(np.isfinite(system.displacement)):
        print("q became non-finite at step", step)

    if not np.all(np.isfinite(system.momentum)):
        print("p became non-finite at step", step)

    # ============================================
    # A
    # ============================================

    system.displacement[1:-1] += 0.5 * dt * system.momentum[1:-1]

    print("\nAFTER A")
    print("q =", system.displacement)
    print("max q =", np.max(np.abs(system.displacement)))
    print("max p =", np.max(np.abs(system.momentum)))

    # ============================================
    # O
    # ============================================

    noise = np.random.normal(size=N - 2)

    system.momentum[1:-1] = constant * system.momentum[1:-1] + sigma * noise

    print("\nAFTER O")
    print("max q =", np.max(np.abs(system.displacement)))
    print("max p =", np.max(np.abs(system.momentum)))

    # ============================================
    # A
    # ============================================

    system.displacement[1:-1] += 0.5 * dt * system.momentum[1:-1]

    print("\nAFTER SECOND A")
    print("q =", system.displacement)
    print("max q =", np.max(np.abs(system.displacement)))
    print("max p =", np.max(np.abs(system.momentum)))

    # ============================================
    # B
    # ============================================

    updated_force = potential_dif(system)

    print("\nUPDATED FORCE")
    print(updated_force)
    print("max force =", np.max(np.abs(updated_force)))

    system.momentum[1:-1] += 0.5 * dt * updated_force[1:-1]

    print("\nFINAL")
    print("q =", system.displacement)
    print("p =", system.momentum)

    return system
