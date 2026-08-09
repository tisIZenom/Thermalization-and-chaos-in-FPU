## This is for how the system evolves.

import numpy as np
from potential_differentiated import potential_dif


def langevindynamics(system, temp, dt, gamma):

    pn1 = (pn) + dt * ( - potential_dif(system, i ) - gamma * pn + )
