## This is a complicated function for the potential at any given point. I think it needs two differnet values

import numpy as np


def potential(system, i):
    beta = 100 / 4
    # Needs to return a value for the langevin and the hamiltonian
    distancer = system.x[i] - system.x[i - 1]

    potenialatpoint = (distancer**2) + beta(distancer**4)
