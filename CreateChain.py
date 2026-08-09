# This is to initialize the 1D chain.
#
# Needs to give out positions and neighbours
#

import numpy as np


def create_chain(N):
    positions = []
    neighbours = []

    for i in range(N):
        positions.append(i)

        left = (i - 1) % N
        right = (i + 1) % N

        neighbours.append([left, right])

    return positions, neighbours


positions, neighbours = create_chain(10)
