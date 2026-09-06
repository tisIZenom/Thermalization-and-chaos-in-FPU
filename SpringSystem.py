## This I would like to be similar to the frustrated spin system tht I havefr

from CreateChain import create_chain
import numpy as np

# create a chain here and have it function as the main idea
# Mass is constant and is set to 1 for all the members so that the transformation matrix is easier to make


class springsystem:
    def __init__(self, members, neighbours, alpha, beta) -> None:
        self.members = members
        self.neighbours = neighbours
        self.alpha = alpha
        self.beta = beta

        # make the postions random in the start then let the langevin dynamics or the motecarlo fix them
        # The momentum will also be fixed by the thermalizer
        self.displacement = np.zeros(len(self.members))

        # trying to force hard boundary conditions.

        self.displacement[0] = 0
        self.displacement[len(self.members) - 1] = 0

        self.momentum = np.random.randn(len(self.members))

        # Need a method to adjust the postions of the system by a small delta

    def trial_move(self, i, epsilon):

        delta = np.random.uniform(-epsilon, epsilon)

        old_config = self.displacement[i].copy()

        new_config = old_config + delta

        self.displacement[i] = new_config

        return old_config

    pass
