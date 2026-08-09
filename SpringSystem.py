## This I would like to be similar to the frustrated spin system tht I havefr

from CreateChain import create_chain
import numpy as np

# create a chain here and have it function as the main idea
#


class springsystem:
    def __init__(self, members, neighbours, mass, alpha, beta) -> None:
        self.members = int(members)
        self.neighbours = neighbours
        self.mass = mass
        self.alpha = alpha
        self.beta = beta

        # Calculate the momentum from the displacements
        #
        self.x = np.zeros(self.members)

        # intializing random positions of the system.

        self.x = np.random.randn(self.members, 1)

        # trying to force hard boundary conditions.

        for i in range(self.members):
            self.x[i] = np.random.random()

        self.x[0] = 0
        self.x[self.members - 1] = 0

        # This data type should have the momentum also attached to it. just in case. Keep them zero?
        #

        pass


positions, neighbours = create_chain(20)

system = springsystem(len(positions), neighbours, 1, 0.1, 0.1)
print(system.x)
