## this is to check whether the thermalization works well or not.

from CreateChain import create_chain
from SpringSystem import springsystem
from BAOAB_vectorized import BAOAB

import numpy as np
import matplotlib.pyplot as plt

members, neighbours = create_chain(108)

system = springsystem(members, neighbours, 0, 1)

system, kinetic, potential, total = BAOAB(system, 10, 10, 0.0001, 0.5)

print("temperature of the system is 10")


print(np.mean(kinetic))
plt.plot(kinetic)
plt.show()
