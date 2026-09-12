import numpy as np
from Velocity_verlet import velver
import copy
from SpringSystem import springsystem

class decorrelator:
    def __init__(self, dt, positional) -> None:
        self.dt = dt 
        self.positional = positional 
        pass


def decorrelator_measure(system, target, time, dt, epsilon): 

    original = copy.deepcopy(system)
    perturbed = copy.deepcopy(system)

    perturbation_kick(target, epsilon)

    time_steps = int( time / dt )

    Decor_matrix = np.zeros(time_steps, len(system.members))

    for i in range(time_steps): 
        original, kino, poto, toto = velver(original, dt, dt)
        perturbed, kinp, potp, totp = velver(perturbed, dt, dt )

        # returns a matrix tho 

        for k in range(len(system.members)): 
            

