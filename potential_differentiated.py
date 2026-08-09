## using the equation that I have differentiated we getL

def potential_dif(system, i ):
    beta = 100 
    distance = system.x[i+1] - system.x[i]

    potential_difference = ( distance ) + *  beta ( distance ** 3)
    
    potential_difference = - potential_difference 

    return potential_difference 
