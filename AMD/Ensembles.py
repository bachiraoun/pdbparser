import numpy as np

def innerVirial(distances, forces):
    """
    calculates the inner virial for clausius virial theorem sake.
    P = 2/(3*V)*(Ekin - E) where E is the inner virial
    E = 1/2*SUM(rij*f(rij)) the sum is for (i<j) 
    f(rij) is the force between particles i and j at a distance rij
    """
    return 0.5*np.sum( np.dot(distances, forces) )