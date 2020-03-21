#!/Library/Frameworks/Python.framework/Versions/7.0/Resources/Python.app/Contents/MacOS/Python
import numpy as np
import ForceFieldTerm

__basename__ = "LennardJones_"
class LennardJones(ForceFieldTerm.ForceFieldTerm):
    """
    Class containing all possible LennardJones calculation function
    to implement a new one, make sure the function name begins with LennardJones_
    """
    
    def __init__(self, parent):
        """
        """
        ForceFieldTerm.ForceFieldTerm.__init__(self, basename = __basename__, parent = parent)
    
    
    def initialize(self):
        """
        initialize needed variables
        """
        ForceFieldTerm.ForceFieldTerm.initialize(self)
        
        # create method specified variables
        if self.__calculation_method__.__name__ == "LennardJones_12_6":
            self.eps = None
            self.sigma = None
        elif self.__calculation_method__.__name__ == "LennardJones_12_6_CHARMM":
            self.eps = None
            self.rmin = None

            
    def LennardJones_12_6(self, distances):
        """
        Lennard-Jones potential power 12 - power 6 format
        V(Lennard-Jones) = 4*Eps,i,j[(sigma,i,j/ri,j)**12 - (sigma,i,j/ri,j)**6]
        epsilon: kcal/mole, Eps,i,j = sqrt(eps,i * eps,j)
        sigma/2: A, sigma,i,j = sigma/2,i + sigma/2,j
        """
        # calculate sigma/distances
        ratio = np.divide(self.sigma, distances)
        
        # calculate potential 
        potentialEnergy = np.dot(4*self.eps, ratio**12 - ratio**6)
    
        # calculate forces
        forces = np.dot( 24.0*self.eps/self.sigma, 2*ratio**13 - ratio**7) 
    
        return potentialEnergy, forces
    
   
    def LennardJones_12_6_CHARMM(self, distances):
        """
        Lennard-Jones potential power 12 - power 6 format
        V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]
        epsilon: kcal/mole, Eps,i,j = sqrt(eps,i * eps,j)
        Rmin/2: A, Rmin,i,j = Rmin/2,i + Rmin/2,j
        """
        # calculate Rmin/distances
        ratio = np.divide(self.rmin, distances)
        
        # calculate potential 
        potentialEnergy = np.dot(self.eps, ratio**12 - 2*ratio**6)
    
        # calculate forces
        forces = np.dot( 12.0*self.eps/self.rmin, ratio**13 - ratio**7 ) 
    
        return potentialEnergy, forces
    
    
    

if __name__ == "__main__":
    LJ = LennardJones()
    LJ.initialize(None)
    LJ()

       