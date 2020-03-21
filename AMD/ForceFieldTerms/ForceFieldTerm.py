"""
"""
import numpy as np
from Utilities import Array

class ForceFieldTerm(object):
    """
    mother class of all forcefield calculation terms
    """
    def __init__(self, basename, parent):
        """
        generates all calculation functions dictionary and general variables
        """
        # parent is the the simulation. so a forcefield term can access all its attributes
        self.parent = parent
        
        # generale calculation possible methods
        self.methodNames = [methodName for methodName in dir(self) if basename in methodName]
        self.methods = {}
        for name in self.methodNames:
            self.methods[name] = self.__getattribute__(name)
        
        # forcefield method of calculation
        self.__calculation_method__ = None   
        
        # general variables
        self.energy = None
        self.force = None
        
        # export variable, list of variable that user wishes to export.
        # By default it's only the energy
        self.export = ["energy"]
        
        # indexes to mask during calculation
        # mask is a list of lists, its length is the total number of atoms
        self.mask = None
            
    
    def __call__(self, *args):
        self.__calculation_method__(*args)
               
    
    def initialize(self):
        """
        initialize needed variables
        """
        self.energy = Array( np.zeros((self.parent.numberOfAtoms)) )
        self.force = Array( np.zeros((self.parent.numberOfAtoms, 3)) )
        print "%s forcefield term is ready"%self.__class__.__name__
        
    
    def export(self):
        """
        method called to write results
        """
        pass
        
        
        