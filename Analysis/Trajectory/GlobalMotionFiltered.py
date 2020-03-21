"""
This module provides classes to correct global motion and export new trajectories.

.. inheritance-diagram:: pdbparser.Analysis.Trajectory.GlobalMotionFiltered
    :parts: 2
"""
# standard libraries imports
from __future__ import print_function

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.log import Logger
from pdbparser.Analysis.Core import Analysis
from pdbparser.Utilities.Math import get_superposition_transformation_elements
from pdbparser.Utilities.Information import get_records_database_property_values
from pdbparser.Utilities.BoundaryConditions import InfiniteBoundaries



class GlobalMotionFiltered(Analysis):
    """
    Computes the global motion filtered trajectory.
    """
    def __init__(self, trajectory, configurationsIndexes,
                       globalMotionAtomsIndexes, targetAtomsIndexes,
                       *args, **kwargs):
        # set trajectory
        super(GlobalMotionFiltered,self).__init__(trajectory, *args, **kwargs)
        # set steps indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        self.numberOfSteps = len(self.configurationsIndexes)
        # set atoms selection
        self.globalMotionAtomsIndexes = self.get_atoms_indexes(globalMotionAtomsIndexes)
        self.targetAtomsIndexes = self.get_atoms_indexes(globalMotionAtomsIndexes)
        # initialize variables
        self.__initialize_variables__()
        # initialize results
        self.__initialize_results__()

    def __initialize_variables__(self):
        # calculate weights
        self.weights = np.array(get_records_database_property_values(self.globalMotionAtomsIndexes, self.structure, "atomicWeight"))
        self.totalWeight = np.sum(self.weights)

    def __initialize_results__(self):
        pass

    def step(self, index):
        """"
        analysis step of calculation method.\n

        :Parameters:
            #. index (int): the step index

        :Returns:
            #. stepData (object): object used in combine method
        """
        # skip first configuration
        if index == 0:
            return index, None
        # get configuration index
        confIdx = self.configurationsIndexes[index]
        refConfIdx = self.configurationsIndexes[index-1]
        # get coordinates
        coordinates = self._trajectory.get_configuration_coordinates(confIdx)
        coords = coordinates[self.globalMotionAtomsIndexes,:]
        refCoords = self._trajectory.get_configuration_coordinates(refConfIdx)[self.globalMotionAtomsIndexes,:]
        # get superposition_fit
        rotationMatrix, refCoordsCOM, coordsCOM, rms = get_superposition_transformation_elements(self.weights, self.totalWeight, refCoords, coords)
        # translate target coordinates to origin
        coordinates[self.targetAtomsIndexes] -= coordsCOM
        # rotate target records
        coordinates[self.targetAtomsIndexes] = np.dot( rotationMatrix,np.transpose(coordinates[self.targetAtomsIndexes]).\
                                                       reshape(1,3,-1)).\
                                                       transpose().\
                                                       reshape(-1,3)
        # translate coords to reference center of mass
        coordinates[self.targetAtomsIndexes] += refCoordsCOM
        # set new coordinates
        self._trajectory.set_configuration_coordinates(confIdx, coordinates)
        return index, None


    def combine(self, index, stepData):
        pass


    def finalize(self):
        # remove all unwanted configurations
        indexesToRemove = list(set(self._trajectory.indexes)-set(self.configurationsIndexes))
        if len(indexesToRemove) > 1:
            subs = 1+np.arange(len(indexesToRemove)-1)
            indexesToRemove = np.array(indexesToRemove)
            indexesToRemove[1:] -= subs
        for idx in list(indexesToRemove):
            self._trajectory.remove_configuration(idx)
        # remove all unwanted atoms
        atomsIndexes = list(set(self._trajectory.atomsIndexes)-set(self.targetAtomsIndexes))
        self._trajectory.remove_atoms(atomsIndexes)
        # transform boundary condition to infinite boundaries
        bc = InfiniteBoundaries()
        for idx in self._trajectory.indexes:
            bc.set_vectors()
        self._trajectory._boundaryConditions = bc
