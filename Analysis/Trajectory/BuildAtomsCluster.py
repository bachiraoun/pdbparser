"""
This module provides trajectory rebuilding atoms clusters analysis

.. inheritance-diagram:: pdbparser.Analysis.Trajectory.BuildAtomsCluster
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
from pdbparser.Utilities.BoundaryConditions import PeriodicBoundaries



class BuildAtomsCluster(Analysis):
    """
    Build atoms cluster trajectory.
    """
    def __init__(self, trajectory, configurationsIndexes,
                       clusterIndexes,
                       clusterToBoxCenter=True, fold=True,
                       *args, **kwargs):
        # set trajectory
        super(BuildAtomsCluster,self).__init__(trajectory, *args, **kwargs)
        # set steps indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        self.numberOfSteps = len(self.configurationsIndexes)
        # set atoms selection
        self.clusterIndexes = self.get_atoms_indexes(clusterIndexes)
        # initialize variables
        self.__initialize_variables__(clusterToBoxCenter, fold)

    def __initialize_variables__(self, clusterToBoxCenter, fold):
        # referenceIndex
        self.restOfAtomsIndexes = list(set(self._trajectory.atomsIndexes)-set(self.clusterIndexes))
        # translateToCenter
        assert isinstance(clusterToBoxCenter, bool), Logger.error("clusterToBoxCenter must be boolean")
        self.clusterToBoxCenter = clusterToBoxCenter
        # fold
        assert isinstance(fold, bool), Logger.error("fold must be boolean")
        self.fold = fold


    def step(self, index):
        """"
        analysis step of calculation method.\n

        :Parameters:
            #. index (int): the step index

        :Returns:
            #. stepData (object): object used in combine method
        """
        if not isinstance(self._trajectory._boundaryConditions, PeriodicBoundaries):
            raise Logger.error("rebuild cluster is not possible with infinite boundaries trajectory")

        # get configuration index
        confIdx = self.configurationsIndexes[index]
        # get coordinates
        boxCoords = self._trajectory.get_configuration_coordinates(confIdx)
        boxCoords = self._trajectory._boundaryConditions.real_to_box_array(realArray=boxCoords, index=confIdx)
        # get box coordinates
        clusterBoxCoords = boxCoords[self.clusterIndexes,:]
        # initialize variables
        incrementalCenter = np.array([0.,0.,0.])
        centerNumberOfAtoms = 0.0
        # incrementally construct cluster
        for idx in range(clusterBoxCoords.shape[0]):
            if idx > 0:
                diff = clusterBoxCoords[idx,:]-(incrementalCenter/centerNumberOfAtoms)
                # remove multiple box distances
                intDiff = diff.astype(int)
                clusterBoxCoords[idx,:] -= intDiff
                diff -= intDiff
                # remove half box distances
                clusterBoxCoords[idx,:] = np.where(np.abs(diff)<0.5, clusterBoxCoords[idx,:], clusterBoxCoords[idx,:]-np.sign(diff))
            incrementalCenter += clusterBoxCoords[idx,:]
            centerNumberOfAtoms += 1.0
        # set cluster atoms new box positions
        boxCoords[self.clusterIndexes,:] = clusterBoxCoords
        # translate cluster in box center
        if self.clusterToBoxCenter:
            # calculate cluster center of mass
            center = np.sum(clusterBoxCoords,0)/len(self.clusterIndexes)
            # translate cluster to center of box
            boxCoords += np.array([0.5, 0.5, 0.5])-center
        # fold all but cluster atoms
        if self.fold:
            boxCoords[self.restOfAtomsIndexes,:] %= 1
        # convert to real coordinates
        coords = self._trajectory._boundaryConditions.box_to_real_array(boxArray=boxCoords, index=confIdx)
        # set new coordinates
        self._trajectory.set_configuration_coordinates(confIdx, coords)
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
