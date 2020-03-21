"""
This module provides classes to correct and export new trajectories.

.. inheritance-diagram:: pdbparser.Analysis.Trajectory.CenterOfMassTranslated
    :parts: 2
"""
# standard libraries imports
from __future__ import print_function
from collections import Counter

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.log import Logger
from pdbparser.Analysis.Core import Analysis, AxisDefinition, CenterDefinition
from pdbparser.Utilities.Database import get_element_property, is_element_property



class CenterOfMassTranslated(Analysis):
    """
    Computes the global center of mass translated trajectory.
    """
    def __init__(self, trajectory, configurationsIndexes,
                       center, fold=True, *args, **kwargs):
        # set trajectory
        super(CenterOfMassTranslated,self).__init__(trajectory, *args, **kwargs)
        # set steps indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        self.numberOfSteps = len(self.configurationsIndexes)
        # initialize variables
        self.__initialize_variables__(center, fold)
        # initialize results
        self.__initialize_results__()


    def __initialize_variables__(self, center, fold):
        assert isinstance(fold, bool), Logger.error("fold must be boolean")
        self.fold = fold
        # center
        self.center = CenterDefinition(self._trajectory, center)

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
        # get configuration index
        confIdx = self.configurationsIndexes[index]
        # set working configuration index
        self._trajectory.set_configuration_index(confIdx)
        # get coordinates
        coordinates = self._trajectory.get_configuration_coordinates(confIdx)
        # get center of mass
        COM = self.center.get_center(coordinates)
        # translate all to COM
        coordinates += self._trajectory._boundaryConditions.get_box_real_center(index=confIdx) - COM
        # fold coordinates
        if self.fold:
            coordinates = self._trajectory._boundaryConditions.fold_real_array(coordinates, index=confIdx)
        # set new coordinates
        self._trajectory.set_configuration_coordinates(confIdx, coordinates)

        # return
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
