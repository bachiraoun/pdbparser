"""
This module provides classes to analyse distances

.. inheritance-diagram:: pdbparser.Analysis.Structure.Distances
    :parts: 2

"""
# pdbparser library imports
from __future__ import print_function
from collections import Counter

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.log import Logger
from pdbparser.Analysis.Core import Analysis, AxisDefinition
from pdbparser.Utilities.Database import get_element_property, is_element_property



class InsideCylinderDistances(Analysis):
    """
    Computes the mean minimum distance between two atoms subset.
    """
    def __init__(self, trajectory, configurationsIndexes,
                       cylinderAtomsIndexes, targetAtomsIndexes,
                       axis=None, *args, **kwargs):
        # set trajectory
        super(InsideCylinderDistances,self).__init__(trajectory, *args, **kwargs)
        # set steps indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        self.numberOfSteps = len(self.configurationsIndexes)
        # set atoms indexes
        self.targetAtomsIndexes = self.get_atoms_indexes(targetAtomsIndexes)
        self.cylinderAtomsIndexes = self.get_atoms_indexes(cylinderAtomsIndexes)
        # initialize variables
        self.__initialize_variables__(axis)
        # initialize results
        self.__initialize_results__()


    def __initialize_variables__(self, axis):
        # check atoms indexes
        assert not len(set.intersection(set(self.cylinderAtomsIndexes), set(self.targetAtomsIndexes))), Logger.error("cylinderAtomsIndexes and targetAtomsIndexes can't have any index in common")
        # set axis
        if axis is None:
            axis = {"principal":self.cylinderAtomsIndexes}
        self.axis = AxisDefinition(self._trajectory, axis)


    def __initialize_results__(self):
        # time
        self.results['time'] = np.array([self.time[idx] for idx in self.configurationsIndexes], dtype=np.float)
        # mean minimum distances
        self.results['mean_minimum_distance'] = np.zeros( self.numberOfSteps, dtype=np.float)
        self.results['minimum_distance']      = np.zeros( self.numberOfSteps, dtype=np.float)
        self.results['mean_shell_thickness']  = np.zeros( self.numberOfSteps, dtype=np.float)
        self.results['cylinder_radius']       = np.zeros( self.numberOfSteps, dtype=np.float)


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
        targetAtomsCoordinates = coordinates[self.targetAtomsIndexes]
        cylinderAtomsCoordinates = coordinates[self.cylinderAtomsIndexes]
        # get center and axis and rotation matrix
        center, rotationMatrix = self.axis.get_center_rotationMatrix(coordinates)
        # translate to center
        targetAtomsCoordinates -= center
        cylinderAtomsCoordinates -= center
        # change coordinates to cylinder axes system
        targetAtomsCoordinates = np.dot(targetAtomsCoordinates, rotationMatrix)
        cylinderAtomsCoordinates = np.dot(cylinderAtomsCoordinates, rotationMatrix)
        # calculate carbon cylinder dimensions
        cylRadiusSquared = np.mean( cylinderAtomsCoordinates[:,1]**2+cylinderAtomsCoordinates[:,2]**2 )
        cylLength        = np.abs(np.max(cylinderAtomsCoordinates[:,0])-np.min(cylinderAtomsCoordinates[:,0]))
        # calculate target atoms radius
        radiiSquared     = targetAtomsCoordinates[:,1]**2+targetAtomsCoordinates[:,2]**2
         # find inside cylinder indexes
        outOfLengthIdx  = list(np.nonzero(targetAtomsCoordinates[:,0]<-cylLength/2.)[0])
        outOfLengthIdx += list(np.nonzero(targetAtomsCoordinates[:,0]>cylLength/2.)[0])
        outOfLengthIdx  = set(outOfLengthIdx)
        outOfRadiusIdx  = set( np.nonzero(radiiSquared>cylRadiusSquared)[0] )
        # substract atomes outside cylinder indexes
        indexes = set(range(len(self.targetAtomsIndexes)))
        indexes -= outOfLengthIdx
        indexes -= outOfRadiusIdx
        indexes = list(indexes)
        # get this frame arrays
        targetAtomsCoordinates = targetAtomsCoordinates[indexes,:]
        # return
        return index, (cylinderAtomsCoordinates, targetAtomsCoordinates, np.sqrt(cylRadiusSquared), cylLength )


    def combine(self, index, stepData):
        """
        analysis combine method called after each step.\n

        :Parameters:
            #. index (int): the index of the last calculated step
            #. stepData (object): the returned data from step method
        """
        cylinderAtomsCoordinates = stepData[0]
        targetAtomsCoordinates   = stepData[1]
        cylRadius                = stepData[2]
        cylLength                = stepData[3]
        # update cylinder radius
        self.results['cylinder_radius'][index] = cylRadius
        # if nothing inside
        if not len(targetAtomsCoordinates):
            self.results['mean_minimum_distance'][index] = np.nan
            self.results['minimum_distance'][index]      = np.nan
        else:
             # initialize distances
            distances = np.zeros(cylinderAtomsCoordinates.shape[0])
            # calculate minimum distances
            for ntIndex in range(cylinderAtomsCoordinates.shape[0]):
                # calculate the distance between subset atoms and target atoms
                # no need to calculate boundary conditions real distances because all atoms are translated to cylinder center
                difference         = targetAtomsCoordinates-cylinderAtomsCoordinates[ntIndex,:]
                distances[ntIndex] = np.sqrt(np.min(np.add.reduce(difference**2,1)))
            # calculate minimum distances at step index
            self.results['mean_minimum_distance'][index] = np.mean(distances)
            self.results['minimum_distance'][index]      = np.min(distances)
        # calculate mean shell distance to cylinder wall
        targetAtomsDistances   = np.sqrt(np.add.reduce(targetAtomsCoordinates[:,1:3]**2,1))
        self.results['mean_shell_thickness'][index] = np.mean(targetAtomsDistances)


    def finalize(self):
        """
        called once all the steps has been run.\n
        """
        pass
