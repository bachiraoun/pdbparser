"""
This module provides classes to analyse cylindrical distributions and densities

.. inheritance-diagram:: pdbparser.Analysis.Structure.CylindricalDensity
    :parts: 2

"""
# standard libraries imports
from __future__ import print_function
from collections import Counter

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.log import Logger
from pdbparser.Analysis.Core import Analysis, AxisDefinition
from pdbparser.Utilities.Database import get_element_property, is_element_property



class CylindricalRadialDensity(Analysis):
    """
    Computes the radial density profile in comparison to an axis, normalized to the total number of selected atoms.
    """
    def __init__(self, trajectory, configurationsIndexes,
                       targetAtomsIndexes, weighting,
                       axis, radii, length, *args, **kwargs):
        # set trajectory
        super(CylindricalRadialDensity,self).__init__(trajectory, *args, **kwargs)
        # set steps indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        self.numberOfSteps = len(self.configurationsIndexes)
        # set targetAtomsIndexes
        self.targetAtomsIndexes = self.get_atoms_indexes(targetAtomsIndexes)
        # initialize variables
        self.__initialize_variables__(weighting, axis, radii, length)
        # initialize results
        self.__initialize_results__()


    def __initialize_variables__(self, weighting, axis, radii, length):
        # get elements
        elements = self._trajectory.elements
        self.elements = [elements[idx] for idx in self.targetAtomsIndexes]
        self.elementsSet = list(set(self.elements))
        # set weighting
        assert is_element_property(weighting), Logger.error("weighting '%s' don't exist in database"%weighting)
        self.weighting = weighting
        self.weights = np.array([get_element_property(el, self.weighting) for el in self.elements])
        self.elementsWeights = dict(zip(self.elementsSet,[get_element_property(el, self.weighting) for el in self.elementsSet]))
        self.elementsNumber = dict(Counter(self.elements))
        # set axis
        self.axis = AxisDefinition(self._trajectory, axis)
        # set radii
        assert isinstance(radii, (list, tuple, set, np.ndarray)), Logger.error("radii must be a list or numpy.array")
        try:
            radii = np.array(sorted(set(radii)), dtype=np.float32)
        except:
            raise Logger.error("radii element must be numbers")
        assert len(radii.shape)==1, Logger.error("radii must be uni-dimensional")
        assert radii[0]>0, Logger.error("all radii array must be positive")
        self.radii = radii
        # set length
        try:
            length = float(length)
        except:
            raise Logger.error("length must be numbers")
        assert length>0, Logger.error("length must be positive")
        self.length = length
        # cylinder volume
        self.cylinderVolume      = self.length*np.pi*self.radii[-1]**2
        self.cylindersVolumes    = 2*np.pi*self.radii[0:-1]*(self.radii[1:]-self.radii[0:-1])*self.length
        self.cumCylindersVolumes = np.pi*self.length*self.radii[1:]**2


    def __initialize_results__(self):
        # time
        self.results['time'] = np.array([self.time[idx] for idx in self.configurationsIndexes], dtype=np.float)
        # radii
        self.results['radii'] = (self.radii[1:]+self.radii[0:-1])/2.
        #volumes
        self.results['cylindersVolume'] = self.cylindersVolumes
        self.results['cumulutiveCylindersVolume'] = self.cumCylindersVolumes
        # Will store the radial_density_profile.
        self.cylindricalDistribution = {}
        self.cylindricalDensity = {}
        self.insideAtomsNumber = {}
        self.numberDensity = {}
        # initialize results per element
        for el in set(self.elements):
            self.results['CumulCylRadialDenDist_%s'%el] = np.zeros( len(self.radii)-1, dtype=np.float)
            self.results['CumulCylRadialDen_%s'%el]     = np.zeros( len(self.radii)-1, dtype=np.float)
            self.results['CylRadialDenDist_%s'%el]      = np.zeros( len(self.radii)-1, dtype=np.float)
            self.results['CylRadialDen_%s'%el]          = np.zeros( len(self.radii)-1, dtype=np.float)
            self.results['insideAtomsNumber_%s'%el]     = np.zeros( self.numberOfSteps, dtype=np.float)
            self.results['numberDensity_%s'%el]         = np.zeros( self.numberOfSteps, dtype=np.float)
            self.results['density_%s'%el]               = np.zeros( self.numberOfSteps, dtype=np.float)
        # initialize totals
        self.results['CumulCylRadialDenDist_total'] = np.zeros( len(self.radii)-1, dtype=np.float)
        self.results['CumulCylRadialDen_total']     = np.zeros( len(self.radii)-1, dtype=np.float)
        self.results['CylRadialDenDist_total']      = np.zeros( len(self.radii)-1, dtype=np.float)
        self.results['CylRadialDen_total']          = np.zeros( len(self.radii)-1, dtype=np.float)
        self.results['insideAtomsNumber_total']     = np.zeros( self.numberOfSteps, dtype=np.float)
        self.results['numberDensity_total']         = np.zeros( self.numberOfSteps, dtype=np.float)
        self.results['density_total']               = np.zeros( self.numberOfSteps, dtype=np.float)


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
        # get center and axis and rotation matrix
        center, rotationMatrix = self.axis.get_center_rotationMatrix(coordinates)
        # translate to center
        targetAtomsCoordinates -= center
         # change coordinates to cylinder axes system
        targetAtomsCoordinates = np.dot(targetAtomsCoordinates, rotationMatrix)
        # calculate distance from axis vector
        radii = np.sqrt(targetAtomsCoordinates[:,1]**2+targetAtomsCoordinates[:,2]**2)
         # find inside cylinder indexes
        indexes = set(range(len(self.targetAtomsIndexes)))
        outOfLengthIdx = set( list(np.nonzero(np.abs(targetAtomsCoordinates[:,0])>self.length/2.)[0]) )
        outOfRadiusIdx = set( np.nonzero(radii>self.radii[-1])[0] )
        # substract outside nanotube
        indexes -= outOfLengthIdx
        indexes -= outOfRadiusIdx
        indexes = list(indexes)
        # get this frame arrays
        radii = radii[indexes]
        # inside nantoube atoms elements
        selectedElements = np.array(self.elements)[indexes]
        # return
        return index, (radii, selectedElements, indexes)


    def combine(self, index, stepData):
        """
        analysis combine method called after each step.\n

        :Parameters:
            #. index (int): the index of the last calculated step
            #. stepData (object): the returned data from step method
        """
        radii = stepData[0]
        insideCylinderElements = stepData[1]
        insideCylinderIndexes = stepData[2]
        insideDensity = 1.0*np.sum(self.weights[insideCylinderIndexes])/self.cylinderVolume
        for el in self.elementsSet:
            # get element indexes
            indexes = np.nonzero(insideCylinderElements == el)[0]
            # calculate histogram
            hist, _ = np.histogram( radii[indexes] ,bins=self.radii, weights=self.weights[indexes])
            # calculate element density
            elementDensity = self.elementsWeights[el]*len(indexes)/self.cylinderVolume
            # update value
            self.results['CylRadialDenDist_%s'%el]        += hist/insideDensity
            self.results['CylRadialDen_%s'%el]            += hist
            self.results['insideAtomsNumber_%s'%el][index] = len(indexes)
            self.results['numberDensity_%s'%el][index]     = len(indexes)/self.cylinderVolume
            self.results['density_%s'%el][index]           = elementDensity


    def finalize(self):
        """
        called once all the steps has been run.\n
        """
        for el in list(set(self.elements)):
            # update value
            self.results['CumulCylRadialDenDist_%s'%el]  = 1.0*np.cumsum(self.results['CylRadialDenDist_%s'%el])/self.cumCylindersVolumes/self.numberOfSteps
            self.results['CumulCylRadialDen_%s'%el]      = 1.0*np.cumsum(self.results['CylRadialDen_%s'%el])/self.cumCylindersVolumes/self.numberOfSteps
            self.results['CylRadialDenDist_%s'%el]      /= self.cylindersVolumes*self.numberOfSteps
            self.results['CylRadialDen_%s'%el]          /= self.cylindersVolumes*self.numberOfSteps
            self.results['CumulCylRadialDenDist_total'] += self.results['CumulCylRadialDenDist_%s'%el]
            self.results['CumulCylRadialDen_total']     += self.results['CumulCylRadialDen_%s'%el]
            self.results['CylRadialDenDist_total']      += self.results['CylRadialDenDist_%s'%el]
            self.results['CylRadialDen_total']          += self.results['CylRadialDen_%s'%el]
            self.results['insideAtomsNumber_total']     += self.results['insideAtomsNumber_%s'%el]
            self.results['numberDensity_total']         += self.results['numberDensity_%s'%el]
            self.results['density_total']               += self.results['density_%s'%el]




class CylindricalDiskDensity(Analysis):
    """
    Computes the radial density profile in comparison to an axis, normalized to the total number of selected atoms.
    """
    def __init__(self, trajectory, configurationsIndexes,
                       targetAtomsIndexes, weighting,
                       axis, radius, lengths, *args, **kwargs):
        # set trajectory
        super(CylindricalDiskDensity,self).__init__(trajectory, *args, **kwargs)
        # set steps indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        self.numberOfSteps = len(self.configurationsIndexes)
        # set targetAtomsIndexes
        self.targetAtomsIndexes = self.get_atoms_indexes(targetAtomsIndexes)
        # initialize variables
        self.__initialize_variables__(weighting, axis, radius, lengths)
        # initialize results
        self.__initialize_results__()


    def __initialize_variables__(self, weighting, axis, radius, lengths):
        # get elements
        elements = self._trajectory.elements
        self.elements = [elements[idx] for idx in self.targetAtomsIndexes]
        self.elementsSet = list(set(self.elements))
        # set weighting
        assert is_element_property(weighting), Logger.error("weighting '%s' don't exist in database"%weighting)
        self.weighting = weighting
        self.weights = np.array([get_element_property(el, self.weighting) for el in self.elements])
        self.elementsWeights = dict(zip(self.elementsSet,[get_element_property(el, self.weighting) for el in self.elementsSet]))
        self.elementsNumber = dict(Counter(self.elements))
        # set axis
        self.axis = AxisDefinition(self._trajectory, axis)
        # set length
        assert isinstance(lengths, (list, tuple, set, np.ndarray)), Logger.error("lengths must be a list or numpy.array")
        try:
            lengths = np.array(sorted(set(lengths)), dtype=np.float32)
        except:
            raise Logger.error("lengths element must be numbers")
        assert len(lengths.shape)==1, Logger.error("lengths must be uni-dimensional")
        self.lengths = lengths
        # set radius
        try:
            radius = float(radius)
        except:
            raise Logger.error("radius must be numbers")
        assert radius>0, Logger.error("radius must be positive")
        self.radius = radius
        # cylinder volume
        self.cylinderVolume  = (self.lengths[-1]-self.lengths[0])*np.pi*self.radius**2
        self.diskVolumes     = (self.lengths[1:]-self.lengths[0:-1])*np.pi*self.radius**2
        self.cumDisksVolumes = np.cumsum(self.diskVolumes)


    def __initialize_results__(self):
        # time
        self.results['time'] = np.array([self.time[idx] for idx in self.configurationsIndexes], dtype=np.float)
        # radii
        self.results['lengths'] = (self.lengths[1:]+self.lengths[0:-1])/2.
        #volumes
        self.results['disksVolume'] = self.diskVolumes
        self.results['cumulutiveDisksVolume'] = self.cumDisksVolumes
        # Will store the radial_density_profile.
        self.cylindricalDistribution = {}
        self.cylindricalDensity = {}
        self.insideAtomsNumber = {}
        self.numberDensity = {}
        # initialize results per element
        for el in set(self.elements):
            self.results['CumulCylDiskDenDist_%s'%el]   = np.zeros( len(self.lengths)-1, dtype=np.float)
            self.results['CumulCylDiskDen_%s'%el]       = np.zeros( len(self.lengths)-1, dtype=np.float)
            self.results['CylDiskDenDist_%s'%el]        = np.zeros( len(self.lengths)-1, dtype=np.float)
            self.results['CylDiskDen_%s'%el]            = np.zeros( len(self.lengths)-1, dtype=np.float)
            self.results['insideAtomsNumber_%s'%el]     = np.zeros( self.numberOfSteps, dtype=np.float)
            self.results['numberDensity_%s'%el]         = np.zeros( self.numberOfSteps, dtype=np.float)
            self.results['density_%s'%el]               = np.zeros( self.numberOfSteps, dtype=np.float)
        # initialize totals
        self.results['CumulCylDiskDenDist_total']   = np.zeros( len(self.lengths)-1, dtype=np.float)
        self.results['CumulCylDiskDen_total']       = np.zeros( len(self.lengths)-1, dtype=np.float)
        self.results['CylDiskDenDist_total']        = np.zeros( len(self.lengths)-1, dtype=np.float)
        self.results['CylDiskDen_total']            = np.zeros( len(self.lengths)-1, dtype=np.float)
        self.results['insideAtomsNumber_total']     = np.zeros( self.numberOfSteps, dtype=np.float)
        self.results['numberDensity_total']         = np.zeros( self.numberOfSteps, dtype=np.float)
        self.results['density_total']               = np.zeros( self.numberOfSteps, dtype=np.float)


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
        # get center and axis and rotation matrix
        center, rotationMatrix = self.axis.get_center_rotationMatrix(coordinates)
        # translate to center
        targetAtomsCoordinates -= center
         # change coordinates to cylinder axes system
        targetAtomsCoordinates = np.dot(targetAtomsCoordinates, rotationMatrix)
        # calculate distance from axis vector
        radii = np.sqrt(targetAtomsCoordinates[:,1]**2+targetAtomsCoordinates[:,2]**2)
         # find inside cylinder indexes
        indexes = set(range(len(self.targetAtomsIndexes)))
        outOfLengthIdx = list(np.nonzero(targetAtomsCoordinates[:,0]>self.lengths[-1])[0])
        outOfLengthIdx += list(np.nonzero(targetAtomsCoordinates[:,0]<self.lengths[0])[0])
        outOfLengthIdx = set(outOfLengthIdx)
        outOfRadiusIdx = set( np.nonzero(radii>self.radius)[0] )
        # substract outside nanotube
        indexes -= outOfLengthIdx
        indexes -= outOfRadiusIdx
        indexes = list(indexes)
        # get this frame arrays
        lengths = targetAtomsCoordinates[indexes,0]
        # inside nantoube atoms elements
        selectedElements = np.array(self.elements)[indexes]
        # return
        return index, (lengths, selectedElements, indexes)


    def combine(self, index, stepData):
        """
        analysis combine method called after each step.\n

        :Parameters:
            #. index (int): the index of the last calculated step
            #. stepData (object): the returned data from step method
        """
        lengths = stepData[0]
        insideCylinderElements = stepData[1]
        insideCylinderIndexes = stepData[2]
        insideDensity = np.sum(self.weights[insideCylinderIndexes])/self.cylinderVolume
        for el in self.elementsSet:
            # get element indexes
            indexes = np.nonzero(insideCylinderElements == el)[0]
            # calculate histogram
            hist, _ = np.histogram( lengths[indexes] ,bins=self.lengths, weights=self.weights[indexes])
            # calculate element density
            elementDensity = 1.0*self.elementsWeights[el]*len(indexes)/self.cylinderVolume
            # update value
            self.results['CylDiskDenDist_%s'%el]          += hist/insideDensity
            self.results['CylDiskDen_%s'%el]              += hist
            self.results['insideAtomsNumber_%s'%el][index] = len(indexes)
            self.results['numberDensity_%s'%el][index]     = len(indexes)/self.cylinderVolume
            self.results['density_%s'%el][index]           = elementDensity


    def finalize(self):
        """
        called once all the steps has been run.\n
        """
        for el in list(set(self.elements)):
            # update value
            self.results['CumulCylDiskDenDist_%s'%el]  = 1.0*np.cumsum(self.results['CylDiskDenDist_%s'%el])/self.cumDisksVolumes/self.numberOfSteps
            self.results['CumulCylDiskDen_%s'%el]      = 1.0*np.cumsum(self.results['CylDiskDen_%s'%el])/self.cumDisksVolumes/self.numberOfSteps
            self.results['CylDiskDenDist_%s'%el]      /= self.diskVolumes*self.numberOfSteps
            self.results['CylDiskDen_%s'%el]          /= self.diskVolumes*self.numberOfSteps
            self.results['CumulCylDiskDenDist_total'] += self.results['CumulCylDiskDenDist_%s'%el]
            self.results['CumulCylDiskDen_total']     += self.results['CumulCylDiskDen_%s'%el]
            self.results['CylDiskDenDist_total']      += self.results['CylDiskDenDist_%s'%el]
            self.results['CylDiskDen_total']          += self.results['CylDiskDen_%s'%el]
            self.results['insideAtomsNumber_total']   += self.results['insideAtomsNumber_%s'%el]
            self.results['numberDensity_total']       += self.results['numberDensity_%s'%el]
            self.results['density_total']             += self.results['density_%s'%el]
