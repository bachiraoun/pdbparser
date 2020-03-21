"""
This module provides all mean square displacement classes.

.. inheritance-diagram:: pdbparser.Analysis.Dynamics.MeanSquareDisplacement
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
from pdbparser.Utilities.Information import get_records_database_property_values
from pdbparser.Utilities.Collection import correlation, get_data_weighted_sum
from pdbparser.Utilities.Database import get_element_property, is_element_property


class MeanSquareDisplacement(Analysis):
    """
    Computes the mean square displacement for a set of atoms.

    :Parameters:
        #. trajectory (pdbTrajectory): pdbTrajectory instance.
        #. configurationsIndexes (list, set, tuple): List of selected indexes of configuration used to perform the analysis.
        #. targetAtomsIndexes (list, set, tuple): Selected target atoms indexes.
        #. weighting (database key): a database property to weight the mean square displacement partials.
    """

    def __init__(self, trajectory, configurationsIndexes,
                       targetAtomsIndexes, weighting="equal", *args, **kwargs):
        # set trajectory
        super(MeanSquareDisplacement,self).__init__(trajectory, *args, **kwargs)
        # set steps indexes
        self.targetAtomsIndexes = self.get_atoms_indexes(targetAtomsIndexes)
        self.numberOfSteps = len(self.targetAtomsIndexes)
        # set configurations indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        # set weighting
        assert is_element_property(weighting), Logger.error("weighting '%s' don't exist in database"%weighting)
        self.weighting = weighting
        # initialize variables
        self.__initialize_variables__()
        # initialize results
        self.__initialize_results__()

    def __initialize_variables__(self):
        self.weights = np.array(get_records_database_property_values(self.targetAtomsIndexes, self.structure, self.weighting))
        elements = self._trajectory.elements
        self.elements = [elements[idx] for idx in self.targetAtomsIndexes]
        elementsSet = set(self.elements)
        self.elementsWeights = dict(zip(elementsSet,[get_element_property(el, self.weighting) for el in elementsSet]))
        self.elementsNumber = dict(Counter(self.elements))

    def __initialize_results__(self):
        # time
        self.results['time'] = np.array([self.time[idx] for idx in self.configurationsIndexes], dtype=np.float)
        # mean square displacements
        for el in set(self.elements):
            self.results['msd_%s' %el] = np.zeros((len(self.configurationsIndexes)), dtype=np.float)

    def step(self, index):
        """
        analysis step of calculation method.\n

        :Parameters:
            #. atomIndex (int): the atom step index

        :Returns:
            #. stepData (object): object used in combine method
        """
        # get atom index
        atomIndex = self.targetAtomsIndexes[index]
        # get atomTrajectory
        atomTrajectory = self._trajectory.get_atom_trajectory(atomIndex, self.configurationsIndexes)
        dsq = np.add.reduce(atomTrajectory*atomTrajectory,1)
        # sum_dsq1 is the cumulative sum of dsq
        sum_dsq1 = np.add.accumulate(dsq)
        # sum_dsq1 is the reversed cumulative sum of dsq
        sum_dsq2 = np.add.accumulate(dsq[::-1])
        # sumsq refers to SUMSQ in the published algorithm
        sumsq = 2.*sum_dsq1[-1]
        # this line refers to the instruction SUMSQ <-- SUMSQ - DSQ(m-1) - DSQ(N - m) of the published algorithm
        # In this case, msd is an array because the instruction is computed for each m ranging from 0 to len(traj) - 1
        # So, this single instruction is performing the loop in the published algorithm
        Saabb  = sumsq - np.concatenate(([0.], sum_dsq1[:-1])) - np.concatenate(([0.], sum_dsq2[:-1]))
        # Saabb refers to SAA+BB/(N-m) in the published algorithm
        # Sab refers to SAB(m)/(N-m) in the published algorithm
        Saabb = Saabb / (len(dsq) - np.arange(len(dsq)))
        Sab   = 2.*correlation(atomTrajectory)
        # The atomic MSD.
        atomicMSD = Saabb - Sab
        return index, atomicMSD

    def combine(self, index, stepData):
        """
        analysis combine method called after each step.\n

        :Parameters:
            #. atomIndex (int): the atomIndex of the last calculated atom
            #. stepData (object): the returned data from step method
        """
        # get atom index
        atomIndex = self.targetAtomsIndexes[index]
        # The symbol of the atom.
        element = self.elements[index]
        if element =="NA":
            print(index, atomIndex)
        # The MSD for element |symbol| is updated.
        self.results['msd_%s'%element] += stepData

    def finalize(self):
        """
        called once all the steps has been run.\n
        """
        # The MSDs per element are averaged.
        data = {}
        for el in set(self.elements):
            self.results['msd_%s' %el] /= len([item for item in self.elements if item==el])
            data[el] = self.results['msd_%s' %el]
        # get msd total
        self.results['msd_total'] = get_data_weighted_sum(data, numbers=self.elementsNumber, weights=self.elementsWeights)




class MeanSquareDisplacementInCylinder(Analysis):
    """
    Computes the mean square displacement for a set of atoms splitting partials between inside and outside of a cylinder.

    :Parameters:
        #. trajectory (pdbTrajectory): pdbTrajectory instance.
        #. configurationsIndexes (list, set, tuple): List of selected indexes of configuration used to perform the analysis.
        #. cylinderAtomsIndexes (list, set, tuple): Selected atoms indexes supposedly forming a cylinder e.g. nanotube.
        #. targetAtomsIndexes (list, set, tuple): Selected target atoms indexes.
        #. weighting (database key): a database property to weight the mean square displacement partials.
        #. axis (None, vector): The cylinder main axis, If None main principal axis of cylinderAtomsIndexes is calculated automatically.
    """
    def __init__(self, trajectory, configurationsIndexes,
                       cylinderAtomsIndexes, targetAtomsIndexes,
                       axis=None, weighting="equal", histBin=1, *args, **kwargs):
        # set trajectory
        super(MeanSquareDisplacementInCylinder,self).__init__(trajectory, *args, **kwargs)
        # set configurations indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        # set atoms indexes
        self.targetAtomsIndexes = self.get_atoms_indexes(targetAtomsIndexes)
        self.cylinderAtomsIndexes = self.get_atoms_indexes(cylinderAtomsIndexes)
        # set steps indexes
        self.numberOfSteps = len(self.targetAtomsIndexes)
        # set weighting
        assert is_element_property(weighting), Logger.error("weighting '%s' don't exist in database"%weighting)
        self.weighting = weighting
        # set residency time histogram bin
        try:
            self.histBin = float(histBin)
        except:
            raise Logger.error("histBin must be number convertible. %s is given."%histBin)
        assert self.histBin%1 == 0, logger.error("histBin must be integer. %s is given."%histBin)
        assert self.histBin>0, logger.error("histBin must be positive. %s is given."%histBin)
        assert self.histBin<len(self.configurationsIndexes), logger.error("histBin must smaller than numberOfConfigurations")
        # initialize variables
        self.__initialize_variables__(axis)
        # initialize results
        self.__initialize_results__()
        # get cylinder centers, matrices, radii, length
        Logger.info("%s --> initializing cylinder parameters along all configurations"%self.__class__.__name__)
        self.cylCenters, self.cylMatrices, self.cylRadii, self.cylLengths = self.__get_cylinder_properties__()

    def __initialize_variables__(self, axis):
        self.weights = np.array(get_records_database_property_values(self.targetAtomsIndexes, self.structure, self.weighting))
        elements = self._trajectory.elements
        self.elements = [elements[idx] for idx in self.targetAtomsIndexes]
        elementsSet = set(self.elements)
        self.elementsWeights = dict(zip(elementsSet,[get_element_property(el, self.weighting) for el in elementsSet]))
        self.elementsNumber = dict(Counter(self.elements))
        # check atoms indexes
        assert not len(set.intersection(set(self.cylinderAtomsIndexes), set(self.targetAtomsIndexes))), Logger.error("cylinderAtomsIndexes and targetAtomsIndexes can't have any index in common")
        # set axis
        if axis is None:
            axis = {"principal":self.cylinderAtomsIndexes}
        self.axis = AxisDefinition(self._trajectory, axis)

    def __initialize_results__(self):
        # time
        self.results['time'] = np.array([self.time[idx] for idx in self.configurationsIndexes], dtype=np.float)
        self.results['histogram_edges'] = self.histBin*np.array(range(int(len(self.configurationsIndexes)/self.histBin)+1))

        # mean square displacements
        for el in set(self.elements):
            self.results['residency_time_inside_%s' %el]  = [0.,0]
            self.results['residency_time_outside_%s' %el] = [0.,0]
            self.results['msd_inside_%s' %el]  = np.zeros((len(self.configurationsIndexes)), dtype=np.float)
            self.results['msd_inside_axial_%s' %el]  = np.zeros((len(self.configurationsIndexes)), dtype=np.float)
            self.results['msd_inside_transversal_%s' %el]  = np.zeros((len(self.configurationsIndexes)), dtype=np.float)
            self.results['msd_outside_%s' %el] = np.zeros((len(self.configurationsIndexes)), dtype=np.float)
            self.results['histogram_inside_%s' %el]  = np.zeros((int(len(self.configurationsIndexes)/self.histBin)+1), dtype=np.float)
            self.results['histogram_outside_%s' %el] = np.zeros((int(len(self.configurationsIndexes)/self.histBin)+1), dtype=np.float)

        # normalization
        self.insideNormalization = {}
        self.outsideNormalization = {}
        for el in set(self.elements):
            self.insideNormalization[el]  = np.zeros((len(self.configurationsIndexes)), dtype=np.float)
            self.outsideNormalization[el] = np.zeros((len(self.configurationsIndexes)), dtype=np.float)

    def __get_cylinder_properties__(self):
        cylCenters  = []
        cylMatrices = []
        cylRadii    = []
        cylLengths  = []
        # get the Frame index
        for confIdx in self.configurationsIndexes:
            # set working configuration index
            self._trajectory.set_configuration_index(confIdx)
            # get coordinates
            coordinates = self._trajectory.get_configuration_coordinates(confIdx)
            # cylinder atomTrajectory
            cylinderAtomsCoordinates = coordinates[self.cylinderAtomsIndexes]
            # get center and axis and rotation matrix
            center, rotationMatrix = self.axis.get_center_rotationMatrix(coordinates)
            # translate cylinder coordinates to center
            cylinderAtomsCoordinates -= center
            # change coordinates to cylinder axes system
            cylinderAtomsCoordinates = np.dot(cylinderAtomsCoordinates, rotationMatrix)
            # get radius and length
            cylRadiusSquared = np.sqrt( np.mean( cylinderAtomsCoordinates[:,1]**2+cylinderAtomsCoordinates[:,2]**2 ) )
            cylLength        = np.abs(np.max(cylinderAtomsCoordinates[:,0])-np.min(cylinderAtomsCoordinates[:,0]))
            # append center and rotation matrix, radius, length
            cylCenters.append(center)
            cylMatrices.append(rotationMatrix)
            cylRadii.append(cylRadiusSquared)
            cylLengths.append(cylLength)
        return np.array(cylCenters), np.array(cylMatrices), np.array(cylRadii), np.array(cylLengths)

    def __split_trajectory_to_inside_outside__(self, atomTrajectory):
        inside = []
        outside = []
        isIn = False
        isOut = False
        inCylBasisAtomTraj = np.empty(atomTrajectory.shape)
        for idx in range(atomTrajectory.shape[0]):
            #center = atomTrajectory[idx]-self.cylCenters[idx]
            center = self._trajectory.boundaryConditions.real_difference(self.cylCenters[idx], atomTrajectory[idx], index = idx)
            # transform to cnt basis
            inCylCoords = np.dot(center, self.cylMatrices[idx])
            # set inCylBasisAtomTraj
            inCylBasisAtomTraj[idx] = inCylCoords + self.cylCenters[idx]
            # check x if outside or inside nanotube
            if np.abs(inCylCoords[0]) > self.cylLengths[idx]/2.:
                if isOut:
                    outside[-1].append(idx)
                else:
                    outside.append([idx])
                isIn = False
                isOut = True
                continue
            # check radius if bigger or smaller than the nanotube's one
            radius = np.sqrt(inCylCoords[1]**2+inCylCoords[2]**2)
            if radius < self.cylRadii[idx]:
                if isIn:
                    inside[-1].append(idx)
                else:
                    inside.append([idx])
                isIn = True
                isOut = False
            else:
                if isOut:
                    outside[-1].append(idx)
                else:
                    outside.append([idx])
                isIn = False
                isOut = True
        return inside, outside, inCylBasisAtomTraj

    def __get_sub_trajectory_msd__(self, atomSubTrajectory):
        dsq = np.add.reduce(atomSubTrajectory*atomSubTrajectory,1)
        # sum_dsq1 is the cumulative sum of dsq
        sum_dsq1 = np.add.accumulate(dsq)
        # sum_dsq1 is the reversed cumulative sum of dsq
        sum_dsq2 = np.add.accumulate(dsq[::-1])
        # sumsq refers to SUMSQ in the published algorithm
        sumsq = 2.*sum_dsq1[-1]
        # this line refers to the instruction SUMSQ <-- SUMSQ - DSQ(m-1) - DSQ(N - m) of the published algorithm
        # In this case, msd is an array because the instruction is computed for each m ranging from 0 to len(traj) - 1
        # So, this single instruction is performing the loop in the published algorithm
        Saabb  = sumsq - np.concatenate(([0.], sum_dsq1[:-1])) - np.concatenate(([0.], sum_dsq2[:-1]))
        # Saabb refers to SAA+BB/(N-m) in the published algorithm
        # Sab refers to SAB(m)/(N-m) in the published algorithm
        Saabb = Saabb / (len(dsq) - np.arange(len(dsq)))
        Sab   = 2.*correlation(atomSubTrajectory)
        # The atomic MSD.
        return Saabb - Sab

    def step(self, index):
        """
        analysis step of calculation method.\n

        :Parameters:
            #. atomIndex (int): the atom step index

        :Returns:
            #. stepData (object): object used in combine method
        """
        # get atom index
        atomIndex = self.targetAtomsIndexes[index]
        # get atomTrajectory
        atomTrajectory = self._trajectory.get_atom_trajectory(atomIndex, self.configurationsIndexes)
        # get inside outside
        inside, outside, inCylBasisAtomTraj = self.__split_trajectory_to_inside_outside__(atomTrajectory)
        # get atom element
        element = self.elements[index]
        # residency time
        residencyInside  = [0,0]
        residencyOutside = [0,0]
        axialTraj = np.zeros((inCylBasisAtomTraj.shape[0], 1))
        transTraj = np.zeros((inCylBasisAtomTraj.shape[0], 2))
        # calculate msd inside
        for item in inside:
            self.results['histogram_inside_%s' %element][int(len(item)/self.histBin)] += 1
            if len(item) == 1:
                continue
            residencyInside[0] += self.results['time'][item[-1]]-self.results['time'][item[0]]
            residencyInside[1] += 1
            # get msd inside
            atomicMSD = self.__get_sub_trajectory_msd__(inCylBasisAtomTraj[item])
            self.results['msd_inside_%s' %element][0:atomicMSD.shape[0]] += atomicMSD
            # get msd axial
            axialTraj[item,0] = inCylBasisAtomTraj[item,0]
            axialAtomicMSD = self.__get_sub_trajectory_msd__(axialTraj[item])
            self.results['msd_inside_axial_%s' %element][0:axialAtomicMSD.shape[0]] += axialAtomicMSD
            # get msd transversal
            transTraj[item,0:] = inCylBasisAtomTraj[item,1:]
            transAtomicMSD = self.__get_sub_trajectory_msd__(transTraj[item])
            self.results['msd_inside_transversal_%s' %element][0:transAtomicMSD.shape[0]] += transAtomicMSD
            # update insideNormalization
            self.insideNormalization[element][0:atomicMSD.shape[0]] += 1
        # calculate msd outside
        for item in outside:
            self.results['histogram_outside_%s' %element][int(len(item)/self.histBin)] += 1
            if len(item) == 1:
                continue
            residencyOutside[0] += self.results['time'][item[-1]]-self.results['time'][item[0]]
            residencyOutside[1] += 1
            atomicMSD = self.__get_sub_trajectory_msd__(atomTrajectory[item])
            self.results['msd_outside_%s' %element][0:atomicMSD.shape[0]] += atomicMSD
            self.outsideNormalization[element][0:atomicMSD.shape[0]] += 1
        # return
        return index, (residencyInside, residencyOutside)

    def combine(self, index, stepData):
        """
        analysis combine method called after each step.\n

        :Parameters:
            #. atomIndex (int): the atomIndex of the last calculated atom
            #. stepData (object): the returned data from step method
        """
        element = self.elements[index]
        self.results['residency_time_inside_%s' %element][0] += stepData[0][0]
        self.results['residency_time_inside_%s' %element][1] += stepData[0][1]
        self.results['residency_time_outside_%s'%element][0] += stepData[1][0]
        self.results['residency_time_outside_%s'%element][1] += stepData[1][1]


    def finalize(self):
        """
        called once all the steps has been run.\n
        """
        # The MSDs per element are averaged.
        for el in set(self.elements):
            whereInside = np.where(self.insideNormalization[el]==0)[0]
            if len(whereInside) == 0:
                self.results['msd_inside_%s' %el][:] /= self.insideNormalization[el]
                self.results['msd_inside_axial_%s' %el][:] /= self.insideNormalization[el]
                self.results['msd_inside_transversal_%s' %el][:] /= self.insideNormalization[el]
            else:
                self.results['msd_inside_%s' %el][:whereInside[0]] /= self.insideNormalization[el][:whereInside[0]]
                self.results['msd_inside_axial_%s' %el][:whereInside[0]] /= self.insideNormalization[el][:whereInside[0]]
                self.results['msd_inside_transversal_%s' %el][:whereInside[0]] /= self.insideNormalization[el][:whereInside[0]]
            whereOutside = np.where(self.outsideNormalization[el]==0)[0]
            if len(whereOutside) == 0:
                self.results['msd_outside_%s' %el][:] /= self.outsideNormalization[el]
            else:
                self.results['msd_outside_%s' %el][:whereOutside[0]] /= self.outsideNormalization[el][:whereOutside[0]]
        # The MSDs per element are averaged.
        msdsInside   = {}
        msdsAxialInside   = {}
        msdsTransInside   = {}
        msdsOutside  = {}
        for el in set(self.elements):
            msdsInside[el]   = self.results['msd_inside_%s' %el]
            msdsAxialInside[el]   = self.results['msd_inside_axial_%s' %el]
            msdsTransInside[el]   = self.results['msd_inside_transversal_%s' %el]
            msdsOutside[el]  = self.results['msd_outside_%s' %el]
        # get msd total
        self.results['msd_inside_total']  = get_data_weighted_sum(msdsInside, numbers=self.elementsNumber, weights=self.elementsWeights)
        self.results['msd_inside_axial_total']  = get_data_weighted_sum(msdsAxialInside, numbers=self.elementsNumber, weights=self.elementsWeights)
        self.results['msd_inside_transversal_total']  = get_data_weighted_sum(msdsTransInside, numbers=self.elementsNumber, weights=self.elementsWeights)
        self.results['msd_outside_total'] = get_data_weighted_sum(msdsOutside, numbers=self.elementsNumber, weights=self.elementsWeights)
        # get residency time
        self.results['residency_time_inside_total'] = np.array([0.0])
        self.results['residency_time_outside_total'] = np.array([0.0])
        for el in set(self.elements):
            if self.results['residency_time_inside_%s' %el][1] == 0:
                self.results['residency_time_inside_%s' %el][1] = 1.
            if self.results['residency_time_outside_%s' %el][1] == 0:
                self.results['residency_time_outside_%s' %el][1] = 1.
            self.results['residency_time_inside_%s' %el]  = np.array([self.results['residency_time_inside_%s' %el][0]/self.results['residency_time_inside_%s' %el][1]])
            self.results['residency_time_outside_%s' %el] = np.array([self.results['residency_time_outside_%s' %el][0]/self.results['residency_time_outside_%s' %el][1]])
            self.results['residency_time_inside_total']  += self.results['residency_time_inside_%s' %el]
            self.results['residency_time_outside_total'] += self.results['residency_time_outside_%s' %el]
        self.results['residency_time_inside_total']  /= len(set(self.elements))
        self.results['residency_time_outside_total'] /= len(set(self.elements))
        # get histogram time
        self.results['histogram_inside_total']  = np.zeros((int(len(self.configurationsIndexes)/self.histBin)+1), dtype=np.float)
        self.results['histogram_outside_total'] = np.zeros((int(len(self.configurationsIndexes)/self.histBin)+1), dtype=np.float)
        for el in set(self.elements):
            self.results['histogram_inside_total']  += self.results['histogram_inside_%s' %el]
            self.results['histogram_outside_total'] += self.results['histogram_outside_%s' %el]
