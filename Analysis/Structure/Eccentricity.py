"""
This moldule provides the Eccentricity analysis

.. inheritance-diagram:: pdbparser.Analysis.Structure.Eccentricity
    :parts: 2

"""
# standard libraries imports
from __future__ import print_function

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.log import Logger
from pdbparser.Analysis.Core import Analysis
from pdbparser.Utilities.Information import get_records_database_property_values


class Eccentricity(Analysis):
    """
    **Short description:**\n
    Computes the eccentricity for a set of atoms.\n

    **Calculation:** \n
    Eccentricity is calculated using the inertia principal axes 'I' along x, y and z: \n
    .. math:: Eccentricity = 1-\\frac{I_{min}}{I_{average}}

    The ratio of largest to smallest is between the biggest inertia to the smallest \n
    .. math:: ratio = \\frac{Imax}{Imin}

    The semiaxes a,b and c are those of an ellipsoid \n
    .. math:: semiaxis_a = \\sqrt{ \\frac{5}{2M} (I_{max}+I_{mid}-I_{min}) }
    .. math:: semiaxis_b = \\sqrt{ \\frac{5}{2M} (I_{max}+I_{min}-I_{mid}) }
    .. math:: semiaxis_c = \\sqrt{ \\frac{5}{2M} (I_{mid}+I_{min}-I_{max}) }

    Where:\n
        - M is the total mass of all the selected atoms
        - :math:`I_{min}` , :math:`I_{mid}` and :math:`I_{max}` are respectively the smallest, the middle and the biggest inertia moment value


    **Output:** \n
    #. moment_of_inertia_xx: the moment of inertia in x direction acting on the surface element with its vector normal in x direction
    #. moment_of_inertia_xy: the moment of inertia in y direction acting on the surface element with its vector normal in x direction
    #. moment_of_inertia_xz: the moment of inertia in z direction acting on the surface element with its vector normal in x direction
    #. moment_of_inertia_yy: the moment of inertia in y direction acting on the surface element with its vector normal in y direction
    #. moment_of_inertia_yz: the moment of inertia in z direction acting on the surface element with its vector normal in y direction
    #. moment_of_inertia_zz: the moment of inertia in z direction acting on the surface element with its vector normal in z direction
    #. semiaxis_a: ellipse biggest axis
    #. semiaxis_b: ellipse middle axis
    #. semiaxis_c: ellipse smallest axis
    #. ratio_of_largest_to_smallest
    #. eccentricity
    #. radius_of_gyration


    **Usage:** \n
    This analysis can be used to study macro-molecules geometry and sphericity .
    Originally conceived to calculate the sphericity of micelles.

    **Acknowledgement and publication:**\n
    AOUN Bachir

    **Job input parameters:** \n
    +------------------------+------------------------+---------------------------------------------+
    | Parameter              | Default                | Description                                 |
    +========================+========================+=============================================+
    | trajectory             |                        | MMTK trajectory path                        |
    +------------------------+------------------------+---------------------------------------------+
    | frames                 | '0:100:1'              | selected frames to perform the calculation  |
    +------------------------+------------------------+---------------------------------------------+
    | running_mode           | 'local:1'              | the job host and number of processors       |
    +------------------------+------------------------+---------------------------------------------+
    | output_file            | 'output_file'          | the analysis output file path               |
    +------------------------+------------------------+---------------------------------------------+
    | output_file_formats    | ['ascii','netcdf']     | the analysis output files formats the user  |
    |                        |                        | wishes to export at the end of the analysis |
    +------------------------+------------------------+---------------------------------------------+
    | atom_selection         |                        | atoms selection formula as defined in       |
    |                        |                        | nmoldyn, used to calculate the moment of    |
    |                        |                        | inertia                                     |
    +------------------------+------------------------+---------------------------------------------+
    | center_of_mass         |                        | atoms selection formula as defined in       |
    |                        |                        | nmoldyn, used to calculate the total mass   |
    |                        |                        | of the system                               |
    +------------------------+------------------------+---------------------------------------------+
    """

    def __init__(self, trajectory, configurationsIndexes,
                       targetAtomsIndexes, *args, **kwargs):
        # set trajectory
        super(Eccentricity,self).__init__(trajectory, *args, **kwargs)
        # set steps indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        self.numberOfSteps = len(self.configurationsIndexes)
        # set targetAtomsIndexes
        self.targetAtomsIndexes = self.get_atoms_indexes(targetAtomsIndexes)
        # initialize variables
        self.__initialize_variables__()
        # initialize results
        self.__initialize_results__()

    def __initialize_variables__(self):
        self.weights = np.array(get_records_database_property_values(self.targetAtomsIndexes, self.structure, "atomicWeight"))
        self.totalWeight = np.sum(self.weights)
        elements = self._trajectory.elements
        self.elements = [elements[idx] for idx in self.targetAtomsIndexes]

    def __initialize_results__(self):
        # time
        self.results['time'] = np.array([self.time[idx] for idx in self.configurationsIndexes], dtype=np.float)
        # moments of inertia
        for axis in ['xx','xy','xz','yy','yz','zz']:
            self.results['moment_of_inertia_%s' %axis] = np.zeros((len(self.configurationsIndexes)), dtype=np.float)
        # semi-axes
        for axis in ['a','b','c']:
            self.results['semiaxis_%s' %axis] = np.zeros((len(self.configurationsIndexes)), dtype=np.float)
        # eccentricity
        self.results['eccentricity'] = np.zeros((len(self.configurationsIndexes)), dtype=np.float)
        # ratio of largest to smallest
        self.results['ratio_of_largest_to_smallest'] = np.zeros((len(self.configurationsIndexes)), dtype=np.float)
        # radius of gyration
        self.results['radius_of_gyration'] = np.zeros((len(self.configurationsIndexes)), dtype=np.float)

    def step(self, index):
        """
        analysis step of calculation method.\n

        :Parameters:
            #. index (int): the step index

        :Returns:
            #. stepData (object): object used in combine method
        """
        # get configuration index
        confIdx = self.configurationsIndexes[index]
        # get coordinates
        coordinates = self._trajectory.get_configuration_coordinates(confIdx)
        targetAtomsCoordinates = coordinates[self.targetAtomsIndexes]
        weightedTargetAtomsCoordinates = np.transpose(np.transpose(targetAtomsCoordinates)*self.weights)
        # get center of mass
        COM = np.sum(weightedTargetAtomsCoordinates,0)/self.totalWeight
        # translate target atoms coordinates to COM
        targetAtomsCoordinates -= COM
        # calculate the inertia moments and the radius of gyration
        rog = np.sum( self.weights/self.totalWeight * np.add.reduce((targetAtomsCoordinates)**2,1) )
        xx  = np.add.reduce(self.weights * (targetAtomsCoordinates[:,1]*targetAtomsCoordinates[:,1] + targetAtomsCoordinates[:,2]*targetAtomsCoordinates[:,2]) )
        xy  = -np.add.reduce(self.weights * (targetAtomsCoordinates[:,0]*targetAtomsCoordinates[:,1]) )
        xz  = -np.add.reduce(self.weights * (targetAtomsCoordinates[:,0]*targetAtomsCoordinates[:,2]) )
        yy  = np.add.reduce(self.weights * (targetAtomsCoordinates[:,0]*targetAtomsCoordinates[:,0] + targetAtomsCoordinates[:,2]*targetAtomsCoordinates[:,2]) )
        yz  = -np.add.reduce(self.weights * (targetAtomsCoordinates[:,1]*targetAtomsCoordinates[:,2]) )
        zz  = np.add.reduce(self.weights * (targetAtomsCoordinates[:,0]*targetAtomsCoordinates[:,0] + targetAtomsCoordinates[:,1]*targetAtomsCoordinates[:,1]) )
        # return step data
        return index, (xx ,xy, xz, yy, yz, zz, rog)

    def combine(self, index, stepData):
        """
        analysis combine method called after each step.\n

        :Parameters:
            #. index (int): the index of the last calculated step
            #. stepData (object): the returned data from step method
        """
        Imin = min(stepData[0], stepData[3], stepData[5])
        Imax = max(stepData[0], stepData[3], stepData[5])
        Imid = [stepData[0], stepData[3], stepData[5]]
        Imid.pop(Imid.index(Imin))
        Imid.pop(Imid.index(Imax))
        Imid = Imid[0]
        # calculate average
        average = (stepData[0]+stepData[3]+stepData[5]) / 3.
        # moment of inertia
        self.results['moment_of_inertia_xx'][index] = stepData[0]
        self.results['moment_of_inertia_xy'][index] = stepData[1]
        self.results['moment_of_inertia_xz'][index] = stepData[2]
        self.results['moment_of_inertia_yy'][index] = stepData[3]
        self.results['moment_of_inertia_yz'][index] = stepData[4]
        self.results['moment_of_inertia_zz'][index] = stepData[5]
        # eccentricity = 0 for spherical objects
        self.results['eccentricity'][index] = 1-Imin/average
        # ratio_of_largest_to_smallest = 1 for spherical objects
        self.results['ratio_of_largest_to_smallest'][index] = Imax/Imin
        # semiaxis
        self.results["semiaxis_a"][index] = np.sqrt( 5.0/(2.0*self.totalWeight) * (Imax+Imid-Imin) )
        self.results["semiaxis_b"][index] = np.sqrt( 5.0/(2.0*self.totalWeight) * (Imax+Imin-Imid) )
        self.results["semiaxis_c"][index] = np.sqrt( 5.0/(2.0*self.totalWeight) * (Imid+Imin-Imax) )
        # radius_of_gyration is a measure of the distribution of the mass
        # of atomic groups or molecules that constitute the aqueous core
        # relative to its center of mass
        self.results['radius_of_gyration'][index] = np.sqrt(stepData[6])

    def finalize(self):
        """
        called once all the steps has been run.\n
        """
        pass
