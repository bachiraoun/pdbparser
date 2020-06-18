"""
This module contains the definition of all possible boundary conditions simulation boxes.

.. inheritance-diagram:: pdbparser.Utilities.BoundaryConditions
    :parts: 2
"""
# standard libraries imports
from __future__ import print_function
from numbers import Number

# external libraries imports
import numpy as np

# pdbparser library imports
import pdbparser
from pdbparser.log import Logger
from pdbparser.Utilities.Collection import get_lattice_vectors
from pdbparser.Utilities.Information import get_coordinates
from pdbparser.Utilities.Geometry import get_min_max


class InfiniteBoundaries(object):
    """
    simulation box class for no boundary conditions universe.
    by definition simulation box is in the positive quadrant
    """
    def __init__(self, *args, **kwargs):
        # simulation box name
        self._name = ''
        # self.directBasisVectors = [np.array(3,3), np.array(3,3), ... numberOfSteps]
        self._directBasisVectors = []
        # self.angles = [alpha(b,c), beta(c,a), gamma(a,b)]
        self._angles = []
        # self.reciprocalBasisVectors = [np.array(3,3), np.array(3,3), ... numberOfSteps]
        self._reciprocalBasisVectors = []
        # self.directVolume = [..., numberOfSteps]
        self._directVolume = []
        # self.directVolume = [..., numberOfSteps]
        self._reciprocalVolume = []

    def __len__(self):
        return len(self._directBasisVectors)

    @property
    def directBasisVectors(self):
        """ Get directBasisVectors list."""
        return self._directBasisVectors

    def set_vectors(self, vectorsArray = None, index = None):
        """
        Creates new box parameters at time index.

        :Parameters:
            #. vectorsArray (number, numpy.ndarray): The box vectors. if number is given, the box is considered cubic of length vectorsArray.
               If numpy.ndarray is given, the shape can be (1,) and its similar to number, if shape is (3,) its an Orthorhombic box, if shape is (9,) it can be anything.
            # index (None, Integer): The time index.
        """
        self._directBasisVectors.append(None)

        # calculate directVolume
        self._directVolume.append(None)

        # calculate reciprocalBasisVectors
        self._reciprocalBasisVectors.append(None)

        # calculate reciprocalVolume
        self._reciprocalVolume.append(None)

    def set_vectors_using_abc_alpha_beta_gamma(self, a=None, b=None, c=None, alpha=None, beta=None, gamma=None, index = None):
        """
        Creates new box parameters at time index using a, b, c, alpha, beta, gamma values
        instead of vectors. Convention is c along z axis

        :Parameters:
            #. a (Number): Length of a vector.
            #. b (Number): Length of b vector.
            #. c (Number): Length of c vector.
            #. alpha (Number): Angle between b and c in degrees.
            #. beta (Number): Angle between c and a in degrees.
            #. gamma (Number): Angle between a and b in degrees.
            #. index (integer, None): the index of the vectors.
        """
        self.set_vectors()

    def delete_vectors(self, index):
        """
        Removes box parameters at time index

        :Parameters:
            # index (None, Integer): The time index.
        """
        self._directBasisVectors.pop(index)
        self._reciprocalBasisVectors.pop(index)
        self._directVolume.pop(index)
        self._reciprocalVolume.pop(index)

    def get_vectors(self, index = -1):
        """
        Get the basis (3,3) array vectors of the box at time index.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'vectors definition' is ambiguous")

    def get_a(self, index = -1):
        """
        Get a the first vector length.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'a definition' is ambiguous")

    def get_b(self, index = -1):
        """
        Get b the second vector length.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'b definition' is ambiguous")

    def get_c(self, index = -1):
        """
        Get c the second vector length.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'c definition' is ambiguous")

    def get_angles(self, index = -1):
        """
        Get the basis alpha(b,c), beta(c,a), gamma(a,b) angles in rad at time index.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'angles definition' is ambiguous")

    def get_alpha(self, index = -1):
        """
        returns alpha(b,c) angle in rad at time index.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'alpha definition' is ambiguous")

    def get_beta(self, index = -1):
        """
        returns beta(c,a) angle in rad at time index.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'beta definition' is ambiguous")

    def get_gamma(self, index = -1):
        """
        returns gamma(a,b) angle in rad at time index.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'gamma definition' is ambiguous")

    def get_reciprocal_vectors(self, index = -1):
        """
        returns the basis (3,3) array reciprocal vectors of the box at time index.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'reciprocal vectors' definition is ambiguous")

    def get_box_volume(self, index = -1):
        """
        returns the volume of the box at time index.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'box volume' definition is ambiguous")

    def get_box_real_center(self, index = -1):
        """
        returns the box center coordinates at time index
        For InfiniteBoundaries it's always [0,0,0]
        """
        return np.array([0,0,0])

    def get_reciprocal_box_volume(self, index = -1):
        """
        returns the reciprocal volume of the box at time index.
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'reciprocal box volume' definition is ambiguous")

    def real_to_box_array(self, realArray, index = -1):
        """
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'real to box' definition is ambiguous")

    def box_to_real_array(self, boxArray, index = -1):
        """
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'box to real' definition is ambiguous")

    def fold_box_array(self, boxArray):
        """
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'fold' definition is ambiguous")

    def fold_real_array(self, realArray, index = -1):
        """
        For InfiniteBoundaries calling this method will return the same realArray.

        :parameter:
            #. realArray (numpy.ndarray): The coordinates array.
        """
        return realArray

    def box_difference(self, boxVector, boxArray):
        """
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'box difference' definition is ambiguous")

    def real_difference(self, realVector, realArray, index = -1):
        """
        Computes the real difference between a vector and an array

        :Parameters:
            #. realVector (numpy.ndarray): the [x,y,z] vector
            #. realArray (numpy.ndarray): the (N,3) coordinates array.
        """
        return realArray-realVector

    def box_distance(self, boxVector, boxArray):
        """
        For InfiniteBoundaries calling this method will raise an error.
        """
        raise Logger.error("Infinite universe 'box distance' definition is ambiguous")

    def real_distance(self, realVector, realArray, index = -1):
        """
        Computes the real distance between a vector and an array

        :Parameters:
            #. realVector (numpy.ndarray): the [x,y,z] vector
            #. realArray (numpy.ndarray): the (N,3) coordinates array.
        """
        difference = self.real_difference(realVector, realArray, index)
        return np.sqrt( np.add.reduce( difference**2, axis = 1) )



class PeriodicBoundaries(InfiniteBoundaries):
    """
    stores box dimension at different time indexes.
    """
    def __init__(self, vectorsArray=None, *args, **kwargs):
         # The base class constructor.
        super(PeriodicBoundaries,self).__init__(*args, **kwargs)
        # set vectors
        if vectorsArray is not None:
            self.set_vectors(vectorsArray)


    def set_vectors(self, vectorsArray, index = None):
        """
        Create a new set of box vectors.\n

        :Parameters:
            #. vectorsArray (number, list, tuple, numpy.ndarray): the box vectors.
               If number is given, the box is considered cubic of length vectorsArray.
               If numpy.ndarray is given, the shape can be (1,) and its similar to number, if shape is (3,) its an Orthorhombic box, if shape is (9,) it can be anything.
            #. index (integer, None): the index of the vectors.
        """
        if index is None:
            index = len(self)
        # check vectorsArray type
        if isinstance(vectorsArray, (list, tuple)):
            vectorsArray = np.array(vectorsArray, dtype = np.float)
        else:
            assert isinstance(vectorsArray, (Number, np.ndarray)), "vectors can be either None, number, numpy.ndarray"
        # get directBasisVectors matrix
        if isinstance(vectorsArray, Number):
            directBasisVectors = np.array( [ [vectorsArray, 0, 0],\
                                             [0, vectorsArray, 0],\
                                             [0, 0, vectorsArray] ], dtype = np.float )
        elif vectorsArray.shape == (1,) or vectorsArray.shape == (1,1):
            directBasisVectors = np.array( [ [vectorsArray[0], 0, 0],\
                                             [0, vectorsArray[0], 0],\
                                             [0, 0, vectorsArray[0]] ], dtype = np.float )
        elif vectorsArray.shape == (3,) or vectorsArray.shape == (3,1):
            directBasisVectors = np.array( [ [vectorsArray[0], 0, 0],\
                                             [0, vectorsArray[1], 0],\
                                             [0, 0, vectorsArray[2]] ], dtype = np.float )
        elif vectorsArray.shape == (9,) or vectorsArray.shape == (9,1):
            directBasisVectors = np.array( [ [vectorsArray[0], vectorsArray[1], vectorsArray[2]],\
                                             [vectorsArray[3], vectorsArray[4], vectorsArray[5]],\
                                             [vectorsArray[6], vectorsArray[7], vectorsArray[8]] ], dtype = np.float )
        elif vectorsArray.shape == (3,3):
            directBasisVectors = np.array(vectorsArray, dtype = np.float)
        else:
            raise ValueError('incompatible vectorsArray format')
        self._directBasisVectors.insert(index, directBasisVectors)
        # calculate angles alpha, beta, gamma
        vectLength = np.array([np.linalg.norm(directBasisVectors[idx,:]) for idx in range(3)])
        normalizedVectors = np.array([directBasisVectors[idx,:]/vectLength[idx] for idx in range(3)])
        angles = [ np.arccos(np.dot(normalizedVectors[1],normalizedVectors[2])),
                   np.arccos(np.dot(normalizedVectors[2],normalizedVectors[0])),
                   np.arccos(np.dot(normalizedVectors[0],normalizedVectors[1])) ]
        self._angles.insert(index, angles)
        # calculate directVolume
        self._directVolume.insert(index, np.dot(directBasisVectors[0,:], np.cross(directBasisVectors[1,:],directBasisVectors[2,:]) ))
        # calculate reciprocalBasisVectors
        reciprocalBasisVectors = np.linalg.inv(directBasisVectors)
        self._reciprocalBasisVectors.insert(index, reciprocalBasisVectors)
        # calculate reciprocalVolume
        self._reciprocalVolume.insert(index, np.dot( reciprocalBasisVectors[0,:], np.cross(reciprocalBasisVectors[1,:],reciprocalBasisVectors[2,:]) ))

    def set_vectors_using_abc_alpha_beta_gamma(self,a,b,c,alpha,beta,gamma, index=None):
        """
        Creates new box parameters at time index using a, b, c, alpha, beta, gamma values
        instead of vectors. Convention is c along z axis

        :Parameters:
            #. a (Number): Length of a vector.
            #. b (Number): Length of b vector.
            #. c (Number): Length of c vector.
            #. alpha (Number): Angle between b and c in degrees.
            #. beta (Number): Angle between c and a in degrees.
            #. gamma (Number): Angle between a and b in degrees.
            #. index (integer, None): the index of the vectors.
        """
        basis, rbasis = get_lattice_vectors(a, b, c, alpha, beta, gamma)
        self.set_vectors(basis, index=index)

    def set_vectors_using_lattice_parameters(self, *args, **kwargs):
        """alias to set_vectors_using_abc_alpha_beta_gamma
        """
        return self.set_vectors_using_abc_alpha_beta_gamma(*args, **kwargs)

    def get_box_real_center(self, index = -1):
        """
        Get the box real center position.

        :Parameters:
            #. index (integer): the index of the vectors.

        :Returns:
            #. center (numpy.ndarray): the box center.
        """
        return self.box_to_real_array(np.array([.5,.5,.5]), index)

    def get_reciprocal_box_volume(self, index = -1):
        """
        Get the reciprocal box volume.

        :Parameters:
            #. index (integer): the index of the vectors.

        :Returns:
            #. volume (float): the reciprocal box volume
        """
        return self._reciprocalVolume[index]

    def get_reciprocal_vectors(self, index = -1):
        """
        Get the reciprocal space box vectors.

        :Parameters:
            #. index (integer): the index of the vectors.

        :Returns:
            #. vectors (numpy.ndarray): the reciprocal space box volume of shape (3,3)
        """
        return self._reciprocalBasisVectors[index]

    def get_vectors(self, index = -1):
        """
        Get the real space box vectors.

        :Parameters:
            #. index (integer): the index of the vectors.

        :Returns:
            #. vectors (numpy.ndarray): the real box volume of shape (3,3)
        """
        return self._directBasisVectors[index]

    def get_a(self, index = -1):
        """
        Get a the first vector length.

        :Parameters:
            #. index (integer): the index of the vectors.

        :Returns:
            #. a (float): the vector length
        """
        return np.linalg.norm(self._directBasisVectors[index][0])

    def get_b(self, index = -1):
        """
        Get b the second vector length.

        :Parameters:
            #. index (integer): the index of the vectors.

        :Returns:
            #. b (float): the vector length
        """
        return np.linalg.norm(self._directBasisVectors[index][1])

    def get_c(self, index = -1):
        """
        Get c the second vector length.

        :Parameters:
            #. index (integer): the index of the vectors.

        :Returns:
            #. c (float): the vector length
        """
        return np.linalg.norm(self._directBasisVectors[index][2])

    def get_angles(self, index = -1):
        """
        Get alpha, beta, gamma angles in rad at time index between respectively the box vectors (b,c) (c,a) and (a,b).

        :Parameters:
            #. index (integer): the index of the vectors.

        :Returns:
            #. angles (list): [alpha, beta, gamma] the angles in radian
        """
        return self._angles[index]

    def get_alpha(self, index = -1, degrees=False):
        """
        Get alpha angle in rad at time index between box vectors (b,c).

        :Parameters:
            #. index (integer): the index of the vectors.
            #. degrees (boolean): whether to convert to degrees

        :Returns:
            #. alpha (float): the angle in rad
        """
        ang = self._angles[index][0]
        if degrees:
            ang *= 180./np.pi
        return ang

    def get_beta(self, index = -1, degrees=False):
        """
        Get beta angle in rad at time index between box vectors (c,a).

        :Parameters:
            #. index (integer): the index of the vectors.
            #. degrees (boolean): whether to convert to degrees

        :Returns:
            #. beta (float): the angle in rad
        """
        ang = self._angles[index][1]
        if degrees:
            ang *= 180./np.pi
        return ang

    def get_gamma(self, index = -1, degrees=False):
        """
        Get gamma angle in rad at time index between box vectors (a,b).

        :Parameters:
            #. index (integer): the index of the vectors.
            #. degrees (boolean): whether to convert to degrees

        :Returns:
            #. gamma (float): the angle in rad
        """
        ang = self._angles[index][2]
        if degrees:
            ang *= 180./np.pi
        return ang

    def get_box_volume(self, index = -1):
        """
        Get the real space box volume.

        :Parameters:
            #. index (integer): the index of the vectors.

        :Returns:
            #. volume (float): the real space box volume
        """
        return self._directVolume[index]

    def real_to_box_array(self, realArray, index = -1):
        """
        Transforms array from real coordinates to box [0,1[.

        :Parameters:
            #. realArray (numpy.ndarray): the coordinates in real space
            #. index (integer): the index of the vectors.

        :Returns:
            #. boxArray (numpy.ndarray): the coordinates in box system [0,1[.
        """
        shape = realArray.shape
        if len(shape) == 2:
            assert shape[1] == 3, "realArray must have three columns for x, y and z coordinates"
        else:
            assert len(shape) == 1 and len(realArray) == 3, "if realArray is a point vector it should be of length 3 for x, y and z coordinates"
            realArray  = realArray.reshape(-1,3)
        rbasis    = self.get_reciprocal_vectors(index)
        boxArray  = np.empty(realArray.shape)
        boxArray[:,0] = realArray[:,0]*rbasis[0,0] + realArray[:,1]*rbasis[1,0] + realArray[:,2]*rbasis[2,0]
        boxArray[:,1] = realArray[:,0]*rbasis[0,1] + realArray[:,1]*rbasis[1,1] + realArray[:,2]*rbasis[2,1]
        boxArray[:,2] = realArray[:,0]*rbasis[0,2] + realArray[:,1]*rbasis[1,2] + realArray[:,2]*rbasis[2,2]
        return boxArray
        #return np.dot(realArray, self.get_reciprocal_vectors(index))

    def box_to_real_array(self, boxArray, index = -1):
        """
        Transforms array from box coordinates [0,1[ to real space.

        :Parameters:
            #. boxArray (numpy.ndarray): the box coordinates.
            #. index (integer): the index of the vectors.

        :Returns:
            #. realArray (numpy.ndarray): the coordinates in real space.
        """
        shape = boxArray.shape
        if len(shape) == 2:
            assert shape[1] == 3, "boxArray must have three columns for x, y and z coordinates"
        else:
            assert len(shape) == 1 and len(boxArray) == 3, "if boxArray is a point vector it should be of length 3 for x, y and z coordinates"
            boxArray  = boxArray.reshape(-1,3)
        basis     = self.get_vectors(index)
        realArray = np.empty(boxArray.shape)
        realArray[:,0] = boxArray[:,0]*basis[0,0] + boxArray[:,1]*basis[1,0] + boxArray[:,2]*basis[2,0]
        realArray[:,1] = boxArray[:,0]*basis[0,1] + boxArray[:,1]*basis[1,1] + boxArray[:,2]*basis[2,1]
        realArray[:,2] = boxArray[:,0]*basis[0,2] + boxArray[:,1]*basis[1,2] + boxArray[:,2]*basis[2,2]
        return realArray
        #return np.dot(boxArray, self.get_vectors(index))

    def fold_box_array(self, boxArray):
        """
        Folds all box coordinates between [0,1[

        :Parameters:
            #. boxArray (numpy.ndarray): the box coordinates.

        :Returns:
            #. foldedArray (numpy.ndarray): the folded into box array.
        """
        return boxArray % 1

    def fold_real_array(self, realArray, index = -1):
        """
        Folds all real coordinates between [0,boxLength[

        :Parameters:
            #. realArray (numpy.ndarray): the real space coordinates.

        :Returns:
            #. foldedArray (numpy.ndarray): the folded into real coordinates box array.
        """
        boxArray = self.real_to_box_array(realArray, index)
        return self.box_to_real_array( self.fold_box_array(boxArray), index )

    def box_difference(self, boxVector, boxArray):
        """
        Get vectorial difference between boxArray and boxVectors points in box space

        :Parameters:
            #. boxVector (numpy.ndarray): the box space coordinates to subtract.
            #. boxArray (numpy.ndarray): the box space coordinates to subtract from.

        :Returns:
            #. difference (numpy.ndarray): the box space difference vectors array.
        """
        difference = self.fold_box_array(boxArray) - self.fold_box_array(boxVector)
        s = np.sign(difference)
        a = np.abs(difference)
        d = np.where(a < 0.5, a, 1-a)
        return s*d

    def real_difference(self, realVector, realArray, index = -1):
        """
        Get vectorial difference between realArray and realVector points in box space

        :Parameters:
            #. realVector (numpy.ndarray): the real space coordinates to subtract.
            #. realArray (numpy.ndarray): the real space coordinates to subtract from.

        :Returns:
            #. difference (numpy.ndarray): the real space difference vectors array.
        """
        boxVector = self.real_to_box_array(realVector, index)
        boxArray = self.real_to_box_array(realArray, index)
        return self.box_to_real_array( self.box_difference(boxVector, boxArray), index )

    def box_distance(self, boxVector, boxArray):
        """
        Get distance between boxArray and boxVectors points in box space

        :Parameters:
            #. boxVector (numpy.ndarray): the box space coordinates to subtract.
            #. boxArray (numpy.ndarray): the box space coordinates to subtract from.

        :Returns:
            #. difference (numpy.ndarray): the box space distance vectors array.
        """
        difference = self.box_difference(boxVector, boxArray).reshape((-1,3))
        return np.sqrt( np.add.reduce( difference**2, axis = 1) )

    def real_distance(self, realVector, realArray, index = -1):
        """
        Get distance between realArray and realVector points in box space

        :Parameters:
            #. realVector (numpy.ndarray): the real space coordinates to subtract.
            #. realArray (numpy.ndarray): the real space coordinates to subtract from.

        :Returns:
            #. difference (numpy.ndarray): the real space distance vectors array.
        """
        difference = self.real_difference(realVector, realArray, index).reshape((-1,3))
        return np.sqrt( np.add.reduce( difference**2, axis = 1) )

    def box_to_indexes_histogram(self, boxArray, boxBinLength, index = -1):
        """
        calculates the histogram of real_array.
        histogram dimension is simulation box size at time index
        histogram values are real_array indexes + 1
        zeros in histogram means no entries
        box_bin_length is the histogram bin dimension in box coordinates system
        """
        bins = np.ceil(1.0/boxBinLength )
        return np.histogramdd(sample = boxArray, bins = bins,\
                              range = [[0,1],[0,1],[0,1]],\
                              weights = range(1,box_array.shape[0]+1))[0]

    def real_to_indexes_histogram(self, realArray, realBinLength = 0.5, index = -1):
        """
        """
        # get coordinates in box space
        boxArray = self.real_to_box_array(realArray = realArray, index = index)
        # calculate the binSize in box space
        boxBinLength = self.real_to_box_array([realBinLength, realBinLength, realBinLength], index)
        # build histogram of indexes
        return self.boxToIndexesHistogram(boxArray = boxArray,\
                                          boxBinLength = boxBinLength,\
                                          index = index)

    def histogram_indexes_within(self, histogramShape, radius, index = -1):
        """
        """
        # get basis vectors at time index
        basisVectors = self.get_vectors(index)
        # build ogrid between -0.5 and 0.5
        x = np.linspace(-0.5, 0.5, histogramShape[0], endpoint=False).reshape((histogramShape[0],1,1))
        y = np.linspace(-0.5, 0.5, histogramShape[1], endpoint=False).reshape((1,histogramShape[1],1))
        z = np.linspace(-0.5, 0.5, histogramShape[2], endpoint=False).reshape((1,1,histogramShape[2]))
        # get all indexes
        indexes = np.transpose((x**2+y**2+z**2>=0).nonzero())
        # calculate distances
        x = x.reshape((histogram_shape[0],1))
        y = y.reshape((histogram_shape[1],1))
        z = z.reshape((histogram_shape[2],1))
        distances = x[indexes[:,0]]*basisVectors[0] + y[indexes[:,1]]*basisVectors[1] + z[indexes[:,2]]*basisVectors[2]
        # return good indexes
        return indexes[ np.where(np.add.reduce(distances**2,1)<=radius**2) ]



class BoundaryConditions(object):
    """
    This class constructs a simulation box for pdbparser file.
    According to the records coordinates, it calculates the best fit a,b, and c vectors of a simulation box
    """

    def __init__(self, pdb):
        """
        Initialise Box class.

        :Parameters:
            #. pdb (pdbparser): pdbparser instance.
        """
        self.set_pdb(pdb)


    def set_pdb(self, pdb):
        """
        set a new pdb. all the variables will be reinitialized.

        :Parameters:
            #. pdb (pdbparser): pdbparser instance.
        """
        assert isinstance(pdb, pdbparser.pdbparser)
        self.__pdb = pdb

        # initialize variables
        self.initialize_variables()


    def initialize_variables(self):
        self.__a = None
        self.__b = None
        self.__c = None
#         self.__alpha = None
#         self.__beta = None
#         self.__gamma = None

    def __set_a__(self, a):
        if len(a) == 1:
            self.__a = np.array([float(a),0,0])
        elif len(a) ==3:
            self.__a = np.array([float(a[0]), float(a[1]), float(a[2])])
        else:
            raise pdbparser.Logger.error( "'a' should be a float or a three dimensions vector")

    def __set_b__(self, b):
        if len(b) == 1:
            assert float(b)
            self.__b = np.array([0,float(b),0])
        elif len(b) == 3:
            self.__b = np.array([float(b[0]), float(b[1]), float(b[2])])
        else:
            raise pdbparser.Logger.error( "'b' should be a float or a three dimensions vector")

    def __set_c__(self, c):
        if len(c) == 1:
            assert float(c)
            self.__c = np.array([0,0,float(c)])
        elif len(c) == 3:
            self.__c = np.array([float(c[0]), float(c[1]), float(c[2])])
        else:
           raise pdbparser.Logger.error( "'c' should be a float or a three dimensions vector")

    @property
    def vectors(self):
        return self.__a, self.__b, self.__c

    def set_vectors(self, a, b, c):
        """
        set simultaneously the box a,b, and c vectors.

        :Parameters:
            #. a (float, list, tuple, numpy.array): it can be either a float number or 3 dimensions vector.
            #. b (float, list, tuple, numpy.array): it can be either a float number or 3 dimensions vector.
            #. c (float, list, tuple, numpy.array): it can be either a float number or 3 dimensions vector.
        """
        self.__set_a__(a)
        self.__set_b__(b)
        self.__set_c__(c)

    def calculate_vectors(self, lattice = "triclinic", bin = 5.0):
        """
        Calculates a, b and c box vectors.

        :Parameters:
            #. lattice (string): force symmetry rules to the box structure.
               Lattice can be [triclinic, orthorhombic, cubic]
            #. bin (float): the space grid bin size, used to calculate the box vectors.
        """
        if lattice not in ["triclinic", "orthorhombic", "cubic"]:
            pdbparser.Logger.error('lattice argument should be one of ["triclinic", "orthorhombic", "cubic"]')
            raise

        bin = float(bin)

        coords = get_coordinates(self.__pdb.indexes, self.__pdb)
        x      = coords[:,0]
        y      = coords[:,1]
        z      = coords[:,2]
        self.__calculate_vector__(x, y, z, bin)
        print("\n\n")
        self.__calculate_vector__(y, x, z, bin)
        print("\n\n")
        self.__calculate_vector__(z, y, x, bin)

        self.__pdb.visualize()


    def __calculate_vector__(self, direction, dim0, dim1, bin = 2.0):
        # get arrays
        direction = np.array(direction)
        dim0 = np.array(dim0)
        dim0 = np.array(dim1)

        # get min and max of dim
        minDim0 = np.min(dim0)
        minDim1 = np.min(dim1)
        maxDim0 = np.max(dim0)
        maxDim1 = np.max(dim1)

        # construct histogram
        histHedges = np.arange(0, np.max([maxDim0-minDim0, maxDim1-minDim1]), bin)
        bins       = np.zeros(len(histHedges))
        occurrence = np.zeros(len(histHedges))

        #print(histHedges)

        # get grid
        grid = np.mgrid[minDim0:maxDim0:bin, minDim1:maxDim1:bin].reshape((2,-1))

        # get direction max distance
        for idx in range(grid.shape[1]):
            dim0Down = np.where( dim0 >= grid[:,idx][0] )[0]
            dim0Up   = np.where( dim0 <= grid[:,idx][0]+bin )[0]
            dim1Down = np.where( dim1 >= grid[:,idx][1] )[0]
            dim1Up   = np.where( dim1 <= grid[:,idx][1]+bin )[0]

            indexes = list( set(dim0Down) & set(dim0Up) & set(dim1Down) & set(dim1Up) )
            coords = direction[indexes]

            if len(coords):
                radius  = np.sqrt(np.sum(grid[:,idx]**2))
                histIdx = int(radius/bin)
                bins[histIdx] += np.max(coords) - np.min(coords)
                occurrence[histIdx] += 1
                #print(radius, , histHedges[int(radius/bin)])
                #print([grid[:,idx][0],grid[:,idx][0]+bin], [grid[:,idx][1],grid[:,idx][1]+bin], np.max(coords) - np.min(coords))

            # delete used indexes for rapidity
            dim0 = np.delete(dim0, indexes)
            dim1 = np.delete(dim1, indexes)
            direction = np.delete(direction, indexes)

        notZeroIdxs = np.where(occurrence>0)[0]
        bins[notZeroIdxs] /= occurrence[notZeroIdxs]

        print(bins)








if __name__ == "__main__":
    import numpy as np
    boxSize = np.array([10,10,10])
    print("box size:", boxSize, " - boxCenter:", boxSize/2.)
    print("================================================\n")

    print("Calculate distances:\n====================")

    # construct box
    box = PeriodicSimulationBox()
    box.set_vectors(boxSize)

    # general tests
    arrayVect = np.array([[0,0,0],[0.5,0.5,0.5],[1,1,1],[2,2,2],[-1,-1,-1],[-2,-2,-2],
                          [1.1,1.1,1.1],[-1.1,-1.1,-1.1],[3.3,3.3,3.3],[-3.3,-3.3,-3.3],
                          [11,11,11],[-11,-11,-11],[22,22,22],[-22,-22,-22]] ).reshape(-1,3)
    positions = np.array([[0,0,0],[3.5,3.5,3.5],[-3.5,-3.5,-3.5]] ).reshape(-1,3)
    for pos in positions:
        for idx in range(arrayVect.shape[0]):
            diff = box.real_difference(pos, arrayVect[idx])
            dist = box.real_distance(pos, arrayVect[idx])
            print("general --> ","pos:", pos, "  array:", arrayVect[idx], "  diff:",diff, "  dist:",dist)
    # commuting vector and array
    pos = np.array([1,1,1])
    arr = np.array([2,2,2])
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("commuting --> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    diff = box.real_difference(arr, pos)
    dist = box.real_distance(arr, pos)
    print("commuting --> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)

    # at box center
    pos  = boxSize/2.
    arr  = boxSize/2.
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box center--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = -boxSize/2.
    arr  = -boxSize/2.
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box center--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize/2.+0.01
    arr  = boxSize/2.
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box center--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize/2.-0.01
    arr  = boxSize/2.
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box center--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize/2.
    arr  = boxSize/2.+0.01
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box center--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize/2.
    arr  = boxSize/2.-0.01
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box center--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)


    # at box edge
    pos  = boxSize
    arr  = boxSize
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = -boxSize
    arr  = -boxSize
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize+0.01
    arr  = boxSize
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize-0.01
    arr  = boxSize
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize
    arr  = boxSize+0.01
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize
    arr  = boxSize-0.01
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = -boxSize
    arr  = -boxSize/2.
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize+0.01
    arr  = boxSize/2.
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize-0.01
    arr  = boxSize/2.
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize
    arr  = boxSize/2.+0.01
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = boxSize
    arr  = boxSize/2.-0.01
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("at box edge--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    # close points
    pos  = np.array([2,2,2])
    arr  = np.array([2,2,2])
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("close points--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = np.array([2,2,2])
    arr  = np.array([2.1,2.1,2.1])
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("close points--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = np.array([12,12,12])
    arr  = np.array([2,2,2])
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("close points--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)
    pos  = np.array([12,12,12])
    arr  = np.array([1.0,1.9,1.9])
    diff = box.real_difference(pos, arr)
    dist = box.real_distance(pos, arr)
    print("close points--> ", "pos:", pos, "  array:", arr, "  diff:",diff, "  dist:",dist)


    # real to box
    print("\n")
    print("Calculate realToBox and foldToBox:\n===================================")
    # at box edges
    pos  = np.array([0,0,0])
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = boxSize
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = -boxSize
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = 2*boxSize
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = -2*boxSize
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = 3*boxSize
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = -3*boxSize
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = np.array([0,0,0])+0.01
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = np.array([0,0,0])-0.01
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = boxSize+0.01
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = -boxSize +0.01
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = 2*boxSize+0.01
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = -2*boxSize+0.01
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = 3*boxSize+0.01
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = -3*boxSize+0.01
    print("at box edge--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    # at center
    pos  = boxSize/2.
    print("at box center--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = -boxSize/2.
    print("at box center--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = boxSize/2.+0.01
    print("at box center--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = -boxSize/2.+0.01
    print("at box center--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = boxSize/2.-0.01
    print("at box center--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))
    pos  = -boxSize/2.-0.01
    print("at box center--> ", "real position:", pos, "  box position:", box.real_to_box_array(pos), "  fold real:", box.fold_real_array(pos), "  fold box:", box.fold_box_array(box.real_to_box_array(pos)))


    box = PeriodicSimulationBox()
    box.set_vectors(np.array([30,30,30]))
    a = np.array([ -9.2572997 , 12.97077196,  -2.76041037])
    b = np.array([-11.18078246 , 11.17413013 ,  0.52458186])
