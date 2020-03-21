"""
This module contains all geometry related methods.
"""

# standard libraries imports
from __future__ import print_function
import parser

# external libraries imports
import numpy as np

# pdbparser library imports
from .Information import *
from ..log import Logger
from ..Utilities.Collection import get_random_perpendicular_vector


def get_intersection(indexes_of, pdb_of, indexes_with, pdb_with, threshold = 3):
    """
        Calculates the intersection between two pdb files.\n
        :Parameters:
            #. indexes_of (list, tuple, numpy.ndarray): the indexes of pdb_of.
            #. pdb_of (pdbparser): the first pdb.
            #. indexes_with (list, tupe, numpy.ndarray): the indexes of pdb_with.
            #. pdb_with (pdbparser): the second pdb.
            #. threshold (float): the distance threshold defining an intersection.

        :Returns:
            #. interection (list): the indexes of pdb_of intersection with pdb_with.
    """
    # get coordinates
    coordsOf = get_coordinates(indexes_of, pdb_of)
    coordsWith = get_coordinates(indexes_with, pdb_with)
    # calculate distance and get indexes
    indexes = []
    for idx in range(len(coordsOf)):
        difference = coordsWith-coordsOf[idx]
        distance = np.sqrt(np.add.reduce(difference*difference,1))
        if len(np.where(distance<threshold)[0])>0:
            indexes.append(indexes_of[idx])
    # return indexes
    return indexes


def get_min_max(indexes, pdb):
    """
        Calculates the boundaries of a pdb.\n

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser): the pdb.

       :Returns:
            #. min_max (list): list of six elements, respectively min(X),max(X),min(Y),max(Y),min(Z),max(Z)
    """
    coords = get_coordinates(indexes, pdb)
    return [min(coords[:,0]),max(coords[:,0]), min(coords[:,1]),max(coords[:,1]), min(coords[:,2]),max(coords[:,2])]


def get_mean(indexes, pdb):
    """
        Calculates the mean position of a pdb.\n

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser): the pdb.

       :Returns:
            #. mean (list): list of three elements, respectively mean(X),mean(Y), mean(Z)
    """
    coords = get_coordinates(indexes, pdb)
    return [np.sum(coords[:,0])/len(coords[:,0]), np.sum(coords[:,1])/len(coords[:,1]), np.sum(coords[:,2])/len(coords[:,2])]


def get_closest_to_origin(indexes, pdb):
    """
        Finds the closest record to the origin of a pdb.\n

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser): the pdb.

       :Returns:
            #. index (integer): the closest record index
    """
    minIdx = np.argmin(np.sqrt( np.add.reduce(get_coordinates(indexes, pdb)**2, 1) ))
    return indexes[minIdx]


def get_farest_to_origin(indexes, pdb):
    """
        Finds the more far record to the origin of a pdb.\n

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser): the pdb.

       :Returns:
            #. index (integer): the farest record index
    """
    maxIdx = np.argmax(np.sqrt( np.add.reduce(get_coordinates(indexes, pdb)**2, 1) ))
    return indexes[maxIdx]


def get_rotation_matrix(rotation_vector, rotation_angle):
    """
        Calculates the rotation (3X3) matrix about an axis by a rotation angle.\n

        :Parameters:
            #. rotation_vector (list, tuple, numpy.ndarray): the rotation vector coordinates.
            #. rotation_angle (float): the rotation angle in rad.

       :Returns:
            #. rotation_matrix (numpy.ndarray): the (3X3) rotation matrix
    """
    rotation_angle = float(rotation_angle)
    axis = rotation_vector/np.sqrt(np.dot(rotation_vector , rotation_vector))
    a = np.cos(rotation_angle/2)
    b,c,d = -axis*np.sin(rotation_angle/2.)
    return np.array( [ [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                       [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                       [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c] ] , dtype = float)


def rotate(indexes, pdb , rotation_matrix):
    """
        Rotates the records of a pdb using a rotation matrix.\n

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be rotated.
            #. pdb (pdbparser): the pdb.
            #. rotation_matrix (numpy.ndarray): the (3X3) rotation matrix
    """
    # rotate records
    for index in indexes:
        atom = pdb.records[index]
        atom['coordinates_x'] , atom['coordinates_y'] , atom['coordinates_z'] =\
        np.dot( rotation_matrix, np.array( [atom['coordinates_x'] , atom['coordinates_y'] , atom['coordinates_z'] ] , dtype = float) )

def orient(indexes, pdb, axis, records_axis):
    """
        Rotates the records of a pdb in order to orient and align with the given axis.\n

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be rotated.
            #. pdb (pdbparser): the pdb.
            #. axis (list, tuple, numpy.ndarray): the axis vector to align with.
            #. records_axis (list, tuple, numpy.ndarray): the records axis defining an initial orientation.
    """
    records = pdb.records

    # normalize vector
    axisNorm = np.linalg.norm(axis)
    assert axisNorm>0, "axis returned 0 norm"
    axis = np.array(axis, dtype=np.float)/axisNorm

    # normalize records_axis
    records_axisNorm = np.linalg.norm(records_axis)
    assert records_axisNorm>0, "record_axis returned 0 norm"
    records_axis = np.array(records_axis, dtype=np.float)/records_axisNorm

    # calculate rotation Angle
    dotProduct = np.dot(records_axis, axis)

    if np.abs(dotProduct-1) <= 10e-6 :
        rotationAngle = 0
    elif np.abs(dotProduct+1) <= 10e-6 :
        rotationAngle = np.pi
    else:
        rotationAngle = np.arccos( dotProduct )
    if np.isnan(rotationAngle) or rotationAngle == 0 :
        return records
    #elif np.abs(rotationAngle-np.pi) <= 10e-6:
    #    # make slight change to the records_axis
    #    records_axis[0] += 0.01
    #    records_axis /= np.linalg.norm(records_axis)
    #    rotationAngle = np.arccos( np.dot(records_axis, axis) )

    ## calculate rotation axis
    #rotationAxis = np.cross(records_axis, axis)
    #rotationAxis /= np.linalg.norm(rotationAxis)
    # calculate rotation axis.
    if np.abs(rotationAngle-np.pi) <= 10e-6:
        rotationAxis = get_random_perpendicular_vector(records_axis)
    else:
        rotationAxis = np.cross(records_axis, axis)

    # calculate rotation matrix
    rotationMatrix  = get_rotation_matrix(rotationAxis, rotationAngle)

    # rotation clock-wise or counter clock-wise
    clockWise = np.abs( (np.dot(rotationMatrix, records_axis) - axis) ) < [1e-6, 1e-6, 1e-6]
    if not  clockWise.all() == True:
        rotationMatrix  = get_rotation_matrix(rotationAxis, -rotationAngle)

    # perform rotation
    rotate(indexes, pdb, rotationMatrix)


def translate(indexes, pdb, vector):
    """
        Translates the records of a pdb.\n

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be translated.
            #. pdb (pdbparser): the pdb.
            #. vector (list, tuple, numpy.ndarray): the translation vector.
    """
    for index in indexes:
        atom = pdb.records[index]
        atom['coordinates_x'] += vector[0]
        atom['coordinates_y'] += vector[1]
        atom['coordinates_z'] += vector[2]


def multiply(indexes, pdb, vector):
    """
        Multiply the records of a pdb by a vector.\n

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be multiplied.
            #. pdb (pdbparser): the pdb.
            #. vector (list, tuple, numpy.ndarray): the multiplication vector.
    """
    for index in indexes:
        atom = pdb.records[index]
        atom['coordinates_x'] *= vector[0]
        atom['coordinates_y'] *= vector[1]
        atom['coordinates_z'] *= vector[2]


def get_axis(indexes, pdb, method = None):
    """
        Calculates pdb records axis using a predefined method.\n

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be multiplied.
            #. pdb (pdbparser): the pdb.
            #. method (python function): method must take two arguments, first one is indexes and second is pdbparser instance. If None is given, the main principal axis will be calculated.

        :Returns:
            #. axis (numpy.ndarray): the records axis

    """
    if isinstance(method, (list, np.ndarray)):
        axis  = np.array(method, dtype = float)
        axis /= np.linalg.norm(axis)

    elif (method is None) or (method == get_principal_axis):
        _, _, _, _, axis, _, _ = get_principal_axis(indexes ,  pdb)

    elif method in globals().values():
        axis = method(indexes, pdb)

    else:
        raise "get axis is not possible, method %r should be either a list, numpy array, None, or an axis definition predefined method " %method

    return axis


def get_closest_farest_axis(indexes, pdb):
    """
        Axis calculation method.\n
        The axis is calculated simply using max-min formula for X/Y/Z axis
        where max is the maximum found distance to origin and min the minimum
        found distance to the origin.

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be multiplied.
            #. pdb (pdbparser): the pdb.

        :Returns:
            #. axis (numpy.ndarray): the normalized records axis
    """
    records = pdb.records
    minDistanceIndex = get_closest_to_origin(indexes, records)
    atom = records[minDistanceIndex]
    [minX, minY, minZ]  = [ atom['coordinates_x'] , atom['coordinates_y'] , atom['coordinates_z'] ]

    maxDistanceIndex = get_farest_to_origin(indexes, records)
    atom = records[maxDistanceIndex]
    [maxX, maxY, maxZ]  = [ atom['coordinates_x'] , atom['coordinates_y'] , atom['coordinates_z'] ]

    vect = [maxX-minX, maxY-minY, maxZ-minZ]

    return vect/np.linalg.norm(vect)


def get_principal_axis(indexes, pdb, weights=None):
    """
        Axis calculation method.\n
        Calculates the principal axis of the given records.

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be multiplied.
            #. pdb (pdbparser): the pdb.
            #. weights (numpy.ndarray, None): the list of weights for the COM calculation.
                                              Must be a numpy.ndarray of numbers of the same length as indexes.
                                              None is accepted for equivalent weighting.

        :Returns:
            #. center (numpy.ndarray): the geometric center of the records.
            #. eval1 (float): the biggest eigen value.
            #. eval2 (float): the second biggest eigen value.
            #. eval3 (float): the smallest eigen value.
            #. axis1 (numpy.ndarray): the principal axis corresponding to the biggest eigen value.
            #. axis2 (numpy.ndarray): the principal axis corresponding to the second biggest eigen value.
            #. axis3 (numpy.ndarray): the principal axis corresponding to the smallest eigen value.

    """
    # create coordinates array
    coord = get_coordinates(indexes, pdb)
    # multiply by weights
    if weights is not None:
        coord[:,0] *= weights
        coord[:,1] *= weights
        coord[:,2] *= weights
        norm = np.sum(weights)
    else:
        norm = len(indexes)
    # compute center
    center = np.sum(coord, 0)/norm
    # coord in center
    coord = coord - center
    # compute principal axis matrix
    inertia = np.dot(coord.transpose(), coord)
    # compute eigen values and eigen vectors
    # warning eigen values are not necessary ordered!
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
    e_values, e_vectors = np.linalg.eig(inertia)
    e_values = list(e_values)
    e_vectors = list(e_vectors.transpose())
    # get eval1 and axis1
    eval1 = max(e_values)
    vect1 = e_vectors.pop(e_values.index(eval1))
    e_values.remove(eval1)
    # get eval1 and axis1
    eval2 = max(e_values)
    vect2 = e_vectors.pop(e_values.index(eval2))
    e_values.remove(eval2)
    # get eval3 and axis3
    eval3 = e_values[0]
    vect3 = e_vectors[0]
    return center, eval1, eval2, eval3, vect1, vect2, vect3


def get_min_max_axis(indexes, pdb):
    """
        Axis calculation method.\n
        The axis is calculated simply using using max-min formula for X/Y/Z axis.

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be multiplied.
            #. pdb (pdbparser): the pdb.

        :Returns:
            #. axis (numpy.ndarray): the normalized records axis
    """
    [minX,maxX,minY,maxY,minZ,maxZ] = get_min_max(indexes, pdb)
    axis = [ maxX-minX, maxY-minY, maxZ-minZ ]
    return axis/np.linalg.norm(axis)


def get_atom_names_axis(indexes, pdb, origin, direction):
    """
        Axis calculation method.\n
        The axis is calculated simply using the geometric centre calculation of two sets of atoms names, origin and direction.

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be multiplied.
            #. pdb (pdbparser): the pdb.
            #. origin (list, tuple): atoms names. The mean coordinates values defines the axis origin
            #. direction (list, tuple): atoms names. The mean coordinates values defines the axis direction

        :Returns:
            #. axis (numpy.ndarray): the normalized records axis
    """
    # make sure origin and direction are lists
    if not isinstance(origin,list):
        origin = list(origin)
    if not isinstance(direction,list):
        direction = list(direction)
    # get atom names
    atomNames = get_records_attribute_values(indexes, pdb, "atom_name")
    # get indexes
    originIdx = [atomNames.index(name) for name in origin]
    directionIdx = [atomNames.index(name) for name in direction]
    # calculate mean position
    minX, minY, minZ = get_mean(originIdx, pdb)
    maxX, maxY, maxZ = get_mean(directionIdx, pdb)
    # calculate axis
    axis = [ maxX-minX, maxY-minY, maxZ-minZ ]
    return axis/np.linalg.norm(axis)


def get_mean_axis(indexes, pdb):
    """
        Axis calculation method.\n
        The axis defined between the origin and the geometric center of the pdb.

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be multiplied.
            #. pdb (pdbparser): the pdb.

        :Returns:
            #. axis (numpy.ndarray): the normalized records axis
    """

    axis = get_mean(indexes, pdb)
    return  axis/np.linalg.norm(axis)


def get_center(indexes, pdb, weights=None):
    """
        Calculates the center of a pdb.

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be multiplied.
            #. pdb (pdbparser): the pdb.
            #. weights (numpy.ndarray, None): the list of weights for the COM calculation.
                                              Must be a numpy.ndarray of numbers of the same length as indexes.
                                              None is accepted for equivalent weighting.

        :Returns:
            #. center (numpy.ndarray): the geometric center.
    """
    # create coordinates array
    coord = get_coordinates(indexes, pdb)
    # multiply by weights
    if weights is not None:
        coord[:,0] *= weights
        coord[:,1] *= weights
        coord[:,2] *= weights
        norm = float(np.sum(weights))
    else:
        norm = float(len(indexes))
    # compute center
    return np.sum(coord, 0)/norm


def get_geometric_center(indexes, pdb):
    """
        Calculates the geometric center of a pdb.

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be multiplied.
            #. pdb (pdbparser): the pdb.

        :Returns:
            #. center (numpy.ndarray): the geometric center.
    """
    return get_center(indexes=indexes, pdb=pdb, weights=None)


def get_satisfactory_records_indexes(indexes, pdb, expression):
    """
        It finds the records that satisfy the expression condition and return their indexes.\n
        This method uses parser module to compile an expression string.\n

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the records indexes of the pdb to be used.
            #. pdb (pdbparser): the pdb.
            #. expression (string): mathematic condition.\n
            e.g: 'np.sqrt(x**2 + y**2 + z**2) >= 15' \n
            (np.sqrt(x**2 + y**2 + z**2)>10) * (np.sqrt(x**2 + y**2 + z**2)<25)

        :Returns:
            #. indexes (list): the records indexes that satisfies the expression.
    """
    # evaluating expression
    code = parser.expr(expression).compile()
    # get coordinates and parse x, y and z
    coords = get_coordinates(indexes, pdb)
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    # evaluate expression
    goodIndexes = eval(code)
    # get all satisfactory atoms indexes
    return [indexes[idx] for idx in range(len(goodIndexes)) if  goodIndexes[idx] ]


def get_closest(indexes_of, pdb_of, indexes_to, pdb_to):
    """
        Calculates the minimum distance found between pdb_of and pdb_to records.

        :Parameters:
            #. indexes_of (list, tuple, numpy.ndarray): the indexes of pdb_of.
            #. pdb_of (pdbparser): the first pdb.
            #. indexes_to (list, tupe, numpy.ndarray): the indexes of pdb_to.
            #. pdb_to (pdbparser): the second pdb.

        :Returns:
            #. minimum (float): the minimum distance found.
            #. indexOf (integer): the index of pdb_of record that has been found to be the closest to a record in pdb_to.
            #. indexTo (integer): the index of pdb_to record that has been found to be the closest to a record in pdb_of.
    """
    # get coords
    ofCoords = np.transpose( get_coordinates(indexes_of, pdb_of) )
    toCoords = np.transpose( get_coordinates(indexes_to, pdb_to) )
    # calculate distances
    distancesList = []
    indexesList = []
    for idx in range(ofCoords.shape[0]):
        differences = toCoords - ofCoords[idx]
        distances = np.sqrt( np.add.reduce(differences*differences, 1) )
        minDistance = np.min(distances)
        distances.append( minDistance )
        indexesList.append( distances.index(distances) )
    # calculate minimum distance found
    minOfAll = np.min( distancesList )
    # get indexes
    indexOf = distanceList.index(minOfAll)
    indexTo = indexesList.index(indexOf)
    # return
    return minOfAll , indexOf , indexTo
