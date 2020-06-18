"""
This module contains a collection of methods used throughout the package.

.. inheritance-diagram:: pdbparser.Utilities.Collection
    :parts: 2
"""
# standard libraries imports
from __future__ import print_function
from collections import Counter
import os,sys,numbers
if sys.version_info[0] >= 3:
    basestring = str

# external libraries imports
import numpy as np
from numpy.fft import fft as FFT
from numpy.fft import ifft as iFFT

# pdbparser library imports
from pdbparser.log import Logger
from pdbparser.Utilities.Database import __atoms_database__, is_element


# parsing string to int or float, returns empty string if not possible
FLOAT = lambda s: float(s)      if s.replace('.','',1).replace('-','',1).strip().isdigit() else np.nan
INT   = lambda s: int(float(s)) if s.replace('.','',1).replace('-','',1).strip().isdigit() else -1

# compare two lists
compare_two_lists = lambda x, y: Counter(x) == Counter(y)


def is_integer(number):
    """
    check if number is convertible to integer.

    :Parameters:
        #. number (str, number): input number

    :Returns:
        #. result (bool): True if convertible, False otherwise
    """
    try:
        number = float(number)
    except:
        return False
    else:
        if number - int(number) < 10e-16:
            return True
        else:
            return False


def is_number(number):
    """
    check if number is convertible to float.

    :Parameters:
        #. number (str, number): input number

    :Returns:
        #. result (bool): True if convertible, False otherwise
    """
    try:
        float(number)
    except:
        return False
    else:
        return True


def get_path(key=None):
    """
    get all information needed about the script, the current, and the python executable path.

    :Parameters:
        #. key (None, string): the path to return. If not None, it can take any of the following\n
                   cwd:                 current working directory
                   script:              the script's total path
                   exe:                 python executable path
                   script_name:         the script name
                   relative_script_dir: the script's relative directory path
                   script_dir:          the script's absolute directory path
                   pdbparser:           pdbparser package path

    :Returns:
        #. path (dictionary, value): if key is not None it returns the value of paths dictionary key. Otherwise all the dictionary is returned.
    """
    import pdbparser
    # check key type
    if key is not None:
        assert isinstance(key, basestring), Logger.error("key must be a string of None")
        key=str(key).lower().strip()
    # create paths
    paths = {}
    paths["cwd"]                 = os.getcwd()
    paths["script"]              = sys.argv[0]
    paths["exe"]                 = os.path.dirname(sys.executable)
    pathname, scriptName         = os.path.split(sys.argv[0])
    paths["script_name"]         = scriptName
    paths["relative_script_dir"] = pathname
    paths["script_dir"]          = os.path.abspath(pathname)
    paths["pdbparser"]           = os.path.split(pdbparser.__file__)[0]
    # return paths
    if key is None:
        return paths
    else:
        assert key in paths, Logger.error("key is not defined")
        return paths[key]


def correlation(data1, data2=None):
    """
    Calculates the numerical correlation between two numpy.ndarray data.

    :Parameters:
        #. data1 (numpy.ndarray): the first numpy.ndarray. If multidimensional the correlation calculation is performed on the first dimension.
        #. data2 (None, numpy.ndarray): the second numpy.ndarray. If None the data1 autocorrelation is calculated.

    :Returns:
        #. correlation (numpy.ndarray): the result of the numerical correlation.
    """
    # The signal must not be empty.
    assert isinstance(data1, np.ndarray), Logger.error("data1 must be a non zero numpy.ndarray")
    # The length of data1 is stored in data1Length
    data1Length = len(data1)
    assert data1Length>0, Logger.error("data1 must be a non zero numpy.ndarray")
    # extendedLength = 2*len(data1)
    extendedLength = 2*data1Length
    # The FCA algorithm:
    # 1) computation of the FFT of data1 zero-padded until extendedLength
    # The computation is done along the 0-axis
    FFTData1 = FFT(data1,extendedLength,0)
    if data2 is None:
        # Autocorrelation case
        FFTData2 = FFTData1
    else:
        # 2) computation of the FFT of data2 zero-padded until extendedLength
        # The computation is  done along the 0-axis
        assert isinstance(data2, np.ndarray), Logger.error("if not None, data2 must be a numpy.ndarray")
        FFTData2 = FFT(data2,extendedLength,0)
    # 3) Product between FFT(data1)* and FFT(data2)
    FFTData1 = np.conjugate(FFTData1)*FFTData2
    # 4) inverse FFT of the product
    # The computation is done along the 0-axis
    FFTData1 = iFFT(FFTData1,len(FFTData1),0)
    # This refers to (1/(N-m))*Sab in the published algorithm.
    # This is the correlation function defined for positive indexes only.
    if len(FFTData1.shape) == 1:
        corr = FFTData1.real[:data1Length] / (data1Length-np.arange(data1Length))
    else:
        corr = np.add.reduce(FFTData1.real[:data1Length],1) / (data1Length-np.arange(data1Length))
    return corr


def get_atomic_form_factor(q, element, charge=0):
    """
        Calculates the Q dependant atomic form factor.\n
        :Parameters:
            #. q (list, tuple, numpy.ndarray): the q vector.
            #. element (str): the atomic element.
            #. charge (int): the expected charge of the element.

        :Returns:
            #. formFactor (numpy.ndarray): the calculated form factor.
    """
    assert is_element(element), "%s is not an element in database" %element
    element = str(element).lower()
    assert charge in __atoms_database__[element]['atomicFormFactor'], Logger.error("atomic form factor for element %s at with %s charge is not defined in database"%(element, charge))
    ff= __atoms_database__[element]['atomicFormFactor'][charge]
    a1= ff['a1']
    b1= ff['b1']
    a2= ff['a2']
    b2= ff['b2']
    a3= ff['a3']
    b3= ff['b3']
    a4= ff['a4']
    b4= ff['b4']
    c= ff['c']
    q = np.array(q)
    qOver4piSquare = (q/(4.*np.pi))**2
    t1=a1*np.exp(-b1*qOver4piSquare)
    t2=a2*np.exp(-b2*qOver4piSquare)
    t3=a3*np.exp(-b3*qOver4piSquare)
    t4=a4*np.exp(-b4*qOver4piSquare)
    return t1+t2+t3+t4+c


#def get_normalized_weighting(numbers, weights):
#    """
#    Calculates the normalized weighting scheme for a set of elements.\n
#    :Parameters:
#        #. numbers (dictionary): The numbers of elements dictionary. keys are the elements and values are the numbers of elements in the system
#        #. weights (dictionary): the weight of every element. keys are the elements and values are the weights. weights must have the same length.
#
#    :Returns:
#        #. normalizedWeights (dictionary): the normalized weighting scheme for every pair of elements.
#    """
#    assert isinstance(numbers, dict), Logger.error("numbers must be a dictionary where values are the number of elements")
#    assert isinstance(weights, dict), Logger.error("weights must be a dictionary where values are the weights of elements")
#    assert set(numbers.keys())==set(weights.keys()), Logger.error("numbers and weights must have the same dictionary keys. numbers:%s    weights:%s"%(numbers.keys(), weights.keys()))
#    elements = list(weights)
#    nelements = [float(numbers[el]) for el in elements]
#    totalNumberOfElements = sum(nelements)
#    molarFraction = [n/totalNumberOfElements for n in nelements]
#    # total weights
#    totalWeight = sum([molarFraction[idx]*weights[elements[idx]] for idx in range(len(elements))])**2
#    # calculate weights
#    normalizedWeights = {}
#    for idx1 in range(len(elements)):
#        el1 = elements[idx1]
#        b1 = weights[el1]
#        mf1 = molarFraction[idx1]
#        for idx2 in range(len(elements)):
#            el2 = elements[idx2]
#            b2 = weights[el2]
#            mf2 = molarFraction[idx2]
#            # get pair elements key
#            pair = el1+'-'+el2
#            if el2+'-'+el1 in normalizedWeights:
#                normalizedWeights[el2+'-'+el1] += mf1*mf2*b1*b2/totalWeight
#            else:
#                normalizedWeights[pair] = mf1*mf2*b1*b2/totalWeight
#    return normalizedWeights

def get_normalized_weighting(numbers, weights, pairsWeight=None):
    """
    Calculates the normalized weighting scheme for a set of elements.

    :Parameters:
        #. numbers (dictionary): The numbers of elements dictionary. keys are the elements and values are the numbers of elements in the system
        #. weights (dictionary): the weight of every element. keys are the elements and values are the weights. weights must have the same length as numbers.
        #. pairsWeight (None, dictionary): the customized interaction weight for element pairs. keys must be a tuple of elements pair and values are the weights.

    :Returns:
        #. normalizedWeights (dictionary): the normalized weighting scheme for every pair of elements.
    """
    if pairsWeight is None:
        pairsWeight = {}
    assert isinstance(pairsWeight, dict), Logger.error("pairsWeigth must be a dictionary where values are the weights of element pairs")
    assert isinstance(numbers, dict), Logger.error("numbers must be a dictionary where values are the number of elements")
    assert isinstance(weights, dict), Logger.error("weights must be a dictionary where values are the weights of elements")
    assert set(numbers)==set(weights), Logger.error("numbers and weights must have the same dictionary keys. numbers:%s    weights:%s"%(numbers.keys(), weights.keys()))
    # check paris weight list
    for p in list(pairsWeight):
        assert isinstance(p, tuple), Logger.error("pairsWeight keys must be tuple")
        assert len(p)==2, Logger.error("pairsWeight keys tuple length must be 2")
        assert p[0] in numbers, Logger.error("pairsWeight keys element '%s' is not defined in given numbers numbers dictionary"%p[0])
        assert p[1] in numbers, Logger.error("pairsWeight keys element '%s' is not defined in given numbers numbers dictionary"%p[0])
        if p[0]!=p[1]:
            assert (p[1],p[0]) not in pairsWeight, Logger.error("pairsWeight key %s is redundant. If (el1,el2) is a key (el2,el1) must not be given"%p)
    # get general system properties
    elements  = list(weights)
    nelements = [float(numbers[el]) for el in elements]
    assert all([n>=0 for n in nelements]), Logger.error("elements dictionary values must be all >=0")
    totalNumberOfElements = sum(nelements)
    molarFraction = [n/totalNumberOfElements for n in nelements]
    # compute total weights
    totalWeight = 0
    for idx1, el1 in enumerate(elements):
        b1  = weights[el1]
        mf1 = molarFraction[idx1]
        for idx2, el2 in enumerate(elements):
            b2 = weights[el2]
            mf2 = molarFraction[idx2]
            pw  = pairsWeight.get((el1,el2), pairsWeight.get((el2,el1),b1*b2) )
            totalWeight += mf1*mf2*pw
    # calculate normalized weights
    normalizedWeights = {}
    for idx1, el1 in enumerate(elements):
        b1  = weights[el1]
        mf1 = molarFraction[idx1]
        for idx2, el2 in enumerate(elements):
            b2  = weights[el2]
            mf2 = molarFraction[idx2]
            pw   = pairsWeight.get( (el1,el2), pairsWeight.get((el2,el1),b1*b2) )
            npw  = mf1*mf2*pw/totalWeight
            if el2+'-'+el1 in normalizedWeights:
                normalizedWeights[el2+'-'+el1] += npw
            else:
                normalizedWeights[el1+'-'+el2] = npw
    # return
    return normalizedWeights


def get_data_weighted_sum(data, numbers, weights):
    """
    Calculates the total weighted sum of all data.\n

    :Parameters:
        #. data (dictionary): The data dictionary. keys are the elements and values are the data.
        #. numbers (dictionary): The number of elements dictionary. keys are the elements and values are the number of elements in the system.
        #. weights (dictionary): The weight of every element. keys are the elements and values are the weights. weights must have the same length.

    :Returns:
        #. weightedSum (np.ndarray): the total weighted sum.
    """
    assert isinstance(data, dict), Logger.error("data must be a dictionary")
    assert isinstance(numbers, dict), Logger.error("numbers must be a dictionary where values are the number of elements")
    assert isinstance(weights, dict), Logger.error("weights must be a dictionary where values are the weights of elements")
    assert set(data.keys())==set(numbers.keys()) and set(numbers.keys())==set(weights.keys()), Logger.error("data, numbers and weights must have the same dictionary keys. data:%s    numbers:%s    weights:%s"%(list(data.keys()), list(numbers.keys()), list(weights.keys())))
    assert not len([False for d in data.values() if not isinstance(d, np.ndarray)]), Logger.error("data must be a dictionary where values are numpy.ndarray")
    s = data[list(data.keys())[0]].shape
    assert not len([False for d in data.values() if not d.shape==s]), Logger.error("data must be a dictionary where values are numpy.ndarray of same shape")
    elements = list(weights.keys())
    nelements = [float(numbers[el]) for el in elements]
    totalNumberOfElements = sum(nelements)
    molarFraction = [n/totalNumberOfElements for n in nelements]
    # total weights
    totalWeight = sum([molarFraction[idx]*weights[elements[idx]] for idx in range(len(elements))])**2
    # calculate weighted sum
    weightedSum = np.zeros(list(data.values())[0].shape)
    for eidx in range(len(elements)):
        el = elements[eidx]
        b = weights[el]
        mf = molarFraction[eidx]
        weightedSum += mf*b/totalWeight*data[el]
    return weightedSum


def generate_sphere_points(radius, nPoints, center = [0,0,0]):
    """
    Returns list of 3d coordinates of points on a sphere using the
    Golden Section Spiral algorithm.
    """
    points = []
    inc = np.pi * (3 - np.sqrt(5))
    offset = 2.0 / float(nPoints)
    for k in range(int(nPoints)):
        y = k * offset - 1 + (offset / 2)
        r = np.sqrt(1 - y*y)
        phi = k * inc
        points.append( [radius*np.cos(phi)*r + center[0],
                        radius*y + center[1],
                        radius*np.sin(phi)*r + center[2]] )

    return points


def generate_circle_points(radius, nPoints, center = [0,0,0]):
    """
    Returns list of 3d coordinates of points on a circle using the
    """
    return [ (center[0]+np.cos(2*np.pi/nPoints*x)*radius,
              center[1]+np.sin(2*np.pi/nPoints*x)*radius,
              center[2])
              for x in xrange(0,nPoints)]


def generate_sphere_points(radius, nPoints, center = [0,0,0]):
    """
    Returns list of coordinates on a sphere using the Golden Section Spiral
    algorithm.
    """
    increm = np.pi * (3. - np.sqrt(5.))
    points = []
    offset = 2. / float(nPoints)
    for k in range(nPoints):
        y = k * offset - 1. + (offset / 2.)
        r = np.sqrt(1 - y*y)
        phi = k * increm
        x = radius * np.cos(phi)*r + center[0]
        y = radius * y + center[1]
        z = radius * np.sin(phi)*r + center[2]
        points.append( [x,y,z] )
    return points

def get_random_perpendicular_vector(vector):
    """
    Get random perpendicular vector to a given vector.

    :Parameters:
        #. vector (numpy.ndarray, list, set, tuple): the vector to compute a random perpendicular vector to it

    :Returns:
        #. perpVector (numpy.ndarray): the perpendicular vector
    """
    vectorNorm = np.linalg.norm(vector)
    assert vectorNorm, Logger.error("vector returned 0 norm")
    # easy cases
    if np.abs(vector[0])<1e-6:
        return np.array([1,0,0], dtype=np.float32)
    elif np.abs(vector[1])<1e-6:
        return np.array([0,1,0], dtype=np.float32)
    elif np.abs(vector[2])<1e-6:
        return np.array([0,0,1], dtype=np.float32)
    # generate random vector
    randVect = 1-2*np.random.random(3)
    randvect = np.array([vector[idx]*randVect[idx] for idx in range(3)])
    # get perpendicular vector
    perpVector = np.cross(randvect,vector)
    # return
    return np.array(perpVector/np.linalg.norm(perpVector), dtype=np.float32)


def get_orthonormal_axes(vector1, vector2, force = False):
    """
    returns 3 orthonormal axes calculated from given 2 vectors.
    vector1 direction is unchangeable.
    vector2 is adjusted in the same plane (vector1, vector2) in order to be perpendicular with vector1.
    """
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    assert vector1.shape in ((3,),(3,1))
    assert vector2.shape in ((3,),(3,1))

    # normalizing
    vector1 /= np.linalg.norm(vector1)
    vector2 /= np.linalg.norm(vector2)

    # calculate vector3
    vector3 = np.cross(vector1, vector2)
    if (np.linalg.norm(vector3) <= 10e-6):
        if not force:
            raise Exception("computing orthogonal vector is impossible with linear vectors")
        else:
            randSign = np.sign(1-np.random.random())
            randSign = np.array([randSign,-randSign, np.sign(1-np.random.random())])
            vector2 += 0.01*randSign*vector2
            vector3 = np.cross(vector1, vector2)

    vector2 = np.cross(vector3, vector1)
    return np.array( [vector1, vector2, vector3] )




def generate_asymmetric_sphere_points(radius, point_2Ddimension, center = [0,0,0]):
    """
    Returns list of 3d coordinates of points on a sphere
    point_2Ddimension = [xwidth, ywidth]
    radius is the sphere radius and dephines zwidth in somehow
    """
    midCirclePeriphery = 2.*np.pi*radius
    points = generate_circle_points(radius, int(midCirclePeriphery/point_2Ddimension[0]))

    # get alpha angle
    alpha = float(point_2Ddimension[1])/float(radius)

    # get radii
    radii = [ ( radius*np.cos(a), radius*np.sin(a) )
              for a in np.linspace(alpha,
                           np.pi/2,
                           np.pi/2/alpha,
                           endpoint=True)]

    # add points
    for t in radii:
        circlePeriphery = 2.*np.pi*t[0]
        points.extend( generate_circle_points(t[0], int(circlePeriphery/point_2Ddimension[0]), center = [0,0,t[1]]) )
        points.extend( generate_circle_points(t[0], int(circlePeriphery/point_2Ddimension[0]), center = [0,0,-t[1]]) )

    return points



"""
Collections of special objects
"""
# Priority dictionary using binary heaps
# David Eppstein, UC Irvine, 8 Mar 2002
class PriorityDictionary(dict):
    def __init__(self):
        '''
        Initialize PriorityDictionary by creating binary heap
        of pairs (value,key).  Note that changing or removing a dict entry will
        not remove the old pair from the heap until it is found by smallest() or
        until the heap is rebuilt.
        '''
        self.__heap = []
        dict.__init__(self)

    def smallest(self):
        '''
        Find smallest item after removing deleted items from heap.
        '''

        if len(self) == 0:
            # smallest of empty PriorityDictionary
            raise IndexError

        heap = self.__heap
        while heap[0][1] not in self or self[heap[0][1]] != heap[0][0]:
            lastItem = heap.pop()
            insertionPoint = 0
            while 1:
                smallChild = 2*insertionPoint+1
                if smallChild+1 < len(heap) and \
                        heap[smallChild] > heap[smallChild+1]:
                    smallChild += 1
                if smallChild >= len(heap) or lastItem <= heap[smallChild]:
                    heap[insertionPoint] = lastItem
                    break
                heap[insertionPoint] = heap[smallChild]
                insertionPoint = smallChild
        return heap[0][1]

    def __iter__(self):
        '''
        Create destructive sorted iterator of PriorityDictionary.
        '''
        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]
        return iterfn()

    def __setitem__(self,key,val):
        '''
        Change value stored in dictionary and add corresponding
        pair to heap.  Rebuilds the heap if the number of deleted items grows
        too large, to avoid memory leakage.
        '''
        dict.__setitem__(self,key,val)
        heap = self.__heap
        if len(heap) > 2 * len(self):
            self.__heap = [(v,k) for k,v in self.iteritems()]
            self.__heap.sort()  # builtin sort likely faster than O(n) heapify
        else:
            newPair = (val,key)
            insertionPoint = len(heap)
            heap.append(None)
            while insertionPoint > 0 and \
                    newPair < heap[(insertionPoint-1)//2]:
                heap[insertionPoint] = heap[(insertionPoint-1)//2]
                insertionPoint = (insertionPoint-1)//2
            heap[insertionPoint] = newPair

    def setdefault(self,key,val):
        '''
        Reimplement setdefault to call our customized __setitem__.
        '''
        if key not in self:
            self[key] = val
        return self[key]



# Dijkstra's algorithm for shortest paths
# David Eppstein, UC Irvine, 4 April 2002
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228
def Dijkstra(G,start,end=None):
    """
    Find shortest paths from the start vertex to all
    vertices nearer than or equal to the end.

    The input graph G is assumed to have the following
    representation: A vertex can be any object that can
    be used as an index into a dictionary.  G is a
    dictionary, indexed by vertices.  For any vertex v,
    G[v] is itself a dictionary, indexed by the neighbors
    of v.  For any edge v->w, G[v][w] is the length of
    the edge.  This is related to the representation in
    <http://www.python.org/doc/essays/graphs.html>
    where Guido van Rossum suggests representing graphs
    as dictionaries mapping vertices to lists of neighbors,
    however dictionaries of edges have many advantages
    over lists: they can store extra information (here,
    the lengths), they support fast existence tests,
    and they allow easy modification of the graph by edge
    insertion and removal.  Such modifications are not
    needed here but are important in other graph algorithms.
    Since dictionaries obey iterator protocol, a graph
    represented as described here could be handed without
    modification to an algorithm using Guido's representation.

    Of course, G and G[v] need not be Python dict objects;
    they can be any other object that obeys dict protocol,
    for instance a wrapper in which vertices are URLs
    and a call to G[v] loads the web page and finds its links.

    The output is a pair (D,P) where D[v] is the distance
    from start to v and P[v] is the predecessor of v along
    the shortest path from s to v.

    Dijkstra's algorithm is only guaranteed to work correctly
    when all edge lengths are positive. This code does not
    verify this property for all edges (only the edges seen
    before the end vertex is reached), but will correctly
    compute shortest paths even for some graphs with negative
    edges, and will raise an exception if it discovers that
    a negative edge has caused it to make a mistake.
    """

    D = {}	# dictionary of final distances
    P = {}	# dictionary of predecessors
    Q = PriorityDictionary()   # est.dist. of non-final vert.
    Q[start] = 0

    for v in Q:
        D[v] = Q[v]
        if v == end:
            break
        for w in G[v]:
            vwLength = D[v] + G[v][w]
            if w in D:
                if vwLength < D[w]:
                    raise ValueError

            #Dijkstra: found better path to already-final vertex
            elif w not in Q or vwLength < Q[w]:
                Q[w] = vwLength
                P[w] = v

    return (D,P)

def shortest_path(G,start,end):
    """
    Find a single shortest path from the given start vertex
    to the given end vertex.
    The input has the same conventions as Dijkstra().
    The output is a list of the vertices in order along
    the shortest path.
    """

    D,P = Dijkstra(G,start,end)
    Path = []
    while True:
        Path.append(end)
        if end == start:
            break
        end = P[end]
    Path.reverse()
    return Path



def get_lattice_vectors(a, b, c, alpha, beta, gamma):
    """
    compute lattice basis vectors (boundary conditions vectors)
    given crystallographic lattice parameters. Convention is c along z axis

    [latice reciprocal matrix] -  3*3 - matrix
    ::

        a*  b*cos(gama*)  c*cos(beta*)
        0   b*sin(gama*) -c*sin(beta*)cosAlpha
        0       0         1/c

    latice matrix vectors -  3*3 - matrix
    ::
        inv( [latice reciprocal matrix] )


    :Parameters:
        #. a (Number): Length of a vector.
        #. b (Number): Length of b vector.
        #. c (Number): Length of c vector.
        #. alpha (Number): Angle between b and c in degrees.
        #. beta (Number):  Angle between a and c in degrees.
        #. gamma (Number): Angle between a and b in degrees.

    :Returns:
        #. basis (numpy.ndarray): (3X3) numpy array basis.
        #. rbasis (numpy.ndarray): (3X3) numpy array normalized by volume reciprocal basis.
    """
    assert is_number(a), "a must be a number"
    assert is_number(b), "b must be a number"
    assert is_number(c), "c must be a number"
    assert is_number(alpha), "alpha must be a number"
    assert is_number(beta), "beta must be a number"
    assert is_number(gamma), "gamma must be a number"
    a     = float(a)
    b     = float(b)
    c     = float(c)
    alpha = float(alpha)
    beta  = float(beta)
    gamma = float(gamma)
    assert a>0, "a must be >0"
    assert b>0, "b must be >0"
    assert c>0, "c must be >0"
    assert 0<alpha<180, "alpha must be >0 and < 180"
    assert 0<beta<180, "beta must be >0 and < 180"
    assert 0<gamma<180, "gamma must be >0 and < 180"
    ### alternative implementation might be as the following
    ### anglesRad = np.radians([alpha, beta, gamma])
    ### cosAlpha, cosBeta, cosGamma = np.cos(anglesRad)
    ### sinAlpha, sinBeta, sinGamma = np.sin(anglesRad)
    ### val = (cosAlpha * cosBeta - cosGamma) / (sinAlpha * sinBeta)
    ### gammaStar = np.arccos(val)
    ### x = [a * sinBeta, 0.0, a * cosBeta]
    ### y = [-b * sinAlpha * np.cos(gammaStar),b * sinAlpha * np.sin(gammaStar),b * cosAlpha,]
    ### z = [0.0, 0.0, float(c)]
    ### # return
    ### return np.array(x), np.array(y), np.array(z)
    # transform to rad
    alpha = alpha * np.pi/180
    beta  = beta  * np.pi/180
    gamma = gamma * np.pi/180
    # compute trigonometry
    cosAlpha = np.cos(alpha)
    sinAlpha = np.sin(alpha)
    cosBeta  = np.cos(beta)
    sinBeta  = np.sin(beta)
    cosGamma = np.cos(gamma)
    sinGamma = np.sin(gamma)
    # compute volume
    vol=a*b*c*np.sqrt(1.-cosAlpha**2-cosBeta**2-cosGamma**2+2.*cosAlpha*cosBeta*cosGamma)
    # normalize by volume
    ar=b*c*sinAlpha/vol
    br=a*c*sinBeta/vol
    cr=a*b*sinGamma/vol
    # compute useful quantities
    cosalfar=(cosBeta*cosGamma-cosAlpha)/(sinBeta*sinGamma)
    cosbetar=(cosAlpha*cosGamma-cosBeta)/(sinAlpha*sinGamma)
    cosgamar=(cosAlpha*cosBeta-cosGamma)/(sinAlpha*sinBeta)
    alfar=np.arccos(cosalfar)
    betar=np.arccos(cosbetar)
    gamar=np.arccos(cosgamar)
    # compute reciprocal basis
    rbasis = np.array( [ [ar, br*np.cos(gamar), cr*np.cos(betar)], \
                       [ 0.0, br*np.sin(gamar), -cr*np.sin(betar)*cosAlpha], \
                       [ 0.0, 0.0, 1.0/c] ], dtype = float)
    # compute normal basis
    basis = np.linalg.inv(rbasis)
    # return
    return basis, rbasis
