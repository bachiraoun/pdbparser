#!/Library/Frameworks/Python.framework/Versions/7.0/Resources/Python.app/Contents/MacOS/Python
import numpy as np
from numbers import Number

class PeriodicSimulationBox(object):
    """
    periodic simulation box class.
    by definition simulation box is in the positif quadrant
    """
    
    def __init__(self):
        self.__name = ''
        # self.directBasisVectors = [np.array(3,3), np.array(3,3), ... numberOfSteps]
        self.__directBasisVectors = []
        # self.reciprocalBasisVectors = [np.array(3,3), np.array(3,3), ... numberOfSteps]
        self.__reciprocalBasisVectors = []
        # self.directVolume = [..., numberOfSteps]
        self.__directVolume = []
        # self.directVolume = [..., numberOfSteps]
        self.__reciprocalVolume = []
        
    
    def __len__(self):
        return len(self.__directBasisVectors)
        
        
    def setVectors(self, vectors_array, index = None):
        """
        Creates new box parameters at time index
        index: the time index. default = None, to append 
        """
        if index is None:
            index = len(self.__directBasisVectors)
            
        # get directBasisVectors matrix
        if isinstance(vectors_array, Number):
            directBasisVectors = np.array( [ [vectors_array, 0, 0],\
                                             [0, vectors_array, 0],\
                                             [0, 0, vectors_array] ] )
        elif vectors_array.shape == (1,) or vectors_array.shape == (1,1):
            directBasisVectors = np.array( [ [vectors_array[0], 0, 0],\
                                             [0, vectors_array[0], 0],\
                                             [0, 0, vectors_array[0]] ] )
        elif vectors_array.shape == (3,) or vectors_array.shape == (3,1):
            directBasisVectors = np.array( [ [vectors_array[0], 0, 0],\
                                             [0, vectors_array[1], 0],\
                                             [0, 0, vectors_array[2]] ] )
        elif vectors_array.shape == (9,) or vectors_array.shape == (9,1):
            directBasisVectors = np.array( [ [vectors_array[0], vectors_array[1], vectors_array[2]],\
                                             [vectors_array[3], vectors_array[4], vectors_array[5]],\
                                             [vectors_array[6], vectors_array[7], vectors_array[8]] ] )
        elif vectors_array.shape == (3,3):
            directBasisVectors = vectors_array
        else:
            raise ValueError('incompatible vectors_array format')
        self.__directBasisVectors.insert(index, directBasisVectors)
        
        # calculate directVolume
        self.__directVolume.insert(index, np.dot( directBasisVectors[0,:], np.cross(directBasisVectors[1,:],directBasisVectors[2,:]) ))
           
        # calculate reciprocalBasisVectors
        reciprocalBasisVectors = np.linalg.inv(directBasisVectors)
        self.__reciprocalBasisVectors.insert(index, reciprocalBasisVectors)
        
        # calculate reciprocalVolume
        self.__reciprocalVolume.insert(index, np.dot( reciprocalBasisVectors[0,:], np.cross(reciprocalBasisVectors[1,:],reciprocalBasisVectors[2,:]) ))
        
        
    def deleteVectors(self, index):
        """
        removes box parameters at time index
        """
        self.__directBasisVectors.pop(index)
        self.__reciprocalBasisVectors.pop(index)
        self.__directVolume.pop(index)
        self.__reciprocalVolume.pop(index)
    
    
    def vectors(self, index = -1):
        """
        returns the basis (3,3) array vectors of the box at time index
        """
        return self.__directBasisVectors[index]
        
    
    def reciprocalVectors(self, index = -1):
        """
        returns the basis (3,3) array reciprocal vectors of the box at time index
        """
        return self.__reciprocalBasisVectors[index]
        
        
    def boxVolume(self, index = -1):
        """
        returns the volume of the box at time index
        """
        return self.__directVolume[index]
    
    
    def reciprocalBoxVolume(self, index = -1):
        """
        returns the reciprocal volume of the box at time index
        """
        return self.__reciprocalVolume[index]
    
    
    def realToBoxArray(self, real_array, index = -1):
        """
        array is the (N,3) coordinates of N atoms
        index is the time index.
        """
        return np.dot(real_array, self.reciprocalVectors(index))
   
   
    def boxToRealArray(self, box_array, index = -1):
        """
        box_array is the (N,3) in box coordinates of N atoms
        index is the time index.
        """
        return np.dot(box_array, self.vectors(index))
        
        
    def wrapBoxArray(self, box_array):
        """
        wrap all box_array coordinates inside the [0,1] box
        """
        return box_array % 1
                         
    
    def wrapRealArray(self, real_array, index = -1):
        """
        wrap all real coordinates array inside the real box
        index is the time index.
        """
        boxArray = self.realToBoxArray(real_array, index)
        return self.boxToRealArray( self.wrapBoxArray(boxArray) )
        
    
    def boxDistance(self, box_vector, box_array):
        """
        calculates the distance between a box_vector an all box_array's vectors
        index is the time index.
        """
        distance = np.sqrt( np.add.reduce( (self.wrapBoxArray(box_array)-self.wrapBoxArray(box_vector))**2, axis = 1) )
        return np.where(distance < 0.5, distance, 1-distance)
        
        
    def realDistance(self, real_vector, real_array, index = -1):
        """
        calculates the distance between a vector and all array's vectors
        index is the time index.
        """
        boxVector = self.realToBoxArray(real_vector, index)
        boxArray = self.realToBoxArray(real_array, index)
        return self.boxToRealArray( self.boxDistance(boxVector, boxArray) )
    
    
    def boxToIndexesHistogram(self, box_array, box_bin_length, index = -1):
        """
        calculates the histogram of real_array.
        histogram dimension is simulation box size at time index
        histogram values are real_array indexes + 1
        zeros in histogram means no entries
        box_bin_length is the histogram bin dimension in box coordinates system
        """
        bins = np.ceil(1.0/box_bin_length )
        return np.histogramdd(sample = box_array, bins = bins,\
                              range = [[0,1],[0,1],[0,1]],\
                              weights = range(1,box_array.shape[0]+1))[0]
    
    
    def realToIndexesHistogram(self, real_array, real_bin_length = 0.5, index = -1):
        """
        """
        # get coordinates in box space
        boxArray = self.realToBoxArray(real_array = real_array, index = index)
        # calculate the binSize in box space
        boxBinLength = self.realToBoxArray([real_bin_length, real_bin_length, real_bin_length], index)
        # build histogram of indexes
        return self.boxToIndexesHistogram(box_array = boxArray,\
                                          box_bin_length = boxBinLength,\
                                          index = index)
        
        
    
    def histogramIndexesWithin(self, histogram_shape, radius, index = -1):
        """
        """
        # get basis vectors at time index
        basisVectors = self.vectors(index)
        # build ogrid between -0.5 and 0.5
        x = np.linspace(-0.5, 0.5, histogram_shape[0], endpoint=False).reshape((histogram_shape[0],1,1))
        y = np.linspace(-0.5, 0.5, histogram_shape[1], endpoint=False).reshape((1,histogram_shape[1],1))
        z = np.linspace(-0.5, 0.5, histogram_shape[2], endpoint=False).reshape((1,1,histogram_shape[2]))
        # get all indexes
        indexes = np.transpose((x**2+y**2+z**2>=0).nonzero())
        # calculate distances 
        x = x.reshape((histogram_shape[0],1))
        y = y.reshape((histogram_shape[1],1))
        z = z.reshape((histogram_shape[2],1))
        distances = x[indexes[:,0]]*basisVectors[0] + y[indexes[:,1]]*basisVectors[1] + z[indexes[:,2]]*basisVectors[2]
        # return good indexes
        return indexes[ np.where(np.add.reduce(distances**2,1)<=radius**2) ]

    
    
    
     
    
if __name__ == "__main__":
    import sys
    sys.path.insert(0,"/Users/AOUN/Desktop/pdbparser")
    import pdbparser
    from Utilities import Array
    
    pdb = pdbparser.pdbparser("examples/Argon.pdb")

    sb = PeriodicSimulationBox()
    sb.setVectors( np.array([[77.395,0,0],[0,77.395,0],[0,0,77.395] ]) )
    
    realArray = Array(pdb.get_coordinates())
    boxArray = sb.wrapBoxArray( sb.realToBoxArray(realArray) )
    
    #minDist = []
    #for idx in range(len(pdb)-1):
    #    dist = realArray[idx+1:,:]-realArray[idx]
    #    dist = np.sqrt(np.add.reduce(dist**2,1))
    #    minDist.append(np.min(dist))
    #print np.min(minDist)
    
    #print realArray
    #print boxArray
    #print "\nbasis vectors: \n",sb.vectors()
    #print "\nreciprocal basis vectors: \n",sb.reciprocalVectors()
    
    boxBinLength = sb.realToBoxArray([0.75,0.75,0.75], index = -1)
    #print boxBinLength
    indexesHistogram = sb.boxToIndexesHistogram(box_array = boxArray, box_bin_length = boxBinLength)
    #print indexesHistogram.shape
    print np.where(indexesHistogram>100)
    print indexesHistogram.shape
    print sb.histogramIndexesWithin(indexesHistogram.shape, 14, index = -1)

    raise
    
    print "\n[0.5 , 0.5 , 0.5] to box: \n", np.ceil(1.0/sb.realToBoxArray(np.array([0.5 , 0.5 , 0.5])))
    
    print "\nwraped array: \n",sb.wrapRealArray(array)
    print "\nwraped box_array: \n",sb.wrapBoxArray(box_array)
    print "\noriginal array: \n",array
    print "\nbox_array: \n",box_array
    print "\nrecalculated original array: \n",sb.boxToRealArray(box_array)
    
