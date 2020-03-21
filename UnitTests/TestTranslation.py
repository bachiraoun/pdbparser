#!/Library/Frameworks/Python.framework/Versions/7.0/Resources/Python.app/Contents/MacOS/Python
from __future__ import print_function
import unittest
import random
import numpy as np

class TestTranslation(unittest.TestCase):

    def setUp(self):
        # get data
        import pdbparser
        self.__pdbData = pdbparser.pdbparser()
        self.__pdbData.records = __import__("Utilities.Database", fromlist=["__WATER__"]).__WATER__
        # get method
        self.__method = __import__("Utilities.Geometry", fromlist=["translate"]).translate


    def test_X(self):
        from Utilities.Information import get_coordinates
        originalCoordinates =  np.transpose( get_coordinates(self.__pdbData._range(), self.__pdbData) )
        # get random translation
        sign = np.sign(random.random()-0.5)
        value = sign*random.random()
        # translate and get data
        self.__method(self.__pdbData._range(), self.__pdbData, [value ,0,0])
        translatedCoordinates = np.transpose( get_coordinates(self.__pdbData._range(), self.__pdbData) )
        # assert translation
        self.assertTrue( np.sum(translatedCoordinates-[value ,0,0] - originalCoordinates) < 10e-6, msg = "Translation along X")


    def test_Y(self):
        from Utilities.Information import get_coordinates
        originalCoordinates =  np.transpose( get_coordinates(self.__pdbData._range(), self.__pdbData) )
        # get random translation
        sign = np.sign(random.random()-0.5)
        value = sign*random.random()
        # translate and get data
        self.__method(self.__pdbData._range(), self.__pdbData, [0,value,0])
        translatedCoordinates = np.transpose( get_coordinates(self.__pdbData._range(), self.__pdbData) )
        # assert translation
        self.assertTrue( np.sum(translatedCoordinates-[0,value,0] - originalCoordinates) < 10e-6, msg = "Translation along Y")


    def test_Z(self):
        from Utilities.Information import get_coordinates
        originalCoordinates =  np.transpose( get_coordinates(self.__pdbData._range(), self.__pdbData) )
        # get random translation
        sign = np.sign(random.random()-0.5)
        value = sign*random.random()
        # translate and get data
        self.__method(self.__pdbData._range(), self.__pdbData, [0,value,0])
        translatedCoordinates = np.transpose( get_coordinates(self.__pdbData._range(), self.__pdbData) )
        # assert translation
        self.assertTrue( np.sum(translatedCoordinates-[0,0,value] - originalCoordinates) < 10e-6, msg = "Translation along Z")


    def test_XYZ(self):
        from Utilities.Information import get_coordinates
        originalCoordinates =  np.transpose( get_coordinates(self.__pdbData._range(), self.__pdbData) )
        # get random translation
        signX = np.sign(random.random()-0.5)
        valueX = signX*random.random()
        signY = np.sign(random.random()-0.5)
        valueY = signY*random.random()
        signZ = np.sign(random.random()-0.5)
        valueZ = signZ*random.random()
        # translate and get data
        self.__method(self.__pdbData._range(), self.__pdbData, [valueX,valueY,valueZ])
        translatedCoordinates = np.transpose( get_coordinates(self.__pdbData._range(), self.__pdbData) )
        # assert translation
        self.assertTrue( np.sum(translatedCoordinates- [valueX,valueY,valueZ]- originalCoordinates) < 10e-6, msg = "Translation along XYZ")


def main():
    unittest.main()

if __name__ == "__main__":
    import sys, os
    path = os.path.join( os.getcwd().split("pdbparser")[0], "pdbparser")
    sys.path.insert(0,path)

    main()
