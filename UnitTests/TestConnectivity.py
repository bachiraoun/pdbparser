#!/Library/Frameworks/Python.framework/Versions/7.0/Resources/Python.app/Contents/MacOS/Python
from __future__ import print_function
import os
import unittest
import random
import copy
import numpy as np

class TestTranslation(unittest.TestCase):

    def setUp(self):
        # get data
        import pdbparser
        from Utilities.Collection import get_path

        pdbparserPath = get_path("pdbparser_path")
        self.__pdb = pdbparser.pdbparser(os.path.join(pdbparserPath,'Data','connectivityTestMolecule.pdb'))
        # get method
        self.__method = __import__("Utilities.Connectivity", fromlist=["Connectivity"]).Connectivity

        # results
        self.__bonds = [[1], [2], [3, 9], [5], [5, 7], [6], [], [8], [], [10], [11, 12], [], [13, 14, 15], [], [], []]
        self.__angles = [[0, 1, 2], [1, 2, 3], [1, 2, 9], [3, 2, 9], [2, 3, 5], [2, 9, 10], [3, 5, 6], [5, 4, 7], [4, 5, 6], [4, 7, 8], [9, 10, 11], [9, 10, 12], [11, 10, 12], [10, 12, 13], [10, 12, 14], [10, 12, 15], [13, 12, 14], [13, 12, 15], [14, 12, 15]]
        self.__dihedrals = [[0, 1, 2, 3], [0, 1, 2, 9], [1, 2, 3, 5], [1, 2, 9, 10], [5, 3, 2, 9], [3, 2, 9, 10], [2, 3, 5, 6], [2, 9, 10, 11], [2, 9, 10, 12], [6, 5, 4, 7], [5, 4, 7, 8], [9, 10, 12, 13], [9, 10, 12, 14], [9, 10, 12, 15], [11, 10, 12, 13], [11, 10, 12, 14], [11, 10, 12, 15]]

        # self.__pdb molecule sketch
        #     0
        #     |
        #     1       6--7--8
        #     |      /
        #     2--3--4
        #     |      \
        #     9       5
        #     |
        # 11--10
        #     |
        # 15--12--13
        #     |
        #     14

    def test_bonds(self):
        connectivity = self.__method(self.__pdb)
        # bonds
        connectivity.calculate_bonds()

        # assert all bonds exist
        bonds = connectivity.get_bonds()[1]
        trueBonds = copy.copy(self.__bonds)
        while bonds:
            bond = bonds.pop(0)
            self.assertTrue( bond in trueBonds)
            trueBonds.remove(bond)

        # assert no more bonds left
        self.assertTrue(not trueBonds)


    def test_angles(self):
        connectivity = self.__method(self.__pdb)
        # bonds
        connectivity.calculate_angles()

        # assert all bonds exist
        angles = connectivity.get_angles()
        trueAngles = copy.copy(self.__angles)
        while angles:
            angle = angles.pop(0)
            self.assertTrue( angle in trueAngles)
            trueAngles.remove(angle)

        # assert no more bonds left
        self.assertTrue(not trueAngles)


    def test_dihedrals(self):
        connectivity = self.__method(self.__pdb)
        # bonds
        connectivity.calculate_dihedrals()

        # assert all bonds exist
        dihedrals = connectivity.get_dihedrals()
        trueDihedrals = copy.copy(self.__dihedrals)
        while dihedrals:
            dihedral = dihedrals.pop(0)
            self.assertTrue( dihedral in trueDihedrals)
            trueDihedrals.remove(dihedral)

        # assert no more bonds left
        self.assertTrue(not trueDihedrals)


def main():
    unittest.main()

if __name__ == "__main__":
    import sys, os
    path = os.path.join( os.getcwd().split("pdbparser")[0], "pdbparser")
    sys.path.insert(0,path)

    main()
