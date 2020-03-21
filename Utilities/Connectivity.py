"""
This module contains the connectivity classes used to calculate the atoms connectivity such as
bonds, angles, dihedrals, etc.

.. inheritance-diagram:: pdbparser.Utilities.Connectivity
    :parts: 2

"""
# standard libraries imports
from __future__ import print_function

# external libraries imports
import numpy as np
import copy

# pdbparser library imports
from pdbparser.log import Logger
from pdbparser import pdbparser
from pdbparser.Utilities.Information import get_coordinates, get_records_database_property_values, get_records_attribute_values
from pdbparser.Utilities.Collection import is_integer

class Connectivity(object):
    '''
    It takes any pdbparser object and calculates the atoms connectivity (bonds, angles, dihedrals).
    '''
    def __init__(self, pdb):
        # set pdb
        self.set_pdb(pdb)

    def set_pdb(self, pdb):
        assert isinstance(pdb, pdbparser), Logger.error("pdb must be a pdbparser instance")
        self.__pdb = pdb
        # initialize variables
        self.initialize_variables()

    def initialize_variables(self):
        # bonds list
        self._bonds = []
        self._numberOfBonds = 0
        # angles list
        self._angles = []
        # dihedrals list
        self._dihedrals = []
        # molecules list
        self._molecules = []

    @property
    def bonds(self):
        return self._bonds

    @property
    def angles(self):
        return self._angles

    @property
    def dihedrals(self):
        return self._dihedrals

    @property
    def numberOfBonds(self):
        return np.sum([len(b) for b in self._bonds])

    @property
    def numberOfAngles(self):
        return len(self._angles)

    @property
    def numberOfDihedrals(self):
        return len(self._dihedrals)

    @property
    def molecules(self):
        return self._molecules

    def calculate_bonds(self, regressive=False, bondingRadii = None, tolerance = 0.25, maxNumberOfBonds=4):
        """
        calculate the bonds list of all the atoms.\n

        :Parameters:
            #. regressive (boolean): If regressive all bonds are counted even those of earlier atoms.
            #. bondingRadii (numpy.array): The distances defining a bond. Must have the same length as indexes.
            #. tolerance (float): The tolerance is defined as the bond maximum stretching
            #. maxNumberOfBonds (integer): Maximum number of bonds allowed per atom
        """
        assert isinstance(regressive, bool), Logger.error("regressive must be boolean")
        assert is_integer(maxNumberOfBonds), Logger.error("maxNumberOfBonds must be integer")
        maxNumberOfBonds = int(float(maxNumberOfBonds))
        assert maxNumberOfBonds>0, Logger.error("maxNumberOfBonds must be bugger than 0")
        # get indexes
        indexes = self.__pdb.indexes
        # get bond radii
        if bondingRadii is None:
            bondingRadii = np.array(get_records_database_property_values(indexes, self.__pdb, "covalentRadius"))
        # get coordinates
        coordinates = get_coordinates(indexes, self.__pdb)
        # initialize bonds
        self._bonds = []
        if regressive:
            indexes = range(coordinates.shape[0])
        else:
            indexes = range(coordinates.shape[0]-1)
        bc = self.__pdb.boundaryConditions
        for idx in indexes:
            if regressive:
                cRadii = 0.5*(bondingRadii + bondingRadii[idx])+tolerance
                distances = bc.real_distance(coordinates[idx], coordinates)
                # find where distance < cRadii
                connected = list( np.where(distances<=cRadii)[0])
                # remove self atom
                connected.remove(idx)
            else:
                cRadii = 0.5*(bondingRadii[idx+1:] + bondingRadii[idx])+tolerance
                distances = bc.real_distance(coordinates[idx], coordinates[idx+1:])
                # find where distance < cRadii
                connected = list( np.where(distances<=cRadii)[0]+idx+1 )
            assert len(connected)<=maxNumberOfBonds , Logger.error("record '%s' is found having more than %i bonds with %s " %(str(idx),maxNumberOfBonds, str(connected)) )
            # set values to connectivity array
            self._bonds.append(connected)
            # increment number of bonds
            self._numberOfBonds += len(connected)
        # add last atom bonds as empty list
        if not regressive:
            self._bonds.append([])

    def calculate_molecules(self):
        """
        calculate the angles list of all the atoms.\n
        """
        def build_chain(chains, key):
            chain=[idx for idx in chains[key]]
            chains[key] = []
            for c in chain:
                chain.extend(build_chain(chains, c))
            return chain
        # get bonds
        if not self._bonds:
            self.calculate_bonds()
        # build chains
        chains = {}
        for idx in self.__pdb.indexes:
            chains[idx]=self._bonds[idx]
        for idx in self.__pdb.indexes:
            chains[idx] = build_chain(chains, idx)
        # build molecules
        self._molecules = []
        for idx, c in chains.items():
            if len(c):
                mol = [idx]
                mol.extend(c)
                self._molecules.append( sorted(set(mol)) )

    def calculate_angles(self):
        """
        calculate the angles list of all the atoms.\n
        """
        # get bonds
        if not self._bonds:
            self.calculate_bonds()
        # get angles
        for atomIdx in self.__pdb.indexes:
            # get first bonded atoms list
            firstBonds = copy.copy(self._bonds[atomIdx])
            while firstBonds:
                firstBond = firstBonds.pop(0)
                for secondBond in firstBonds:
                    # update angles list
                    self._angles.append([firstBond,atomIdx,secondBond])
                # get second bonded atoms list
                secondBonds = copy.copy(self._bonds[firstBond])
                while secondBonds:
                    secondBond = secondBonds.pop(0)
                    # update angles list
                    self._angles.append([atomIdx,firstBond,secondBond])
        # get number of angles
        self._numberOfAngles = len(self._angles)

    def calculate_dihedrals(self):
        """
        calculate the dihedrals list of all the atoms.\n
        """
        # get angles
        if not self._angles:
            self.calculate_angles()
        # get dihedrals
        sortedDihedrals = []
        for angle in self._angles:
            # get first atom bonded atoms list indexes
            for idx in self._bonds[angle[0]]:
                dihedral = list(set( [idx,angle[0],angle[1],angle[2]] ))
                if len(dihedral) == 4 and dihedral not in sortedDihedrals:
                    sortedDihedrals.append( dihedral )
                    self._dihedrals.append([idx,angle[0],angle[1],angle[2]])
            # get last atom bonded atoms list indexes
            for idx in self._bonds[angle[2]]:
                dihedral = list(set( [angle[0],angle[1],angle[2],idx] ))
                if len(dihedral) == 4 and dihedral not in sortedDihedrals:
                    sortedDihedrals.append( dihedral )
                    self._dihedrals.append([angle[0],angle[1],angle[2],idx])

    def get_bonds(self, key = None):
        """
        get bonds lists using key attribute to match with the pdb attributes.\n

        :Parameters:
            #. key (str): any pdbparser.records attribute.

        :Returns:
            #. connectRecord (list): The first records in the bonds.
            #. connectedTo (list): List of lists where every item list is the atoms bonded to the connectRecords item of the same index in list.
        """
        # if connectivity not built yet
        if not self._bonds:
            self.calculate_bonds()
        # get key attributes
        if key is None:
            connectRecord = self.__pdb.indexes
            connectedTo = self._bonds
        else:
            connectRecord = get_records_attribute_values(self.__pdb.indexes, self.__pdb, key)
            connectedTo = [get_records_attribute_values(item, self.__pdb, key) for item in self._bonds]
        return connectRecord, connectedTo

    def get_angles(self, key = None):
        """
        get angles list using key attribute to match with the pdb attributes.\n

        :Parameters:
            #. key (str): any pdbparser.records attribute.

        :Returns:
            #. angles (list): The list of bonds.
        """
        # if connectivity not built yet
        if not self._bonds:
            self.calculate_angles()
        # get key attributes
        if key is None:
            angles = self._angles
        else:
            angles = [get_records_attribute_values(item, self.__pdb, key) for item in self._angles]
        return angles

    def get_dihedrals(self, key = None):
        """
        get dihedrals list using key attribute to match with the pdb attributes.\n

        :Parameters:
            #. key (str): any pdbparser.records attribute.

        :Returns:
            #. dihedrals (list): The list of dihedrals.
        """
        # if connectivity not built yet
        if not self._dihedrals:
            self.calculate_dihedrals()
        # get key attributes
        if key is None:
            dihedrals = self._dihedrals
        else:
            dihedrals = [get_records_attribute_values(item, self.__pdb, key) for item in self._dihedrals]

        return dihedrals

    def export_atoms(self, filePath, indexesOffset = 1, format = "NAMD_PSF", closeFile = True):
        """
        Exports atoms to ascii file.\n

        :Parameters:
            #. filePath (path): the file path.
            #. indexesOffset (int): atoms indexing starts from zero. this adds an offset
            #. format (str): The format of exportation. Exisiting formats are: NAMD_PSF,
        """
        try:
            fd = open(filePath, 'w')
        except:
            raise Logger.error( "cannot open file %r for writing" %filePath)

        if format is "NAMD_PSF":
            self.__NAMD_PSF_export_atoms__(fd, indexesOffset = indexesOffset)
        else:
            fd.close()
            raise Logger.error( "format %r is not defined" %format)
        # close file
        if closeFile:
            fd.close()

    def export_bonds(self, filePath, key = "atom_name", indexesOffset = 1, format = "NAMD_PSF", closeFile = True):
        """
        Exports bonds to ascii file.\n

        :Parameters:
            #. filePath (path): the file path.
            #. indexesOffset (int): atoms indexing starts from zero. this adds an offset. applies only to NAMD_PSF
            #. key (str): any pdbparser.records attribute. applies only to NAMD_TOP
            #. format (str): The format of exportation. Exisiting formats are: NAMD_PSF, NAMD_TOP
        """
        try:
            fd = open(filePath, 'w')
        except:
            raise Logger.error( "cannot open file %r for writing" %filePath)
        if format is "NAMD_PSF":
            self.__NAMD_PSF_export_bonds__(fd, indexesOffset = indexesOffset)
        elif format is "NAMD_TOP":
            self.__NAMD_TOP_export_bonds__(fd, key = key)
        else:
            fd.close()
            raise Logger.error( "format %r is not defined" %format)
        # close file
        if closeFile:
            fd.close()

    def export_angles(self, filePath, indexesOffset = 1, key = "atom_name", format = "NAMD_PSF", closeFile = True):
        """
        Exports angles to ascii file.\n

        :Parameters:
            #. filePath (path): the file path.
            #. indexesOffset (int): atoms indexing starts from zero. this adds an offset. applies only to NAMD_PSF
            #. key (str): any pdbparser.records attribute. applies only to NAMD_TOP
            #. format (str): The format of exportation. Exisiting formats are: NAMD_PSF, NAMD_TOP
        """
        try:
            fd = open(filePath, 'w')
        except:
            raise Logger.error( "cannot open file %r for writing" %filePath)

        if format is "NAMD_PSF":
            self.__NAMD_PSF_export_angles__(fd, indexesOffset = indexesOffset)
        elif format is "NAMD_TOP":
            self.__NAMD_TOP_export_angles__(fd, key = key)
        else:
            fd.close()
            raise Logger.error( "format %r is not defined" %format )

        # close file
        if closeFile:
            fd.close()

    def export_dihedrals(self, filePath, indexesOffset = 1,  key = "atom_name", format = "NAMD_PSF", closeFile = True):
        """
        Exports dihedrals to ascii file.\n

        :Parameters:
            #. filePath (path): the file path.
            #. indexesOffset (int): atoms indexing starts from zero. this adds an offset. applies only to NAMD_PSF
            #. key (str): any pdbparser.records attribute. applies only to NAMD_TOP
            #. format (str): The format of exportation. Exisiting formats are: NAMD_PSF, NAMD_TOP
        """
        try:
            fd = open(filePath, 'w')
        except:
            raise Logger.error( "cannot open file %r for writing" %filePath)
        if format is "NAMD_PSF":
            self.__NAMD_PSF_export_dihedrals__(fd, indexesOffset = indexesOffset)
        elif format is "NAMD_TOP":
            self.__NAMD_TOP_export_dihedrals__(fd, key = key)
        else:
            fd.close()
            raise Logger.error( "format %r is not defined" %format)
        # close file
        if closeFile:
            fd.close()

    def __NAMD_PSF_export_atoms__(self, fd, indexesOffset = 1):
        # write the title line with the number of bonds
        fd.write( "\n%8d !NATOM\n" %len(self.__pdb) )
        # get indexes
        indexes = self.__pdb.indexes
        # get atoms weights
        atomsWeights = np.array(get_records_database_property_values(indexes, self.__pdb, "atomicWeight"))
        # start writing bonds
        for idx in indexes:
            at = self.__pdb[idx]
            atomLine  = str( "%i"%(idx+indexesOffset)).rjust(8, " ")
            atomLine += str( " ") # empty space
            atomLine += str( "%s"%at["segment_identifier"]).ljust(4, " ")
            atomLine += str( " ") # empty space
            atomLine += str( "%i"%at["sequence_number"]).ljust(5, " ")
            atomLine += str( "%s"%at["residue_name"]).ljust(5, " ")
            atomLine += str( "%s"%at["atom_name"]).ljust(5, " ") # atome name
            atomLine += str( "%s"%at["atom_name"]).ljust(5, " ") # force field type
            atomLine += str( "  0.000000") # charge
            atomLine += str( "      ") # empty space
            atomLine += str( "%8.4f"%atomsWeights[idx])
            atomLine += str( "           0")
            atomLine += str( "\n")
            fd.write( atomLine )

    def __NAMD_PSF_export_bonds__(self, fd, indexesOffset = 1):
        # get bonds
        bonds = self.get_bonds()
        # write the title line with the number of bonds
        fd.write( "\n%8d !NBOND: bonds\n" %self._numberOfBonds )
        # start writing bonds
        numberOfBonds = 1
        for atomIdx in bonds[0]:
            for bondedIdx in bonds[1][atomIdx]:
                bondsStr = str("%8d" %(atomIdx+indexesOffset) +
                               "%8d" %(bondedIdx+indexesOffset) +
                               (not numberOfBonds%4)* "\n")
                # write to psf file
                fd.write( bondsStr )
                # increment number of bonds
                numberOfBonds += 1

    def __NAMD_TOP_export_bonds__(self, fd, key = "atom_name"):
        # get bonds
        connectRecord, connectedTo = self.get_bonds(key = key)
        # start writing bonds
        bonds = ""
        for idx in range(len(connectRecord)):
            for to in connectedTo[idx]:
                bonds += str("%s"%connectRecord[idx]).ljust(5) + " " + str("%s"%to).ljust(5) + "    "
            if bonds:
                fd.write( "BOND %s\n"%bonds )
                bonds = ""

    def __NAMD_PSF_export_angles__(self, fd, indexesOffset = 1):
        # get angles
        angles = self.get_angles()
        # write the title line with the number of angles
        fd.write( "\n%8d !NTHETA: angles\n" %self._numberOfAngles )
        # start writing angles
        numberOfAngles = 1
        for angle in angles:
            anglesStr = str("%8d" %(angle[0]+indexesOffset) +
                            "%8d" %(angle[1]+indexesOffset) +
                            "%8d" %(angle[2]+indexesOffset) +
                            (not numberOfAngles%3)* "\n")
            # write to psf file
            fd.write( anglesStr )
            # increment number of bonds
            numberOfAngles += 1

    def __NAMD_TOP_export_angles__(self, fd, key = "atom_name"):
        # get bonds
        angles = self.get_angles(key = key)
        # number of angles per line is 3
        count = 3
        angleLine = ""
        while angles:
            if count:
                angle = angles.pop(0)
                count -=1
                angleLine += str("%s"%angle[0]).ljust(5) + " " + str("%s"%angle[1]).ljust(5) + " " + str("%s"%angle[2]).ljust(5) +"    "
            else:
                fd.write( "ANGLE %s\n"%angleLine )
                angleLine = ""
                count = 3
        # last unsave angle
        if angleLine:
            fd.write( "ANGLE %s\n"%angleLine )

    def __NAMD_PSF_export_dihedrals__(self, fd, indexesOffset = 1):
        # get dihedrals
        dihedrals = self.get_dihedrals()
        # write the title line with the number of angles
        fd.write( "\n%8d !NPHI: dihedrals\n" %self.numberOfDihedrals )
        # start writing angles
        numberOfDihedrals = 1
        for dihedral in dihedrals:
            dihedralsStr = str("%8d" %(dihedral[0]+indexesOffset) +
                               "%8d" %(dihedral[1]+indexesOffset) +
                               "%8d" %(dihedral[2]+indexesOffset) +
                               "%8d" %(dihedral[3]+indexesOffset) +
                               (not numberOfDihedrals%2)* "\n")
            # write to psf file
            fd.write( dihedralsStr )
            # increment number of bonds
            numberOfDihedrals += 1

    def __NAMD_TOP_export_dihedrals__(self, fd, key = "atom_name"):
        # get bonds
        dihedrals = self.get_dihedrals(key = key)
        # number of dihedrals per line is 2
        count = 2
        dihedralLine = ""
        while dihedrals:
            if count:
                dihedral = dihedrals.pop(0)
                count -=1
                dihedralLine += str("%s"%dihedral[0]).ljust(5) + " " + str("%s"%dihedral[1]).ljust(5) + " " + str("%s"%dihedral[2]).ljust(5) + " " + str("%s"%dihedral[3]).ljust(5) + "    "
            else:
                fd.write( "DIHE %s\n"%dihedralLine )
                dihedralLine = ""
                count = 2
        # last unsave angle
        if dihedralLine:
            fd.write( "ANGLE %s\n"%dihedralLine )
