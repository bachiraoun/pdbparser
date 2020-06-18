"""
This module contains classes used to construct new pdb systems.

.. inheritance-diagram:: pdbparser.Utilities.Construct
    :parts: 2
"""
# standard libraries imports
from __future__ import print_function
import copy, parser, sys
if sys.version_info[0] >= 3:
    basestring = str

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser import pdbparser
from pdbparser.Utilities.Information import get_records_database_property_values
from pdbparser.log import Logger
from .Geometry import *
from .Information import *
from .Modify import *
from .BoundaryConditions import InfiniteBoundaries, PeriodicBoundaries
from .Collection import generate_sphere_points, generate_asymmetric_sphere_points
from .Database import __avogadroNumber__, __interMolecularMinimumDistance__, __ATOM__, __WATER__



class Construct(object):
    """
    The mother class of all construct classes
    """
    __defaults__ = {}
    __defaults__["pdbs"]                          = pdbparser
    __defaults__["insertionNumber"]               = 50
    __defaults__["density"]                       = 1  # in g/cm-3 and 1 is the density of water
    __defaults__["pdbsAxis"]                      = None
    __defaults__["flipPdbs"]                      = False
    __defaults__["boxSize"]                       = np.array([50,50,50])
    __defaults__["boxCenter"]                     = np.array([0,0,0])
    __defaults__["periodicBoundaries"]            = True
    __defaults__["priorities"]                    = None
    __defaults__["exisitingPdbs"]                 = None
    __defaults__["restrictions"]                  = None
    __defaults__["bondLength"]                    = None
    __defaults__["interMolecularMinimumDistance"] = None


    def __init__(self, *args, **kwargs):
        # initialize self._pdb
        object.__setattr__(self, "_pdb", pdbparser())

        # set kwargs attributes
        for kwarg, value in kwargs.items():
            self.__setattr__(kwarg, value)

        # get default attributes
        self.initialize_default_attributes()


    def __setattr__(self, name, value):
        if name == "_pdb":
            Logger.error("attribute %r is protected, user set_pdb method instead"%name)
        else:
            object.__setattr__(self, name, value)


    @property
    def pdb(self):
        """
        get the constructed pdb.\n

        :Returns:
            #. self._pdb (pdbparser): The constructed pdb.
        """
        return self._pdb


    def get_pdb(self):
        """
        get the constructed pdb.\n

        :Returns:
            #. self._pdb (pdbparser): The constructed pdb.
        """
        return self._pdb


    def set_pdb(self, pdb):
        """
        set the constructed pdb.\n

        :Parameters:
            #. pdb (pdbparser): The pdb instance replacing the constructed self._pdb.
        """
        assert isinstance(pdb, pdbparser)
        object.__setattr__(self, "_pdb", pdb)


    def status(self, step, numberOfSteps, stepIncrement = 1, mode = True, logEvery = 10):
        """
        This method is used to log construction status.\n

        :Parameters:
            #. step (int): The current step number
            #. numberOfSteps (int): The total number of steps
            #. stepIncrement (int): The incrementation between one step and another
            #. mode (bool): if False no status logging
            #. logEvery (float): the frequency of status logging. its a percent number.
        """
        if not mode:
            return
        if step-1 == -1:
            return
        actualPercent = int( float(step)/float(numberOfSteps)*100)
        previousPercent = int(float(step-stepIncrement)/float(numberOfSteps)*100)
        if actualPercent//logEvery != previousPercent//logEvery:
            Logger.info("%s --> %s%% completed. %s left out of %s" %(self.__class__.__name__, actualPercent, numberOfSteps-step, numberOfSteps))


    def start_insertion_engine(self):
        """
        starts the insertion engine. The insertion engine creates a grid to check if a
        certain pdb can be inserted or not\n
        """

        # initialize commonly useful variables
        self.initialize_useful_pdbs_data()

        # calculate boxSize and insertionNumber according to given priority and numberDensity
        self.set_boxsize_and_insertion_number()

        # check that all pdbs size does not exceed boxSize/2
        self.check_pdbs_size()

        # initialize grid variables
        self.initialize_grid_variables()

        # initialize grid variables
        self.initialize_records_variables()


    def initialize_useful_pdbs_data(self):
        # get pdbs data
        self.pdbsData = [self.get_pdb_data(self.pdbs[idx]) for idx in range(len(self.pdbs))]

        # get pdbs information
        self.pdbsMolarWeight = self.get_pdbs_molar_weight()

        # update variables
        self.initialize_useful_pdbs_variables()


    def initialize_useful_pdbs_variables(self):
        # pdbs dict keys are pdbs indexes in self.pdbs and values are number of atoms
        numberofAtomsPerPdb = [len(pdb) for pdb in self.pdbs]
        self.pdbsNumberofAtomsDict = dict(zip(range(len(self.pdbs)), numberofAtomsPerPdb))

        # pdbs dict keys are pdbs indexes in self.pdbs and values are number of atoms
        self.pdbsNumberOfInsertionDict = dict(zip(range(len(self.pdbs)), self.insertionNumber))

        # pdbs indexes list ordered in number of atoms
        sortedLen = sorted(self.pdbs, key = lambda item: len(item))
        self.pdbsIndexesNumberOfAtomsSorted = [self.pdbs.index(item) for item in sortedLen]

        # get total number of atoms
        totalNumberOfAtoms = [self.pdbsNumberofAtomsDict[idx]*self.pdbsNumberOfInsertionDict[idx]
                              for idx in range(len(self.pdbs))]
        self.pdbsNumberOfRecordsInsertionDict = dict(zip(range(len(self.pdbs)), totalNumberOfAtoms))


    def update_useful_pdbs_variables(self):
        self.initialize_useful_pdbs_variables()

    def initialize_grid_variables(self, gridUnitSize = None):
        # get gridUnitSize
        if gridUnitSize is None:
            self.gridUnitSize = np.array([1.,1.,1.])
        else:
            try:
                self.gridUnitSize = np.array(gridUnitSize)
            except:
                Logger.error("gridUnitSize is expected to be a vector")

            assert len(self.gridUnitSize)==3

        # self.box3DGrid. it should be initialized if needed. It contains records indexes occupying the space
        # self.box3DGrid =  -1*np.ones(np.ceil(self.boxSize), dtype=int)
        self.box3DGrid = -1*np.ones(np.ceil(np.array(self.boxSize)/self.gridUnitSize).astype(int), dtype=int)
        # this is a slice of box3DGrid
        self.subBox3DGrid = np.array((), dtype=int)

        # get restricted grid unit value
        self.restrictedUnitValue =  np.iinfo(self.box3DGrid.dtype).max
        self.fetchRestricted     = False
        self.fetchExisting       = False

    def initialize_records_variables(self):
        # self.generatedRecords
        numberOfRecords = np.sum(list(self.pdbsNumberOfRecordsInsertionDict.values()))
        self.generatedRecords = np.zeros((numberOfRecords,3))
        # self.exisitingPdbsRecords
        self.exisitingPdbsRecords = np.zeros((0,3))
        # constructed records pdb data
        self.initialize_records_pdb_data(numberOfRecords)


    def initialize_records_pdb_data(self, number):
        # the insertion index
        self.insertionIndex = 0
        # models ranges
        self.modelRange = []
        # pdb data
        self.recordName        = np.chararray((number), itemsize = 6)
        self.serialNumber      = np.zeros((number)).astype(int)
        self.atomName          = np.chararray((number), itemsize = 4)
        self.locationIndicator = np.chararray((number), itemsize = 1)
        self.residueName       = np.chararray((number), itemsize = 3)
        self.chainIndentifier  = np.chararray((number), itemsize = 1)
        self.sequenceNumber    = np.zeros((number)).astype(int)
        self.codeOfInsertion   = np.chararray((number), itemsize = 1)
        self.coordinatesX      = np.zeros((number)).astype(float)
        self.coordinatesY      = np.zeros((number)).astype(float)
        self.coordinatesZ      = np.zeros((number)).astype(float)
        self.occupancy         = np.zeros((number)).astype(float)
        self.temperatureFactor = np.zeros((number)).astype(float)
        self.segmentIdentifier = np.chararray((number), itemsize = 4)
        self.elementSymbol     = np.chararray((number), itemsize = 2)
        self.charge            = np.chararray((number), itemsize = 2)


    def initialize_pdbs_position_and_orientation(self, position = None, orientation = None):
        # orient and translate pdb to origin
        for pdb in self.pdbs:
            # get molecule axis and center
            center,_,_,_,moleculeAxis,_,_ = get_principal_axis(pdb.indexes , pdb)
            if position is not None:
                center = postion
            if orientation is not None:
                orientationVect = orientation
            else:
                orientationVect = [1,0,0]

            # center coordinates to the origin
            translate(pdb.indexes , pdb, -center)
            # orient molecule to ox axis
            orient(pdb.indexes, pdb, orientationVect, moleculeAxis)


    def check_pdbs_size(self):
        # test length of molecules
        for pdb in self.pdbs:
            minMax = get_min_max(pdb.indexes, pdb)
            if ((minMax[1]-minMax[0])>self.boxSize[0]/2.) or ((minMax[3]-minMax[2])>self.boxSize[1]/2.) or ((minMax[5]-minMax[4])>self.boxSize[2]/2.):
                Logger.error("Records size of pdb %r exceed the half size of the box" %pdb.name)


    def set_boxsize_and_insertion_number(self):
        # get priorities
        boxSizePriority = self.priorities.get("boxSize", True)
        insertionNumberPriority = self.priorities.get("insertionNumber", False)
        densityPriority = self.priorities.get("density", True)

        # water density in g/A3 is 1e-24
        waterDensity = 1e-24

        if boxSizePriority and densityPriority and not insertionNumberPriority:
            # get insertion number of all pdbs
            self.insertionNumber = (self.density*waterDensity*np.prod(self.boxSize)*__avogadroNumber__)/(np.array(self.pdbsMolarWeight))
            self.insertionNumber = np.array(self.insertionNumber/len(self.pdbs)).astype(int)
            self.density = np.sum(np.array(self.insertionNumber)*np.array(self.pdbsMolarWeight))/np.prod(self.boxSize)/waterDensity/__avogadroNumber__
            #Logger.info("Box size is %r"%(self.boxSize))
        elif insertionNumberPriority and densityPriority and not boxSizePriority:
            totalMass = float( np.sum(np.array(self.insertionNumber)*np.array(self.pdbsMolarWeight)) )
            cubicBoxSideLength = (totalMass/waterDensity/__avogadroNumber__/self.density)**(1/3.)
            self.boxSize = np.array([cubicBoxSideLength,cubicBoxSideLength,cubicBoxSideLength])
            # update construction box
            self.constructionBox = PeriodicBoundaries()
            self.constructionBox.set_vectors(self.boxSize)

            #Logger.info("Cubic box side = %r calculated according to molecules number %r and number density %r"%(cubicBoxSideLength, self.insertionNumber, self.density))
            self.density = totalMass/np.prod(self.boxSize)/waterDensity/__avogadroNumber__

        elif boxSizePriority and insertionNumberPriority and not densityPriority:
            #Logger.info("building system with box size %r and number of molecules %r regardless density set to %r"%(self.boxSize, self.insertionNumber, self.density))
            self.density = np.sum(np.array(self.insertionNumber)*np.array(self.pdbsMolarWeight))/np.prod(self.boxSize)/waterDensity/__avogadroNumber__
        else:
            Logger.error("Two of the three priorities %r must be set to True" %["boxSize","insertionNumber","density"])
            raise

        Logger.info("According to priorities %r, Box size is %r, insertion number is %r, density is %r" %(self.priorities, self.boxSize, self.insertionNumber, self.density) )

        # self.translateToBoxCenterVector
        self.translateToBoxCenterVector = np.array(self.boxCenter).astype(float) - np.array(self.boxSize/2.).astype(float)

        self.update_useful_pdbs_variables()


    def get_pdbs_molar_weight(self):
        pdbsMolarWeight = []
        for pdb in self.pdbs:
            pdbsMolarWeight.append(np.sum(get_records_database_property_values(pdb.indexes, pdb, "atomicWeight")) )

        return pdbsMolarWeight


    def get_pdb_data(self, pdb, indexes = None, attributes = None):
        """
        get pdb attributes dictionary.\n

        :Parameters:
            #. pdb (pdbparser): The pdb instance.
            #. indexes (list of integers): the pdb's used records indexes
            #. attributes (list of strings): attributes name
        :Returns:
            #. pdbData (dict): The pdb data dictionary.
        """

        if attributes is None:
            attributes = ["record_name", "serial_number", "atom_name",
                          "location_indicator", "residue_name", "chain_identifier",
                          "sequence_number", "code_of_insertion", "temperature_factor",
                          "occupancy", "segment_identifier", "element_symbol", "charge"]
        if indexes is None:
            indexes = pdb.indexes

        pdbData = {}
        for attr in attributes:
            pdbData[attr] = get_records_attribute_values(indexes, pdb, attr)

        return pdbData


    def initialize_default_attributes(self):
        # initialize common variables
        self.forbiddenVolume = 0

        # self.name
        if not hasattr(self, "name"):
            self.name = self.__class__.__name__
        self._pdb.set_name(self.name)
        # self.pdbs
        if not hasattr(self, "pdbs"):
            self.pdbs = [self.__defaults__["pdbs"]()]
        else:
            if not isinstance(self.pdbs, (list,tuple)):
                assert isinstance(self.pdbs, pdbparser)
                self.pdbs = [self.pdbs]
            else:
                for pdb in self.pdbs:
                    assert isinstance(pdb, pdbparser)
        # self.insertionNumber
        if not hasattr(self, "insertionNumber"):
            self.insertionNumber = [self.__defaults__["insertionNumber"] for idx in range(len(self.pdbs))]
        elif not self.insertionNumber:
            self.insertionNumber = [self.__defaults__["insertionNumber"] for idx in range(len(self.pdbs))]
        else:
            if not isinstance(self.insertionNumber, (list,tuple)):
                assert isinstance(self.insertionNumber, int)
                self.insertionNumber = [self.insertionNumber]*len(self.pdbs)
            else:
                for insertionNumber in self.insertionNumber:
                    assert isinstance(insertionNumber, int)
        # self.flipPdbs
        if not hasattr(self, "flipPdbs"):
            self.flipPdbs = [self.__defaults__["flipPdbs"] for idx in  range(len(self.pdbs))]
        elif not self.flipPdbs:
            self.flipPdbs = [self.__defaults__["flipPdbs"] for idx in range(len(self.pdbs))]
        else:
            if not isinstance(self.flipPdbs, (list,tuple)):
                assert isinstance(self.flipPdbs, bool)
                self.flipPdbs = [self.flipPdbs]*len(self.pdbs)
            else:
                for flipPdbs in self.flipPdbs:
                    assert isinstance(flipPdbs, bool)
        # self.pdbsAxis
        if not hasattr(self, "pdbsAxis"):
            self.pdbsAxis = [self.__defaults__["pdbsAxis"] for idx in range(len(self.pdbs))]
        elif not self.pdbsAxis:
            self.pdbsAxis = [self.__defaults__["pdbsAxis"] for idx in range(len(self.pdbs))]
        else:
            if not isinstance(self.pdbsAxis, (list,tuple)):
                self.pdbsAxis = [self.pdbsAxis]*len(self.pdbs)
        # self.boxSize
        if not hasattr(self, "boxSize"):
            self.boxSize = self.__defaults__["boxSize"]
        elif self.boxSize is None:
            self.boxSize = self.__defaults__["boxSize"]
        else:
            self.boxSize = np.array(self.boxSize)
            assert self.boxSize.shape in ((3,),(3,1))
        # self.boxCenter
        if not hasattr(self, "boxCenter"):
            self.boxCenter = self.__defaults__["boxCenter"]
        elif self.boxCenter is None:
            self.boxCenter = self.__defaults__["boxCenter"]
        else:
            self.boxCenter = np.array(self.boxCenter)
            assert self.boxCenter.shape == (3,)
        # periodicBoundaries
        if not hasattr(self, "periodicBoundaries"):
            self.periodicBoundaries = self.__defaults__["periodicBoundaries"]
        assert isinstance(self.periodicBoundaries, bool)
        # constructionBox
        if self.periodicBoundaries:
            self.constructionBox = PeriodicBoundaries()
            self.constructionBox.set_vectors(self.boxSize)
        else:
            self.constructionBox = InfiniteBoundaries()
        # self.exisitingPdbs
        if not hasattr(self, "exisitingPdbs"):
            self.exisitingPdbs = [self.__defaults__["exisitingPdbs"]]
        else:
            if not isinstance(self.exisitingPdbs, (list,tuple)):
                if self.exisitingPdbs:
                    assert isinstance(self.exisitingPdbs, pdbparser)
                    self.exisitingPdbs = [self.exisitingPdbs]
                else:
                    self.exisitingPdbs = [self.__defaults__["exisitingPdbs"]]
            else:
                for pdb in self.exisitingPdbs:
                    assert isinstance(pdb, pdbparser)
        # self.restrictions
        if not hasattr(self, "restrictions"):
            self.restrictions = [self.__defaults__["restrictions"]]
        else:
            if not isinstance(self.restrictions, (list,tuple)):
                if self.restrictions:
                    assert isinstance(self.restrictions, basestring)
                    self.restrictions = [self.restrictions]
            else:
                for restriction in self.restrictions:
                    assert isinstance(restriction, basestring)
        # self.bondLength
        if not hasattr(self, "bondLength"):
            self.bondLength = [self.__defaults__["bondLength"]]
        else:
            try:
                self.bondLength = float(self.bondLength)
            except:
                Logger.error("bondLength argument is expected to be numeric and positif")
                raise
            assert self.bondLength > 0
        # self.priorities
        if not hasattr(self, "priorities"):
            self.priorities = {"boxSize":True, "insertionNumber":False, "density":True}
        elif self.priorities is None:
            self.priorities = {"boxSize":True, "insertionNumber":False, "density":True}
        else:
            assert isinstance(self.priorities, dict)
            for value in self.priorities.values():
                assert isinstance(value, bool)
        # self.density
        if not hasattr(self, "density"):
            self.density = self.__defaults__["density"]
        else:
            try:
                self.density = float(self.density)
            except:
                Logger.error("density argument is expected to be numeric and positif")
                raise
            assert self.density > 0
        # self.interMolecularMinimumDistance
        if not hasattr(self, "interMolecularMinimumDistance"):
            self.interMolecularMinimumDistance = self.__defaults__["interMolecularMinimumDistance"]
        if self.interMolecularMinimumDistance is None:
            self.interMolecularMinimumDistance = __interMolecularMinimumDistance__
        assert isinstance(self.interMolecularMinimumDistance,(float,int))
        assert self.interMolecularMinimumDistance>0


    def combine(self, coordinates, pdbData, incrementSequenceNumber = True, model = True):
        """
        this method can be called at any time to combine constructed with the constructed PDB data
        """
        # get added records length
        l = coordinates.shape[0]

        # get sequence number
        if incrementSequenceNumber:
            if self.insertionIndex:
                sequenceNumber = self.sequenceNumber[self.insertionIndex-1]+1
            else:
                sequenceNumber = 1
        else:
            sequenceNumber = pdbData["sequence_number"]

        # update box3DGrid
        posIndexes = np.array(np.array(coordinates-self.translateToBoxCenterVector)%self.boxSize).astype(int)
        self.box3DGrid[posIndexes[:,0],posIndexes[:,1],posIndexes[:,2]] = range(self.insertionIndex,self.insertionIndex+l)

        # set pdb datas
        self.recordName[self.insertionIndex:self.insertionIndex+l]        = pdbData["record_name"]
        self.serialNumber[self.insertionIndex:self.insertionIndex+l]      = pdbData["serial_number"]
        self.atomName[self.insertionIndex:self.insertionIndex+l]          = pdbData["atom_name"]
        self.locationIndicator[self.insertionIndex:self.insertionIndex+l] = pdbData["location_indicator"]
        self.residueName[self.insertionIndex:self.insertionIndex+l]       = pdbData["residue_name"]
        self.chainIndentifier[self.insertionIndex:self.insertionIndex+l]  = pdbData["chain_identifier"]
        self.sequenceNumber[self.insertionIndex:self.insertionIndex+l]    = sequenceNumber
        self.codeOfInsertion[self.insertionIndex:self.insertionIndex+l]   = pdbData["code_of_insertion"]
        self.occupancy[self.insertionIndex:self.insertionIndex+l]         = pdbData["occupancy"]
        self.temperatureFactor[self.insertionIndex:self.insertionIndex+l] = pdbData["temperature_factor"]
        self.segmentIdentifier[self.insertionIndex:self.insertionIndex+l] = pdbData["segment_identifier"]
        self.elementSymbol[self.insertionIndex:self.insertionIndex+l]     = pdbData["element_symbol"]
        self.charge[self.insertionIndex:self.insertionIndex+l]            = pdbData["charge"]
        # set coordinates
        self.generatedRecords[self.insertionIndex:self.insertionIndex+l]  = coordinates

        # set modelRange
        if model:
            self.modelRange.append([self.insertionIndex, self.insertionIndex+l])

        # increment insertion index
        self.insertionIndex += l


    def finalize(self):
        thisPDB = pdbparser()
        thisPDB._boundaryConditions = self.constructionBox

        #for idx in range(self.recordName.shape[0]):
        for idx in range(self.insertionIndex):
            thisPDB.records.append( { "record_name"       : self.recordName[idx] ,\
                                      "serial_number"     : self.serialNumber[idx] ,\
                                      "atom_name"         : self.atomName[idx] ,\
                                      "location_indicator": self.locationIndicator[idx] ,\
                                      "residue_name"      : self.residueName[idx]  ,\
                                      "chain_identifier"  : self.chainIndentifier[idx] ,\
                                      "sequence_number"   : self.sequenceNumber[idx] ,\
                                      "code_of_insertion" : self.codeOfInsertion[idx]  ,\
                                      "coordinates_x"     : self.generatedRecords[idx,0] ,\
                                      "coordinates_y"     : self.generatedRecords[idx,1] ,\
                                      "coordinates_z"     : self.generatedRecords[idx,2] ,\
                                      "occupancy"         : self.occupancy[idx] ,\
                                      "temperature_factor": self.temperatureFactor[idx] ,\
                                      "segment_identifier": self.segmentIdentifier[idx]  ,\
                                      "element_symbol"    : self.elementSymbol[idx]  ,\
                                      "charge"            : self.charge[idx]  ,\
                                     } )
        # set constructed pdb
        self.set_pdb(thisPDB)

        for modelRange in self.modelRange:
            self.get_pdb().define_model(model_start = modelRange[0], model_end = modelRange[1])


    def construct(self):
        pass


    def parse_existing_pdbs(self):
        if not isinstance(self.box3DGrid, np.ndarray):
            Logger.info("Impossible to parse existing pdbs before initializing box 3D grid")
            self.fetchExisting = False
            return
        # self.box3DGrid must be of type integers and stores records indexes
        assert np.issubdtype(self.box3DGrid.dtype, np.dtype(int).type)

        Logger.info("Parsing existing pdbs")

        nExistingRecords = int(np.sum([len(pdb) for pdb in self.exisitingPdbs if isinstance(pdb, pdbparser)]))

        self.exisitingPdbsRecords = np.empty((nExistingRecords,3))

        recordsIdx = 0
        for pdb in self.exisitingPdbs:
            if not isinstance(pdb, pdbparser):
                continue
            Logger.info("Evaluating %r pdb records positions" %pdb.name)

            # get coords
            coords = np.array(pdb.coordinates)
            # get indexes of records within self.boxSize to make correct modulos
            goodIndexes = np.where( np.prod((coords- self.translateToBoxCenterVector>0),axis = 1)*
                                    np.prod((coords- self.translateToBoxCenterVector<self.boxSize),axis = 1) )[0]

            # save coords to self.exisitingPdbsRecords
            self.exisitingPdbsRecords[recordsIdx:recordsIdx+len(goodIndexes),:] = coords[goodIndexes,:]

            boxCoords = ((coords[goodIndexes,:] - self.translateToBoxCenterVector)%self.boxSize)/self.gridUnitSize
            boxCoords = np.array(boxCoords).astype(self.box3DGrid.dtype)
            indexes = -1*(np.array(range(len(goodIndexes)))+recordsIdx)-2

            self.box3DGrid[boxCoords[:,0],boxCoords[:,1],boxCoords[:,2]] = indexes

            recordsIdx += len(goodIndexes)

        # get fobidden volume
        self.forbiddenVolume += len(np.where(self.box3DGrid <= -2)[0])*np.prod(self.gridUnitSize)

        if not recordsIdx:
            self.fetchExisting = False
        else:
            self.fetchExisting = True


    def parse_restrictions(self):
        if not self.restrictions:
            return
        if not isinstance(self.box3DGrid, np.ndarray):
            return
        # self.box3DGrid must be of type integers and stores records indexes
        assert np.issubdtype(self.box3DGrid.dtype, np.dtype(int).type)

        Logger.info("Building restrictions")
        xyz = np.array(list(np.ndindex(self.box3DGrid.shape)))*self.gridUnitSize + self.translateToBoxCenterVector
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]

        for expression in self.restrictions:
            Logger.info("Evaluating %r restriction expression" %expression)
            code = parser.expr(expression).compile()
            indexes = eval(code)
            goodIndexes = np.array([idx for idx in range(len(indexes)) if  indexes[idx] ])
            goodIndexes = np.array(xyz[goodIndexes] - self.translateToBoxCenterVector).astype(int)
            self.box3DGrid[goodIndexes[:,0],goodIndexes[:,1],goodIndexes[:,2]] = self.restrictedUnitValue

        # get fobidden volume
        self.forbiddenVolume += len(np.where(self.box3DGrid == self.restrictedUnitValue)[0])*np.prod(self.gridUnitSize)

        if len(self.restrictions):
            self.fetchRestricted = True
        else:
            self.fetchRestricted = False

    def get_box3DGrid_slicing(self, posIdx, margin):
        return [ np.array( range(posIdx[0]-margin, posIdx[0]+margin+1) ).reshape(-1,1,1)%self.box3DGrid.shape[0],
                 np.array( range(posIdx[1]-margin, posIdx[1]+margin+1) ).reshape(1,-1,1)%self.box3DGrid.shape[1],
                 np.array( range(posIdx[2]-margin, posIdx[2]+margin+1) ).reshape(1,1,-1)%self.box3DGrid.shape[2] ]

    def any_restricted(self):
        if not self.fetchRestricted:
            return False

        if self.restrictedUnitValue in self.subBox3DGrid:
            return True
        else:
            return False

    def get_defined_indexes_in_subBox3DGrid(self):
        return np.array(self.subBox3DGrid[np.nonzero( (self.subBox3DGrid >=0) * (self.subBox3DGrid!=self.restrictedUnitValue) )] )

    def get_exisiting_indexes_in_subBox3DGrid(self):
        if not self.fetchExisting:
            return []
        exisitingIndexes = np.array(self.subBox3DGrid[np.nonzero( (self.subBox3DGrid <=-2) * (self.subBox3DGrid!=self.restrictedUnitValue) )] )
        exisitingIndexes += 2
        exisitingIndexes *= -1
        return exisitingIndexes

    def is_molrecords_accepted(self, molRecords, definedIndexes, exisitingIndexes, minimumDistanceSquared):
        # when no surrounding atoms found
        if not len(definedIndexes) and not len(exisitingIndexes):
            return True
        # calculate distances between atoms
        continueLoop = True
        for molIdx in range(molRecords.shape[0]):
            # for already defined records
            if len(definedIndexes):
                #difference = self.generatedRecords[definedIndexes] - molRecords[molIdx,:]
                difference = self.constructionBox.real_difference(molRecords[molIdx,:], self.generatedRecords[definedIndexes])
                distances = np.add.reduce( difference**2, axis = 1)
                if np.min(distances) < minimumDistanceSquared:
                    continueLoop = False
                    break
            # for existing pdbs
            if len(exisitingIndexes):
                #difference = self.exisitingPdbsRecords[exisitingIndexes] - molRecords[molIdx,:]
                difference = self.constructionBox.real_difference(molRecords[molIdx,:], self.exisitingPdbsRecords[exisitingIndexes])
                distances = np.add.reduce( difference**2, axis = 1)
                if np.min(distances) < minimumDistanceSquared:
                    continueLoop = False
                    break
        # return
        return continueLoop


    def solvate(self, pdbs = None, insertionNumber = None, thikness = 10, density = 1, restrictions = None, priorities = None, recursionLimit = 10000):
        """
        construct a solvation shell around constructedPDB records
        """
        Logger.info("Solvating system")
        if not pdbs:
            pdbs = pdbparser()
            pdbs.records = __WATER__
            pdbs.set_name("water")
        # get coordinates min ad max
        minMax = get_min_max(self.get_pdb().indexes, self.get_pdb())
        boxSize = np.array( [minMax[1]-minMax[0],
                             minMax[3]-minMax[2],
                             minMax[5]-minMax[4]] )+2*thikness
        boxCenter = np.array( [minMax[1]+minMax[0],
                               minMax[3]+minMax[2],
                               minMax[5]+minMax[4]] )/2.
        # build amorphous system and concatenate
        self.get_pdb().concatenate( AmorphousSystem( pdbs = pdbs,
                                                     insertionNumber = insertionNumber,
                                                     exisitingPdbs = self.get_pdb(),
                                                     boxSize = boxSize,
                                                     boxCenter = boxCenter,
                                                     density = density,
                                                     priorities = priorities,
                                                     restrictions = restrictions).construct().get_pdb() )
        # return
        return self




class AmorphousSystem(Construct):
    """
    Constructs amorphous system from the given pdbs records.
    """
    def __init__(self, pdbs,
                       insertionNumber = None,
                       boxSize = None,
                       boxCenter = None,
                       density = 1, # in g/cm-3 and 1 is the density of water
                       recursionLimit = 10000,
                       periodicBoundaries = True,
                       restrictions = None,
                       exisitingPdbs = None,
                       priorities = None,
                       *args, **kwargs):

        # get common arguments
        self.pdbs = pdbs
        self.insertionNumber = insertionNumber
        self.boxSize = boxSize
        self.boxCenter = boxCenter
        self.restrictions = restrictions
        self.exisitingPdbs = exisitingPdbs
        self.density = density
        self.priorities = priorities
        self.periodicBoundaries = periodicBoundaries

        # get specific argument
        assert recursionLimit > 0
        self.recursionLimit = recursionLimit

        # The base class constructor.
        super(AmorphousSystem,self).__init__(*args, **kwargs)

        # initialize
        self.initialize()


    def initialize(self):

        # start insertion engine
        self.start_insertion_engine()

        # initialize pdbs, center to origin and oriented towards [1,0,0]
        self.initialize_pdbs_position_and_orientation()

        # Parse existing pdbs
        self.parse_existing_pdbs()

        # Parse restricted positions
        self.parse_restrictions()

        Logger.info("%r construct successfully initialized" %self.name)


    def construct(self):
        minimumDistanceSquared = self.interMolecularMinimumDistance**2
        totalAtomsNumber = int(np.sum(list(self.pdbsNumberOfRecordsInsertionDict.values())))

        # recalculate pdbsNumberOfInsertionDict when forbiddenVolume
        if self.priorities["density"]:
            if self.forbiddenVolume >0:
                insertionRatio = float(np.prod(self.boxSize)-self.forbiddenVolume)/np.prod(self.boxSize)
            else:
                insertionRatio = 1.0
            insertionDict = dict(zip(range(len(self.pdbs)), np.array(np.array(self.insertionNumber)*insertionRatio).astype(int) ))
            Logger.info("a volume of %r is forbidden out of total volume of %r. Density %r has the priority "%(self.forbiddenVolume, np.prod(self.boxSize), self.density))
            Logger.info("insertion number is set to %r "%list(insertionDict.values()))
        else:
            insertionDict = self.pdbsNumberOfInsertionDict
            if self.forbiddenVolume >0:
                densityRatio = float(np.prod(self.boxSize))/float(np.prod(self.boxSize)-self.forbiddenVolume)
                Logger.info("a volume of %r is forbidden out of total volume of %r. Density %r has no priority. Final system density is %r"%(self.forbiddenVolume, np.prod(self.boxSize), self.density, self.density*densityRatio))

        # start construction
        Logger.info("Building amorphous system using random insertion with recursion limit of %s"%self.recursionLimit)
        for pdbIndex in list(reversed(self.pdbsIndexesNumberOfAtomsSorted)):

            minMax = get_min_max(self.pdbs[pdbIndex].indexes, self.pdbs[pdbIndex])
            margin = np.int( np.ceil(np.max( [minMax[1]-minMax[0], minMax[3]-minMax[2], minMax[5]-minMax[4]])+self.interMolecularMinimumDistance) )
            Logger.info("Adding %r %r pdb of records length %r. margin set to %r"%(insertionDict[pdbIndex], self.pdbs[pdbIndex].name, self.pdbsNumberofAtomsDict[pdbIndex], margin))

            for molNum in range(insertionDict[pdbIndex]):
                self.status(self.insertionIndex, totalAtomsNumber, len(self.pdbs[pdbIndex]) )
                # initialize recursion limit
                recurLim = self.recursionLimit

                while recurLim > 0:
                    # get random orientation vector
                    signs = np.sign( np.random.rand(3)-0.5 )
                    rotationVectors = np.random.rand(3) * signs
                    # get random position vector
                    positionVectors = np.random.rand(3)*self.boxSize
                    # get position index in moleculesIndexesArray
                    posIdx = np.array(positionVectors, dtype = int)

                    # translate to box_center
                    positionVectors += self.translateToBoxCenterVector

                    # check if position is already occupied
                    if self.box3DGrid[posIdx[0], posIdx[1], posIdx[2]] >= 0:
                        continue
                    elif self.box3DGrid[posIdx[0], posIdx[1], posIdx[2]] == self.restrictedUnitValue:
                        continue

                    # get a copy of the molecule
                    pdbMOL = self.pdbs[pdbIndex].get_copy()

                    # get slicing
                    sl = self.get_box3DGrid_slicing(posIdx, margin)

                    # get surrounding self.subBox3DGrid
                    self.subBox3DGrid = self.box3DGrid[sl[0],sl[1],sl[2]]

                    # In case restriction found
                    if self.any_restricted():
                        recurLim -= 1
                        continue

                    # Indexes matching with already defined records
                    definedIndexes = self.get_defined_indexes_in_subBox3DGrid()

                    # Indexes matching with already existing records
                    exisitingIndexes = self.get_exisiting_indexes_in_subBox3DGrid()

                    # orient
                    orient(pdbMOL.indexes, pdbMOL, rotationVectors, [1,0,0])
                    # translate
                    translate(pdbMOL.indexes, pdbMOL, positionVectors)
                    # get records
                    molRecords = np.array(pdbMOL.coordinates)

                    # when no surrounding atoms found
                    if self.is_molrecords_accepted(molRecords, definedIndexes, exisitingIndexes, minimumDistanceSquared):
                        self.combine(molRecords, self.pdbsData[pdbIndex])
                        break
                    else:
                        recurLim -= 1
                        continue

                # recursion limit test
                if recurLim == 0:
                    Logger.warn("Recursion limit %r reached for inserted %r of pdb number %r" %(self.recursionLimit, molNum, len(self._pdb)) )
                    continue

        Logger.info("Amorphous system built successfully")

        # finalize and construct pdb
        self.finalize()

        # return self
        return self




class Micelle(Construct):
    """
    Constructs a spherical micelle
    """
    def __init__(self, pdbs,
                       insertionNumber = 60,
                       pdbsAxis = None,
                       flipPdbs = False,
                       micelleRadius = None,
                       positionsGeneration = None,
                       *args, **kwargs):

        # get common arguments
        self.pdbs = pdbs
        self.insertionNumber = insertionNumber
        self.pdbsAxis = pdbsAxis
        self.flipPdbs = flipPdbs
        # get specific argument
        self.micelleRadius = micelleRadius
        if self.micelleRadius:
            assert isinstance(self.micelleRadius, (float,int))

        if positionsGeneration is None:
            self.positionsGeneration = "symmetric"
        else:
            self.positionsGeneration = positionsGeneration
        assert isinstance(self.positionsGeneration, basestring)
        self.positionsGeneration = str(self.positionsGeneration).lower()
        assert self.positionsGeneration in ("symmetric", "asymmetric")

        # The base class constructor.
        super(Micelle,self).__init__(*args, **kwargs)

        # initialize
        self.initialize()


    def initialize(self):

        # orient and pdb to origin
        minRadius = 0
        self.minMax = [0,0,0,0,0,0]
        self.moleculeAxis = []

        for idx in range(len(self.pdbs)):
            # get pdb
            pdb = self.pdbs[idx]

            # get molecule axis and orient pdb
            moleculeAxis = get_axis(pdb.indexes,  pdb, self.pdbsAxis[idx])
            orient(indexes=pdb.indexes, pdb=pdb, axis=[1-2*self.flipPdbs[idx],0,0], records_axis=moleculeAxis)
            self.moleculeAxis.append([1,0,0])

            # translate to positive quadrant
            atomToOriginIndex = get_closest_to_origin(pdb.indexes,  pdb)
            atom = pdb.records[atomToOriginIndex]
            [minX, minY, minZ]  = [ atom['coordinates_x'] , atom['coordinates_y'] , atom['coordinates_z'] ]
            translate(pdb.indexes,  pdb, [-minX, -minY, -minZ])

            # get minimum acceptable radius
            [minX,maxX,minY,maxY,minZ,maxZ] = get_min_max(pdb.indexes,  pdb)
            translate(pdb.indexes,  pdb, [-minX, 0, 0])
            moleculeWidth = (np.max(np.abs([maxY-minY, maxZ-minZ])) + self.interMolecularMinimumDistance )**2
            minRadius = np.max([minRadius, np.sqrt( (np.sum(self.insertionNumber)*moleculeWidth)/(4*np.pi) ) ])

             # adjust Y and Z of the farest atom on X axis
            [minX,maxX,minY,maxY,minZ,maxZ] = get_min_max(pdb.indexes,  pdb)
            self.minMax[0] = np.min([self.minMax[0],minX])
            self.minMax[1] = np.max([self.minMax[1],maxX])
            self.minMax[2] = np.min([self.minMax[2],minY])
            self.minMax[3] = np.max([self.minMax[3],maxY])
            self.minMax[4] = np.min([self.minMax[4],minZ])
            self.minMax[5] = np.max([self.minMax[5],maxZ])
            recordIndex = get_records_indexes_in_attribute_values(pdb.indexes,  pdb, 'coordinates_x', [maxX])[0]
            yCoord = pdb.records[recordIndex]['coordinates_y']
            zCoord = pdb.records[recordIndex]['coordinates_z']
            translate(pdb.indexes,  pdb, [0, -yCoord, -zCoord])

        # get micelle_radius
        if self.micelleRadius is None:
            self.micelleRadius = minRadius
        else:
            if self.micelleRadius < minRadius:
                self.micelleRadius = minRadius


    def construct(self):
        Logger.info('Micelle inner radius = %r' %(self.micelleRadius))
        Logger.info('Micelle outer radius = %r' %(self.micelleRadius+self.minMax[1]))

        # get micelle sphere points
        if self.positionsGeneration == "symmetric":
            spherePoints = generate_sphere_points(self.micelleRadius, np.sum(self.insertionNumber))
        elif self.positionsGeneration == "asymmetric":
            moleculeWidth_Z = self.minMax[5]-self.minMax[4] + self.interMolecularMinimumDistance/2.
            moleculeWidth_Y = self.minMax[3]-self.minMax[2] + self.interMolecularMinimumDistance/2.
            spherePoints = generate_asymmetric_sphere_points(self.micelleRadius, [moleculeWidth_Y, moleculeWidth_Z])
        else:
            Logger.error("positionsGeneration can either be 'symmetric' or 'asymmetric' ")
            raise

        # fix number of insertion, it might change is asymmetric point generation is chosen
        totalNumberOfMolecules = np.sum(self.insertionNumber)
        moleculesIndexes = []
        for idx in range(len(self.insertionNumber)):
            insertionPercentage = float(self.insertionNumber[idx])/totalNumberOfMolecules
            self.insertionNumber[idx] = int( np.ceil(insertionPercentage*len(spherePoints)) )
            moleculesIndexes.extend([idx]*self.insertionNumber[idx])

        Logger.info('Micelle number of insertion = %r' %len(spherePoints))
        for idx in range(len(spherePoints)):
            # log the status
            self.status(idx, len(spherePoints), stepIncrement = 1 )
            # get random molecule
            molIdx = int(np.random.random(1)[0]*len(moleculesIndexes))
            molIdx = moleculesIndexes.pop(molIdx)
            # get molecule
            pdbMOL = self.pdbs[molIdx].get_copy()
            pdbMOL.models = {}
            pdbMOL.define_model(0, len(pdbMOL))
            pdbMOL.models[list(pdbMOL.models.keys())[0]]["MODEL_NAME"] =  'MOL %s' %idx
            increment_sequence_number(pdbMOL.indexes, pdbMOL, idx)
            # get point
            point = spherePoints[idx]
            # translate pdbMOL
            translationVector = [point[0] , point[1] , point[2]]
            # orient
            orient(indexes=pdbMOL.indexes, pdb=pdbMOL, axis=point, records_axis=self.moleculeAxis[molIdx])
            # translate
            translate(pdbMOL.indexes, pdbMOL, translationVector)
            # extend amorphous box
            self.get_pdb().concatenate(pdbMOL)

        Logger.info("Micelle built successfully")

        return self



class Liposome(Construct):
    """
    Constructs a spherical Liposome
    """
    def __init__(self, pdbs,
                       innerInsertionNumber,
                       innerRadius = None,
                       outerInsertionNumber = None,
                       pdbsAxis = None,
                       flipPdbs = False,
                       positionsGeneration = None,
                       interlayerDistance = None,
                       *args, **kwargs):

        # get general arguments
        self.pdbs = pdbs
        self.pdbsAxis = pdbsAxis
        self.flipPdbs = flipPdbs

        # get specific argument
        self.innerInsertionNumber = innerInsertionNumber
        self.innerRadius = innerRadius
        self.outerInsertionNumber = outerInsertionNumber
        self.interlayerDistance = interlayerDistance
        self.positionsGeneration = positionsGeneration

        # The base class constructor.
        super(Liposome,self).__init__(*args, **kwargs)

        # intialize specific arguments
        self.initialize()


    def initialize(self):
        # get interlayer distance
        if self.interlayerDistance is None:
            self.interlayerDistance = self.interMolecularMinimumDistance
        else:
            self.interlayerDistance = interlayerDistance
            assert isinstance(interlayerDistance,(float,int))
            assert self.interlayerDistance>0

        if self.positionsGeneration is None:
            self.positionsGeneration = ["asymmetric","asymmetric"]
        else:
            if isinstance(self.positionsGeneration, (list,tuple)):
                if len(self.positionsGeneration) == 1:
                    assert isinstance(positionsGeneration[0], basestring)
                    assert positionsGeneration[0].lower() in ("symmetric", "asymmetric")
                    self.positionsGeneration = 2*[positionsGeneration[0].lower()]
                elif len(positionsGeneration) == 2:
                    assert isinstance(self.positionsGeneration[0], basestring)
                    assert self.positionsGeneration[0].lower() in ("symmetric", "asymmetric")
                    assert isinstance(self.positionsGeneration[1], basestring)
                    assert self.positionsGeneration[1].lower() in ("symmetric", "asymmetric")
                    self.positionsGeneration = [self.positionsGeneration[0].lower(), self.positionsGeneration[1].lower()]
                else:
                    Logger.error("positionsGeneration should be a list of length =2 containg only 'symmetric' or 'asymmetric' ")
                    raise
            else:
                assert isinstance(self.positionsGeneration, basestring)
                assert self.positionsGeneration.lower() in ("symmetric", "asymmetric")
                self.positionsGeneration = 2*[self.positionsGeneration.lower()]


    def construct(self):
        Logger.info('Constructing liposome inner micellar part')
        pdbINNER = Micelle(copy.deepcopy(self.pdbs),
                           insertionNumber = self.innerInsertionNumber,
                           pdbsAxis = self.pdbsAxis,
                           flipPdbs = self.flipPdbs,
                           micelleRadius = self.innerRadius,
                           positionsGeneration = self.positionsGeneration[0]).construct().get_pdb()

        # get inner micelle radius and calculate outer radius
        [minX,maxX,minY,maxY,minZ,maxZ] = get_min_max(pdbINNER.indexes, pdbINNER)
        innerMicelleRadius = np.max(np.abs([minX,maxX,minY,maxY,minZ,maxZ]))
        outerLiposomeRadius =  innerMicelleRadius + self.interlayerDistance

        if self.outerInsertionNumber == None:
            # generate proportional molecules number
            self.outerInsertionNumber = int(np.floor(outerLiposomeRadius/innerMicelleRadius)*self.innerInsertionNumber)

        Logger.info('Constructing liposome outer micellar part')
        pdbOUTER = Micelle(copy.deepcopy(self.pdbs),
                           insertionNumber = self.outerInsertionNumber,
                           pdbsAxis = self.pdbsAxis,
                           flipPdbs = [not flip for flip in self.flipPdbs],
                           micelleRadius = outerLiposomeRadius,
                           positionsGeneration = self.positionsGeneration[1]).construct().get_pdb()

        pdbINNER.concatenate(pdbOUTER)
        self.set_pdb(pdbINNER)

        return self



class Sheet(Construct):
    """
    construct hexagonal sheet.
    length and width are the dimensions of the sheet.
    orientation can either be arm-chair or zigzag
    atom is the type of atom used. default is carbon, which makes the sheet a graphene sheet
    bond-length is the theoretical bond length between the sheet's atoms. default is 1.42, the length of Sp2 carbon-carbon bond
    """

    def __init__(self, element = None,
                       length = 50,
                       width = 50,
                       orientation = None,
                       bondLength = 1.42,
                       *args, **kwargs):

        assert isinstance(length,(float,int))
        assert length > 0
        self.length = length

        assert isinstance(width,(float,int))
        assert width > 0
        self.width = width

        assert isinstance(bondLength,(float,int))
        assert bondLength > 0
        self.bondLength = bondLength

        assert self.width > 2*self.bondLength
        assert self.length > 2*self.bondLength

        if not orientation:
            self.orientation = "arm-chair"
        else:
            self.orientation = orientation.lower()
            assert self.orientation in ("arm-chair","zig-zag", "armchair", "zigzag")

        if not element:
            self.element = "C"

        if self.orientation in ("zig-zag", "zigzag"):
            junk = self.length
            self.length = self.width
            self.width = junk

        self.atom = __ATOM__
        self.atom['element_symbol'] = self.element
        self.atom['atom_name'] = self.element
        self.atom['residue_name'] = 'GSH'

        # The base class constructor.
        super(Sheet,self).__init__(*args, **kwargs)

    def construct(self):
        """
        """
        carbonsInLine = []
        position = 0
        while position<=self.width:
            at = copy.deepcopy(self.atom)
            at['coordinates_y'] = position
            carbonsInLine.append(at)
            # add another atom, so the the final number of atoms is even
            position += self.bondLength
            at = copy.deepcopy(self.atom)
            at['coordinates_y'] = position
            carbonsInLine.append(at)
            # increment position
            position += 2*self.bondLength
        # create pdbLine
        pdbLINE = pdbparser()
        pdbLINE.records = carbonsInLine
        recordsIndexes = pdbLINE.indexes
        # building sheet
        translateLength = 1.5*self.bondLength
        position = 0

        while position<=self.length:
            # get pdbLine
            pdbL = pdbLINE.get_copy()
            # translate
            #print('before', [position, 0, 0])
            #print(pdbL.coordinates)
            translate(recordsIndexes, pdbL, [position, 0, 0])
            #print(pdbL.coordinates)
            # extend graphene sheet
            self.get_pdb().records.extend(pdbL)
            # increment position
            position += self.bondLength

            # add another line, so the the final number of lines is even
            pdbL = pdbLINE.get_copy()
            # translate
            translate(recordsIndexes, pdbL, [position, translateLength, 0])
            # extend graphene sheet
            self.get_pdb().records.extend(pdbL)
            # increment position
            position += self.bondLength

        if self.orientation in ("zig-zag", "zigzag"):
            rotationMatrix = get_rotation_matrix([0,0,1], np.pi/2)
            rotate(self.get_pdb().indexes, self.get_pdb(), rotationMatrix)

        reset_records_serial_number(self.get_pdb())
        [minX,maxX, minY,maxY, minZ,maxZ]  = get_min_max(self.get_pdb().indexes, self.get_pdb())
        translate(self.get_pdb().indexes, self.get_pdb(), vector = [-minX, -minY, -minZ])

        return self


    def wrap(self):
        """
        wrap the constructed sheet around a cylinder. the cylinder is along X axis
        the sheet must be in XY plane
        the cylinder diameter is calculated as bond_length+(Ymax-Ymin)/(np.pi)
        """
        [minX,maxX, minY,maxY, minZ,maxZ]  = get_min_max(self.get_pdb().indexes, self.get_pdb())
        translate(self.get_pdb().indexes, self.get_pdb(), vector = [-minX, -minY, -minZ])

        yLength = self.bondLength/2.0 + np.abs(maxY-minY)
        cylinderRadius = yLength/(2*np.pi)

        #print(self.bondLength,[minX,maxX, minY,maxY, minZ,maxZ], yLength, cylinderRadius)

        for idx in self.get_pdb().indexes:
            atom = self.get_pdb().records[idx]
            rotationMatrix = get_rotation_matrix([1,0,0], 2.0*np.pi*atom['coordinates_y']/yLength)
            translate([idx], self.get_pdb(), vector = [0, cylinderRadius-atom['coordinates_y'], 0])
            rotate([idx], self.get_pdb(), rotationMatrix)

        return self




class Nanotube(Sheet):
    """
    construct a single wall nanotube
    """

    def __init__(self, element = None,
                       length = 50,
                       radius = 10,
                       orientation = None,
                       bondLength = 1.42,
                       *args, **kwargs):

     # The base class constructor.
     super(Nanotube,self).__init__(element = element,
                                   length = length,
                                   width = 2.*np.pi*radius,
                                   orientation = orientation,
                                   bondLength = bondLength,
                                   *args, **kwargs)


    def construct(self):
        super(Nanotube,self).construct()
        set_residue_name(self.get_pdb().indexes,self.get_pdb(),"CNT")
        super(Nanotube,self).wrap()

        # translate nanotube to the center of the box
        minMax = get_min_max(self.get_pdb().indexes, self.get_pdb())
        translate(self.get_pdb().indexes, self.get_pdb(), [-(minMax[1]+minMax[0])/2., -(minMax[3]+minMax[2])/2., -(minMax[5]+minMax[4])/2.])

        return self



class MultipleWallNanotube(Construct):
    """
    construct a multiple wall nanotube
    """
    def __init__(self, element = None,
                       length = 50,
                       radius = 10,
                       orientation = None,
                       bondLength = 1.42,
                       interlayerDistance = 3.4,
                       wallsNumber = 1,
                       *args, **kwargs):

        self.element = element
        self.length = length
        self.radius = radius
        self.bondLength = bondLength

        assert isinstance(interlayerDistance,(float,int))
        assert interlayerDistance > 0
        self.interlayerDistance = interlayerDistance

        assert isinstance(wallsNumber,int)
        assert wallsNumber > 0
        self.wallsNumber = wallsNumber

        if not orientation:
            self.orientation = ["arm-chair"]
        else:
            self.orientation = orientation

        if not isinstance(self.orientation, (list, tuple)):
            self.orientation = [orientation]
        if len(self.orientation) == 1:
            self.orientation = self.orientation*self.wallsNumber
        assert len(self.orientation) == self.wallsNumber

        super(MultipleWallNanotube,self).__init__(*args, **kwargs)


    def construct(self):

        for layerNumber in range(self.wallsNumber):
            # construct nanotube
            cnt = Nanotube(element = self.element,
                           length = self.length,
                           radius = self.radius + layerNumber*self.interlayerDistance,
                           orientation = self.orientation[layerNumber],
                           bondLength = self.bondLength).construct().get_pdb()
            # set residue name
            set_residue_name(cnt.indexes, cnt, "NT%s"%layerNumber)
            # concatenate
            self.get_pdb().concatenate(cnt)

        return self





class SuperCell(Construct):
    """
    Constructs a super cell out of a pdb. SuperCell will be generated in positive quadrant
    """
    def __init__(self, pdb,
                       cellAxes,
                       supercellSize,
                       *args, **kwargs):
        self.pdbs = pdb
        self.cellAxes = cellAxes
        self.supercellSize = supercellSize

        # initialize
        self.initialize()

        # The base class constructor.
        super(SuperCell,self).__init__(*args, **kwargs)


    def initialize(self):
        assert isinstance(self.cellAxes, (list, tuple, np.ndarray))
        self.cellAxes = np.array([np.array(vector) for vector in self.cellAxes])
        if self.cellAxes.shape  in ((3,),(3,1)):
            self.cellAxes = np.array([[self.cellAxes[0],0,0], [self.cellAxes[0],0,0], [0,0,self.cellAxes[2]]])
        else:
            assert self.cellAxes.shape == (3,3)

        assert isinstance(self.supercellSize, (list, tuple))
        assert len(self.supercellSize) == 3
        self.supercellSize = [int(item) for item in self.supercellSize]
        assert self.supercellSize[0] >= 1
        assert self.supercellSize[1] >= 1
        assert self.supercellSize[2] >= 1


    def construct(self):
        Logger.info("Constructing %r supercell"%self.pdbs[0].name)
        # translate to origin and initiate pdb
        minMax = get_min_max(self.pdbs[0].indexes, self.pdbs[0])
        pdb = self.pdbs[0].get_copy()
        translate(pdb.indexes, pdb, [-minMax[0], -minMax[2], -minMax[4]])

        finalRecordsNumber = self.supercellSize[0]*len(pdb)
        finalRecordsNumber += (self.supercellSize[1]-1)*finalRecordsNumber
        finalRecordsNumber += (self.supercellSize[2]-1)*finalRecordsNumber
        for direction in range(3):
            pdbCopy = pdb.get_copy()
            stepIncrement = len(pdb)

            for insertionNumber in range(self.supercellSize[direction]-1):
                self.status(len(pdb), finalRecordsNumber, stepIncrement = stepIncrement)
                thisCopy = pdbCopy.get_copy()

                for idx in thisCopy.indexes:
                    thisCopy.records[idx]["sequence_number"] += len(pdb)

                translate(thisCopy.indexes, thisCopy, (insertionNumber+1)*self.cellAxes[direction,:])
                pdb.concatenate(thisCopy)
        # set boundary conditions
        BC = PeriodicBoundaries(np.array([self.cellAxes[0,:]*self.supercellSize[0],
                                          self.cellAxes[1,:]*self.supercellSize[1],
                                          self.cellAxes[2,:]*self.supercellSize[2]]))
        pdb.set_boundary_conditions( BC )
        # set pdb
        self.set_pdb(pdb)
        Logger.info("Supercell successfully constructed. Initial number of records is %r, final number of records is %r "%(len(self.pdbs[0]), len(self._pdb)))
        return self


    def orthorhombic(self):
        """
        """
        Logger.info("Transforming cell to orthorhombic")
        # get axis
        cellAxes = np.array([self.cellAxes[idx]*self.supercellSize[idx] for idx in range(3)])
        axesLength = np.array([np.linalg.norm(vec) for vec in cellAxes])
        Logger.info("Orthorhombic vectors %r"%axesLength)
        # get pdb
        pdb = self.pdb.get_copy()
        # translate all records to positive coordinates
        minMax = get_min_max(pdb.indexes, pdb)
        translate(pdb.indexes, pdb, [-minMax[0], -minMax[2], -minMax[4]])
        # create box
        sb = PeriodicBoundaries()
        sb.setVectors(axesLength)
        # get coordinates
        realCoords = pdb.coordinates
        # fold into box
        cubicCoords = sb.foldRealArray(realCoords)
        set_coordinates(pdb.indexes, pdb, cubicCoords)

        return pdb



    def cubic(self):
        pdb = self.orthorhombic()

        #pdb.visualize()
