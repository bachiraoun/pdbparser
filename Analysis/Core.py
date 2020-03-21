"""
This module provides the mother Analysis class and other definitions and classes used in analysis calculations.

.. inheritance-diagram:: pdbparser.Analysis.Core
    :parts: 2
"""
# standard libraries imports
from __future__ import print_function
import tempfile
import os
import zipfile
import pickle

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.log import Logger
from pdbparser.Utilities.Geometry import *
from pdbparser.Utilities.Information import *
from pdbparser.pdbparser import pdbparser, pdbTrajectory
from pdbparser.Utilities.BoundaryConditions import InfiniteBoundaries, PeriodicBoundaries
from pdbparser.Utilities.Database import get_element_property, is_element_property

class Analysis(object):
    """
    The mother class of all analysis classes. It can be initialized to load and visualize saved analysis only
    """
    def __init__(self, trajectory, *args, **kwargs):
        # set trajectory instance
        self.set_trajectory(trajectory)
        # set kwargs attributes
        self.numberOfSteps = 0
        # create results dictionary
        self.results = {}
        # create formats
        self.__formats__ = {"cdl":self.__load_cdl__,
                            "binary":self.__load_binary__,
                            "datasheet":self.__load_datasheet__}

    @property
    def structure(self):
        if isinstance(self._trajectory, pdbparser):
            return self._trajectory
        elif isinstance(self._trajectory, pdbTrajectory):
            return self._trajectory._structure
        else:
            raise Logger.error("trajectory must be a pdbparser or pdbTrajectory instance")

    @property
    def time(self):
        if isinstance(self._trajectory, pdbparser):
            return [0]
        elif isinstance(self._trajectory, pdbTrajectory):
            return self._trajectory._time
        else:
            raise Logger.error("trajectory must be a pdbparser or pdbTrajectory instance")

    @property
    def numberOfConfigurations(self):
        if isinstance(self._trajectory, pdbparser):
            return 1
        elif isinstance(self._trajectory, pdbTrajectory):
            return len(self._trajectory)
        else:
            raise Logger.error("trajectory must be a pdbparser or pdbTrajectory instance")

    @property
    def numberOfAtoms(self):
        if isinstance(self._trajectory, pdbparser):
            return len(self._trajectory)
        elif isinstance(self._trajectory, pdbTrajectory):
            return self._trajectory.numberOfAtoms
        else:
            raise Logger.error("trajectory must be a pdbparser or pdbTrajectory instance")

    def get_trajectory_indexes(self, indexes):
        """
        check and return indexes if they are in trajectory's range.\n

        :Parameters:
            #. indexes (list): The list of indexes

        :Returns:
            #. indexes (list): the verified list of indexes
        """
        assert isinstance(indexes, (list, set, tuple)), Logger.error("indexes must be a list of positive integers smaller than trajectory's length")
        indexes = sorted(set(indexes))
        assert not len([False for idx in indexes if (idx%1!=0 or idx<0 or idx>=self.numberOfConfigurations)]), Logger.error("indexes must be a list of positive integers smaller than trajectory's length")
        return [int(idx) for idx in indexes]

    def get_atoms_indexes(self, indexes, sort=True, removeRedundancy=True):
        """
        check and return indexes if they are in trajectory number of atoms range.\n

        :Parameters:
            #. indexes (list): The list of indexes
            #. sort (boolean): Sort indexes from smaller to bigger number
            #. removeRedundancy (set): Remove all redundant indexes

        :Returns:
            #. indexes (list): the verified list of indexes
        """
        assert isinstance(indexes, (list, set, tuple)), Logger.error("indexes must be a list of positive integers smaller than number of atoms")
        indexes = list(indexes)
        if removeRedundancy:
            indexes = set(indexes)
        if sort:
            indexes = sorted(indexes)
        assert not len([False for idx in indexes if (idx%1!=0 or idx<0 or idx>=self.numberOfAtoms)]), Logger.error("indexes must be a list of positive integers smaller than number of atoms")
        return [int(idx) for idx in indexes]

    def clear_results(self):
        """ clears all results """
        self.results = {}

    def set_trajectory(self, trajectory):
        """
        set the trajectory for analysis.\n

        :Parameters:
            #. pdb (pdbparser): The pdb instance replacing the constructed self.pdb.
        """
        assert isinstance(trajectory, (pdbparser, pdbTrajectory)), Logger.error("trajectory must be a pdbparser or pdbTrajectory instance")
        self._trajectory = trajectory
        self._boundaryConditions = self._trajectory.simulationBox

    def set_simulation_box(self, simulationBox):
        """
        set the simulation box for the current pdb analysis.\n

        :Parameters:
            #. simulationBox (pdbparser.simulationBox): The simulationBox instance
        """
        assert isinstance(simulationBox, (InfiniteBoundaries, PeriodicBoundaries)), Logger.error("simulationBox must be a InfiniteBoundaries or PeriodicBoundaries instance")
        # create PeriodicBoundaries
        if isinstance(self._trajectory, pdbparser):
            assert len(simulationBox) == 1, Logger.error("trajectory is a simngle pdb, simulationBox length must be 1")
        else:
            assert len(simulationBox) == len(self._trajectory), Logger.error("simulationBox length must be equal to length of trajectory")
        self._boundaryConditions = simulationBox

    def run(self):
        assert self.numberOfSteps>0 and self.numberOfSteps%1==0, Logger.error("numberOfSteps must be a positive integer, '%s' is given"%self.numberOfSteps)
        # run steps
        for idx in range(self.numberOfSteps):
            # log status
            self.status(step=idx, logFrequency = 10)
            # run step and combine
            self.combine(*self.step(idx))
        # finalize
        self.status(step=self.numberOfSteps, logFrequency = 10)
        self.finalize()

    def step(self, index):
        """
        analysis step of calculation method.\n

        :Parameters:
            #. index (int): the step index
        """
        raise Logger.error("step method is not implemented")

    def combine(self, index, stepData):
        """
        analysis combine method called after each step.\n

        :Parameters:
            #. index (int): the index of the last calculated step
            #. stepData (object): the returned data from step method
        """
        pass

    def finalize(self):
        """
        called once all the steps has been run.\n
        """
        pass

    def status(self, step, logFrequency = 10):
        """
        This method is used to log analysis status.\n

        :Parameters:
            #. step (int): The current step number
            #. logFrequency (float): the frequency of status logging. its a percent number.
        """
        if not step:
            Logger.info("%s --> analysis started %s steps to go" %(self.__class__.__name__, self.numberOfSteps))
        elif step == self.numberOfSteps:
            Logger.info("%s --> analysis steps finished" %(self.__class__.__name__))
        else:
            actualPercent = int( float(step)/float(self.numberOfSteps)*100)
            previousPercent = int(float(step-1)/float(self.numberOfSteps)*100)
            if actualPercent/logFrequency != previousPercent/logFrequency:
                Logger.info("%s --> %s%% completed. %s left out of %s" %(self.__class__.__name__, actualPercent, self.numberOfSteps-step, self.numberOfSteps))

    def save(self, path, formats=None):
        """
        Used to export the analysis results stored in self.results dictionary.\n

        :Parameters:
            #. path (str): The saving path.
            #. format (str): The export format. used formats are ascii or bin
        """
        if formats is None:
            formats = ["ascii"]
        elif isinstance(formats, str):
            formats = [formats]
        else:
            assert isinstance(formats, (list, tuple))
            formats = list(formats)

        for f in formats:
            if f == "ascii":
                self.__save_ascii__(str(path)+".zip")
            elif f == "datasheet":
                self.__save_datasheet__(str(path)+".xls")
            elif f == "bin":
                self.__save_binary__(str(path)+".pkl")
            else:
                raise Logger.error("Unknown saving format %r. only %s formats are acceptable" %(f,["ascii","bin",'datasheet']))
        return self

    def load(self, paths, format=None):
        """
        Used to import the analysis results and updates self.results dictionary.\n
        If same result data key or name is found, it will be automatically updated

        :Parameters:
            #. paths (str, list, tuple): a list of analysis paths to load.
        """
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            if format is not None:
                self.__formats__[format](path)
            else:
                if zipfile.is_zipfile(path):
                    self.__load_ascii__(path)
                else:
                    try:
                        pickle.load(open(path,'r'))
                    except:
                        func = self.__load_datasheet__
                    else:
                        func = self.__load_binary__
                    func(path)
        return self

    def plot(self, x, y):
        """
        Simple plotting tool to visualize the analysis data.\n
        for this function matplotlib must be installed

        :Parameters:
            #. x (key, numpy.array): a key from self.results or a numpy.array that will be x-axis of the plot
            #. y (key, numpy.array): a key from self.results or a numpy.array that will be the y-axis of the plot
        """
        try:
            import matplotlib.pyplot as plt
        except:
            Logger.warn("matplotlib is not installed. Plotting cannot be proceeded")
            return
        if isinstance(x, str):
            assert x in self.results
            xLabel = x
            x = self.results[x]
        else:
            assert isinstance(x, (list, tuple, np.ndarray))
            xLabel = "x"
        if isinstance(y, str):
            assert y in self.results
            yLabel = y
            y = self.results[y]
        else:
            assert isinstance(y, (list, tuple, np.ndarray))
            yLabel = "y"
        xData = np.array(x)
        yData = np.array(y)
        assert xData.shape == yData.shape

        # plot
        plt.plot(xData,yData)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()
        plt.show()

    def __save_ascii__(self, path):
        tempDir = tempfile.gettempdir()
        # open zip file
        zf = zipfile.ZipFile(path, 'w')
        for fileName, data in self.results.items():
            np.savetxt(os.path.join(tempDir, fileName+".dat"), data, header = "%s"%fileName)
            zf.write(os.path.join(tempDir, fileName+".dat"))

    def __save_datasheet__(self, path):
        keys = list(self.results.keys())
        resultsLength = None
        for key in keys:
            resSize = np.sum(self.results[key].shape)
            if len(self.results[key].shape)>1:
                raise Logger.error("result %r is of dimension %s, only one dimensional results can be saved to datasheet. Try using other formats." %(key,self.results[key].shape))
            else:
                if resultsLength is None:
                    resultsLength = resSize
                elif resSize != resultsLength:
                    raise Logger.error("All results must have the same size.")
        resultsArray = np.empty((resultsLength,len(keys)))
        for idx in range(len(keys)):
            resultsArray[:,idx] = self.results[keys[idx]]
        np.savetxt(path, resultsArray, header = '  ;  '.join(keys), delimiter='  ;  ')

    def __save_binary__(self, path):
        fd = open(path,'w')
        pickle.dump(self.results, fd)
        fd.close()

    def __load_ascii__(self, path):
        tempDir = tempfile.gettempdir()
        # create temp dir
        tempDir = os.path.join(tempDir,"pdbparserTmpDir")
        if not os.path.exists(tempDir):
            try:
                os.makedirs(tempDir)
            except:
                raise Logger.error("Couldn't create temporary folder %r to extract data" %tempDir)
        # open zipfile
        try:
            zf = zipfile.ZipFile(path, 'r')
        except:
            raise Logger.error("Couldn't open analysis ascii zip file %r." %path)

        # get files names and extract them all to tempDir
        files = zf.namelist()
        zf.extractall(tempDir)

        # read analysis files data
        for file in files:
            if os.path.basename(file) in self.results:
                Logger.warn("analysis name %r already exists. Previous values will be erased and updated with the new ones" %file)
            self.results[os.path.basename(file)] = np.loadtxt(os.path.join(tempDir,file))

    def __load_binary__(self, path):
        # open file
        try:
            fd = open(path,'r')
        except:
            raise Logger.error("Couldn't open analysis binary file %r." %path)
        # read file
        try:
            resDict = pickle.load(fd)
        except:
            fd.close()
            raise Logger.error("Couldn't read analysis binary file data %r." %path)
        else:
            fd.close()
            for key, values in resDict.items():
                if key in self.results:
                    Logger.warn("analysis name %r already exists. Previous values will be erased and updated with the new ones" %key)
                self.results[key] = values

    def __load_datasheet__(self, path):
        # open file
        try:
            fd = open(path,'r')
        except:
            raise Logger.error("Couldn't open analysis file %r." %path)
        # read keys
        firstLine = fd.readline()
        fd.close()
        try:
            keys = [key.strip() for key in firstLine.split('#')[1].split(';')]
        except:
            raise Logger.error("Couldn't read the 'first line' from analysis datasheet file %r first line." %path)
        # read values
        try:
            values = np.loadtxt(path, delimiter = ";")
        except:
            raise Logger.error("Couldn't read the 'data' from analysis datasheet file %r." %path)
        # test size
        if values.shape[1] != len(keys):
            raise Logger.error("values and keys length doesn't match, datasheet file %r seems corrupted." %path)
        # update results
        for idx in range(len(keys)):
            key = keys[idx]
            if key in self.results:
                Logger.warn("analysis name %r already exists. Previous values will be erased and updated with the new ones" %key)
            self.results[key] = values[:,idx]

    def __load_cdl__(self, path):
        # open file
        try:
            fd = open(path,'r')
        except:
            raise Logger.error("Couldn't open analysis file %r." %path)
        # read lines
        lines = fd.readlines()
        fd.close()

        # find dimensions
        while lines:
            line = lines.pop(0).strip()
            if "dimensions:" == line:
                break
        if not len(lines):
            raise Logger.error("Couldn't find the data dimensions in cdl file %r"%path)
        dimensions = {}
        while lines:
            line = lines.pop(0).strip()
            if "variables:" == line:
                break
            else:
                splitted = line.split("=")
                assert len(splitted) == 2, "dimensions line %r has bad format"%line
                dimensions[splitted[0].strip()] = int(float(splitted[1].split(";")[0]))

        # find variables name
        while lines:
            line = lines.pop(0).strip()
            if "// global attributes:" == line:
                break
            elif not len(line):
                break
            _, var = line.split()[0:2]
            varName, dim = var.split("(")
            dim = dim.split(")")[0]
            self.results[varName] = dimensions[dim]
            lines.pop(0)


        while lines:
            line = lines.pop(0).strip()
            if "data:" == line:
                break
        if not len(lines):
            raise Logger.error("Couldn't find the data start in cdl file %r"%path)
        # get data
        dataName = None
        dataVar = []
        for l in lines:
            line = l.strip()
            # empty line
            if not len(line):
                continue
            # end of data
            if "}" in line:
                break
            # end of variable data
            if ";" in line:
                line = line.split(";")[0]
                try:
                    var = [float(d) for d in line.split(",") if len(d)]
                except:
                    raise Logger.error("Couldn't convert data line %r"%line)
                dataVar.extend( var )
                assert self.results[dataName] == len(dataVar), "length of data found and variable %s dimension %s don't match"%(dataName,self.results[dataName])
                self.results[dataName] = np.array(dataVar)
                continue
            # body of variable
            if "=" not in line:
                assert dataName is not None, "data format is bad"
                try:
                    var = [float(d) for d in line.split(",") if len(d)]
                except:
                    raise Logger.error("Couldn't convert data line %r"%line)
                dataVar.extend( var )
            # beginning of variable
            else:
                dataName, data = line.split("=")
                dataName = dataName.strip()
                assert dataName in self.results.keys(), "data name found not declared in file header"
                dataVar = [float(d) for d in data.split(",") if len(d)]






class Definition(object):
    def __init__(self, trajectory, definition):
        assert isinstance(trajectory, (pdbparser, pdbTrajectory)), Logger.error("trajectory must be pdbparser or pdbTrajectory instance")
        self.__trajectory = trajectory
        self.__definition = self.get_definition(definition)

    @property
    def definition(self):
        return self.__definition

    @property
    def trajectory(self):
        return self.__trajectory

    def get_definition(self):
         raise Logger.error("This method must be overloaded")


class CenterDefinition(Definition):
    def __init__(self, trajectory, definition):
        """
        initialize center Definition.\n

        :Parameters:
            #. trajectory (pdbparser, pdbTrajectory): The AxisDefinition parent.
            #. definition (dictionary): The center definition
        """
        super(CenterDefinition,self).__init__(trajectory, definition)

    def get_definition(self, center):
        """
        alias to get_axis_definition
        """
        return self.get_center_definition(center)

    def get_center_definition(self, center):
        """
        check and return axis definition.\n

        :Parameters:
            #. center (dictionary): The center definition

        :returns:
            #. center (dictionary): The verified center definition
        """
        assert isinstance(center, dict), Logger.error("center must be a dictionary")
        assert list(center.keys()) in (["fixed"], ["selection"]), Logger.error("center can have one of two keys '%s'" %(["fixed", "selection"]))
        key = list(center.keys())[0]
        value = list(center.values())[0]
        # fixed center
        if key == "fixed":
            assert isinstance(value, (list, tuple, set, numpy.array)), Logger.error("fixed center value must be a list of three floats")
            value = list(value)
            assert len(value)==3, Logger.error("fixed center value must be a list of three floats")
            try:
                value = np.array([float(val) for val in value])
            except:
                raise Logger.error("fixed center value must be a list of three floats")
        # selection center
        elif key == "selection":
            assert isinstance(value, dict), Logger.error("selection value must be a dictionary")
            assert sorted(value.keys()) == (sorted(["indexes","weighting"])), Logger.error("center selection value dictionary must have two keys '%s'" %(["indexes","weighting"]))
            indexes = get_atoms_indexes(self.trajectory, value["indexes"])
            weighting = value["weighting"]
            assert is_element_property(weighting), Logger.error("weighting '%s' don't exist in database"%weighting)
            elements = self.trajectory.elements
            weights = np.array([get_element_property(elements[idx], weighting) for idx in indexes])
            value = {"indexes":indexes, "weighting":weighting, "weights":weights}
        else:
            raise Logger.error("center definition not valid")
        return {key:value}

    def get_center(self, coordinates):
        """
        return the center.\n

        :Parameters:
            #. coordinates (numpy.array): The atoms coordinates

        :returns:
            #. center (numpy.array): the center
        """
        # fixed center definition
        if list(self.definition.keys())[0] == "fixed":
            return list(self.definition.values())[0]
        # selection center definition
    elif list(self.definition.keys())[0] == "selection":
            indexes = list(self.definition.values())[0]["indexes"]
            weights = list(self.definition.values())[0]["weights"]
            return  np.sum(weights*np.transpose(coordinates[indexes,:]),1)/len(indexes)

class AxisDefinition(Definition):
    def __init__(self, trajectory, definition):
        """
        initialize Axis Definition.\n

        :Parameters:
            #. trajectory (pdbparser, pdbTrajectory): The AxisDefinition parent.
            #. definition (dictionary): The axis definition
        """
        super(AxisDefinition,self).__init__(trajectory, definition)

    def get_definition(self, axis):
        """
        alias to get_axis_definition
        """
        return self.get_axis_definition(axis)

    def get_axis_definition(self, axis):
        """
        check and return axis definition.\n

        :Parameters:
            #. axis (dictionary): The axis definition

        :returns:
            #. axis (dictionary): The verified axis definition
        """
        assert isinstance(axis, dict), Logger.error("axis must be a dictionary")
        assert list(axis.keys()) in (["principal"], ["vector"], ["selection"]), Logger.error("axis can have one of three keys '%s'" %(["principal", "vector", "selection"]))
        key = list(axis.keys())[0]
        value = list(axis.values())[0]
        # principal axis definition
        if key == "principal":
            value = get_atoms_indexes(self.trajectory, value)
            assert len(value), Logger.error("principal axis values must be a not empty list of indexes")
        # fixed vector definition
        elif key == "vector":
            assert isinstance(value, (list,tuple,set)), Logger.error("vector axis values must be a list")
            value = list(value)
            assert len(value)==2, Logger.error("vector axis values must be a list of 2 points coordinates")
            assert len([list(item) for item in value if isinstance(item, (list, tuple, set, np.ndarray))])==2, Logger.error("every vector axis point coordinates must be a list")
            try:
                value = [np.array(item, dtype=np.float32) for item in value if len(item)==3]
            except:
                raise Logger.error("every vector axis point coordinates must be a list of three numbers")
            else:
                X = value[1]-value[0]
                norm = np.linalg.norm(X)
                assert norm>10**-6, Logger.error("vector axis can't return 0 vector")
                X /= norm
                Y = np.array([1,1,1])+np.random.random(3)*X
                Y = np.cross(X,Y)
                norm = np.linalg.norm(Y)
                Y /= norm
                assert np.dot(X,Y)<10**-6, Logger.error("vector axis can't return non orthogonal X and Y vectors")
                Z = np.cross(X, Y)
                norm = np.linalg.norm(Z)
                assert norm>10**-6, Logger.error("vector axis can't return 0 Z vector")
                Z /= norm
                RM = np.linalg.inv(np.array([X,Y,Z]))
                value = {"X":X, "Y":Y, "Z":Z, "center":0.5*(value[1]+value[0]), "rotationMatrix":RM}
        # atoms selection definition
        elif key == "selection":
            assert isinstance(value, (list,tuple,set)), Logger.error("selection axis values must be a list")
            value = list(value)
            assert len(value)==2, Logger.error("selection axis values must be a list of selections")
            try:
                value = [get_atoms_indexes(self.trajectory, item) for item in value]
            except:
                raise Logger.error("every axis selection value must be a list of atoms indexes")
            else:
                assert len([list(item) for item in value if len(item)])==2, Logger.error("none of both axis selection values can return and empty atoms indexes")
        else:
            raise Logger.error("axis definition not valid")
        return {key:value}

    def get_center_rotationMatrix(self, coordinates):
        """
        return the center and the rotation matrix.\n

        :Parameters:
            #. coordinates (numpy.array): The atoms coordinates

        :returns:
            #. center (numpy.array): the center
            #. rotationMatrix (numpy.array): the (3X3) rotation matrix
        """
        # principal axis definition
        if list(self.definition.keys())[0] == "principal":
            indexes = list(self.definition.values())[0]
            center,_,_,_,vect1,vect2,vect3 = get_principal_axis(indexes, self.trajectory)
            rotationMatrix = np.linalg.inv(np.array([vect1, vect2, vect3]))
        # vector definition
    elif list(self.definition.keys())[0] == "vector":
            values = list(self.definition.values())[0]
            center = values["center"]
            rotationMatrix = values["rotationMatrix"]
        # selection definition
    elif list(self.definition.keys())[0] == "selection":
            selections = list(self.definition.values())[0]
            X0 = np.sum(coordinates[selections[0]],0)/len(selections[0])
            X1 = np.sum(coordinates[selections[1]],0)/len(selections[1])
            X = X1-X0
            norm = np.linalg.norm(X)
            assert norm>10**-6, Logger.error("vector axis can't return 0 vector")
            X /= norm
            Y = np.array([1,1,1])+np.random.random(3)*X
            Y = np.cross(X,Y)
            norm = np.linalg.norm(Y)
            Y /= norm
            assert np.dot(X,Y)<10**-6, Logger.error("vector axis can't return non orthogonal X and Y vectors")
            Z = np.cross(X, Y)
            norm = np.linalg.norm(Z)
            assert norm>10**-6, Logger.error("vector axis can't return 0 Z vector")
            Z /= norm
            center = 0.5*(X1+X0)
            rotationMatrix = np.linalg.inv(np.array([X,Y,Z]))
        return center, rotationMatrix
