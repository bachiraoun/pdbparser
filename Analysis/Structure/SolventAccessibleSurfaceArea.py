"""
This module provides all solvent accessible surface area analysis classes.

.. inheritance-diagram:: pdbparser.Analysis.Structure.SolventAccessibleSurfaceArea
    :parts: 2
"""
# standard libraries imports
from __future__ import print_function
import os, atexit, copy

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.log import Logger
from pdbparser.Analysis.Core import Analysis
from pdbparser.Utilities.Information import get_records_database_property_values
from pdbparser.Utilities.Collection import is_number, generate_sphere_points


class _AtomsData(object):
    def __init__(self, tempdir=None):
        # create tempdir
        self.__tempdir = tempdir
        if tempdir is None:
            self.__dirExists = True
        else:
            assert isinstance(tempdir, str), 'tempdir must be a string'
            if os.path.isdir(tempdir):
                self.__dirExists = True
            else:
                self.__dirExists = False
                os.makedirs(tempdir)
            atexit.register(self.on_exit)

        # data keys are radii and values and spheres
        self.__data    = {}
        # keys are atoms index
        self.__points  = {} # values are atoms sphere points
        self.__radius  = {} # values are spheres radius

    def __get_sphere_points(self, radius, resolution, center):
        if radius not in self.__data:
            surface    = 4. * np.pi * radius**2
            npoints    = int(np.ceil( surface/resolution ))
            self.__data[radius] = np.array(generate_sphere_points(radius=radius, nPoints=npoints, center=center)).astype(float)
        return self.__data[radius]

    def get(self, idx, radius, center, resolution):
        if idx not in self.__points:
            points = self.__get_sphere_points(radius=radius, resolution=resolution, center=[0,0,0]) + center
            if self.__tempdir is None:
                self.__points[idx] = points
            else:
                pointsPath = os.path.join( self.__tempdir, '%i_points'%idx )
                np.save( pointsPath, points )
                self.__points[idx]  = pointsPath+'.npy'
            self.__radius[idx]  = radius
        elif self.__tempdir is not None:
            points = np.load( self.__points[idx] )
        else:
            points = self.__points[idx]
        # return
        return self.__radius[idx], points

    def get_copy(self, idx, radius, center, resolution):
        radius, points = self.get(idx=idx, radius=radius, center=center, resolution=resolution)
        if self.__tempdir is None:
            points = copy.deepcopy(points)
        return radius, points

    def update(self, idx, points):
        assert idx in self.__points, 'points for atom %i are not defined'%idx
        if self.__tempdir is None:
            self.__points[idx] = points
        else:
            np.save( self.__points[idx], points )

    def get_all_points(self):
        points = list(self.__points.values())
        if self.__tempdir is not None:
            points = [np.load(path) for path in points]
        return np.concatenate(points, axis=0)

    def number_of_points(self):
        points = list(self.__points.values())
        if self.__tempdir is None:
            pointsLen = [v.shape[0] for v in points]
        else:
            pointsLen = [np.load(path).shape[0] for path in points]
        return sum(pointsLen)

    def reset(self):
        if self.__tempdir is not None:
            [os.remove(path) for path in self.__points.values()]
        self.__data    = {}
        self.__points  = {}
        self.__radius  = {}

    def on_exit(self):
        self.reset()
        if not self.__dirExists:
            os.rmdir(self.__tempdir)


class SolventAccessibleSurfaceArea(Analysis):
    """
    """
    def __init__(self, trajectory, configurationsIndexes,
                       targetAtomsIndexes, atomsRadius='vdwRadius', makeContiguous=False,
                       probeRadius=1.5, resolution=0.5,
                       storeSurfacePoints=False, tempdir=None,
                       *args, **kwargs):
        # set trajectory
        super(SolventAccessibleSurfaceArea,self).__init__(trajectory, *args, **kwargs)
        # set steps indexes
        self.targetAtomsIndexes = self.get_atoms_indexes(targetAtomsIndexes)
        # set configurations indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        self.numberOfSteps = len(self.configurationsIndexes)
        # initialize variables
        self.__initialize_variables__(makeContiguous = makeContiguous,
                                      atomsRadius    = atomsRadius,
                                      probeRadius    = probeRadius,
                                      resolution     = resolution,
                                      storeSurfacePoints = storeSurfacePoints,
                                      tempdir        = tempdir)
        # initialize results
        self.__initialize_results__()

    def __initialize_variables__(self, makeContiguous, atomsRadius, probeRadius, resolution, storeSurfacePoints, tempdir):
        assert isinstance(makeContiguous, bool), 'makeContiguous must be boolean'
        self.__makeContiguous = makeContiguous
        # check probeRadius
        assert is_number(probeRadius), "probeRadius must be a number"
        probeRadius = float(probeRadius)
        assert probeRadius>=0, 'probeRadius must be bigger or equal to 0'
        # check atomsRadius
        if isinstance(atomsRadius, str):
            atomsRadius = get_records_database_property_values(indexes=self.targetAtomsIndexes, pdb=self.structure, property=atomsRadius)
        assert len(atomsRadius) == len(self.targetAtomsIndexes), 'atomsRadius must have the same number of input as target atoms'
        assert all([is_number(v) for v in atomsRadius]), 'all atomsRadius must be numbers'
        atomsRadius = [float(v) for v in atomsRadius]
        assert all([v>0 for v in atomsRadius]), 'all atomsRadius must be >0'
        # set atoms radius
        self.__atomsRadius = np.array(atomsRadius).astype(float) + probeRadius
        # set atoms data
        self.__atomsData = _AtomsData(tempdir=tempdir)
        # check probeRadius
        assert is_number(resolution), "resolution must be a number"
        resolution = float(resolution)
        assert resolution>0, 'resolution must be > 0'
        self.__resolution = resolution
        assert isinstance(storeSurfacePoints, bool), 'storeSurfacePoints must be boolean'
        self.__storeSurfacePoints = storeSurfacePoints

    def __initialize_results__(self):
        # time
        self.results['time'] = np.array([self.time[idx] for idx in self.configurationsIndexes], dtype=np.float)
        # sasa
        self.results['sasa'] =  np.zeros((len(self.configurationsIndexes)), dtype=np.float)
        # surface points
        self.results['surface points'] =  []

    @property
    def atomsData(self):
        return self.__atomsData

    def step(self, index):
        """
        analysis step of calculation method.\n

        :Parameters:
            #. atomIndex (int): the atom step index

        :Returns:
            #. stepData (object): object used in combine method
        """
        self.__atomsData.reset()
        # get configuration index
        confIdx = self.configurationsIndexes[index]
        # set working configuration index
        self._trajectory.set_configuration_index(confIdx)
        # get coordinates
        if self.__makeContiguous:
            coordinates = self._trajectory.get_contiguous_configuration_coordinates(confIdx)
        else:
            coordinates = self._trajectory.get_configuration_coordinates(confIdx)
        # get sub-selection
        coordinates = coordinates[self.targetAtomsIndexes]

        for idx0 in range(coordinates.shape[0]-1):
            center0 = coordinates[idx0,:]
            # get atom 0 data
            radius0, sphere0 = self.__atomsData.get_copy( idx=idx0, radius=self.__atomsRadius[idx0], center=center0, resolution=self.__resolution )
            nPoints0 = sphere0.shape[0]
            # compute distances
            diff = coordinates[idx0+1:,:] - center0
            dist = np.sqrt( np.sum(diff**2, axis=1) )
            # get atoms closer than vdw0 + vdw1+ 2*probeRadius
            comp = self.__atomsRadius[idx0+1:] + self.__atomsRadius[idx0]
            ddif = dist-comp
            # loop atoms
            #print('before', idx0, sphere0.shape)
            for i, dval in enumerate(ddif):
                if dval>0:
                    continue
                idx1 = i+idx0+1
                # get tolerance distance
                tolerance = self.__atomsRadius[idx0]+self.__atomsRadius[idx1]
                center1 = coordinates[idx1,:]
                # get atom 1 data
                radius1, sphere1 = self.__atomsData.get_copy( idx=idx1, radius=self.__atomsRadius[idx1], center=center1, resolution=self.__resolution )
                nPoints1 = sphere1.shape[0]
                # get points of sphere0 to remove
                #print(idx0, idx1, pdbSDS.elements[idx0], pdbSDS.elements[idx1], radius0, radius1)
                dist0 = np.sqrt( np.sum( (sphere0-center1)**2, axis=1 ) )
                sphere0 = sphere0[dist0>radius1,:]
                dist1 = np.sqrt( np.sum( (sphere1-center0)**2, axis=1 ) )
                sphere1 = sphere1[dist1>radius0,:]
                # update sphere 1
                if nPoints1>sphere1.shape[0]:
                    #print('1', idx1, nPoints1, sphere1.shape[0])
                    self.__atomsData.update(idx1, sphere1)
            # update sphere 0
            #print('after', idx0, sphere0.shape)
            if nPoints0>sphere0.shape[0]:
                #print('0', idx0, nPoints0, sphere0.shape[0])
                self.__atomsData.update(idx0, sphere0)
        # return
        sasa = self.__atomsData.number_of_points()*self.__resolution
        surfacePoints = None
        if self.__storeSurfacePoints:
            surfacePoints = self.__atomsData.get_all_points()
        return index, (sasa, surfacePoints)

    def combine(self, index, stepData):
        """
        analysis combine method called after each step.\n

        :Parameters:
            #. atomIndex (int): the atomIndex of the last calculated atom
            #. stepData (object): the returned data from step method
        """
        self.results['sasa'][index] = stepData[0]
        self.results['surface points'].append( stepData[1] )

    def finalize(self):
        """
        called once all the steps has been run.\n
        """
        pass
