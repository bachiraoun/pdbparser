"""
This module provides classes to convert molecular dynamics simulation outputs to pdbTrajectory

.. inheritance-diagram:: pdbparser.IO.TrajectoryConverter
    :parts: 2

"""
# standard libraries imports
from __future__ import print_function

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.pdbparser import pdbTrajectory
from pdbparser.IO.Core import Converter, DCDFile
from pdbparser.Utilities.BoundaryConditions import InfiniteBoundaries, PeriodicBoundaries
from pdbparser.log import Logger


class DCDConverter(Converter):

    def __init__(self, pdb, dcd, indexes=None, format="charmm"):
        """
        Read new simulation trajectory

        :Parameters:
            #. pdb (string): NAMD pdb file used as trajectory structure file
            #. dcd (string): NAMD DCD output file
            #. indexes (list): the configuration indexes to convert. None converts all configurations
            #. format (string): the known formats. only charmm and dcd are supported.
        """
        # initialize converter
        super(DCDConverter,self).__init__()
        # log some info
        Logger.info("Converting NAMD trajectory")
        Logger.info("pdb file path: %s"%pdb)
        Logger.info("dcd file path: %s"%dcd)
        # check format
        assert isinstance(format, str), Logger.error("format must be a string")
        self.format = str(format).lower()
        assert self.format in ("charmm", "namd"), Logger.error("format must be either charmm or namd")
        # check indexes
        if indexes is not None:
            assert isinstance(indexes, (list, tuple, set, np.ndarray)), Logger.error("indexes must be a list of 3 integers [start, end, step]")
            indexes = [int(idx) for idx in sorted(set(indexes))]
            assert indexes[0]>=0, Logger.error("indexes start must be positive")
        self.indexes = indexes
        # check pdb file
        try:
            fd = open(pdb,'r')
        except:
            raise Logger.error("cannot open pdb file")
        else:
            fd.close()
            self.pdb = pdb
        # check dcd file
        try:
            fd = open(dcd,'r')
        except:
            raise Logger.error("cannot open dcd file")
        else:
            fd.close()
            self.dcd=dcd
        # create trajectory
        self.trajectory = None

    def __unit_cell_to_basis_vectors__(self, a, b, c, alpha, beta, gamma):
        # By construction the a vector is aligned with the x axis.
        X = np.array([a, 0.0, 0.0])
        # By construction the Y vector is in the xy plane.
        Y = b*np.array([np.cos(gamma), np.sin(gamma), 0.0])
        # calculate the Z vector
        Zx = np.cos(beta)
        Zy = (np.cos(alpha)-np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
        Zz = np.sqrt(1.0 - Zx**2 - Zy**2)
        Z = c*np.array([Zx, Zy, Zz])
        return np.array((X, Y, Z))

    def __convert_charmm__(self):
        # create new trajectory instance
        traj = pdbTrajectory()
        # set structure
        traj.set_structure(self.pdb)
        # Open the DCD trajectory file for reading.
        dcd = DCDFile(self.dcd)
        # set boundary conditions
        if dcd.has_pbc_data:
            traj._boundaryConditions = PeriodicBoundaries()
        else:
            traj._boundaryConditions = InfiniteBoundaries()
        # set indexes
        if self.indexes is None:
            self.indexes = range(dcd.numberOfConfigurations)
        elif  self.indexes[-1]>=dcd.numberOfConfigurations:
            Logger.warn("Some of the given indexes exceed '%s' which is the number of configurations in dcd file"%dcd.numberOfConfigurations)
            self.indexes = [index for index in self.indexes if index<dcd.numberOfConfigurations]
        # check number of atoms in dcd and structure
        assert dcd.natoms ==  traj.numberOfAtoms, Logger.error("pdb file and dcd file must have the same number of atoms")
        # The starting step number.
        step = dcd.istart
        # The step increment.
        stepIncrement = dcd.nsavc
        # The MD time steps round it to 6 decimals to avoid noise.
        dt = np.around(dcd.delta, 6)
        # store trajectory info
        info = {}
        info["software"]='charmm'
        info["software_version"] = dcd.charmmVersion
        traj._info = info
        # The cell parameters a, b, c, alpha, beta and gamma (stored in |unit_cell|)
        # and the x, y and z values of the first frame.
        # log conversion start
        self.status(0, dcd.fileSize)
        confIdx = -1
        while self.indexes:
            confIdx += 1
            if isinstance(self.indexes, list):
                idx = self.indexes.pop(0)
            else:
                idx = confIdx
            # check configuration number
            while confIdx < idx:
                try:
                    dcd.skip_step()
                except:
                    Logger.warn("file reading ended unexpectedly. Trajectory conversion stopped. all recorded data are still valid")
                    break
                confIdx += 1
            # read step
            try:
                unit_cell, x, y, z = dcd.read_step()
            except:
                Logger.warn("file reading ended unexpectedly. Trajectory conversion stopped. all recorded data are still valid")
                break
            # append coordinates
            traj._coordinates.append(np.transpose([x,y,z]))
            # append boundary conditions
            if dcd.has_pbc_data:
                traj._boundaryConditions.set_vectors(self.__unit_cell_to_basis_vectors__(*unit_cell))
            else:
                traj._boundaryConditions.set_vectors()
            # append time
            traj._time.append(confIdx*stepIncrement*dt)
            # log status
            self.status(dcd.currentPosition, dcd.fileSize)
        return traj

    def convert(self):
        if self.format in ('charmm','namd'):
            self.trajectory = self.__convert_charmm__()
        else:
            raise Logger.error("unsupported dcd format")
        self.trajectory._filePath = self.dcd
        return self.trajectory
