from __future__ import print_function
import os
import tempfile
import copy

import numpy as np

from pdbparser import pdbparser
from pdbparser.log import Logger
from pdbparser.Utilities.Database import __ATOM__
from pdbparser.Utilities.Construct import AmorphousSystem
from pdbparser.Utilities.Collection import get_path
from pdbparser.Utilities.Simulate import Simulation


Ar1 = copy.deepcopy(__ATOM__)
Ar1['atom_name'] = "Ar1"
Ar1['residue_name'] = "Ar"
Ar1['element_symbol'] = "Ar"
Ar2 = copy.deepcopy(Ar1)
Ar2['atom_name'] = "Ar2"
Ar2["coordinates_x"] = 1
pdbAr = pdbparser()
pdbAr.records = [Ar1]

boxSize = np.array([20,20,20])
pdb = AmorphousSystem([pdbAr], boxSize = boxSize, interMolecularMinimumDistance = 2.5, periodicBoundaries = True).construct().get_pdb()

# create simulation
sim = Simulation(pdb, logStatus = True, logExport = False,
                 numberOfSteps = 100, outputFrequency = 100,
                 boxVectors = boxSize, foldCoordinatesIntoBox = True,
                 exportInitialConfiguration = True, outputPath = tempfile.mktemp(".xyz"))


# initial parameters
sim.bonds_indexes = []
sim.nBondsThreshold = [[] for ids in pdb.indexes]
sim.angles_indexes = []
sim.dihedrals_indexes = []
sim.atomsCharge *= 0

# minimize energy
#Logger.info("minimization at %s fm per step" % (sim.timeStep) )
#sim.outputFrequency = 1
#sim.minimize_steepest_descent(99)

# equilibration
sim.exportInitialConfiguration = True
sim.outputFrequency = 1
sim.logExport = True
sim.timeStep = 0.1
Logger.info("equilibration at %s fm per step" % (sim.timeStep) )
sim.simulate(100)

# production
sim.exportInitialConfiguration = False
sim.outputFrequency = 1
sim.logExport = True
sim.timeStep = 1
Logger.info("production at %s fm per step" % (sim.timeStep) )
sim.simulate(3000, initializeVelocities = False)


# visualize molecule
sim.visualize_trajectory(sim.outputPath)
