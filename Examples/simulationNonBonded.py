from __future__ import print_function
import os
import tempfile
import copy

import numpy as np

from pdbparser import pdbparser
from pdbparser.log import Logger
from pdbparser.Utilities.Database import __ATOM__
from pdbparser.Utilities.Collection import get_path
from pdbparser.Utilities.Simulate import Simulation

at1 = copy.deepcopy(__ATOM__)
at2 = copy.deepcopy(__ATOM__)
at1['atom_name'] = "h1"
at2['atom_name'] = "h2"
at1['residue_name'] = "h2"
at2['residue_name'] = "h2"
at1['element_symbol'] = "h"
at2['element_symbol'] = "h"
at1['coordinates_x'] = -0.125
at2['coordinates_x'] =  0.125

# import molecule
pdb1 = pdbparser()
pdb1.records = [at1, at2]

# create simulation
sim = Simulation(pdb1, logStatus = False, logExport = False,
                 stepTime = 0.2, numberOfSteps = 10, outputFrequency = 1,
                 exportInitialConfiguration = True, outputPath = tempfile.mktemp(".xyz"))
# remove all bonded interactions
sim.bonds_indexes = []
sim.angles_indexes = []
sim.dihedrals_indexes = []
sim.nBondsThreshold = [[],[]]
# setting charges to 0
sim.atomsCharge = [0,0]

# initial parameters
Logger.info("minimizing %s steps at %s fm per step, with atoms charge %s, VDW forces push atoms to equilibrium distance %s" % (sim.numberOfSteps, sim.timeStep, sim.atomsCharge, 2*sim.__LJ__['h']['rmin/2']) )
sim.minimize_steepest_descent()

# add charges and change stepTime
sim.atomsCharge = [0.15,0.15]
sim.stepTime = 0.02
sim.exportInitialConfiguration = False

# re-minimize parameters
Logger.info("minimizing %s steps at %s fm per step, with atoms charge %s, VDW forces push atoms to equilibrium distance %s" % (sim.numberOfSteps, sim.timeStep, sim.atomsCharge, 2*sim.__LJ__['h']['rmin/2']) )
sim.minimize_steepest_descent()

# add charges and change stepTime
sim.atomsCharge = [-0.15,0.15]
#
# # re-minimize parameters
Logger.info("minimizing %s steps at %s fm per step, with atoms charge %s, VDW forces push atoms to equilibrium distance %s" % (sim.numberOfSteps, sim.timeStep, sim.atomsCharge, 2*sim.__LJ__['h']['rmin/2']) )
sim.minimize_steepest_descent()

# add charges and change stepTime
sim.atomsCharge = [-0.15,-0.15]

# re-minimize parameters
Logger.info("minimizing %s steps at %s fm per step, with atoms charge %s, VDW forces push atoms to equilibrium distance %s" % (sim.numberOfSteps, sim.timeStep, sim.atomsCharge, 2*sim.__LJ__['h']['rmin/2']) )
sim.minimize_steepest_descent()

# add charges and change stepTime
sim.atomsCharge = [0.15,-0.15]

# re-minimize parameters
Logger.info("minimizing %s steps at %s fm per step, with atoms charge %s, VDW forces push atoms to equilibrium distance %s" % (sim.numberOfSteps, sim.timeStep, sim.atomsCharge, 2*sim.__LJ__['h']['rmin/2']) )
sim.minimize_steepest_descent()

# minimze molecule
sim.visualize_trajectory(sim.outputPath)
