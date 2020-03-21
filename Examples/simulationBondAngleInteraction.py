from __future__ import print_function
import os
import tempfile

import numpy as np

from pdbparser.log import Logger
from pdbparser.Utilities.Collection import get_path
from pdbparser import pdbparser
from pdbparser.Utilities.Simulate import Simulation


# import molecule
pdb = pdbparser(os.path.join(get_path("pdbparser"),"Data/WATER.pdb" ) )

# create simulation
sim = Simulation(pdb, logStatus = False, logExport = False,
                 numberOfSteps = 100, outputFrequency = 1,
                 outputPath = tempfile.mktemp(".xyz"))

# initial parameters
Logger.info("minimizing 100 step with H-O-H angle %s and O-H bond %s" % (sim.__ANGLE__['h o h']['theta0'], sim.__BOND__['h o']['b0']))
sim.minimize_steepest_descent()

# change parameters to 120
sim.__ANGLE__['h o h']['theta0'] = 120
sim.set_angles_parameters()
sim.set_bonds_parameters()
Logger.info("minimizing 100 step with H-O-H angle %s and O-H bond %s" % (sim.__ANGLE__['h o h']['theta0'], sim.__BOND__['h o']['b0']))
sim.minimize_steepest_descent()

# change parameters to 120
sim.__BOND__['h o']['b0'] = 3
sim.set_angles_parameters()
sim.set_bonds_parameters()
Logger.info("minimizing 100 step with H-O-H angle %s and O-H bond %s" % (sim.__ANGLE__['h o h']['theta0'], sim.__BOND__['h o']['b0']))
sim.minimize_steepest_descent()


# change parameters to 120
sim.__ANGLE__['h o h']['theta0'] = 10
sim.__BOND__['h o']['b0'] = 0.92
sim.set_angles_parameters()
sim.set_bonds_parameters()
Logger.info("minimizing 100 step with H-O-H angle %s and O-H bond %s" % (sim.__ANGLE__['h o h']['theta0'], sim.__BOND__['h o']['b0']))
sim.minimize_steepest_descent()

# change parameters to 120
sim.__ANGLE__['h o h']['theta0'] = 104.52
sim.__BOND__['h o']['b0'] = 0.92
sim.set_angles_parameters()
sim.set_bonds_parameters()
Logger.info("minimizing 100 step with H-O-H angle %s and O-H bond %s" % (sim.__ANGLE__['h o h']['theta0'], sim.__BOND__['h o']['b0']))
sim.minimize_steepest_descent()


# visualize molecule
sim.visualize_trajectory(sim.outputPath)
